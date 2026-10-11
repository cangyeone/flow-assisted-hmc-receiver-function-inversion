# Our CPU-only archived research implementation; experimental, not default inference.
"""Remote GPU calibration, physical-gradient training and matched checks."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import argparse, json, hashlib, platform, time
from pathlib import Path
import numpy as np
import torch
import scipy
from scipy.optimize import minimize
from scipy.stats import qmc
from rf_physics_surrogate import ROOT, OUT, DeviceBasinTarget, ConditionalTarget, LocalLikelihoodNetwork, load_calibrated, box_jacobian
from rf_surrogate_hmc import surrogate_hmc_sample
from neural_rf_experiment import noise_covariance, write_json, sha, plain
from basin_rf import BasinTarget, event_data
from fit_refinement import load_events
from grtm.rf_hmc import hmc_sample, rank_diagnostics
import grtm.rf_cpu as grtm_rf_cpu
import grtm.rf_hmc as grtm_rf_hmc
torch.set_num_threads(2)
torch.set_default_dtype(torch.float64)

def sync(device):
    pass

def build_target(case, device):
    if case == 'conditional':
        folder = ROOT / 'results/neural-rf-v1/in-distribution'
        cfg = json.loads((folder / 'protocol.json').read_text())
        observed = np.load(folder / 'observations.npz')['observed']
        target = ConditionalTarget(observed, cfg['p'], cfg['p_sd'], cfg['sigma'][0], noise_covariance(), device)
        initial = np.array([40.0, 1.78])
        source = folder / 'observations.npz'
        info = dict(model='fixed-hs/Vs H/kappa conditional', rf_sha256=sha(source), truth=cfg['truth'], p=cfg['p'], p_sd=cfg['p_sd'])
    else:
        data, groups = load_events(case)
        full, _ = event_data(data, groups, 'train')
        source = ROOT / 'results/basin-assessment-v1/real' / f'{case}.json'
        saved = json.loads(source.read_text())
        key = saved['selected']
        if not key.startswith('basin_'):
            raise ValueError('Prespecified regional comparison requires an existing sediment-aware target')
        c = next((c for c in saved['checks'] if c['key'] == key))
        target = DeviceBasinTarget(full, c['model'], device)
        initial = np.array(c['full_candidates'][0]['x'])
        info = dict(model=key, source_sha256=sha(source), rf_sha256=hashlib.sha256(full['rf'].tobytes()).hexdigest(), event_archive_sha256=sha(ROOT / 'data/source' / f'ears-{case}.zip'), processed_sha256=sha(ROOT / 'data/processed' / f'{case}.npz'), fitting_event_members=[e['member'] for g in groups for e in g if e['split'] == 'train'], outer_event_members=[e['member'] for g in groups for e in g if e['split'] == 'test'], covariance='Original diagonal composite density with power 0.2 and original priors; no calibration coverage claim')
    return (target, initial, info)

def smoke(case, device):
    target, x0, info = build_target(case, device)
    cpu, _, _ = build_target(case, 'cpu')
    rng = np.random.default_rng(32100)
    span = target.high - target.low
    points = np.clip(x0 + rng.normal(size=(4, len(x0))) * 0.0002 * span, target.low + 1e-07 * span, target.high - 1e-07 * span)
    x = torch.tensor(points, dtype=torch.float64, device=device, requires_grad=True)
    lp = target(x)
    g = torch.autograd.grad(lp.sum(), x)[0]
    xc = torch.tensor(points, requires_grad=True)
    lc = cpu(xc)
    gc = torch.autograd.grad(lc.sum(), xc)[0]
    np.testing.assert_allclose(lp.detach().cpu(), lc.detach(), rtol=2e-10, atol=2e-09)
    np.testing.assert_allclose(g.detach().cpu(), gc.detach(), rtol=2e-07, atol=2e-06)
    if case != 'conditional':
        d, groups = load_events(case)
        full, _ = event_data(d, groups, 'train')
        old = BasinTarget(full, target.kind)
        with torch.no_grad():
            old_lp = old(xc)
            old_rf = old.predict(xc)
            new_rf = cpu.predict(xc)
        np.testing.assert_allclose(old_lp, lc.detach(), rtol=1e-11, atol=1e-10)
        np.testing.assert_allclose(old_rf, new_rf, rtol=0, atol=1e-11)
    OUT.mkdir(parents=True, exist_ok=True)
    print(case, 'CPU/CPU density and gradient alignment passed', flush=True)

def freeze(case, target, info, device):
    folder = OUT / case
    folder.mkdir(parents=True, exist_ok=True)
    p = dict(version=2, case=case, device=device, gpu=None, runtime=dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__, scipy=scipy.__version__, host=platform.node(), grtm_rf_source_sha256=sha(grtm_rf_cpu.__file__), hmc_source_sha256=sha(grtm_rf_hmc.__file__), native_library_sha256=None), train=2048 if case == 'conditional' else 4096, validation=512, test=512, seeds=dict(train=32101, validation=32102, test=32103, network=32104), epochs=1200, batch=256, width=128, learning_rate=0.001, gradient_loss_weight=1.0, local_coordinate='Logit box coordinates whitened using finite differences of exact physical first derivatives; eigenvalue floor 0.2, ceiling 1e8', sampling_scales=[0.25, 0.7, 1.2, 2.0], sampling_scale_probabilities=[0.1, 0.3, 0.5, 0.1], loss='Physical energy and gradient MSE in whitened coordinates, weights 1/(1+abs(E-E0)/20)^2. Validation selects checkpoint. No classification or automatic bad labels.', selection='Best weighted validation energy/gradient loss; fixed 1200 epochs, test never chooses weights or architecture.', sampling=dict(steps=2200, burn_in=800, leapfrog_steps=8, target_accept=0.9, inverse_mass='Same frozen teacher curvature for all arms', seeds=[32120, 32121, 32122] if case == 'conditional' else [32120]), arms=['physical HMC', 'physical quadratic proposal with exact correction', 'neural proposal with exact correction'], scope='Observation-specific local likelihood network, not a new universal arbitrary-layer classifier. Each physical parameter-to-layer map can have any fixed layer count; ordinary HMC dimension remains fixed. Exact correction keeps original posterior; no extra geological information or global-mode claim.', info=info, names=target.names, bounds=[target.low, target.high], sources={n: sha(ROOT / 'scripts' / n) for n in ('rf_physics_surrogate.py', 'rf_surrogate_hmc.py', 'train_rf_physics_surrogate.py', 'basin_rf.py', 'fit_refinement.py', 'cpu_hmc.py', 'neural_rf_experiment.py')})
    f = folder / 'protocol.json'
    if f.exists():
        assert json.loads(f.read_text()) == plain(p), 'Frozen protocol changed; preserve this run and create another version'
    else:
        write_json(f, p)
    return (folder, p)

def calibrate(target, x0, folder, p, device):
    if (folder / 'calibration.json').exists():
        return
    lo = torch.tensor(target.low, device=device)
    span = torch.tensor(target.high - target.low, device=device)
    zinit = torch.logit(((torch.tensor(x0, device=device) - lo) / span).clamp(1e-06, 0.999999))

    def energy_z(z):
        return -(target(lo + span * torch.sigmoid(z)) + box_jacobian(z, span))
    start = time.perf_counter()
    calls = 0

    def objective(z):
        nonlocal calls
        v = torch.tensor(z, device=device, requires_grad=True)
        e = energy_z(v[None])[0]
        g = torch.autograd.grad(e, v)[0]
        calls += 1
        return (float(e.detach()), g.detach().cpu().numpy())
    opt = minimize(objective, zinit.cpu().numpy(), jac=True, method='L-BFGS-B', options=dict(maxiter=150, ftol=1e-11, gtol=1e-06))
    z0 = opt.x
    dim = len(z0)
    h = 0.001
    points = np.r_[z0[None], z0[None] + h * np.eye(dim), z0[None] - h * np.eye(dim)]
    energies = []
    gradients = []
    for first in range(0, len(points), 8):
        v = torch.tensor(points[first:first + 8], device=device, requires_grad=True)
        e = energy_z(v)
        g = torch.autograd.grad(e.sum(), v)[0]
        energies.append(e.detach().cpu().numpy())
        gradients.append(g.detach().cpu().numpy())
    E = np.concatenate(energies)
    G = np.concatenate(gradients)
    H = (G[1:1 + dim] - G[1 + dim:]).T / (2 * h)
    H = (H + H.T) / 2
    eigen, U = np.linalg.eigh(H)
    positive = np.clip(eigen, 0.2, 100000000.0)
    transform = U / np.sqrt(positive)[None, :]
    sync(device)
    seconds = time.perf_counter() - start
    write_json(folder / 'calibration.json', dict(z0=z0, transform=transform, energy0=E[0], linear=G[0] @ transform, lower=target.low, upper=target.high, width=p['width'], hessian_eigenvalues=eigen, regularized_eigenvalues=positive, optimizer_success=bool(opt.success), optimizer_message=str(opt.message), optimization_evaluations=calls, hessian_gradient_points=len(points), seconds=seconds, gradient_at_center=G[0], note='Common physical preconditioning/calibration, not independent observation or guaranteed global mode'))
    print('calibration', folder.name, seconds, 'seconds', opt.success, 'min eigen', eigen.min(), flush=True)

def configuration(folder):
    c = json.loads((folder / 'calibration.json').read_text())
    return {k: c[k] for k in ('z0', 'transform', 'energy0', 'linear', 'lower', 'upper', 'width')}

def generate(target, folder, p, device):
    c = configuration(folder)
    z0 = torch.tensor(c['z0'], device=device)
    T = torch.tensor(c['transform'], device=device)
    lo = torch.tensor(target.low, device=device)
    span = torch.tensor(target.high - target.low, device=device)
    for split in ('train', 'validation', 'test'):
        path = folder / f'{split}.npz'
        if path.exists():
            continue
        start = time.perf_counter()
        rng = np.random.default_rng(p['seeds'][split])
        n = p[split]
        dim = len(z0)
        scales = rng.choice(p['sampling_scales'], n, p=p['sampling_scale_probabilities'])
        y = rng.normal(size=(n, dim)) * scales[:, None]
        E = []
        G = []
        for first in range(0, n, 16):
            v = torch.tensor(y[first:first + 16], device=device, requires_grad=True)
            z = z0 + v @ T.T
            e = -(target(lo + span * torch.sigmoid(z)) + box_jacobian(z, span))
            g = torch.autograd.grad(e.sum(), v)[0]
            if not bool(torch.isfinite(e).all() & torch.isfinite(g).all()):
                raise FloatingPointError('Nonfinite exact teacher labels')
            E.append(e.detach().cpu().numpy())
            G.append(g.detach().cpu().numpy())
            if (first + 16) % 512 == 0:
                print(folder.name, split, first + 16, n, flush=True)
        np.savez_compressed(path, y=y, energy=np.concatenate(E), gradient=np.concatenate(G), scale=scales)
        sync(device)
        write_json(folder / f'{split}-generation.json', dict(count=n, seconds=time.perf_counter() - start, sha256=sha(path)))

def train(folder, p, device):
    if (folder / 'network.pt').exists():
        return
    torch.manual_seed(p['seeds']['network'])
    model = LocalLikelihoodNetwork(**configuration(folder)).to(device)

    def data(split):
        d = np.load(folder / f'{split}.npz')
        return [torch.tensor(d[k], device=device) for k in ('y', 'energy', 'gradient')]
    yt, et, gt = data('train')
    yv, ev, gv = data('validation')
    optimizer = torch.optim.AdamW(model.parameters(), lr=p['learning_rate'], weight_decay=1e-05)
    generator = torch.Generator(device=device).manual_seed(p['seeds']['network'] + 1)

    def loss(y, e, g, training):
        y = y.detach().requires_grad_()
        pred = model.energy_y(y)
        grad = torch.autograd.grad(pred.sum(), y, create_graph=training)[0]
        w = 1 / (1 + (e - model.energy0).abs() / 20) ** 2
        value = (w * (pred - e).square()).mean()
        derivative = (w[:, None] * (grad - g).square()).mean()
        return (value + p['gradient_loss_weight'] * derivative, value, derivative)
    best = np.inf
    history = []
    start = time.perf_counter()
    state = None
    for epoch in range(1, p['epochs'] + 1):
        order = torch.randperm(len(yt), generator=generator, device=device)
        total = 0.0
        for first in range(0, len(order), p['batch']):
            ix = order[first:first + p['batch']]
            l, _, _ = loss(yt[ix], et[ix], gt[ix], True)
            optimizer.zero_grad()
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            total += float(l.detach()) * len(ix)
        if epoch == 1 or epoch % 20 == 0:
            vl, ve, vg = loss(yv, ev, gv, False)
            v = float(vl.detach())
            entry = dict(epoch=epoch, train_loss=total / len(yt), validation_loss=v, validation_energy_loss=float(ve.detach()), validation_gradient_loss=float(vg.detach()))
            history.append(entry)
            if v < best:
                best = v
                state = {k: q.detach().cpu().clone() for k, q in model.state_dict().items()}
                best_epoch = epoch
            print(folder.name, 'epoch', epoch, 'train', entry['train_loss'], 'validation', v, flush=True)
    sync(device)
    meta = dict(selected_epoch=best_epoch, best_validation_loss=best, train_seconds=time.perf_counter() - start, device=device, gpu=None, torch=str(torch.__version__), host=platform.node(), protocol_sha256=sha(folder / 'protocol.json'), train_sha256=sha(folder / 'train.npz'), validation_sha256=sha(folder / 'validation.npz'), meaning='Local approximation to the exact observation-conditioned physical energy, not probability of geological truth')
    torch.save(dict(state_dict=state, configuration=configuration(folder), metadata=meta), folder / 'network.pt')
    write_json(folder / 'training.json', dict(metadata=meta, history=history))

def assess(folder, device):
    if (folder / 'assessment.json').exists():
        return
    model, _ = load_calibrated(folder / 'network.pt', device)
    d = np.load(folder / 'test.npz')
    y = torch.tensor(d['y'], device=device, requires_grad=True)
    E = torch.tensor(d['energy'], device=device)
    G = torch.tensor(d['gradient'], device=device)
    arms = {}
    for name, qonly in [('quadratic', True), ('neural', False)]:
        e = model.energy_y(y, quadratic_only=qonly)
        g = torch.autograd.grad(e.sum(), y)[0]
        de = (e - E).detach().cpu().numpy()
        dg = (g - G).detach().cpu().numpy()
        w = 1 / (1 + abs(d['energy'] - float(model.energy0)) / 20) ** 2
        gradient_relative = np.linalg.norm(dg, axis=1) / np.maximum(1.0, np.linalg.norm(d['gradient'], axis=1))
        rng = np.random.default_rng(32105)
        other = rng.permutation(len(de))
        exact_gap = d['energy'] - d['energy'][other]
        pred = e.detach().cpu().numpy()
        gap = pred - pred[other]
        keep = abs(exact_gap) > 1e-08
        arms[name] = dict(energy_rmse=float(np.sqrt(np.mean(de ** 2))), weighted_energy_rmse=float(np.sqrt(np.sum(w * de ** 2) / w.sum())), energy_abs_error_quantiles=np.quantile(abs(de), [0.5, 0.9, 0.99]), gradient_relative_quantiles=np.quantile(gradient_relative, [0.5, 0.9, 0.99]), weighted_gradient_rmse=float(np.sqrt(np.sum(w[:, None] * dg ** 2) / (w.sum() * dg.shape[1]))), pair_order_agreement=float(((gap[keep] > 0) == (exact_gap[keep] > 0)).mean()), test_points=len(de))
        np.savez_compressed(folder / f'{name}-assessment.npz', energy=pred, gradient=g.detach().cpu().numpy(), energy_error=de)
    write_json(folder / 'assessment.json', dict(arms=arms, checkpoint_sha256=sha(folder / 'network.pt'), test_sha256=sha(folder / 'test.npz'), scope='Independent local coordinates for the same supplied RF, not new unseen stations/structures or a universal classifier'))
    print('assessment', folder.name, plain(arms), flush=True)

def conditional_grid(folder, device):
    if folder.name != 'conditional' or (folder / 'grid-summary.json').exists():
        return
    net, _ = load_calibrated(folder / 'network.pt', device)
    reference = np.load(ROOT / 'results/neural-rf-v1/in-distribution/physical-grid-129.npz')
    axes = reference['axes']
    xy = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 2)
    lp = []
    with torch.no_grad():
        for first in range(0, len(xy), 1024):
            lp.append(net.log_physical(torch.tensor(xy[first:first + 1024], device=device)).cpu().numpy())
    lp = np.concatenate(lp).reshape(129, 129)
    w = np.exp(lp - lp.max())
    w[[0, -1], :] *= 0.5
    w[:, [0, -1]] *= 0.5
    w /= w.sum()
    mean = (xy * w.ravel()[:, None]).sum(0)
    sd = np.sqrt(((xy - mean) ** 2 * w.ravel()[:, None]).sum(0))
    wp = reference['weights']
    mp = (xy * wp.ravel()[:, None]).sum(0)
    sp = np.sqrt(((xy - mp) ** 2 * wp.ravel()[:, None]).sum(0))
    np.savez_compressed(folder / 'neural-grid-129.npz', axes=axes, log_prob=lp, weights=w)
    write_json(folder / 'grid-summary.json', dict(neural_mean=mean, neural_sd=sd, mean_error_in_physical_sd=(mean - mp) / sp, sd_ratio=sd / sp, total_variation=float(0.5 * abs(w - wp).sum()), note='Full existing conditional box; standalone surrogate is approximate. Corrected chains still use the exact physical target.'))

def sample(target, folder, p, device):
    net, _ = load_calibrated(folder / 'network.pt', device)
    c = configuration(folder)
    T = np.array(c['transform'])
    metric = T @ T.T
    spec = p['sampling']
    settings = dict(steps=spec['steps'], burn_in=spec['burn_in'], leapfrog_steps=spec['leapfrog_steps'], target_accept=spec['target_accept'], inverse_mass=metric, step_size=0.2, device=device)
    for seed in spec['seeds']:
        rng = np.random.default_rng(seed)
        z0 = np.array(c['z0'])
        z = z0 + rng.normal(size=(4, len(z0))) * 0.7 @ T.T
        initial = target.low + (target.high - target.low) / (1 + np.exp(-z))
        for arm in ('physical', 'quadratic', 'neural'):
            path = folder / f'{arm}-seed{seed}.json'
            if path.exists():
                continue
            sync(device)
            start = time.perf_counter()
            callback = lambda n, total: print(folder.name, arm, seed, n, total, flush=True)
            if arm == 'physical':
                r = hmc_sample(target, initial, target.low, target.high, adapt_mass=False, seed=seed, progress=lambda n, total, lp: callback(n, total), progress_interval=500, **settings)
            else:
                r = surrogate_hmc_sample(target, lambda q: net.log_z(q, quadratic_only=arm == 'quadratic'), initial, target.low, target.high, seed=seed, progress=callback, **settings)
            sync(device)
            seconds = time.perf_counter() - start
            post = r['posterior'].detach().cpu().numpy()
            flat = post.reshape(-1, post.shape[-1])
            diag = rank_diagnostics(post)
            np.savez_compressed(path.with_suffix('.npz'), chain=r['chain'].detach().cpu().numpy(), log_prob=r['log_prob'].detach().cpu().numpy(), initial=initial)
            result = dict(arm=arm, seed=seed, sampling_seconds=seconds, mean=flat.mean(0), sd=flat.std(0), diagnostics=diag, acceptance=r['acceptance'].detach().cpu().numpy(), production_divergences=r['production_divergences'].detach().cpu().numpy(), gradient_evaluations=r['gradient_evaluations'], physical_gradient_evaluations=r.get('physical_gradient_evaluations', r['gradient_evaluations']), physical_endpoint_evaluations=r.get('physical_endpoint_evaluations', 0), checkpoint_sha256=sha(folder / 'network.pt'), same_target=True)
            if folder.name == 'conditional':
                ref = json.loads((ROOT / 'results/neural-rf-v1/in-distribution/summary.json').read_text())['references']['physical'][-1]
                result['mean_error_in_reference_sd'] = (flat.mean(0) - ref['mean']) / ref['sd']
                result['sd_ratio'] = flat.std(0) / ref['sd']
            write_json(path, result)
            print(folder.name, arm, 'seconds', seconds, 'converged', diag['converged'], flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('case', choices=['conditional', 'V20A', 'X20A'])
    ap.add_argument('stage', choices=['smoke', 'calibrate', 'generate', 'train', 'assess', 'sample', 'all'])
    ap.add_argument('--device', default='cpu')
    a = ap.parse_args()
    if a.stage == 'smoke':
        smoke(a.case, a.device)
        return
    target, x0, info = build_target(a.case, a.device)
    folder, p = freeze(a.case, target, info, a.device)
    if a.stage in ('calibrate', 'all'):
        calibrate(target, x0, folder, p, a.device)
    if a.stage in ('generate', 'all'):
        generate(target, folder, p, a.device)
    if a.stage in ('train', 'all'):
        train(folder, p, a.device)
    if a.stage in ('assess', 'all'):
        assess(folder, a.device)
        conditional_grid(folder, a.device)
    if a.stage in ('sample', 'all'):
        sample(target, folder, p, a.device)
if __name__ == '__main__':
    main()
