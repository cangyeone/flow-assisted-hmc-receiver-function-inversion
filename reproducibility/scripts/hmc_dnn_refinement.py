# Our CPU-only archived research implementation; experimental, not default inference.
"""Physical HMC followed by frozen RF/model DNN constrained point refinement.

The second stage is a predictive point-estimate experiment, not a posterior
update. A zero-DNN-weight control has identical starts and trust constraints.
Only internal validation events select the weight; outer RF and truth are
scored after selection. Existing failed V1/V2 experiments remain untouched.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import argparse, json, time, platform, hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from scipy.stats import qmc
from neural_rf import ROOT, load_network
from neural_rf_experiment import sha, write_json, plain
from rf_physics_surrogate import DeviceBasinTarget, box_jacobian
from basin_rf import event_data, period_starts, profile
from basin_assessment import synthetic_data
from fit_refinement import load_events, score
from grtm.rf_hmc import hmc_sample, rank_diagnostics
import grtm.rf_cpu as grtm_rf_cpu
import grtm.rf_hmc as grtm_rf_hmc
torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)
OUT = ROOT / 'results/hmc-dnn-refine-v1'
WEIGHTS = [0.0, 0.3, 3.0]
CASES = ['shallow-low', 'deep-low', 'shallow-high', 'deep-high', 'V20A', 'X20A']
SOURCES = ['hmc_dnn_refinement.py', 'neural_rf.py', 'neural_rf_experiment.py', 'rf_physics_surrogate.py', 'basin_rf.py', 'basin_assessment.py', 'fit_refinement.py', 'cpu_hmc.py', 'synthetic_fit_check.py']

def sync(device):
    pass

def freeze(device):
    OUT.mkdir(parents=True, exist_ok=True)
    p = dict(version=1, cases=CASES, weights=WEIGHTS, checkpoint_sha256=sha(ROOT / 'results/neural-rf-v1/network.pt'), sources={s: sha(ROOT / 'scripts' / s) for s in SOURCES}, device=device, runtime=dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__, host=platform.node(), gpu=None, grtm_rf_sha256=sha(grtm_rf_cpu.__file__), hmc_sha256=sha(grtm_rf_hmc.__file__), native_sha256=None), synthetic=dict(noise_low=0.025, noise_high=0.06, seed=33021, fine_generator='Original basin generator: 32 sediment + 64 crust segments, dt .05 s, nfft4096, checked against8192. New noise seed; previously explored truth families. Inverse target damped, 3 sediment +4 crust layers.'), initialization=dict(starts=8, maxiter=250, seed=33022, policy='Physical-only Sobol/period-hint fits of the current fitting events; no previous full-training model initialization'), hmc=dict(steps=3000, burn=1200, chains=4, leapfrog=10, target_accept=0.95, adapt_mass=True, seed_base=33030), refinement=dict(starts=4, maxiter=150, trust_coordinate='Median logit center and shrinkage covariance from current retained HMC cloud', covariance_shrinkage=0.2, coordinate_limit=3.0, trust_penalty=0.02, physical_scale='max(1, HMC physical NLL 90%-10% spread)', neural_scale='max(.02, HMC frozen compatibility logit 90%-10% spread)', objective='physical_NLL / physical_scale + weight * softplus((median_logit - logit)/neural_scale) + .01*sum(whitened_logit_coordinate^2)', meaning='Heuristic constrained point refinement; classification score is not new independent geological evidence and does not define calibrated posterior intervals.'), selection='Pick positive weight only if its inner-validation RF RMSE is at least 1% smaller than the zero-weight control. Otherwise zero. Weight selected before full-training HMC/refinement and outer/truth scoring.', partitions='Original disjoint train/test events; every fifth train event is inner validation. Separate inner-fit HMC then separate full-training HMC. No full-training HMC chain or previous fitted parameters enter inner selection.', scope='Continued exploratory model development; 4 noisy synthetic cases in previously used basin families and 2 previously inspected field stations. All failed HMC/optimizer checks retained. Not blind geology validation or global-mode proof.')
    path = OUT / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == plain(p), 'Frozen experiment changed; use a new version'
    else:
        write_json(path, p)
    return p

def cache_data(folder, case):
    if (folder / 'data-info.json').exists():
        return json.loads((folder / 'data-info.json').read_text())
    if case in ('V20A', 'X20A'):
        original, groups = load_events(case)
        previous = ROOT / 'results/basin-assessment-v1/real' / f'{case}.json'
        selection = json.loads(previous.read_text())
        kind = next((c['model'] for c in selection['checks'] if c['key'] == selection['selected']))
        if not selection['selected'].startswith('basin_'):
            raise ValueError('Need a basin target')
        info = dict(case=case, kind=kind, cohort='field', previous_model_selection_sha256=sha(previous), archive_sha256=sha(ROOT / 'data/source' / f'ears-{case}.zip'), processed_sha256=sha(ROOT / 'data/processed' / f'{case}.npz'), note='Previously selected model kind fixed; no previously fitted parameters reused')
    else:
        basin, level = case.split('-')
        noise = 0.025 if level == 'low' else 0.06
        original, groups, truth, parameters = synthetic_data(basin, noise, 33021)
        np.savez_compressed(folder / 'truth.npz', layers=truth)
        write_json(folder / 'truth.json', parameters)
        info = dict(case=case, kind='damped', cohort='synthetic', seed=33021, noise=noise, truth_sha256=sha(folder / 'truth.npz'))
    memberships = {}
    hashes = {}
    for phase in ('inner', 'validation', 'train', 'test'):
        data, selected = event_data(original, groups, phase)
        members = [e['member'] for g in selected for e in g]
        memberships[phase] = members
        counts = np.array([len(g) for g in selected])
        rays = np.concatenate(data['event_p'])
        np.savez_compressed(folder / f'data-{phase}.npz', time=data['time'], fit_mask=data['fit_mask'], rf=data['rf'], cov=data['cov'], p=data['p'], rays=rays, counts=counts)
        hashes[phase] = sha(folder / f'data-{phase}.npz')
        if info['cohort'] == 'synthetic' and phase == 'test':
            np.savez_compressed(folder / 'test-noiseless.npz', rf=np.stack([np.mean([e['exact'] for e in g], 0) for g in selected]))
    assert set(memberships['inner']).isdisjoint(memberships['validation'])
    assert set(memberships['train']) == set(memberships['inner'] + memberships['validation'])
    assert set(memberships['train']).isdisjoint(memberships['test'])
    info.update(memberships=memberships, data_sha256=hashes)
    write_json(folder / 'data-info.json', info)
    return info

def data_target(folder, phase, kind, device):
    d = dict(np.load(folder / f'data-{phase}.npz'))
    cuts = np.r_[0, d['counts'].cumsum()]
    d['event_p'] = [d['rays'][cuts[j]:cuts[j + 1]] for j in range(3)]
    return DeviceBasinTarget(d, kind, device)

class CompatibilityGuide:

    def __init__(self, network, target):
        self.network = network
        d = target.data
        self.target = target
        self.rf = target.forward.tensor(d['rf'])
        self.p = target.forward.tensor(d['p'])
        self.ps = target.forward.tensor([np.std(a) for a in d['event_p']])
        self.sigma = target.forward.tensor(np.sqrt(np.median(np.diagonal(d['cov'], axis1=1, axis2=2), axis=1)))

    def __call__(self, x):
        return self.network.score(self.rf, self.target.model(x), self.p, self.sigma, tau=x[:, -1], p_sd=self.ps)

    def metadata(self):
        p, ps, sigma = [a.cpu().numpy() for a in (self.p, self.ps, self.sigma)]
        flags = []
        if np.any((ps < 0.0003) | (ps > 0.003)):
            flags.append('event ray spread outside V1 simulation range')
        if np.any((sigma < 0.015) | (sigma > 0.12)):
            flags.append('stack noise proxy outside V1 simulation range')
        if np.any((p < np.array([0.041, 0.051, 0.067])) | (p > np.array([0.049, 0.065, 0.083]))):
            flags.append('ray means outside V1 simulation range')
        return dict(p=p, p_sd=ps, sigma_proxy=sigma, domain_flags=flags)

def physical_initialization(target, p):
    spec = p['initialization']
    lo = target.low
    span = target.high - lo
    device = target.device
    starts = qmc.Sobol(len(lo), scramble=True, seed=spec['seed']).random_base2(3)
    hints = [(1.5, 2.4)] + [(min(9.0, 0.5 * t * v), v) for t in period_starts(target.data) for v in (0.5, 1.0, 2.0)]
    for i, (hs, vs) in enumerate(hints[:len(starts)]):
        x = [hs, vs, 40.0, 0.0, 0.0, 0.0, 3.4, 3.55, 3.7, 3.85, 1.78, 2.1, 2.2]
        if target.kind != 'sediment':
            x += [0.5]
        if target.kind == 'damped':
            x += [np.log(60.0)]
        x += [0.0]
        starts[i] = (np.array(x) - lo) / span
    starts = np.clip(starts, 0.0001, 0.9999)
    answers = []
    begin = time.perf_counter()
    tl = torch.tensor(lo, device=device)
    ts = torch.tensor(span, device=device)

    def objective(u):
        v = torch.tensor(u, device=device, requires_grad=True)
        loss = -target((tl + ts * v)[None])[0]
        g = torch.autograd.grad(loss, v)[0]
        if not bool(torch.isfinite(loss) & torch.isfinite(g).all()):
            raise FloatingPointError('Initialization density/gradient')
        return (float(loss.detach()), g.detach().cpu().numpy())
    for i, u in enumerate(starts):
        result = minimize(objective, u, jac=True, method='L-BFGS-B', bounds=[(0.0001, 0.9999)] * len(lo), options=dict(maxiter=spec['maxiter'], ftol=1e-09, gtol=1e-05))
        answers.append(dict(start=i, x=lo + span * result.x, nll=float(result.fun), success=bool(result.success), message=str(result.message), iterations=int(result.nit)))
    answers.sort(key=lambda a: a['nll'])
    sync(device)
    return (answers, time.perf_counter() - begin)

def hmc_stage(target, folder, phase, p, seed):
    path = folder / f'{phase}-hmc.json'
    cache = folder / f'{phase}-hmc.npz'
    if path.exists():
        return (json.loads(path.read_text()), dict(np.load(cache)))
    begin = time.perf_counter()
    answers, initial_seconds = physical_initialization(target, p)
    lo = target.forward.tensor(target.low)
    span = target.forward.tensor(target.high - target.low)
    initial = np.array([a['x'] for a in answers[:4]])
    u = np.clip((initial - target.low) / (target.high - target.low), 0.002, 0.998)
    initial = target.low + (target.high - target.low) * u
    center = np.log(u[0] / (1 - u[0]))
    dim = len(center)
    points = np.r_[center[None] + 0.002 * np.eye(dim), center[None] - 0.002 * np.eye(dim)]
    gradient = []
    for first in range(0, len(points), 8):
        q = torch.tensor(points[first:first + 8], device=target.device, requires_grad=True)
        density = target(lo + span * torch.sigmoid(q)) + box_jacobian(q, span)
        gradient.append(torch.autograd.grad(density.sum(), q)[0].detach().cpu().numpy())
    g = np.concatenate(gradient)
    H = -(g[:dim] - g[dim:]).T / 0.004
    H = (H + H.T) / 2
    e, U = np.linalg.eigh(H)
    metric = U / np.clip(e, 0.2, 100000000.0)[None, :] @ U.T
    sync(target.device)
    setup_seconds = time.perf_counter() - begin
    spec = p['hmc']
    begin = time.perf_counter()
    result = hmc_sample(target, initial, target.low, target.high, steps=spec['steps'], burn_in=spec['burn'], leapfrog_steps=spec['leapfrog'], target_accept=spec['target_accept'], inverse_mass=metric, adapt_mass=True, step_size=0.1, seed=seed, device=str(target.device), progress=lambda n, total, lp: print(folder.name, phase, 'HMC', n, total, flush=True), progress_interval=1000)
    sync(target.device)
    seconds = time.perf_counter() - begin
    chain = result['chain'].detach().cpu().numpy()
    lp = result['log_prob'].detach().cpu().numpy()
    post = chain[spec['burn']:]
    best_per_chain = np.array([post[np.argmax(lp[spec['burn']:, c]), c] for c in range(4)])
    median = np.median(post, axis=(0, 1))
    best = best_per_chain[np.argmax([lp[spec['burn']:, c].max() for c in range(4)])]
    np.savez_compressed(cache, chain=chain, log_prob=lp, initial=initial, best_per_chain=best_per_chain, median=median, best=best)
    record = dict(phase=phase, seed=seed, steps=spec['steps'], burn=spec['burn'], diagnostics=rank_diagnostics(post), acceptance=result['acceptance'], production_divergences=result['production_divergences'], setup_seconds=setup_seconds, physical_initialization_seconds=initial_seconds, sampling_seconds=seconds, initialization=answers, hessian_eigenvalues=e, interpretation='Candidate cloud only if unconverged; no validated posterior uncertainty')
    write_json(path, record)
    print(folder.name, phase, 'HMC diagnostics', record['diagnostics']['converged'], flush=True)
    return (plain(record), dict(np.load(cache)))

class RefinementObjective:

    def __init__(self, physical, guide, center, transform, physical_scale, neural_center, neural_scale, weight, trust_penalty=0.02):
        self.physical, self.guide = (physical, guide)
        self.center = center
        self.transform = transform
        self.physical_scale = physical_scale
        self.neural_center = neural_center
        self.neural_scale = neural_scale
        self.weight = weight
        self.trust_penalty = trust_penalty
        self.lo = center.new_tensor(physical.low)
        self.span = center.new_tensor(physical.high - physical.low)

    def x(self, y):
        return self.lo + self.span * torch.sigmoid(self.center + y @ self.transform.T)

    def __call__(self, y):
        value = -self.physical(self.x(y)) / self.physical_scale + 0.5 * self.trust_penalty * y.square().sum(-1)
        if self.weight:
            value = value + self.weight * F.softplus((self.neural_center - self.guide(self.x(y))) / self.neural_scale)
        return value

def refinement_geometry(target, guide, saved, burn):
    post = saved['chain'][burn:].reshape(-1, len(target.low))
    u = np.clip((post - target.low) / (target.high - target.low), 1e-12, 1 - 1e-12)
    q = np.log(u / (1 - u))
    center = np.median(q, axis=0)
    cov = np.cov(q, rowvar=False)
    cov = 0.8 * cov + 0.2 * np.diag(np.diag(cov)) + np.eye(len(center)) * 1e-05
    e, U = np.linalg.eigh(cov)
    T = U * np.sqrt(np.maximum(e, 1e-06))[None, :]
    physical_scale = max(1.0, float(np.quantile(-saved['log_prob'][burn:], [0.9, 0.1])[0] - np.quantile(-saved['log_prob'][burn:], [0.9, 0.1])[1]))
    ix = np.linspace(0, len(post) - 1, 256, dtype=int)
    values = []
    with torch.no_grad():
        for first in range(0, len(ix), 32):
            values.append(guide(torch.tensor(post[ix[first:first + 32]], device=target.device)).cpu().numpy())
    values = np.concatenate(values)
    neural_center = float(np.median(values))
    neural_scale = max(0.02, float(np.quantile(values, 0.9) - np.quantile(values, 0.1)))
    return dict(center=center, transform=T, physical_scale=physical_scale, neural_center=neural_center, neural_scale=neural_scale, neural_logit_quantiles=np.quantile(values, [0.1, 0.5, 0.9]), neural_context=guide.metadata())

def fine_stage(target, validation, network, saved, folder, phase, p):
    path = folder / f'{phase}-refinement.json'
    if path.exists():
        return json.loads(path.read_text())
    guide = CompatibilityGuide(network, target)
    geometry = refinement_geometry(target, guide, saved, p['hmc']['burn'])
    c = np.array(geometry['center'])
    T = np.array(geometry['transform'])
    inverse = np.linalg.inv(T)
    u = np.clip((saved['best_per_chain'] - target.low) / (target.high - target.low), 1e-12, 1 - 1e-12)
    starts = (np.log(u / (1 - u)) - c) @ inverse.T
    starts = np.clip(starts, -3.0, 3.0)
    rows = []
    device = target.device
    mask = (target.data['time'] >= 0.799) & (target.data['time'] <= 26.001)
    for weight in p['weights']:
        begin = time.perf_counter()
        objective = RefinementObjective(target, guide, torch.tensor(c, device=device), torch.tensor(T, device=device), geometry['physical_scale'], geometry['neural_center'], geometry['neural_scale'], weight)

        def f(y):
            leaf = torch.tensor(y, device=device, requires_grad=True)
            loss = objective(leaf[None])[0]
            g = torch.autograd.grad(loss, leaf)[0]
            if not bool(torch.isfinite(loss) & torch.isfinite(g).all()):
                raise FloatingPointError('Refinement loss/gradient')
            return (float(loss.detach()), g.detach().cpu().numpy())
        candidates = []
        for i, start in enumerate(starts):
            opt = minimize(f, start, jac=True, method='L-BFGS-B', bounds=[(-3.0, 3.0)] * len(c), options=dict(maxiter=p['refinement']['maxiter'], ftol=1e-10, gtol=1e-05))
            with torch.no_grad():
                x = objective.x(torch.tensor(opt.x, device=device)[None])[0].cpu().numpy()
            candidates.append(dict(start=i, x=x, y=opt.x, objective=float(opt.fun), success=bool(opt.success), iterations=int(opt.nit), message=str(opt.message)))
        best = min(candidates, key=lambda a: a['objective'])
        x = torch.tensor(best['x'], device=device)[None]
        with torch.no_grad():
            train = target.predict(x)[0].cpu().numpy()
            valid = validation.predict(x)[0].cpu().numpy()
            nll = float(-target(x)[0])
            logit = float(guide(x)[0])
        sync(device)
        seconds = time.perf_counter() - begin
        rows.append(dict(weight=weight, candidates=candidates, selected_start=best['start'], x=best['x'], y=best['y'], optimizer_success=best['success'], training=score(train, target.data['rf'], mask), validation=score(valid, validation.data['rf'], mask), physical_nll=nll, neural_logit=logit, seconds=seconds))
        print(folder.name, phase, 'fine weight', weight, 'seconds', seconds, 'validation', rows[-1]['validation']['rmse'], flush=True)
    record = dict(phase=phase, geometry=geometry, rows=rows, meaning='Constrained point optimization, no posterior samples or calibrated confidence intervals')
    write_json(path, record)
    return plain(record)

def choose_weight(rows):
    zero = next((r for r in rows if r['weight'] == 0.0))
    best = min(rows, key=lambda r: (r['validation']['rmse'], r['weight']))
    return best['weight'] if best['weight'] > 0 and best['validation']['rmse'] <= 0.99 * zero['validation']['rmse'] else 0.0

def final_scores(folder, info, target, test, full_hmc, full_fine, selected_weight):
    arrays = dict(time=target.data['time'], rf_train=target.data['rf'], rf_test=test.data['rf'], test_cov=test.data['cov'], fit_mask=target.data['fit_mask'])
    saved = np.load(folder / 'train-hmc.npz')
    models = {'HMC_best': saved['best'], 'HMC_median': saved['median']}
    for r in full_fine['rows']:
        models[f"fine_weight{r['weight']:g}"] = np.array(r['x'])
    depth = np.arange(0.05, 50.0, 0.1)
    rows = []
    mask = (target.data['time'] >= 0.799) & (target.data['time'] <= 26.001)
    truth = None
    if info['cohort'] == 'synthetic':
        truth = np.load(folder / 'truth.npz')['layers']
        parameters = json.loads((folder / 'truth.json').read_text())
        arrays['truth_profile'] = profile(truth, depth)
        arrays['rf_test_exact'] = np.load(folder / 'test-noiseless.npz')['rf']
    for name, x in models.items():
        xt = torch.tensor(x, device=target.device)[None]
        with torch.no_grad():
            train = target.predict(xt)[0].cpu().numpy()
            prediction = test.predict(xt)[0].cpu().numpy()
            layers = target.model(xt)[0].cpu().numpy()
        arrays[name + '_train'] = train
        arrays[name + '_test'] = prediction
        arrays[name + '_profile'] = profile(layers, depth)
        row = dict(method=name, x=x, train=score(train, target.data['rf'], mask), test=score(prediction, test.data['rf'], mask))
        if truth is not None:
            shallow = depth < parameters['hs']
            row.update(noiseless=score(prediction, arrays['rf_test_exact'], mask), Vs_profile_rmse=float(np.sqrt(np.mean((profile(layers, depth) - arrays['truth_profile']) ** 2))), sediment_Vs_rmse=float(np.sqrt(np.mean((profile(layers, depth[shallow]) - profile(truth, depth[shallow])) ** 2))), H_abs_error_km=float(abs(x[2] - parameters['H'])), hs_abs_error_km=float(abs(x[0] - parameters['hs'])))
        rows.append(row)
    arrays['depth'] = depth
    np.savez_compressed(folder / 'predictions.npz', **arrays)
    return dict(case=info['case'], cohort=info['cohort'], model=info['kind'], selected_weight=selected_weight, selected_method=f'fine_weight{selected_weight:g}', rows=rows, inner_hmc_converged=json.loads((folder / 'inner-hmc.json').read_text())['diagnostics']['converged'], full_hmc_converged=full_hmc['diagnostics']['converged'], full_hmc_divergences=full_hmc['production_divergences'], interpretation='Predictive point comparison; unresolved HMC clouds are not validated posteriors. Refinement cannot restore their uncertainty validity.', prediction_sha256=sha(folder / 'predictions.npz'))

def run(case, p, device):
    folder = OUT / case
    folder.mkdir(exist_ok=True)
    if (folder / 'result.json').exists():
        print(case, 'already complete', flush=True)
        return
    info = cache_data(folder, case)
    for phase, h in info['data_sha256'].items():
        assert sha(folder / f'data-{phase}.npz') == h
    network, _ = load_network(dtype=torch.float32)
    network.to(device)
    inner = data_target(folder, 'inner', info['kind'], device)
    valid = data_target(folder, 'validation', info['kind'], device)
    seed = p['hmc']['seed_base'] + 10 * CASES.index(case)
    inner_hmc, inner_saved = hmc_stage(inner, folder, 'inner', p, seed)
    inner_fine = fine_stage(inner, valid, network, inner_saved, folder, 'inner', p)
    selected = choose_weight(inner_fine['rows'])
    write_json(folder / 'selection.json', dict(selected_weight=selected, rule=p['selection'], input_sha256=sha(folder / 'inner-refinement.json')))
    del inner, valid
    full = data_target(folder, 'train', info['kind'], device)
    full_hmc, full_saved = hmc_stage(full, folder, 'train', p, seed + 1)
    full_fine = fine_stage(full, full, network, full_saved, folder, 'train', p)
    test = data_target(folder, 'test', info['kind'], device)
    result = final_scores(folder, info, full, test, full_hmc, full_fine, selected)
    write_json(folder / 'result.json', result)
    print(case, 'completed selected weight', selected, flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=CASES)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    p = freeze(args.device)
    for case in [args.case] if args.case else CASES:
        run(case, p, args.device)
if __name__ == '__main__':
    main()
