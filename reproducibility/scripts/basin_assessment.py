# Our CPU research program; ported imports use the public GRTM RF preview.
"""Preserve v3 targets; assess sediment physics using inner events only."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import argparse, hashlib, json, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from fit_refinement import ROOT, RefinedTarget, load_events, partition_data, optimize as old_optimize, score, plain
from basin_rf import KINDS, BasinTarget, event_data, optimize, choose, profile
from synthetic_fit_check import generate_rf
from grtm import JointRFForward
from grtm.rf_cpu import _surface_torch
OUT = ROOT / 'results/basin-assessment-v1'
DEPTH = np.arange(0.05, 50.0, 0.1)

def baseline_checks(original, groups):
    inner, validation, _ = partition_data(original, groups)
    fullinner, _ = event_data(original, groups, 'inner')
    checks = []
    for kind, covariance in [('legacy', 'full'), ('flex5', 'diagonal')]:
        target = RefinedTarget(inner if kind == 'legacy' else fullinner, kind, covariance=covariance)
        candidates, secs = old_optimize(target, 16, [1.5, 2.4, 40, 1.78], maxiter=400)
        with torch.no_grad():
            pred = target.predict(torch.tensor(candidates[0]['x'])[None], target.make_forward(validation['p']))[0].numpy()
        checks.append(dict(key=kind + '_' + covariance, model=kind, covariance=covariance, names=target.names, bounds=[target.low, target.high], inner_candidates=candidates, inner_seconds=secs, validation=score(pred, validation['rf'], original['fit_mask'])))
    return checks

def comparison(original, groups, oldchecks):
    inner, _ = event_data(original, groups, 'inner')
    valid, _ = event_data(original, groups, 'validation')
    checks = [{k: v for k, v in c.items() if k in ['key', 'model', 'covariance', 'names', 'bounds', 'inner_candidates', 'inner_seconds', 'validation']} for c in oldchecks]
    for kind in KINDS:
        target = BasinTarget(inner, kind)
        candidates, seconds = optimize(target)
        with torch.no_grad():
            prediction = target.predict(torch.tensor(candidates[0]['x'])[None], target.event_forward(valid))[0].numpy()
        c = dict(key='basin_' + kind, model=kind, covariance='diagonal_composite_power_0.2', names=target.names, bounds=[target.low, target.high], inner_candidates=candidates, inner_seconds=seconds, validation=score(prediction, valid['rf'], original['fit_mask']))
        checks.append(c)
        print(kind, 'inner RMSE', c['validation']['rmse'], flush=True)
    selected = choose(checks)
    return (selected, checks)

def refit(original, groups, c):
    full, _ = event_data(original, groups, 'train')
    test, _ = event_data(original, groups, 'test')
    if c['key'].startswith('basin_'):
        target = BasinTarget(full, c['model'])
        candidates, seconds = optimize(target, warm=c['inner_candidates'][0]['x'])
        forward_test = target.event_forward(test)
    else:
        target = RefinedTarget(original if c['model'] == 'legacy' else full, c['model'], covariance=c['covariance'])
        candidates, seconds = old_optimize(target, 16, [1.5, 2.4, 40, 1.78], warm=c['inner_candidates'][0]['x'], maxiter=400)
        forward_test = target.make_forward(test['p'])
    x = torch.tensor(candidates[0]['x'])[None]
    with torch.no_grad():
        train = target.predict(x)[0].numpy()
        prediction = target.predict(x, forward_test)[0].numpy()
        layers = target.model(x)[0].numpy()
    c.update(full_candidates=candidates, full_seconds=seconds, train=score(train, original['rf'], original['fit_mask']), test=score(prediction, test['rf'], original['fit_mask']), layers=layers)
    if c['key'].startswith('basin_'):
        wider = BasinTarget(full, c['model'], nfft=2048)
        with torch.no_grad():
            refined = wider.predict(x, wider.event_forward(test))[0].numpy()
        c['nfft1024_2048_max_test_rf_difference'] = float(np.max(np.abs(prediction - refined)))
    return (train, prediction, layers)

def uncertainty(groups, which):
    return np.stack([np.std([e['rf'] for e in g if e['split'] == which], axis=0, ddof=1) / np.sqrt(sum((e['split'] == which for e in g))) for g in groups])

def run_station(station):
    folder = OUT / 'real'
    folder.mkdir(exist_ok=True, parents=True)
    if (folder / f'{station}.json').exists():
        return
    original, groups = load_events(station)
    old = json.loads((ROOT / 'results/fit-refinement-v3' / f'{station}.json').read_text())
    selected, checks = comparison(original, groups, old['checks'])
    c = next((c for c in checks if c['key'] == selected))
    saved = dict(np.load(ROOT / 'results/fit-refinement-v3' / f'{station}-predictions.npz'))
    if selected.startswith('basin_'):
        train, test, layers = refit(original, groups, c)
    else:
        previous = next((c for c in old['checks'] if c['key'] == selected))
        c.update(full_candidates=previous['full_candidates'], full_seconds=0.0, train=previous['train'], test=previous['test'])
        train = saved[selected + '_train']
        test = saved[selected + '_test']
        t = RefinedTarget(original, c['model'], covariance=c['covariance'])
        layers = t.model(torch.tensor(c['full_candidates'][0]['x'])[None])[0].detach().numpy()
        c['layers'] = layers
    _, _, splits = partition_data(original, groups)
    oldchosen = next((c for c in old['checks'] if c['selected']))
    oldtarget = RefinedTarget(original, oldchosen['model'], covariance=oldchosen['covariance'])
    oldlayers = oldtarget.model(torch.tensor(oldchosen['full_candidates'][0]['x'])[None])[0].detach().numpy()
    arrays = dict(time=original['time'], rf_train=original['rf'], rf_test=original['test_rf'], train_se=uncertainty(groups, 'train'), test_se=uncertainty(groups, 'test'), original_train=saved['original_map_train'], original_test=saved['original_map_test'], compact_train=saved[old['selected'] + '_train'], compact_test=saved[old['selected'] + '_test'], selected_train=train, selected_test=test, layers=layers, depth=DEPTH, selected_profile=profile(layers, DEPTH), compact_profile=profile(oldlayers, DEPTH), p_train=original['p'], p_test=original['test_p'])
    result = dict(station=station, selected=selected, previous_selected=old['selected'], checks=checks, split=splits, original_test=score(arrays['original_test'], original['test_rf'], original['fit_mask']), compact_test=score(arrays['compact_test'], original['test_rf'], original['fit_mask']), selected_test=score(test, original['test_rf'], original['fit_mask']), selected_train=score(train, original['rf'], original['fit_mask']), source_sha256=old['source_sha256'], target_sha256=target_hash(), script_sha256=script_hash(), scope='Continued exploratory MAP assessment; original event splits, no new blind validation, no posterior claim')
    np.savez_compressed(folder / f'{station}.npz', **arrays)
    (folder / f'{station}.json').write_text(json.dumps(plain(result), indent=2))
    print(station, selected, result['selected_test'], flush=True)

def synthetic_data(case, noise, seed):
    hs, H = (0.8, 40.0) if case == 'shallow' else (4.0, 40.0)
    v0, v1, ks, rho, qs = (0.45, 2.15, 2.45, 1.95, 60.0) if case == 'shallow' else (0.65, 2.6, 2.15, 2.2, 30.0)
    edges = np.linspace(0, hs, 33)
    middle = (edges[:-1] + edges[1:]) / 2
    vs = v0 + (v1 - v0) * middle / hs
    sed = np.c_[edges[:-1], ks * vs, vs, np.full_like(vs, rho), np.full_like(vs, 2 * qs), np.full_like(vs, qs)]
    edges = np.linspace(hs, H, 65)
    vs = 3.45 + 0.4 * ((edges[:-1] + edges[1:]) / 2 - hs) / (H - hs)
    crust = np.c_[edges[:-1], 1.78 * vs, vs, np.full_like(vs, 2.8), np.full_like(vs, np.inf), np.full_like(vs, np.inf)]
    truth = np.vstack((sed, crust, [H, 8.0, 4.5, 3.3, np.inf, np.inf]))
    tau = 0.08
    rays = np.r_[np.linspace(0.041, 0.048, 16), np.linspace(0.051, 0.064, 16), np.linspace(0.068, 0.083, 16)]

    def generate(nfft):
        f = JointRFForward(rays, gaussians=[2.5], periods=[0.0], dt=0.05, n_samples=601, t0=-2.0, nfft=nfft, device='cpu', attenuation='constant_q')
        with torch.no_grad():
            x = f.prepare_layers(torch.tensor(truth)[None])
            r, z = _surface_torch(x, f.p, f.omega, 'constant_q')
            power = z.abs().square()
            den = torch.maximum(power, f.water_level * power.amax(-1, keepdim=True))
            norm = torch.fft.irfft(power / den * f.filters[0], n=nfft)[..., :1]
            y = torch.fft.irfft(r * z.conj() / den * f.filters[0] / norm * f.shift * torch.exp(-1j * f.omega * tau), n=nfft)[0, :, :601].numpy()[:, ::4]
            return y / y[:, 10:11]
    exact = generate(4096)
    refine = float(np.max(np.abs(exact - generate(8192))))
    assert refine < 0.002, refine
    t = np.arange(-2, 28.01, 0.2)
    rng = np.random.default_rng(seed)
    corr = 0.85 * np.exp(-0.5 * ((t[:, None] - t[None, :]) / 0.35) ** 2) + 0.15 * np.eye(len(t))
    obs = exact + noise * (rng.normal(size=exact.shape) @ np.linalg.cholesky(corr).T)
    obs /= obs[:, 10:11]
    groups = [[dict(rf=obs[b * 16 + j], exact=exact[b * 16 + j], p=rays[b * 16 + j], member=f'{case}-{b}-{j}', split='test' if j % 4 == 0 else 'train') for j in range(16)] for b in range(3)]
    original = dict(time=t, fit_mask=(t >= 0.799) & (t <= 26.001))
    original, _ = event_data(original, groups, 'train', full=False)
    return (original, groups, truth, dict(hs=hs, H=H, Vss_surface=v0, Vss_base=v1, kappa_s=ks, rho_s=rho, Qs=qs, tau=tau, fft_refinement=refine))

def run_synthetic(case, noise, seed):
    folder = OUT / 'synthetic'
    folder.mkdir(exist_ok=True, parents=True)
    key = f'{case}-noise{noise:.3f}-seed{seed}'
    if (folder / f'{key}.json').exists():
        return
    original, groups, truth, parameters = synthetic_data(case, noise, seed)
    base = baseline_checks(original, groups)
    selected, checks = comparison(original, groups, base)
    arrays = {}
    actual = profile(truth, DEPTH)
    for key_model in set(['legacy_full', 'flex5_diagonal', selected]):
        c = next((c for c in checks if c['key'] == key_model))
        train, test, layers = refit(original, groups, c)
        arrays[key_model + '_train'] = train
        arrays[key_model + '_test'] = test
        arrays[key_model + '_profile'] = profile(layers, DEPTH)
        c['Vs_profile_rmse'] = float(np.sqrt(np.mean((profile(layers, DEPTH) - actual) ** 2)))
        c['shallow_Vs_rmse'] = float(np.sqrt(np.mean((profile(layers, DEPTH[DEPTH < parameters['hs']]) - profile(truth, DEPTH[DEPTH < parameters['hs']])) ** 2)))
        c['H_abs_error_km'] = float(abs(c['full_candidates'][0]['x'][2] - parameters['H']))
    test, gt = event_data(original, groups, 'test', full=False)
    arrays.update(time=original['time'], rf_train=original['rf'], rf_test=test['rf'], test_se=uncertainty(groups, 'test'), rf_test_exact=np.stack([np.mean([e['exact'] for e in g], 0) for g in gt]), depth=DEPTH, truth_profile=actual, truth_layers=truth)
    result = dict(case=case, noise=noise, seed=seed, truth=parameters, selected=selected, checks=checks, target_sha256=target_hash(), script_sha256=script_hash(), scope='Basin synthetic MAP model check with fine generating layers and event rays; shared approximate constant-Q physics, no calibrated uncertainty or HMC comparison')
    np.savez_compressed(folder / f'{key}.npz', **arrays)
    (folder / f'{key}.json').write_text(json.dumps(plain(result), indent=2))
    print(key, selected, flush=True)

def target_hash():
    return hashlib.sha256((ROOT / 'scripts/basin_rf.py').read_bytes()).hexdigest()

def script_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--station')
    p.add_argument('--case', choices=['shallow', 'deep'])
    p.add_argument('--noise', type=float)
    p.add_argument('--seed', type=int)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--cohort', choices=['real', 'synthetic'])
    p.add_argument('--stations', nargs='+')
    a = p.parse_args()
    OUT.mkdir(exist_ok=True, parents=True)
    if a.station:
        run_station(a.station)
        return
    if a.case:
        run_synthetic(a.case, a.noise, a.seed)
        return
    if not a.cohort:
        raise ValueError('Specify cohort or individual case')
    stations = a.stations or [m['station'] for m in json.loads((ROOT / 'data/processed/manifest.json').read_text())]
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), starts=16, maxiter=400, seed=1730, models=list(KINDS), basin_priors='kappa_s N(2.1,.45),rho_s N(2.2,.25),log_Qs N(log60,1), endpoint velocity difference SD2; v3 crust/roughness/arrival priors retained', event_prediction='Average individual normalized synthetic event RFs at actual fitting/validation/test ray parameters; nfft1024, refinement2048', selection='Minimum dimension within 2% of best inner-validation RMSE; every fifth training event; unchanged outer events and 0.8-26 s score', real_scope='All 42 stations assessed; previous outer results informed model development; continued exploratory predictive assessment', synthetic_cases=['shallow', 'deep'], noise=[0.025, 0.06], seeds=[200, 201, 202], target_sha256=target_hash(), script_sha256=script_hash(), legacy_target_sha256=hashlib.sha256((ROOT / 'scripts/fit_refinement.py').read_bytes()).hexdigest())
    frozen = OUT / 'protocol.json'
    if frozen.exists():
        previous = json.loads(frozen.read_text())
        assert previous['target_sha256'] == target_hash() and previous['script_sha256'] == script_hash()
    else:
        frozen.write_text(json.dumps(protocol, indent=2))
    jobs = [['--station', s] for s in stations] if a.cohort == 'real' else [['--case', c, '--noise', str(n), '--seed', str(s)] for c in protocol['synthetic_cases'] for n in protocol['noise'] for s in protocol['seeds']]

    def launch(job):
        label = job[1] if a.cohort == 'real' else f'{job[1]}-noise{float(job[3]):.3f}-seed{job[5]}'
        with (OUT / f'{a.cohort}-{label}.log').open('w') as f:
            r = subprocess.run([sys.executable, str(Path(__file__).resolve()), *job], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        print(a.cohort, label, r.returncode, flush=True)
        return dict(label=label, returncode=r.returncode)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        status = list(pool.map(launch, jobs))
    (OUT / f'{a.cohort}-status.json').write_text(json.dumps(status, indent=2))
    assert not any((s['returncode'] for s in status)), status
if __name__ == '__main__':
    main()
