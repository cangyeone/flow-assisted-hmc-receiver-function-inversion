# Our CPU research program; ported imports use the public GRTM RF preview.
"""Synthetic RF fit/structure checks before the real-data interpretation.

Three fixed structures x two event-noise levels x three seeds. Forward truth
uses finer layers/sampling/FFT and event-specific rays. Outer event RFs are
never used for model selection. This is a MAP model check, not HMC superiority.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import argparse, hashlib, json, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from fit_refinement import ROOT, RefinedTarget, MODELS, stack, optimize, score, plain
from grtm import JointRFForward
from grtm.rf_cpu import _surface_torch
OUT = ROOT / 'results/synthetic-fit-v1'
DEPTH = np.arange(0.05, 50.0, 0.1)

def truth_layers(case):
    hs, H, vs_s, k = (1.5, 40.0, 2.0, 1.78)
    edges = np.unique(np.r_[np.linspace(hs, H, 65), 12.0, 22.0])
    mid = (edges[:-1] + edges[1:]) / 2
    if case == 'uniform':
        vs = np.full_like(mid, 6.5 / k)
        tau = 0.0
    elif case == 'gradient':
        vs = 3.1 + 0.9 * (mid - hs) / (H - hs)
        tau = 0.12
    elif case == 'lvz':
        vs = np.where(mid < 12.0, 3.6, np.where(mid < 22.0, 3.05, 3.85))
        tau = -0.08
    else:
        raise ValueError(case)
    crust = np.c_[edges[:-1], k * vs, vs, np.full_like(vs, 2.8)]
    return (np.vstack(([0, 1.8 * vs_s, vs_s, 2.5], crust, [H, 8.0, 4.5, 3.3])), tau)

def generate_rf(layers, p, tau, nfft=4096):
    forward = JointRFForward(p, gaussians=[2.5], periods=[0.0], dt=0.05, n_samples=601, t0=-2.0, nfft=nfft, device='cpu')
    with torch.no_grad():
        a = forward.prepare_layers(torch.tensor(layers)[None])
        r, z = _surface_torch(a, forward.p, forward.omega, 'elastic')
        power = z.abs().square()
        denom = torch.maximum(power, forward.water_level * power.amax(-1, keepdim=True))
        norm = torch.fft.irfft(power / denom * forward.filters[0], n=nfft)[..., :1]
        freq = r * z.conj() / denom * forward.filters[0] / norm * forward.shift * torch.exp(-1j * forward.omega * tau)
        rf = torch.fft.irfft(freq, n=nfft)[0, :, :601].numpy()[:, ::4]
        return rf / rf[:, 10:11]

def dataset(case, noise, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(-2, 28.01, 0.2)
    layers, tau = truth_layers(case)
    rays = np.r_[np.linspace(0.041, 0.048, 16), np.linspace(0.051, 0.064, 16), np.linspace(0.068, 0.083, 16)]
    exact = generate_rf(layers, rays, tau)
    corr = 0.85 * np.exp(-0.5 * ((t[:, None] - t[None, :]) / 0.35) ** 2) + 0.15 * np.eye(len(t))
    perturb = rng.normal(size=exact.shape) @ np.linalg.cholesky(corr).T * noise
    observed = exact + perturb
    observed = observed / observed[:, 10:11]
    groups = []
    for b in range(3):
        g = []
        for j in range(16):
            index = 16 * b + j
            g.append(dict(rf=observed[index], exact=exact[index], p=rays[index], member=f'synthetic-{b}-{j:02d}', split='test' if j % 4 == 0 else 'train'))
        groups.append(g)

    def make(which, full=False):
        mask = (t >= (-0.601 if full else 0.799)) & (t <= 26.001)
        picked = []
        for g in groups:
            train = [e for e in g if e['split'] == 'train']
            if which == 'inner':
                part = [e for j, e in enumerate(train) if j % 5 != 0]
            elif which == 'validation':
                part = [e for j, e in enumerate(train) if j % 5 == 0]
            elif which == 'train':
                part = train
            elif which == 'test':
                part = [e for e in g if e['split'] == 'test']
            else:
                raise ValueError(which)
            picked.append(part)
        vals = [stack(g, mask) for g in picked]
        return (dict(time=t, fit_mask=mask, rf=np.stack([a[0] for a in vals]), p=np.array([a[1] for a in vals]), cov=np.stack([a[2] for a in vals])), picked)
    test_p = np.array([np.mean([e['p'] for e in g if e['split'] == 'test']) for g in groups])
    fine = generate_rf(layers, test_p, tau)
    finer = generate_rf(layers, test_p, tau, nfft=8192)
    max_refinement = float(np.max(np.abs(finer - fine)))
    if max_refinement > 0.002:
        raise RuntimeError(f'Truth FFT refinement insufficient: {max_refinement}')
    return (make, groups, layers, tau, max_refinement)

def profile(layers):
    return layers[np.searchsorted(layers[:, 0], DEPTH, side='right') - 1, 2]

def run_case(case, noise, seed, starts=16):
    key = f'{case}-noise{noise:.3f}-seed{seed}'
    if (OUT / f'{key}.json').exists():
        return
    make, groups, truth, tau, refinement = dataset(case, noise, seed)
    train, _ = make('train')
    test, test_groups = make('test')
    inner, _ = make('inner')
    validation, _ = make('validation')
    fulltrain, _ = make('train', True)
    fullinner, _ = make('inner', True)
    reference = [1.5, 2.4, 40.0, 1.78]
    checks = []
    for kind in MODELS:
        for covariance in ['full', 'diagonal']:
            target = RefinedTarget(inner if kind == 'legacy' else fullinner, kind, covariance=covariance)
            candidates, seconds = optimize(target, starts, reference, maxiter=300)
            x = torch.tensor(candidates[0]['x'])[None]
            with torch.no_grad():
                pred = target.predict(x, target.make_forward(validation['p']))[0].numpy()
            checks.append(dict(model=kind, key=kind + '_' + covariance, covariance=covariance, names=target.names, inner_candidates=candidates, inner_seconds=seconds, validation=score(pred, validation['rf'], train['fit_mask'])))
    best = min((c['validation']['rmse'] for c in checks))
    eligible = [c for c in checks if c['validation']['rmse'] <= 1.02 * best]
    selected = min(eligible, key=lambda c: (len(c['names']), c['validation']['rmse']))['key']
    predictions = {}
    true_profile = profile(truth)
    noiseless_test = np.stack([np.mean([e['exact'] for e in g], 0) for g in test_groups])
    for c in checks:
        target = RefinedTarget(train if c['model'] == 'legacy' else fulltrain, c['model'], covariance=c['covariance'])
        candidates, seconds = optimize(target, starts, reference, warm=c['inner_candidates'][0]['x'], maxiter=300)
        x = torch.tensor(candidates[0]['x'])[None]
        with torch.no_grad():
            ptr = target.predict(x)[0].numpy()
            pte = target.predict(x, target.make_forward(test['p']))[0].numpy()
            estimated_layers = target.model(x)[0].numpy()
        curve = profile(estimated_layers)
        key_model = c['key']
        predictions[key_model + '_train'] = ptr
        predictions[key_model + '_test'] = pte
        predictions[key_model + '_profile'] = curve
        c.update(selected=key_model == selected, full_candidates=candidates, full_seconds=seconds, train=score(ptr, train['rf'], train['fit_mask']), test=score(pte, test['rf'], train['fit_mask']), noiseless_test=score(pte, noiseless_test, train['fit_mask']), profile_rmse_km_s=float(np.sqrt(np.mean((curve - true_profile) ** 2))), H_abs_error_km=float(abs(candidates[0]['x'][2] - 40.0)))
    result = dict(case=case, event_noise_sd=noise, seed=seed, selected=selected, truth_H_km=40.0, truth_time_shift_s=tau, truth_layer_count=len(truth), truth_dt_s=0.05, truth_nfft=4096, inverse_dt_s=0.2, inverse_nfft=512, truth_refinement_4096_8192_max_rf=refinement, checks=checks, score_window_s=[0.8, 26.0], profile_depth_window_km=[0.0, 50.0], source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), target_hash=hashlib.sha256((ROOT / 'scripts/fit_refinement.py').read_bytes()).hexdigest(), scope='Synthetic MAP model/likelihood assessment; common GRTM physics with finer generation discretization; no independent physics solver, HMC risk superiority, or calibrated coverage claim')
    np.savez_compressed(OUT / f'{key}.npz', time=train['time'], rf_train=train['rf'], rf_test=test['rf'], rf_test_exact=noiseless_test, p_train=train['p'], p_test=test['p'], depth=DEPTH, truth_profile=true_profile, truth_layers=truth, **predictions)
    (OUT / f'{key}.json').write_text(json.dumps(plain(result), indent=2))
    print(key, selected, flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case', choices=['uniform', 'gradient', 'lvz'])
    p.add_argument('--noise', type=float)
    p.add_argument('--seed', type=int)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    OUT.mkdir(exist_ok=True, parents=True)
    if a.case:
        run_case(a.case, a.noise, a.seed)
        return
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), cases=['uniform', 'gradient', 'lvz'], noise_sd=[0.015, 0.05], seeds=[100, 101, 102], stations_or_cases=18, events_per_case=48, event_split='3 slowness bins; every fourth event held out (36 train,12 test); every fifth training event for inner validation', selection='same v3 10 candidates; 16 starts; fixed inner reference; minimum dimension within 2% of best inner RMSE', profile_score='0-50 km shear-speed RMSE at 0.1 km spacing; Moho absolute error separately', forward='finer crustal layering, dt .05, nfft4096; refinement8192; inversion dt.2 nfft512; event-specific truth rays, mean-bin inverse rays', script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), target_sha256=hashlib.sha256((ROOT / 'scripts/fit_refinement.py').read_bytes()).hexdigest())
    f = OUT / 'protocol.json'
    if f.exists():
        old = json.loads(f.read_text())
        assert old['script_sha256'] == protocol['script_sha256'] and old['target_sha256'] == protocol['target_sha256']
    else:
        f.write_text(json.dumps(protocol, indent=2))
    jobs = [(c, n, s) for c in protocol['cases'] for n in protocol['noise_sd'] for s in protocol['seeds']]

    def launch(job):
        c, n, s = job
        key = f'{c}-noise{n:.3f}-seed{s}'
        with (OUT / f'{key}.log').open('w') as log:
            r = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--case', c, '--noise', str(n), '--seed', str(s)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        print(key, r.returncode, flush=True)
        return dict(case=c, noise=n, seed=s, returncode=r.returncode)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        results = list(pool.map(launch, jobs))
    (OUT / 'status.json').write_text(json.dumps(results, indent=2))
    if any((r['returncode'] for r in results)):
        raise RuntimeError('Synthetic job failed; inspect individual logs')
if __name__ == '__main__':
    main()
