# Our CPU research program; ported imports use the public GRTM RF preview.
"""CPU RF model assessment; choose complexity using training events only.

The outer event split and normalized 0.8--26 s score are unchanged. Candidate
selection uses every fifth *training* event as inner validation. Results are
MAP predictive checks, not replacement posterior intervals or geology truth.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import argparse, hashlib, io, json, sys, time, zipfile
from pathlib import Path
import numpy as np
import torch
from obspy import read
from scipy.optimize import minimize
from scipy.signal import resample_poly
from scipy.stats import qmc
from grtm import JointRFForward
from grtm.rf_cpu import _surface_torch
from cpu_hmc import LOW as OLD_LOW, HIGH as OLD_HIGH, layers as old_layers, plain
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/fit-refinement'
MODELS = ('legacy', 'timing', 'layer3', 'layer5', 'flex5')

def specification(kind):
    if kind in ('legacy', 'timing'):
        lo, hi = (OLD_LOW.copy(), OLD_HIGH.copy())
        names = ['hs', 'Vs_s', 'H', 'kappa']
    else:
        n = 2 if kind == 'layer3' else 4
        count = 3 if kind == 'flex5' else 1
        depth_names = [f'crust_thickness_logratio_{i + 1}' for i in range(count)] if kind == 'flex5' else ['crust_interface_fraction']
        names = ['hs', 'Vs_s', 'H'] + depth_names + [f'Vs_c{i + 1}' for i in range(n)] + ['kappa']
        lo = np.array([0.05, 0.7, 15.0] + ([-2.5] * count if kind == 'flex5' else [0.15]) + [2.7] * n + [1.55])
        hi = np.array([10.0, 3.5, 75.0] + ([2.5] * count if kind == 'flex5' else [0.85]) + [4.3] * n + [2.05])
    if kind != 'legacy':
        lo = np.r_[lo, -0.6]
        hi = np.r_[hi, 0.6]
        names += ['P_time_shift_s']
    return (lo, hi, names)

def stack(traces, mask):
    ar = np.stack([a['rf'] for a in traces])
    tt = np.arange(-2, 28.01, 0.2)[mask]
    empirical = np.cov(ar[:, mask], rowvar=False) / len(ar)
    corr = np.exp(-0.5 * ((tt[:, None] - tt[None, :]) / 0.5) ** 2)
    cov = 0.75 * empirical + 0.25 * np.diag(np.diag(empirical)) + 0.03 ** 2 * corr + 0.005 ** 2 * np.eye(len(tt))
    return (ar.mean(0), np.mean([a['p'] for a in traces]), cov)

def load_events(station):
    manifest = json.loads((ROOT / 'data/processed/manifest.json').read_text())
    meta = next((a for a in manifest if a['station'] == station))
    original = dict(np.load(ROOT / 'data/processed' / f'{station}.npz'))
    groups = [[], [], []]
    cursor = 0
    with zipfile.ZipFile(ROOT / 'data/source' / f'ears-{station}.zip') as archive:
        for b, (n, nt) in enumerate(zip(meta['train_counts'], meta['test_counts'])):
            for e in meta['events'][cursor:cursor + n + nt]:
                tr = read(io.BytesIO(archive.read(e['member'])), format='SAC')[0]
                y = resample_poly(tr.data.astype(float), 1, round(0.2 / tr.stats.delta))
                t = float(tr.stats.sac.b - tr.stats.sac.a) + np.arange(len(y)) * 0.2
                rf = np.interp(original['time'], t, y) / np.interp(0, t, y)
                groups[b].append(dict(e, rf=rf))
            cursor += n + nt
    for b, g in enumerate(groups):
        for split, key in [('train', 'rf'), ('test', 'test_rf')]:
            mean = np.mean([e['rf'] for e in g if e['split'] == split], 0)
            np.testing.assert_allclose(mean, original[key][b], atol=2e-12, rtol=0)
    return (original, groups)

def partition_data(original, groups):
    inner, valid, inner_names, valid_names = ([], [], [], [])
    for g in groups:
        train = [e for e in g if e['split'] == 'train']
        v = [e for i, e in enumerate(train) if i % 5 == 0]
        fit = [e for i, e in enumerate(train) if i % 5 != 0]
        inner.append(fit)
        valid.append(v)
        inner_names += [e['member'] for e in fit]
        valid_names += [e['member'] for e in v]

    def make(gs):
        d = original.copy()
        values = [stack(g, original['fit_mask']) for g in gs]
        d.update(rf=np.stack([a[0] for a in values]), p=np.array([a[1] for a in values]), cov=np.stack([a[2] for a in values]))
        return d
    assert set(inner_names).isdisjoint(valid_names)
    outer = [e['member'] for g in groups for e in g if e['split'] == 'test']
    assert set(inner_names + valid_names).isdisjoint(outer)
    return (make(inner), make(valid), dict(inner_fit=inner_names, inner_validation=valid_names, outer_test=outer))

class RefinedTarget:

    def __init__(self, data, kind, smooth=1.0, covariance='full'):
        self.data, self.kind, self.smooth, self.covariance = (data, kind, float(smooth), covariance)
        self.low, self.high, self.names = specification(kind)
        self.mask = torch.tensor(data['fit_mask'])
        self.y = torch.tensor(data['rf'][:, data['fit_mask']])
        if covariance == 'full':
            self.invchol = torch.linalg.inv(torch.linalg.cholesky(torch.tensor(data['cov'])))
        elif covariance == 'diagonal':
            self.invchol = torch.diag_embed(torch.diagonal(torch.tensor(data['cov']), dim1=-2, dim2=-1).rsqrt()) * np.sqrt(0.2)
        else:
            raise ValueError(covariance)
        self.forward = self.make_forward(data['p'])

    @staticmethod
    def make_forward(p):
        return JointRFForward(p, gaussians=[2.5], periods=[0.0], dt=0.2, n_samples=151, t0=-2.0, nfft=512, device='cpu')

    def model(self, x):
        if self.kind in ('legacy', 'timing'):
            return old_layers(x[:, :4])
        hs, vss, H = x[:, :3].unbind(-1)
        start = 6 if self.kind == 'flex5' else 4
        controls = x[:, start:-2]
        k = x[:, -2]
        zero = hs * 0
        one = zero + 1
        if self.kind == 'flex5':
            logits = torch.cat((x[:, 3:6], zero[:, None]), -1)
            thickness = 0.04 + 0.84 * torch.softmax(logits, -1)
            relative = torch.cat((zero[:, None], thickness.cumsum(-1)[:, :-1]), -1)
        elif controls.shape[1] == 2:
            fraction = x[:, 3]
            relative = torch.stack((zero, fraction), -1)
        else:
            fraction = x[:, 3]
            relative = torch.stack((zero, 0.5 * fraction, fraction, 0.5 * (1 + fraction)), -1)
        tops = hs[:, None] + relative * (H - hs)[:, None]
        crust = torch.stack((tops, controls * k[:, None], controls, torch.ones_like(controls) * 2.8), -1)
        sediment = torch.stack((zero, 1.8 * vss, vss, one * 2.5), -1)[:, None]
        mantle = torch.stack((H, one * 8.0, one * 4.5, one * 3.3), -1)[:, None]
        return torch.cat((sediment, crust, mantle), 1)

    def predict(self, x, forward=None):
        f = forward or self.forward
        a = f.prepare_layers(self.model(x), check=False)
        radial, vertical = _surface_torch(a, f.p, f.omega, 'elastic')
        power = vertical.abs().square()
        denominator = torch.maximum(power, f.water_level * power.amax(-1, keepdim=True))
        reference = power / denominator * f.filters[0]
        norm = torch.fft.irfft(reference, n=f.nfft)[..., :1]
        shift = f.shift
        if self.kind != 'legacy':
            shift = shift[None, None] * torch.exp(-1j * f.omega[None, None] * x[:, -1, None, None])
        spectrum = radial * vertical.conj() / denominator * f.filters[0] / norm * shift
        rf = torch.fft.irfft(spectrum, n=f.nfft)[..., :f.n]
        return rf / rf[..., 10:11]

    def penalties(self, x):
        penalty = x[:, 0] * 0
        if self.kind != 'legacy':
            penalty = (x[:, -1] / 0.25).square()
        if self.kind in ('layer3', 'layer5', 'flex5'):
            c = x[:, 6 if self.kind == 'flex5' else 4:-2]
            penalty = penalty + ((x[:, -2] - 1.78) / 0.12).square()
            penalty = penalty + 0.25 * ((c - 3.6) / 0.5).square().sum(-1)
            penalty = penalty + self.smooth * ((c[:, 1:] - c[:, :-1]) / 0.8).square().sum(-1)
            if c.shape[1] > 2:
                penalty = penalty + self.smooth * ((c[:, 2:] - 2 * c[:, 1:-1] + c[:, :-2]) / 0.4).square().sum(-1)
            if self.kind == 'flex5':
                penalty = penalty + (x[:, 3:6] / 1.5).square().sum(-1)
        return penalty

    def __call__(self, x):
        residual = self.predict(x)[:, :, self.mask] - self.y
        white = torch.einsum('rij,brj->bri', self.invchol, residual)
        return -0.5 * (white.square().sum((1, 2)) + self.penalties(x))

def seeds(target, n, previous):
    lo, span = (target.low, target.high - target.low)
    initial = qmc.Sobol(len(lo), scramble=True, seed=728).random_base2(int(np.ceil(np.log2(n))))[:n]
    old = np.array(previous)
    if target.kind in ('legacy', 'timing'):
        initial[0, :4] = (old - lo[:4]) / span[:4]
        if target.kind == 'timing':
            initial[0, -1] = 0.5
    else:
        n = 2 if target.kind == 'layer3' else 4
        positions = [0.0, 0.0, 0.0] if target.kind == 'flex5' else [0.5]
        base = [old[0], old[1], old[2]] + positions + [6.5 / old[3]] * n + [old[3], 0.0]
        initial[0] = (np.array(base) - lo) / span
        for i in range(1, min(5, n)):
            v = 3.3 + 0.12 * i
            positions = [-0.5, 0.0, 0.5] if target.kind == 'flex5' else [0.3 + 0.1 * i]
            b = [min(old[0], 8), old[1], [30.0, 40.0, 50.0, 60.0][i - 1]] + positions + list(np.linspace(v, v + 0.15, n)) + [1.75, 0.1]
            initial[i] = (np.array(b) - lo) / span
    return np.clip(initial, 0.0002, 0.9998)

def optimize(target, n, previous, warm=None, maxiter=250):
    lo = torch.tensor(target.low)
    span = torch.tensor(target.high - target.low)
    initial = seeds(target, n, previous)
    if warm is not None:
        initial[0] = (np.array(warm) - target.low) / (target.high - target.low)
    answers = []
    start = time.perf_counter()

    def fn(u):
        q = torch.tensor(u, requires_grad=True)
        loss = -target((lo + span * q)[None])[0]
        g = torch.autograd.grad(loss, q)[0]
        if not torch.isfinite(loss) or not torch.isfinite(g).all():
            raise FloatingPointError('nonfinite bounded model target/gradient')
        return (float(loss.detach()), g.detach().numpy())
    for i, u in enumerate(initial):
        opt = minimize(fn, u, jac=True, method='L-BFGS-B', bounds=[(0.0001, 0.9999)] * len(lo), options=dict(maxiter=maxiter, ftol=1e-09, gtol=1e-05))
        answers.append(dict(start=i, x=target.low + (target.high - target.low) * opt.x, nll=float(opt.fun), success=bool(opt.success), message=str(opt.message), iterations=int(opt.nit)))
    answers.sort(key=lambda a: a['nll'])
    return (answers, time.perf_counter() - start)

def score(pred, observed, mask):
    pred, observed = (np.asarray(pred), np.asarray(observed))
    residual = pred[:, mask] - observed[:, mask]
    return dict(rmse=float(np.sqrt(np.mean(residual ** 2))), correlation=float(np.corrcoef(pred[:, mask].ravel(), observed[:, mask].ravel())[0, 1]))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('station')
    p.add_argument('--starts', type=int, default=12)
    p.add_argument('--maxiter', type=int, default=300)
    p.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    p.add_argument('--output', default='results/fit-refinement-v3')
    args = p.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    station = args.station
    output = out / f'{station}.json'
    if output.exists():
        print(station, 'already complete', flush=True)
        return
    original, groups = load_events(station)
    inner, valid, splits = partition_data(original, groups)
    full_original = original.copy()
    full_original['fit_mask'] = (original['time'] >= -0.601) & (original['time'] <= 26.001)
    values = [stack([e for e in g if e['split'] == 'train'], full_original['fit_mask']) for g in groups]
    full_original['cov'] = np.stack([a[2] for a in values])
    full_inner, full_valid, _ = partition_data(full_original, groups)
    old_map = json.loads((ROOT / 'results' / f'{station}-map.json').read_text())['candidates'][0]['x']
    checks = []
    predictions = {}
    fits = {}
    for kind in args.models:
        for covariance in ('full', 'diagonal'):
            key = kind + '_' + covariance
            dfit = inner if kind == 'legacy' else full_inner
            dvalid = valid if kind == 'legacy' else full_valid
            target = RefinedTarget(dfit, kind, covariance=covariance)
            candidates, secs = optimize(target, args.starts, [1.5, 2.4, 40.0, 1.78], maxiter=args.maxiter)
            x = torch.tensor(candidates[0]['x'])[None]
            with torch.no_grad():
                pr = target.predict(x, target.make_forward(dvalid['p']))[0].numpy()
            check = dict(model=kind, covariance=covariance, key=key, names=target.names, bounds=[target.low, target.high], smooth=target.smooth, fit_window_s=[0.8, 26.0] if kind == 'legacy' else [-0.6, 26.0], inner_candidates=candidates, inner_seconds=secs, validation=score(pr, valid['rf'], original['fit_mask']))
            checks.append(check)
            print(station, key, 'inner', check['validation'], flush=True)
    best = min((c['validation']['rmse'] for c in checks))
    eligible = [c for c in checks if c['validation']['rmse'] <= 1.02 * best]
    chosen = min(eligible, key=lambda c: (len(c['names']), c['validation']['rmse']))['key']
    for check in checks:
        kind = check['model']
        key = check['key']
        target = RefinedTarget(original if kind == 'legacy' else full_original, kind, covariance=check['covariance'])
        warm = check['inner_candidates'][0]['x']
        candidates, secs = optimize(target, args.starts, old_map, warm=warm, maxiter=args.maxiter)
        x = torch.tensor(candidates[0]['x'])[None]
        with torch.no_grad():
            train = target.predict(x)[0].numpy()
            test = target.predict(x, target.make_forward(original['test_p']))[0].numpy()
        predictions[key + '_train'] = train
        predictions[key + '_test'] = test
        check.update(full_candidates=candidates, full_seconds=secs, selected=key == chosen, train=score(train, original['rf'], original['fit_mask']), test=score(test, original['test_rf'], original['fit_mask']))
        print(station, key, 'outer', check['test'], flush=True)
    with torch.no_grad():
        baseline = RefinedTarget(original, 'legacy')
        bx = torch.tensor(old_map)[None]
        predictions['original_map_train'] = baseline.predict(bx)[0].numpy()
        predictions['original_map_test'] = baseline.predict(bx, baseline.make_forward(original['test_p']))[0].numpy()
    result = dict(station=station, device='cpu', thread_count=1, starts=args.starts, maxiter=args.maxiter, seed=728, selection='lowest inner validation RMSE; prefer lower dimension within 2% of minimum', selected=chosen, checks=checks, split=splits, score_window_s=[0.8, 26.0], original_map_test=score(predictions['original_map_test'], original['test_rf'], original['fit_mask']), source_sha256=hashlib.sha256((ROOT / 'data/source' / f'ears-{station}.zip').read_bytes()).hexdigest(), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), scope='MAP model assessment; outer events are unused by selection; no replacement posterior or geological truth claim')
    np.savez_compressed(out / f'{station}-predictions.npz', time=original['time'], **predictions)
    output.write_text(json.dumps(plain(result), indent=2))
    print(station, 'SELECTED', chosen, flush=True)
if __name__ == '__main__':
    main()
