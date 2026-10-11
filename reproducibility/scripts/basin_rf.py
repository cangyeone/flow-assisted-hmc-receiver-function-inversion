# Our CPU research program; ported imports use the public GRTM RF preview.
"""Compact sediment-aware CPU RF targets, with actual event-ray averaging.

No observed trace is shifted, dereverberated, or smoothed. All multiple
reflections remain in the GRTM response. This is predictive MAP assessment.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import json, time
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.signal import find_peaks
from scipy.stats import qmc
from fit_refinement import RefinedTarget, ROOT, stack, score, plain
from grtm.rf_cpu import _surface_torch
from grtm import JointRFForward
KINDS = ('sediment', 'gradient', 'damped')

def specification(kind):
    if kind not in KINDS:
        raise ValueError(kind)
    names = ['hs', 'Vs_s_top', 'H'] + [f'crust_thickness_logratio_{i + 1}' for i in range(3)] + [f'Vs_c{i + 1}' for i in range(4)] + ['kappa', 'kappa_s', 'rho_s']
    low = [0.05, 0.3, 15] + [-2.5] * 3 + [2.7] * 4 + [1.55, 1.6, 1.6]
    high = [10, 3.4, 75] + [2.5] * 3 + [4.3] * 4 + [2.05, 3.2, 2.7]
    if kind != 'sediment':
        names += ['sediment_gradient_fraction']
        low += [0.0]
        high += [1.0]
    if kind == 'damped':
        names += ['log_sediment_Qs']
        low += [np.log(10.0)]
        high += [np.log(300.0)]
    return (np.array(low + [-0.6]), np.array(high + [0.6]), names + ['P_time_shift_s'])

def event_data(data, groups, which, full=True):
    """Rebuild covariance and ray lists from the same disjoint event members."""
    selected = []
    for group in groups:
        train = [e for e in group if e['split'] == 'train']
        if which == 'inner':
            pick = [e for i, e in enumerate(train) if i % 5 != 0]
        elif which == 'validation':
            pick = [e for i, e in enumerate(train) if i % 5 == 0]
        elif which == 'train':
            pick = train
        elif which == 'test':
            pick = [e for e in group if e['split'] == 'test']
        else:
            raise ValueError(which)
        selected.append(pick)
    d = data.copy()
    d['fit_mask'] = (data['time'] >= (-0.601 if full else 0.799)) & (data['time'] <= 26.001)
    vals = [stack(g, d['fit_mask']) for g in selected]
    d.update(rf=np.stack([v[0] for v in vals]), p=np.array([v[1] for v in vals]), cov=np.stack([v[2] for v in vals]), event_p=[np.array([e['p'] for e in g]) for g in selected])
    return (d, selected)

class BasinTarget(RefinedTarget):

    def __init__(self, data, kind, nfft=1024):
        super().__init__(data, 'flex5', covariance='diagonal')
        self.kind = kind
        self.low, self.high, self.names = specification(kind)
        self.attenuation = 'constant_q' if kind == 'damped' else 'elastic'
        self.nfft = nfft
        self.forward = self.event_forward(data)

    def event_forward(self, data):
        groups = data['event_p']
        p = np.concatenate(groups)
        f = JointRFForward(p, gaussians=[2.5], periods=[0.0], dt=0.2, n_samples=151, t0=-2.0, nfft=self.nfft, device='cpu', attenuation=self.attenuation)
        weights = np.zeros((len(groups), len(p)))
        offset = 0
        for i, g in enumerate(groups):
            weights[i, offset:offset + len(g)] = 1 / len(g)
            offset += len(g)
        f.basin_weights = torch.tensor(weights)
        return f

    def model(self, x):
        hs, vss, H = x[:, :3].unbind(-1)
        zero = hs * 0
        one = zero + 1
        weights = 0.04 + 0.84 * torch.softmax(torch.cat((x[:, 3:6], zero[:, None]), -1), -1)
        tops = hs[:, None] + (H - hs)[:, None] * torch.cat((zero[:, None], weights.cumsum(-1)[:, :-1]), -1)
        vc = x[:, 6:10]
        k = x[:, 10]
        crust = torch.stack((tops, vc * k[:, None], vc, torch.ones_like(vc) * 2.8), -1)
        if self.kind == 'sediment':
            st = zero[:, None]
            sv = vss[:, None]
        else:
            st = hs[:, None] * torch.tensor([0.0, 1 / 3, 2 / 3])
            sv = vss[:, None] + (3.5 - vss)[:, None] * x[:, 13:14] * torch.tensor([1 / 6, 1 / 2, 5 / 6])
        sed = torch.stack((st, sv * x[:, 11:12], sv, x[:, 12:13].expand_as(sv)), -1)
        mantle = torch.stack((H, one * 8, one * 4.5, one * 3.3), -1)[:, None]
        layers = torch.cat((sed, crust, mantle), 1)
        if self.kind == 'damped':
            qs = torch.exp(x[:, 14:15]).expand(-1, 3)
            qcrust = torch.full_like(layers[:, 3:, 0], float('inf'))
            qs = torch.cat((qs, qcrust), -1)
            layers = torch.cat((layers, (2 * qs)[..., None], qs[..., None]), -1)
        return layers

    def predict(self, x, forward=None):
        f = forward or self.forward
        layers = f.prepare_layers(self.model(x), check=False)
        r, z = _surface_torch(layers, f.p, f.omega, self.attenuation)
        power = z.abs().square()
        denom = torch.maximum(power, f.water_level * power.amax(-1, keepdim=True))
        norm = torch.fft.irfft(power / denom * f.filters[0], n=f.nfft)[..., :1]
        phase = f.shift[None, None] * torch.exp(-1j * f.omega[None, None] * x[:, -1, None, None])
        freq = r * z.conj() / denom * f.filters[0] / norm * phase
        rf = torch.fft.irfft(freq, n=f.nfft)[..., :f.n]
        rf = rf / rf[..., 10:11]
        return torch.einsum('gr,brt->bgt', f.basin_weights, rf)

    def penalties(self, x):
        vc = x[:, 6:10]
        penalty = (x[:, -1] / 0.25).square() + ((x[:, 10] - 1.78) / 0.12).square()
        penalty = penalty + ((vc - 3.6) / 1.0).square().sum(-1) + ((vc[:, 1:] - vc[:, :-1]) / 0.8).square().sum(-1)
        penalty = penalty + ((vc[:, 2:] - 2 * vc[:, 1:-1] + vc[:, :-2]) / 0.4).square().sum(-1) + (x[:, 3:6] / 1.5).square().sum(-1)
        penalty = penalty + ((x[:, 11] - 2.1) / 0.45).square() + ((x[:, 12] - 2.2) / 0.25).square()
        if self.kind != 'sediment':
            penalty = penalty + ((3.5 - x[:, 1]) * x[:, 13] / 2.0).square()
        if self.kind == 'damped':
            penalty = penalty + ((x[:, 14] - np.log(60.0)) / 1.0).square()
        return penalty

def period_starts(data):
    """Period hints use fitting events only, and carry no geological label."""
    y = data['rf'][:, data['fit_mask']]
    y = y - y.mean(-1, keepdims=True)
    ac = np.mean([np.correlate(a, a, mode='full')[len(a) - 1:] for a in y], 0)
    ac = ac / max(ac[0], 1e-12)
    peaks, _ = find_peaks(ac)
    peaks = [i for i in peaks if 1.0 <= i * 0.2 <= 12.0]
    peaks = sorted(peaks, key=lambda i: ac[i], reverse=True)
    return [i * 0.2 for i in peaks[:3]] or [2.0, 4.0, 6.0]

def optimize(target, starts=16, warm=None, maxiter=400):
    lo, span = (target.low, target.high - target.low)
    u = qmc.Sobol(len(lo), scramble=True, seed=1730).random_base2(int(np.ceil(np.log2(starts))))[:starts]
    refs = []
    for hs, vss, period in [(1.5, 2.4, None)] + [(min(9.5, 0.5 * t * v), v, t) for t in period_starts(target.data) for v in [0.5, 1.0, 2.0]]:
        x = [hs, vss, 40.0, 0.0, 0.0, 0.0, 3.4, 3.55, 3.7, 3.85, 1.78, 2.1, 2.2]
        if target.kind != 'sediment':
            x += [0.5]
        if target.kind == 'damped':
            x += [np.log(60.0)]
        refs.append(x + [0.0])
    for i, x in enumerate(refs[:min(len(refs), starts)]):
        u[i] = (np.array(x) - lo) / span
    if warm is not None:
        u[0] = (np.array(warm) - lo) / span
    u = np.clip(u, 0.0001, 0.9999)
    answers = []
    t0 = time.perf_counter()

    def objective(v):
        q = torch.tensor(v, requires_grad=True)
        loss = -target((torch.tensor(lo) + torch.tensor(span) * q)[None])[0]
        grad = torch.autograd.grad(loss, q)[0]
        if not torch.isfinite(loss) or not torch.isfinite(grad).all():
            raise FloatingPointError('nonfinite basin target')
        return (float(loss.detach()), grad.detach().numpy())
    for i, init in enumerate(u):
        opt = minimize(objective, init, jac=True, method='L-BFGS-B', bounds=[(0.0001, 0.9999)] * len(lo), options=dict(maxiter=maxiter, ftol=1e-09, gtol=1e-05))
        answers.append(dict(start=i, x=lo + span * opt.x, nll=float(opt.fun), success=bool(opt.success), message=str(opt.message), iterations=int(opt.nit)))
    answers.sort(key=lambda c: c['nll'])
    return (answers, time.perf_counter() - t0)

def choose(checks):
    best = min((c['validation']['rmse'] for c in checks))
    eligible = [c for c in checks if c['validation']['rmse'] <= 1.02 * best]
    return min(eligible, key=lambda c: (len(c['names']), c['validation']['rmse']))['key']

def profile(layers, depth):
    return layers[np.searchsorted(layers[:, 0], depth, side='right') - 1, 2]
