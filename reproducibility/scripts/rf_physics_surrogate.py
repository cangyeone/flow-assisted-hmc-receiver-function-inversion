# Our CPU-only archived research implementation; experimental, not default inference.
"""Observation-calibrated likelihood network with exact GRTM teachers.

The observation and parameter-to-layer map define each calibration. This is
not an amortized geology classifier. Production weights stay frozen; the
separate sampler always accepts using the original physical posterior.
"""
import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from grtm import JointRFForward
from basin_rf import BasinTarget, specification
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/neural-rf-physics-v2'

def box_jacobian(z, span):
    return (span.log() - F.softplus(z) - F.softplus(-z)).sum(-1)

class DeviceBasinTarget(BasinTarget):
    """Same selected BasinTarget, with native CPU response and device tensors."""

    def __init__(self, data, kind, device='cpu', nfft=1024):
        super().__init__(data, kind, nfft)
        self.device = torch.device(device)
        self.y = self.y.to(self.device)
        self.mask = torch.as_tensor(np.flatnonzero(data['fit_mask']), dtype=torch.long, device=self.device)
        self.invchol = self.invchol.to(self.device)
        rays = np.concatenate(data['event_p'])
        self.forward = JointRFForward(rays, gaussians=[2.5], periods=[0.0], dt=0.2, n_samples=151, t0=-2.0, nfft=nfft, device=device, attenuation=self.attenuation)
        weights = np.zeros((3, len(rays)))
        offset = 0
        for j, p in enumerate(data['event_p']):
            weights[j, offset:offset + len(p)] = 1 / len(p)
            offset += len(p)
        self.forward.basin_weights = self.forward.tensor(weights)
        self.sediment_tops = self.forward.tensor([0.0, 1 / 3, 2 / 3])
        self.sediment_midpoints = self.forward.tensor([1 / 6, 1 / 2, 5 / 6])

    def model(self, x):
        hs, vss, H = x[:, :3].unbind(-1)
        zero = hs * 0
        one = zero + 1
        w = 0.04 + 0.84 * torch.softmax(torch.cat((x[:, 3:6], zero[:, None]), -1), -1)
        tops = hs[:, None] + (H - hs)[:, None] * torch.cat((zero[:, None], w.cumsum(-1)[:, :-1]), -1)
        vc = x[:, 6:10]
        k = x[:, 10]
        crust = torch.stack((tops, vc * k[:, None], vc, torch.ones_like(vc) * 2.8), -1)
        if self.kind == 'sediment':
            st = zero[:, None]
            sv = vss[:, None]
        else:
            st = hs[:, None] * self.sediment_tops
            sv = vss[:, None] + (3.5 - vss)[:, None] * x[:, 13:14] * self.sediment_midpoints
        sed = torch.stack((st, sv * x[:, 11:12], sv, x[:, 12:13].expand_as(sv)), -1)
        mantle = torch.stack((H, one * 8.0, one * 4.5, one * 3.3), -1)[:, None]
        result = torch.cat((sed, crust, mantle), 1)
        if self.kind == 'damped':
            qs = torch.exp(x[:, 14:15]).expand(-1, 3)
            qs = torch.cat((qs, torch.full_like(result[:, 3:, 0], float('inf'))), -1)
            result = torch.cat((result, (2 * qs)[..., None], qs[..., None]), -1)
        return result

    def predict(self, x, forward=None):
        f = forward or self.forward
        spectra = f._evaluate(self.model(x), check=False)
        r, z = (spectra['radial_spectrum'], spectra['vertical_spectrum'])
        power = z.abs().square()
        den = torch.maximum(power, f.water_level * power.amax(-1, keepdim=True))
        norm = torch.fft.irfft(power / den * f.filters[0], n=f.nfft)[..., :1]
        phase = f.shift[None, None] * torch.exp(-1j * f.omega[None, None] * x[:, -1, None, None])
        y = torch.fft.irfft(r * z.conj() / den * f.filters[0] / norm * phase, n=f.nfft)[..., :f.n]
        y = y / y[..., 10:11]
        return torch.einsum('gr,brt->bgt', f.basin_weights, y)

class ConditionalTarget:
    """Original two-parameter known-noise example, on CPU or native CPU."""

    def __init__(self, observed, p, p_sd, sigma, covariance, device='cpu'):
        self.device = torch.device(device)
        self.low = np.array([38.0, 1.72])
        self.high = np.array([42.0, 1.84])
        self.names = ['Moho_depth_km', 'crust_VpVs']
        mask = (np.arange(-2.0, 28.01, 0.2) >= 0.799) & (np.arange(-2.0, 28.01, 0.2) <= 26.001)
        self.mask = torch.as_tensor(np.flatnonzero(mask), dtype=torch.long, device=device)
        rays = (np.array(p)[:, None] + np.array(p_sd)[:, None] * np.sqrt(1.5) * [-1.0, 0.0, 1.0]).ravel()
        self.forward = JointRFForward(rays, gaussians=[2.5], periods=[0.0], dt=0.2, n_samples=151, t0=-2.0, nfft=1024, device=device)
        self.y = torch.as_tensor(observed[:, mask], dtype=torch.float64, device=device)
        self.invchol = torch.linalg.inv(torch.linalg.cholesky(torch.as_tensor(covariance[np.ix_(mask, mask)] * sigma ** 2, dtype=torch.float64, device=device)))

    def model(self, x):
        H, k = x.unbind(-1)
        one = torch.ones_like(H)
        zero = H * 0
        return torch.stack((torch.stack((zero, one * 4.32, one * 2.4, one * 2.5), -1), torch.stack((one * 1.5, one * 6.5, 6.5 / k, one * 2.8), -1), torch.stack((H, one * 8.0, one * 4.5, one * 3.3), -1)), -2)

    def predict(self, x):
        y = self.forward._evaluate(self.model(x), check=False)['rf'][:, :, 0]
        y = y / y[:, :, 10:11]
        return y.reshape(len(x), 3, 3, 151).mean(2)

    def __call__(self, x):
        white = torch.einsum('ij,brj->bri', self.invchol, self.predict(x)[:, :, self.mask] - self.y)
        return -0.5 * white.square().sum((1, 2))

class LocalLikelihoodNetwork(nn.Module):
    """Physical quadratic plus smooth learned correction in whitened logits.

    Bounds and the observation-specific calibration are part of the checkpoint.
    Different layer counts are handled by the teacher's parameter-to-layer map;
    each production run still has a fixed parameter dimension.
    """

    def __init__(self, z0, transform, energy0, linear, lower, upper, width=128):
        super().__init__()
        for name, value in dict(z0=z0, transform=transform, inverse_transform=np.linalg.inv(np.asarray(transform)), energy0=energy0, linear=linear, lower=lower, upper=upper).items():
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float64))
        d = len(z0)
        self.width = width
        self.correction = nn.Sequential(nn.Linear(d, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1)).float()
        self.quadratic_correction = nn.Parameter(torch.zeros(d, d, dtype=torch.float32))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def energy_y(self, y, quadratic_only=False):
        base = self.energy0 + (y * self.linear).sum(-1) + 0.5 * y.square().sum(-1)
        if quadratic_only:
            return base
        dtype = self.correction[0].weight.dtype
        r2 = y.square().sum(-1)
        delta = self.correction(torch.tanh(y / 3).to(dtype)).squeeze(-1).to(y.dtype)
        yy = y.to(dtype)
        a = (self.quadratic_correction + self.quadratic_correction.T) / 2
        quadratic = 0.5 * (yy * (yy @ a)).sum(-1).to(y.dtype) / (1 + r2 / 25)
        return base + r2 / (1 + r2) * delta + quadratic

    def y_from_z(self, z):
        return (z - self.z0) @ self.inverse_transform.T

    def log_z(self, z, quadratic_only=False):
        return -self.energy_y(self.y_from_z(z), quadratic_only)

    def log_physical(self, x, quadratic_only=False):
        span = self.upper - self.lower
        z = torch.logit(((x - self.lower) / span).clamp(1e-14, 1 - 1e-14))
        return self.log_z(z, quadratic_only) - box_jacobian(z, span)

def load_calibrated(path, device='cpu', double_network=False):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    c = saved['configuration']
    net = LocalLikelihoodNetwork(**c)
    net.load_state_dict(saved['state_dict'])
    net.to(device)
    if double_network:
        net.correction.double()
        net.quadratic_correction.data = net.quadratic_correction.data.double()
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return (net, saved['metadata'])
