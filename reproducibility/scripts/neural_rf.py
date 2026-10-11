# Our CPU research program; ported imports use the public GRTM RF preview.
"""Differentiable, subdivision-invariant RF/structure neural ratio prototype.

Scores are matched-source versus independent-pair logits, not truth
probabilities. The learned target replaces, rather than multiplies, the RF
likelihood. HMC dimensions stay fixed even though this encoder accepts K layers.
"""
import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/neural-rf-v1'
TIME = np.arange(-2.0, 28.01, 0.2)
MASK = (TIME >= 0.799) & (TIME <= 26.001)
EDGES = np.r_[np.linspace(0.0, 10.0, 101), np.arange(10.5, 80.01, 0.5)]

def depth_projection(layers, edges=None):
    """Exact interval averages of Vp, Vs, rho, 1/Qp, 1/Qs to 80 km.

    Arbitrarily many finite layers are accepted. The last row is a halfspace.
    min/max overlap derivatives move interfaces without hard depth indexing.
    Splitting a homogeneous layer leaves every average exactly unchanged.
    These averages do not preserve structure below the chosen depth resolution.
    """
    if layers.ndim == 2:
        layers = layers[None]
    if layers.ndim != 3 or layers.shape[-1] not in (4, 6):
        raise ValueError('Expected [batch, layer, 4 or 6], with final halfspace')
    e = torch.as_tensor(EDGES if edges is None else edges, dtype=layers.dtype, device=layers.device)
    top = layers[..., 0]
    bottom = torch.cat((top[:, 1:], e[-1].expand(len(layers), 1)), -1)
    overlap = (torch.minimum(bottom[..., None], e[1:]) - torch.maximum(top[..., None], e[:-1])).clamp_min(0)
    qinv = layers[..., 4:].reciprocal() if layers.shape[-1] == 6 else torch.zeros_like(layers[..., :2])
    values = torch.cat((layers[..., 1:4], qinv), -1)
    return torch.einsum('bkd,bkc->bcd', overlap, values) / (e[1:] - e[:-1])

def context(p, sigma, tau, p_sd=None):
    p = torch.as_tensor(p)
    if p.ndim == 1:
        p = p[None]
    sigma = torch.as_tensor(sigma, dtype=p.dtype, device=p.device).reshape(-1, 3)
    tau = torch.as_tensor(tau, dtype=p.dtype, device=p.device).reshape(-1, 1)
    p_sd = torch.zeros_like(p) if p_sd is None else torch.as_tensor(p_sd, dtype=p.dtype, device=p.device).reshape(-1, 3)
    batch = max(len(p), len(sigma), len(tau), len(p_sd))
    return torch.cat(((p.expand(batch, -1) - 0.06) / 0.02, sigma.expand(batch, -1).log() - np.log(0.05), tau.expand(batch, -1) / 0.25, p_sd.expand(batch, -1) / 0.005), -1)

class RatioNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.register_buffer('rf_center', torch.zeros(1, 3, len(TIME)))
        self.register_buffer('rf_scale', torch.ones(1, 3, len(TIME)))
        self.register_buffer('rf_mask', torch.as_tensor(MASK)[None, None])
        self.register_buffer('profile_center', torch.tensor([6.0, 3.4, 2.7, 0.0, 0.0])[None, :, None])
        self.register_buffer('profile_scale', torch.tensor([2.0, 1.0, 0.5, 0.02, 0.04])[None, :, None])
        self.rf_encoder = nn.Sequential(nn.Conv1d(3, 16, 7, padding=3), nn.SiLU(), nn.Conv1d(16, 24, 5, stride=2, padding=2), nn.SiLU(), nn.Conv1d(24, 32, 5, stride=2, padding=2), nn.SiLU(), nn.AdaptiveAvgPool1d(20), nn.Flatten(), nn.Linear(640, 96), nn.SiLU())
        self.model_encoder = nn.Sequential(nn.Conv1d(5, 16, 5, padding=2), nn.SiLU(), nn.Conv1d(16, 24, 5, stride=2, padding=2), nn.SiLU(), nn.Conv1d(24, 32, 5, stride=2, padding=2), nn.SiLU(), nn.AdaptiveAvgPool1d(30), nn.Flatten(), nn.Linear(960, 96), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(96 * 4 + 10, 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, rf, profile, conditions):
        a = self.rf_encoder((rf - self.rf_center) / self.rf_scale * self.rf_mask)
        b = self.model_encoder((profile - self.profile_center) / self.profile_scale)
        return self.head(torch.cat((a, b, a * b, (a - b).square(), conditions), -1)).squeeze(-1)

    def score(self, rf, layers, p, sigma, tau=0.0, p_sd=None):
        dtype = self.rf_center.dtype
        layers = torch.as_tensor(layers, dtype=dtype, device=self.rf_center.device)
        profile = depth_projection(layers)
        rf = torch.as_tensor(rf, dtype=dtype, device=layers.device)
        if rf.ndim == 2:
            rf = rf[None]
        batch = len(profile)
        cond = context(torch.as_tensor(p, dtype=dtype, device=layers.device), sigma, torch.as_tensor(tau, dtype=dtype, device=layers.device).expand(batch), p_sd)
        return self(rf.expand(batch, -1, -1), profile, cond.expand(batch, -1))

def load_network(checkpoint=None, dtype=torch.float64):
    file = Path(checkpoint or OUT / 'network.pt')
    saved = torch.load(file, map_location='cpu', weights_only=True)
    model = RatioNetwork().to(dtype=dtype)
    model.load_state_dict(saved['state_dict'])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return (model, saved['metadata'])

class NeuralRatioTarget:
    """Approximate physical-coordinate posterior: log prior + neural log ratio.

    The caller supplies a differentiable parameter-to-layer mapping and a
    normalized/unnormalized log prior. Bounds are applied by the HMC sampler.
    This does not assert calibrated intervals or an exact simulator posterior.
    """

    def __init__(self, network, rf, p, sigma, model, prior=None, tau=None, p_sd=None):
        self.network, self.rf, self.p, self.sigma = (network, rf, p, sigma)
        self.model, self.prior, self.tau, self.p_sd = (model, prior, tau, p_sd)

    def __call__(self, x):
        shift = self.tau(x) if self.tau is not None else torch.zeros_like(x[:, 0])
        value = self.network.score(self.rf, self.model(x), self.p, self.sigma, shift, self.p_sd)
        return value if self.prior is None else value + self.prior(x)

def auc(positive, negative):
    from scipy.stats import rankdata
    a, b = (np.asarray(positive), np.asarray(negative))
    rank = rankdata(np.r_[a, b])
    return float((rank[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))
