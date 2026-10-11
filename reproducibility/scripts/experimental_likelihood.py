"""Our experimental, observation-specific CPU likelihood approximation."""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
def box_jacobian(z, span):
    return (span.log() - F.softplus(z) - F.softplus(-z)).sum(-1)

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
