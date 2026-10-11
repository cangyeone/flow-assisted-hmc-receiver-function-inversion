"""Bijective coordinates for variable-layer solid Earth structures.

Density measure: finite-layer (thickness, Vs, Vp/Vs, rho, Qs) and P shift.
The halfspace is fixed. Arbitrary finite-layer counts are possible in this
transform; the learned layer-count distribution has explicitly finite support.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F

HMIN, HMAX, THICKNESS_MIN = 12., 80., .02


def active_mask(count, max_layers, dtype=torch.float32):
    m = (torch.arange(max_layers, device=count.device)[None] < count[:, None])
    mask = torch.zeros(len(count), max_layers + 1, 5, device=count.device, dtype=dtype)
    mask[:, :max_layers] = m[..., None]
    mask[:, -1, 0] = 1
    return mask


def decode(u, count):
    """Return padded GRTM layers, shifts and log|d physical / d u|.

    Last active thickness coordinate is total depth, previous coordinates are
    additive log ratios of thickness fractions to the last layer. Padding is
    ignored. Halfspace lives immediately after the active finite layers.
    """
    b, slots, _ = u.shape
    kmax = slots - 1
    ix = torch.arange(kmax, device=u.device)[None]
    valid = ix < count[:, None]
    last = ix == count[:, None] - 1
    count_float = count.to(u.dtype)
    uh = (u[:, :kmax, 0] * last).sum(-1)
    sh = torch.sigmoid(uh)
    H = HMIN + (HMAX - HMIN) * sh
    if kmax * THICKNESS_MIN >= HMIN:
        raise ValueError('Increase HMIN for this many layers')
    weights = torch.softmax(torch.where(last, 0., u[:, :kmax, 0]).masked_fill(~valid, -torch.inf), -1)
    h = (THICKNESS_MIN + (H - count_float * THICKNESS_MIN)[:, None] * weights) * valid
    s = torch.sigmoid(u[:, :kmax, 1:])
    vs = .3 + 4. * s[..., 0]
    upper = torch.minimum(torch.full_like(vs, 3.2), 8.8 / vs)
    ratio = 1.55 + (upper - 1.55) * s[..., 1]
    rho = 1.6 + 1.6 * s[..., 2]
    qs = 10. * torch.exp(math.log(100.) * s[..., 3])
    tops = F.pad(h.cumsum(-1), (1, 0))[:, :kmax]
    finite = torch.stack((tops, vs * ratio, vs, rho, 2 * qs, qs), -1)
    layers = torch.zeros(b, kmax + 1, 6, dtype=u.dtype, device=u.device)
    layers[:, :kmax] = finite * valid[..., None]
    layers[:, :, 4:] = torch.where(layers[:, :, 4:] > 0, layers[:, :, 4:], torch.full_like(layers[:, :, 4:], torch.inf))
    half = torch.stack((H, H * 0 + 8., H * 0 + 4.5, H * 0 + 3.3,
                        H * 0 + torch.inf, H * 0 + torch.inf), -1)
    layers[torch.arange(b, device=u.device), count] = half
    tau = .3 * torch.tanh(u[:, -1, 0])
    # Stable sigmoid Jacobians, including the Vs-dependent upper ratio bound.
    log_sig = -F.softplus(-u[:, :kmax, 1:]) - F.softplus(u[:, :kmax, 1:])
    widths = torch.stack((torch.full_like(vs, 4.), upper - 1.55,
                          torch.full_like(vs, 1.6), qs * math.log(100.)), -1)
    lj = ((widths.log() + log_sig) * valid[..., None]).sum((1, 2))
    lj = lj + (count - 1) * (H - count_float * THICKNESS_MIN).log()
    lj = lj + torch.where(valid, weights.clamp_min(torch.finfo(u.dtype).tiny).log(), 0.).sum(-1)
    lj = lj + math.log(HMAX - HMIN) - F.softplus(uh) - F.softplus(-uh)
    t = u[:, -1, 0]
    lj = lj + math.log(.3) + 2 * (math.log(2.) - t - F.softplus(-2 * t))
    return layers, tau, lj


def encode(layers, count, tau, max_layers):
    """Inverse of decode on its physical support (throws outside support)."""
    out = torch.zeros(len(count), max_layers + 1, 5, dtype=layers.dtype, device=layers.device)
    for k in count.unique().tolist():
        take = count == k
        m = layers[take, :k + 1]
        h = m[:, 1:, 0] - m[:, :-1, 0]
        H = m[:, -1, 0]
        vs = m[:, :-1, 2]
        ratio = m[:, :-1, 1] / vs
        upper = torch.minimum(torch.full_like(vs, 3.2), 8.8 / vs)
        scaled = torch.stack(((vs - .3) / 4., (ratio - 1.55) / (upper - 1.55),
                              (m[:, :-1, 3] - 1.6) / 1.6,
                              (m[:, :-1, 5] / 10.).log() / math.log(100.)), -1)
        hs = (H - HMIN) / (HMAX - HMIN)
        if not bool(((scaled > 0) & (scaled < 1)).all() & (hs > 0).all() & (hs < 1).all() & (h > THICKNESS_MIN).all()):
            raise ValueError('Structure is outside the configured coordinate support')
        fractions = (h - THICKNESS_MIN) / (H - k * THICKNESS_MIN)[:, None]
        q = out[take].clone()
        if k > 1:
            q[:, :k - 1, 0] = (fractions[:, :-1] / fractions[:, -1:]).log()
        q[:, k - 1, 0] = torch.logit(hs)
        q[:, :k, 1:] = torch.logit(scaled)
        out[take] = q
    if not bool((tau.abs() < .3).all()):
        raise ValueError('Shift outside (-.3,.3) s')
    out[:, -1, 0] = torch.atanh(tau / .3)
    return out


def sample_prior(rng, count, max_layers):
    """Declared simulator prior, with explicit sediment and LVZ mixtures.

    Density and Q are nuisance parameters inferred rather than silently fixed.
    Five parameters per finite layer plus a shift; no truth-dependent filtering.
    """
    n = len(count)
    u = np.zeros((n, max_layers + 1, 5), np.float32)
    for i, k in enumerate(count):
        H = rng.uniform(13., 79.)
        sediment = int(rng.integers(1, min(4, k) + 1)) if k > 1 and rng.random() < .8 else 0
        if sediment >= k:
            sediment = k - 1
        if sediment:
            hs = rng.uniform(max(.05, sediment * .021), min(10., .6 * H))
            hs_parts = .021 + (hs - sediment * .021) * rng.dirichlet(np.full(sediment, 2.))
            nc = k - sediment
            hc_parts = .021 + (H - hs - nc * .021) * rng.dirichlet(np.full(nc, 2.))
            h = np.r_[hs_parts, hc_parts]
            v0 = rng.uniform(.35, 2.8)
            sv = v0 + (3.4 - v0) * rng.uniform(.1, .9) * (np.arange(sediment) + .5) / sediment
        else:
            h = .021 + (H - k * .021) * rng.dirichlet(np.full(k, 2.))
            sv = np.empty(0)
        nc = k - sediment
        cv = np.clip(rng.uniform(2.9, 3.6) + rng.uniform(-.15, .6) * np.linspace(0, 1, nc)
                     + rng.normal(0, .12, nc), 2.65, 4.2)
        if nc >= 3 and rng.random() < .35:
            cv[int(rng.integers(1, nc))] -= rng.uniform(.15, .5)
        vs = np.r_[sv, cv]
        ratio = np.r_[rng.uniform(1.65, 3., sediment), rng.uniform(1.58, 2.03, nc)]
        upper = np.minimum(3.2, 8.8 / vs)
        ratio = np.minimum(ratio, upper - .02)
        rho = np.r_[rng.uniform(1.65, 2.6, sediment), rng.uniform(2.65, 3.05, nc)]
        qs = np.r_[np.exp(rng.uniform(np.log(12), np.log(250), sediment)),
                    np.exp(rng.uniform(np.log(250), np.log(950), nc))]
        frac = (h - THICKNESS_MIN) / (H - k * THICKNESS_MIN)
        u[i, :k - 1, 0] = np.log(frac[:-1] / frac[-1])
        u[i, k - 1, 0] = np.log((H - HMIN) / (HMAX - H))
        s = np.stack(((vs - .3) / 4., (ratio - 1.55) / (upper - 1.55),
                      (rho - 1.6) / 1.6, np.log(qs / 10.) / np.log(100.)), -1)
        u[i, :k, 1:] = np.log(s / (1 - s))
        u[i, -1, 0] = np.arctanh(rng.uniform(-.29, .29) / .3)
    return u
