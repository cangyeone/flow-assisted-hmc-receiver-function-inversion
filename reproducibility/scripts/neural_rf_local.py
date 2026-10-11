# Our CPU research program; ported imports use the public GRTM RF preview.
"""Local pair discrimination, supervised by the physical RF likelihood.

A changed model is not automatically a negative: noise can make it preferable
to its source. The shared score accepts arbitrary layer counts through the
unchanged depth projection. Scores after this training are heuristic ranking
scores, not calibrated likelihood ratios or probabilities of true geology.
"""
import numpy as np
import torch
import torch.nn.functional as F
from neural_rf import load_network
SCALES = [0.0025, 0.005, 0.01, 0.02, 0.05]
MODES = ['joint_velocity', 'vp_vs_ratio', 'interfaces', 'time_shift']

def perturb_model(layers, tau, scale, mode, direction):
    """Same direction at each scale; preserve positive thickness and Vp > Vs.

    Interface changes operate on thickness fractions, preserving the total
    Moho depth. Velocity changes leave the halfspace unchanged. Time shift has
    an absolute scale of 2*scale seconds. No noise-dependent accept/reject step.
    """
    m = np.asarray(layers, float).copy()
    n = len(m) - 1
    d = np.asarray(direction, float)
    t = float(tau)
    if mode == 'joint_velocity':
        m[:-1, 1:3] *= np.exp(scale * d[:n, None])
    elif mode == 'vp_vs_ratio':
        m[:-1, 1] *= np.exp(0.5 * scale * d[:n])
        m[:-1, 2] *= np.exp(-0.5 * scale * d[:n])
        if np.any(m[:, 1] <= m[:, 2]):
            raise ValueError('Perturbation crossed Vp=Vs')
    elif mode == 'interfaces':
        thickness = np.diff(m[:, 0]) * np.exp(scale * d[:n])
        thickness *= m[-1, 0] / thickness.sum()
        m[1:, 0] = np.cumsum(thickness)
    elif mode == 'time_shift':
        t += 2 * scale * d[0]
    else:
        raise ValueError(mode)
    return (m, t)

def local_pair_loss(scores, physical_log_likelihood, temperature=1.0, gap_weight=0.05, gap_clip=8.0):
    """All candidate pairs; source identity never supplies a good/bad label.

    Soft targets sigmoid(delta physical log likelihood / temperature) avoid
    assigning high confidence to nearly tied models. The bounded gap loss
    teaches relative strength without letting distant mismatches dominate.
    """
    a, b = torch.triu_indices(scores.shape[1], scores.shape[1], 1, device=scores.device)
    predicted = scores[:, a] - scores[:, b]
    exact = (physical_log_likelihood[:, a] - physical_log_likelihood[:, b]).detach()
    preference = torch.sigmoid(exact / temperature)
    rank = F.binary_cross_entropy_with_logits(predicted, preference)
    gap = F.smooth_l1_loss(predicted, (exact / temperature).clamp(-gap_clip, gap_clip))
    return (rank + gap_weight * gap, rank, gap)

class LocalPairDiscriminator:
    """Shared scalar scoring gives antisymmetric, transitive pair logits."""

    def __init__(self, network):
        self.network = network

    def pair_logits(self, rf, first, second, p, sigma, tau_first=0.0, tau_second=0.0, p_sd=None):
        return self.network.score(rf, first, p, sigma, tau_first, p_sd) - self.network.score(rf, second, p, sigma, tau_second, p_sd)

def load_local_network(checkpoint, dtype=torch.float64, device='cpu'):
    network, metadata = load_network(checkpoint, dtype)
    if metadata.get('training_objective') != 'global_BCE_and_physical_local_pairs':
        raise ValueError('Expected a physical-supervised local-pair checkpoint')
    return (network.to(device), metadata)
