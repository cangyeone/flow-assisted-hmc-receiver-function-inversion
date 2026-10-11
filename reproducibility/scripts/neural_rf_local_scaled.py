# Our CPU research program; ported imports use the public GRTM RF preview.
"""Scale-aware local comparison loss, preserving physical preference signs.

Nearly identical encoder inputs otherwise cancel in score differences and
their parameter gradients. Divide the predicted pair gap by its declared
perturbation distance during training. Targets still come from the physical
likelihood; source identity supplies no label. Production uses the shared
scalar score as a heuristic guide, not a calibrated density ratio.
"""
import torch
import torch.nn.functional as F
from neural_rf_local import SCALES

def scaled_local_pair_loss(scores, physical_log_likelihood, temperature=1.0, gap_weight=0.05, gap_clip=8.0):
    if scores.shape[1] != len(SCALES) + 1:
        raise ValueError('Expected source plus the five declared local scales')
    a, b = torch.triu_indices(scores.shape[1], scores.shape[1], 1, device=scores.device)
    scale = scores.new_tensor([0.0] + SCALES)
    distance = (scale[a] - scale[b]).abs().clamp_min(min(SCALES))
    predicted = (scores[:, a] - scores[:, b]) / distance
    exact = (physical_log_likelihood[:, a] - physical_log_likelihood[:, b]).detach() / temperature
    rank = F.binary_cross_entropy_with_logits(predicted, torch.sigmoid(exact))
    gap = F.smooth_l1_loss(predicted, exact.clamp(-gap_clip, gap_clip))
    return (rank + gap_weight * gap, rank, gap)
