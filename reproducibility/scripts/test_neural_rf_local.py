# Our CPU-only archived research implementation; experimental, not default inference.
"""Physical labels, layer invariance, and CPU/CPU teacher alignment."""
import os
import unittest
import numpy as np
import torch
from neural_rf import TIME, RatioNetwork, depth_projection
from neural_rf_local import perturb_model, local_pair_loss, LocalPairDiscriminator, SCALES, MODES
from neural_rf_experiment import rf_forward, noise_covariance
from train_neural_rf_local import cpu_forward, physical_log_likelihood

class LocalTests(unittest.TestCase):

    def setUp(self):
        self.layers = np.array([[0.0, 4.32, 2.4, 2.1, 120.0, 60.0], [2.13, 6.1, 3.4, 2.8, np.inf, np.inf], [20.37, 6.8, 3.8, 2.8, np.inf, np.inf], [40.28, 8.0, 4.5, 3.3, np.inf, np.inf]])

    def test_physical_preference_can_favor_perturbation(self):
        score = torch.zeros((1, 2), dtype=torch.float64, requires_grad=True)
        loss, _, _ = local_pair_loss(score, torch.tensor([[-5.0, -1.0]]))
        grad = torch.autograd.grad(loss, score)[0]
        self.assertGreater(float(grad[0, 0]), 0.0)
        self.assertLess(float(grad[0, 1]), 0.0)
        score = torch.zeros((1, 2), dtype=torch.float64, requires_grad=True)
        loss, _, _ = local_pair_loss(score, torch.zeros_like(score))
        torch.testing.assert_close(torch.autograd.grad(loss, score)[0], torch.zeros_like(score))

    def test_interfaces_and_halfspace_remain_valid(self):
        for mode in MODES:
            for scale in SCALES:
                m, t = perturb_model(self.layers, 0.04, scale, mode, [0.5, -0.8, 1.2])
                self.assertTrue((np.diff(m[:, 0]) > 0).all())
                self.assertTrue((m[:, 1] > m[:, 2]).all())
                np.testing.assert_allclose(m[-1], self.layers[-1], atol=1e-12)
                self.assertTrue(np.isfinite(t))

    def test_shared_pair_score_invariance_and_gradient(self):
        torch.manual_seed(99)
        net = RatioNetwork().double().eval()
        pair = LocalPairDiscriminator(net)
        first = torch.tensor(self.layers)
        second = first.clone()
        second[0, 2] *= 1.003
        rf = torch.tensor(np.sin(TIME)[None].repeat(3, 0))
        args = ([0.044, 0.057, 0.075], [0.03] * 3)
        a = pair.pair_logits(rf, first, second, *args)
        b = pair.pair_logits(rf, second, first, *args)
        torch.testing.assert_close(a, -b, rtol=0, atol=1e-12)
        row = first[0].clone()
        row[0] = 0.91
        split = torch.cat((first[:1], row[None], first[1:]))
        torch.testing.assert_close(depth_projection(split), depth_projection(first), rtol=0, atol=1e-12)
        torch.testing.assert_close(pair.pair_logits(rf, split, second, *args), a, rtol=0, atol=1e-12)
        x = second.clone().requires_grad_()
        value = pair.pair_logits(rf, first, x, *args).sum()
        g = torch.autograd.grad(value, x)[0]
        h = 1e-05
        up = second.clone()
        down = second.clone()
        up[0, 2] += h
        down[0, 2] -= h
        fd = (pair.pair_logits(rf, first, up, *args) - pair.pair_logits(rf, first, down, *args)) / (2 * h)
        self.assertLess(abs(float(fd) - float(g[0, 2])), 2e-08)
if __name__ == '__main__':
    unittest.main()
