# Our CPU-only archived research implementation; experimental, not default inference.
"""Scale normalization retains preference and amplifies local gradients."""
import unittest
import numpy as np
import torch
from neural_rf_local_scaled import scaled_local_pair_loss

class Tests(unittest.TestCase):

    def test_ties_and_physical_direction(self):
        scores = torch.zeros(1, 6, dtype=torch.float64, requires_grad=True)
        value, _, _ = scaled_local_pair_loss(scores, torch.zeros_like(scores))
        torch.testing.assert_close(torch.autograd.grad(value, scores)[0], torch.zeros_like(scores))
        scores = torch.zeros(1, 6, dtype=torch.float64, requires_grad=True)
        physical = torch.tensor([[-10.0, -1.0, -10.0, -10.0, -10.0, -10.0]], dtype=torch.float64)
        value, _, _ = scaled_local_pair_loss(scores, physical)
        gradient = torch.autograd.grad(value, scores)[0]
        self.assertLess(float(gradient[0, 1]), 0.0)
        self.assertGreater(float(gradient[0, 0]), 0.0)

    def test_gradient_finite_difference(self):
        x = torch.tensor([[0.001, -0.0021, 0.003, -0.001, 0.002, -0.003]], dtype=torch.float64, requires_grad=True)
        lp = torch.tensor([[-1.0, -3.0, -2.0, -5.0, -8.0, -20.0]], dtype=torch.float64)
        value, _, _ = scaled_local_pair_loss(x, lp)
        gradient = torch.autograd.grad(value, x)[0].numpy()[0]
        numeric = []
        with torch.no_grad():
            for j in range(6):
                d = torch.zeros_like(x)
                d[0, j] = 1e-07
                numeric.append(float((scaled_local_pair_loss(x + d, lp)[0] - scaled_local_pair_loss(x - d, lp)[0]) / 2e-07))
        np.testing.assert_allclose(gradient, numeric, rtol=1e-07, atol=1e-07)
if __name__ == '__main__':
    unittest.main()
