# Our CPU-only archived research implementation; experimental, not default inference.
"""Exact-target correction and native CPU graph smoke tests."""
import os
import unittest
import numpy as np
import torch
from rf_surrogate_hmc import surrogate_hmc_sample
from rf_physics_surrogate import LocalLikelihoodNetwork, box_jacobian
from grtm.rf_hmc import rank_diagnostics

class SurrogateTests(unittest.TestCase):

    def test_misleading_force_still_targets_physical_density(self):
        torch.set_num_threads(2)
        lo = np.full(2, -6.0)
        hi = np.full(2, 6.0)

        def physical(x):
            return -0.5 * x.square().sum(-1)

        def proposal(z):
            x = -6 + 12 * torch.sigmoid(z)
            return -0.5 * ((x - x.new_tensor([0.5, -0.3])) / 1.1).square().sum(-1) + box_jacobian(z, x.new_tensor([12.0, 12.0]))
        initial = np.array([[-1.0, 0.5], [1.0, -0.5], [0.2, -1.0], [0.5, 1.0]])
        r = surrogate_hmc_sample(physical, proposal, initial, lo, hi, steps=10000, burn_in=1500, leapfrog_steps=6, seed=32110)
        a = r['posterior'].numpy()
        flat = a.reshape(-1, 2)
        diag = rank_diagnostics(a)
        np.testing.assert_allclose(flat.mean(0), 0, atol=0.05)
        np.testing.assert_allclose(flat.std(0), 1, atol=0.05)
        np.testing.assert_allclose(r['log_prob'].numpy(), -0.5 * (r['chain'].numpy() ** 2).sum(-1), rtol=1e-12, atol=1e-12)
        self.assertTrue(diag['converged'], diag)
        self.assertEqual(r['physical_gradient_evaluations'], 0)
        self.assertEqual(r['physical_endpoint_evaluations'], 10001)

    def test_center_anchor_and_independent_chain_rows(self):
        m = LocalLikelihoodNetwork([0.0, 0.0], np.eye(2), 3.0, [0.1, -0.2], [-1.0, -1.0], [1.0, 1.0])
        y = torch.tensor([[0.0, 0.0], [0.3, 0.2]], requires_grad=True)
        e = m.energy_y(y)
        g = torch.autograd.grad(e.sum(), y)[0]
        self.assertEqual(float(e[0].detach()), 3.0)
        np.testing.assert_allclose(g[0].detach(), [0.1, -0.2], atol=1e-12)
        leaf = torch.tensor([[0.3, 0.2]], requires_grad=True)
        g1 = torch.autograd.grad(m.energy_y(leaf).sum(), leaf)[0]
        np.testing.assert_allclose(g1.detach(), g[1:2].detach(), atol=1e-12)
if __name__ == '__main__':
    unittest.main()
