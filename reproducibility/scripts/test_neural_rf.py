# Our CPU-only archived research implementation; experimental, not default inference.
"""Scientific invariants for the variable-layer neural target."""
import unittest
import numpy as np
import torch
from neural_rf import RatioNetwork, NeuralRatioTarget, depth_projection, TIME, OUT

class NeuralRFTests(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(99)
        self.network = RatioNetwork().double().eval()
        for p in self.network.parameters():
            p.requires_grad_(False)
        self.layers = torch.tensor([[0.0, 4.0, 2.0, 2.1, 120.0, 60.0], [2.13, 6.1, 3.4, 2.8, 100000.0, 100000.0], [20.37, 6.8, 3.8, 2.8, 100000.0, 100000.0], [40.28, 8.0, 4.5, 3.3, 100000.0, 100000.0]], dtype=torch.float64)
        self.rf = torch.tensor(np.sin(TIME)[None].repeat(3, axis=0), dtype=torch.float64)

    def test_homogeneous_subdivision_invariance(self):
        rows = []
        for i, row in enumerate(self.layers):
            if i == len(self.layers) - 1:
                rows.append(row)
                continue
            for top in np.linspace(float(row[0]), float(self.layers[i + 1, 0]), 9)[:-1]:
                copy = row.clone()
                copy[0] = top
                rows.append(copy)
        split = torch.stack(rows)
        np.testing.assert_allclose(depth_projection(split), depth_projection(self.layers), atol=2e-13, rtol=0)
        a = self.network.score(self.rf, self.layers, [0.044, 0.057, 0.075], [0.03] * 3, 0.08)
        b = self.network.score(self.rf, split, [0.044, 0.057, 0.075], [0.03] * 3, 0.08)
        np.testing.assert_allclose(a, b, atol=1e-12, rtol=0)

    def test_interface_and_property_gradients(self):
        x = self.layers.clone().requires_grad_()
        score = self.network.score(self.rf, x, [0.044, 0.057, 0.075], [0.03] * 3, 0.08).sum()
        grad = torch.autograd.grad(score, x)[0]
        self.assertTrue(bool(torch.isfinite(grad).all()))
        for row, col in [(1, 0), (2, 0), (3, 0), (0, 2), (1, 1), (0, 5)]:
            h = 1e-05
            plus = self.layers.clone()
            minus = self.layers.clone()
            plus[row, col] += h
            minus[row, col] -= h
            fd = (self.network.score(self.rf, plus, [0.044, 0.057, 0.075], [0.03] * 3, 0.08) - self.network.score(self.rf, minus, [0.044, 0.057, 0.075], [0.03] * 3, 0.08)) / (2 * h)
            self.assertLess(abs(float(fd) - float(grad[row, col])), 2e-08)

    def test_independent_chain_rows_and_target_gradient(self):

        def model(x):
            rows = self.layers[None].repeat(len(x), 1, 1)
            rows[:, 3, 0] = x[:, 0]
            return rows
        target = NeuralRatioTarget(self.network, self.rf, [0.044, 0.057, 0.075], [0.03] * 3, model, prior=lambda x: -0.5 * ((x[:, 0] - 40.0) / 10.0).square())
        x = torch.tensor([[39.31], [43.17]], dtype=torch.float64, requires_grad=True)
        lp = target(x)
        grad = torch.autograd.grad(lp[0], x)[0]
        self.assertEqual(float(grad[1, 0]), 0.0)
        np.testing.assert_allclose(lp.detach(), torch.cat([target(row[None]).detach() for row in x]), atol=1e-12)

    def test_frozen_data_partitions_and_projection(self):
        ids = []
        for split in ('train', 'validation', 'test'):
            file = OUT / f'{split}.npz'
            if not file.exists():
                self.skipTest('Dataset not generated yet')
            a = dict(np.load(file))
            ids.append(set(a['source_id']))
            for i in [0, len(a['rf']) // 2, len(a['rf']) - 1]:
                m = torch.tensor(a['layers'][i, :a['lengths'][i]], dtype=torch.float64)
                np.testing.assert_allclose(depth_projection(m)[0], a['profile'][i], atol=5e-07, rtol=1e-07)
                self.assertTrue(np.isfinite(a['rf'][i]).all())
        self.assertTrue(ids[0].isdisjoint(ids[1]) and ids[0].isdisjoint(ids[2]) and ids[1].isdisjoint(ids[2]))
if __name__ == '__main__':
    unittest.main()
