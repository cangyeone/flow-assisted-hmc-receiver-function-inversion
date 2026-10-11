# Our CPU research program; ported imports use the public GRTM RF preview.
"""Known-posterior CPU test: H and kappa free, sediment fixed.

Quadrature and samplers share the same RF target and common multistart setup.
This tests sampler approximation error, not four-dimensional field accuracy.
"""
import sys, time, json, argparse
from pathlib import Path
import numpy as np, torch
from scipy.optimize import minimize
from scipy.stats import qmc
from cpu_hmc import Target, LOW, HIGH, plain, ROOT, hmc_sample, dram_sample, rank_diagnostics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=100)
    a = ap.parse_args()
    out = ROOT / 'results/conditional'
    out.mkdir(exist_ok=True, parents=True)
    d = dict(np.load(ROOT / 'data/processed/X26A.npz'))
    tt = d['time'][d['fit_mask']]
    C = 0.035 ** 2 * (0.9 * np.exp(-0.5 * ((tt[:, None] - tt[None, :]) / 0.5) ** 2) + 0.1 * np.eye(len(tt)))
    d['cov'] = np.stack([C] * 3)
    base = Target(d)
    truth = torch.tensor([[1.5, 2.4, 40.0, 1.78]])
    with torch.no_grad():
        clean = base.predict(truth)[0].numpy()
    rng = np.random.default_rng(a.seed)
    d['rf'] = clean.copy()
    d['rf'][:, d['fit_mask']] += rng.normal(size=(3, len(tt))) @ np.linalg.cholesky(C).T
    target4 = Target(d)
    lo = np.array([38.0, 1.72])
    hi = np.array([42.0, 1.84])

    def target(x):
        return target4(torch.cat((torch.tensor([[1.5, 2.4]]).expand(len(x), -1), x), -1))
    starts = qmc.Sobol(2, scramble=True, seed=728).random_base2(4)
    modes = []
    t0 = time.perf_counter()
    for s in starts:

        def fn(u):
            v = torch.tensor(u, requires_grad=True)
            l = -target((torch.tensor(lo) + torch.tensor(hi - lo) * v)[None])[0]
            g = torch.autograd.grad(l, v)[0]
            return (float(l.detach()), g.detach().numpy())
        m = minimize(fn, s, jac=True, method='L-BFGS-B', bounds=[(0.0001, 0.9999)] * 2, options={'maxiter': 150})
        modes.append({'x': lo + (hi - lo) * m.x, 'nll': m.fun, 'success': bool(m.success)})
    modes.sort(key=lambda m: m['nll'])
    setup = time.perf_counter() - t0
    best = modes[0]['x']
    initial = np.clip(np.array(best)[None] + rng.normal(size=(4, 2)) * 0.001 * (hi - lo), lo + 1e-05 * (hi - lo), hi - 1e-05 * (hi - lo))
    quadrature = []
    for n in (129, 257):
        axes = [np.linspace(l, h, n) for l, h in zip(lo, hi)]
        grid = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 2)
        lp = []
        with torch.no_grad():
            for start in range(0, len(grid), 128):
                lp.append(target(torch.tensor(grid[start:start + 128])).numpy())
        lp = np.concatenate(lp).reshape(n, n)
        w = np.exp(lp - lp.max())
        w[[0, -1], :] *= 0.5
        w[:, [0, -1]] *= 0.5
        w /= w.sum()
        flat = w.ravel()
        mean = (grid * flat[:, None]).sum(0)
        sd = np.sqrt(((grid - mean) ** 2 * flat[:, None]).sum(0))
        quadrature.append({'nodes': n, 'mean': mean, 'sd': sd, 'edge_mass': w[0].sum() + w[-1].sum() + w[:, 0].sum() + w[:, -1].sum()})
        np.savez_compressed(out / f'grid-{a.seed}-{n}.npz', axes=axes, density=w)
    results = []
    for method in ('hmc', 'dram'):
        t0 = time.perf_counter()
        if method == 'hmc':
            r = hmc_sample(target, initial, lo, hi, steps=4000, burn_in=1500, leapfrog_steps=8, target_accept=0.9, seed=a.seed, device='cpu')
        else:
            r = dram_sample(target, initial, lo, hi, steps=30000, burn_in=5000, proposal_scale=0.015, seed=a.seed, device='cpu')
        seconds = time.perf_counter() - t0
        p = r['posterior'].numpy()
        mean = p.mean((0, 1))
        diag = rank_diagnostics(p)
        np.savez_compressed(out / f'{method}-{a.seed}-chains.npz', chain=r['chain'].numpy(), log_prob=r['log_prob'].numpy())
        results.append({'method': method, 'seconds': seconds, 'steps': r['steps'], 'burn': r['burn_in'], 'diagnostics': diag, 'mean': mean, 'sd': p.reshape(-1, 2).std(0), 'mean_error_in_reference_sd': (mean - quadrature[-1]['mean']) / quadrature[-1]['sd'], 'truth_error': mean - [40, 1.78], 'acceptance': r.get('acceptance', r.get('acceptance_first', 0) + r.get('acceptance_delayed', 0)), 'production_divergences': r.get('production_divergences', [])})
    result = {'seed': a.seed, 'device': 'cpu', 'threads': 1, 'truth': [40.0, 1.78], 'fixed_sediment': [1.5, 2.4], 'bounds': [lo, hi], 'setup_seconds': setup, 'modes': modes, 'quadrature': quadrature, 'results': results}
    (out / f'check-{a.seed}.json').write_text(json.dumps(plain(result), indent=2))
    print(json.dumps(plain(result)), flush=True)
if __name__ == '__main__':
    main()
