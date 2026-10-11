# Our CPU research program; ported imports use the public GRTM RF preview.
"""CPU-only RF inversion with identical physical targets for HMC and DRAM.

Station model: sediment Vs and thickness, total crustal thickness and crustal
Vp/Vs; Vp_crust=6.5, sediment Vp/Vs=1.8. All timings include warmup.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import argparse, json, time, sys, platform, hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.stats import qmc
from grtm import JointRFForward
from grtm.rf_hmc import hmc_sample, rank_diagnostics
from grtm.rf_bayes import dram_sample
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parents[1]
LOW = np.array([0.2, 1.5, 22.0, 1.6])
HIGH = np.array([8.0, 3.2, 65.0, 2.0])
NAMES = ['sediment_thickness_km', 'sediment_Vs_km_s', 'Moho_depth_km', 'crust_VpVs']

def layers(x):
    hs, vs, H, k = x.unbind(-1)
    zero = hs * 0
    one = zero + 1
    return torch.stack((torch.stack((zero, 1.8 * vs, vs, one * 2.5), -1), torch.stack((hs, one * 6.5, 6.5 / k, one * 2.8), -1), torch.stack((H, one * 8.0, one * 4.5, one * 3.3), -1)), -2)

class Target:

    def __init__(self, data):
        self.data = data
        self.mask = torch.tensor(data['fit_mask'])
        self.y = torch.tensor(data['rf'][:, data['fit_mask']])
        self.invchol = torch.linalg.inv(torch.linalg.cholesky(torch.tensor(data['cov'])))
        self.forward = JointRFForward(data['p'], gaussians=[2.5], periods=[0.0], dt=0.2, n_samples=151, t0=-2.0, nfft=512, device='cpu')

    def predict(self, x, forward=None):
        rf = (forward or self.forward)._evaluate(layers(x), check=False)['rf'][:, :, 0, :]
        return rf / rf[:, :, 10:11]

    def __call__(self, x):
        residual = self.predict(x)[:, :, self.mask] - self.y
        white = torch.einsum('rij,brj->bri', self.invchol, residual)
        return -0.5 * white.square().sum((1, 2))

def plain(x):
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, (np.ndarray, torch.Tensor)):
        return x.tolist()
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    return x

def multistart(target, n=16, seed=728):
    starts = qmc.Sobol(4, scramble=True, seed=seed).random_base2(int(np.ceil(np.log2(n))))[:n]
    answers = []
    t0 = time.perf_counter()
    for i, u in enumerate(starts):

        def fn(z):
            v = torch.tensor(z, requires_grad=True)
            loss = -target((torch.tensor(LOW) + torch.tensor(HIGH - LOW) * v)[None])[0]
            g = torch.autograd.grad(loss, v)[0]
            return (float(loss.detach()), g.detach().numpy())
        out = minimize(fn, u, jac=True, method='L-BFGS-B', bounds=[(0.0001, 0.9999)] * 4, options={'maxiter': 180, 'ftol': 1e-09, 'gtol': 1e-05})
        answers.append({'start_index': i, 'initial': LOW + (HIGH - LOW) * u, 'x': LOW + (HIGH - LOW) * out.x, 'negative_log_density': float(out.fun), 'success': bool(out.success), 'iterations': out.nit})
    answers.sort(key=lambda a: a['negative_log_density'])
    return (answers, time.perf_counter() - t0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('station')
    ap.add_argument('--method', choices=['hmc', 'dram', 'map'], default='hmc')
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--burn', type=int, default=1000)
    ap.add_argument('--starts', type=int, default=16)
    ap.add_argument('--seed', type=int, default=728)
    ap.add_argument('--out', default='results')
    args = ap.parse_args()
    outdir = ROOT / args.out
    outdir.mkdir(exist_ok=True, parents=True)
    data = dict(np.load(ROOT / 'data/processed' / f'{args.station}.npz'))
    target = Target(data)
    mpath = outdir / f'{args.station}-map.json'
    if mpath.exists():
        saved = json.loads(mpath.read_text())
        modes = saved['candidates']
        map_sec = saved['map_seconds']
    else:
        modes, map_sec = multistart(target, args.starts, args.seed)
        mpath.write_text(json.dumps(plain({'candidates': modes, 'map_seconds': map_sec, 'seed': args.seed, 'bounds': [LOW, HIGH], 'names': NAMES}), indent=2))
    print(args.station, 'MAP', map_sec, modes[0]['negative_log_density'], flush=True)
    if args.method == 'map':
        return
    best = np.array(modes[0]['x'])
    rng = np.random.default_rng(args.seed)
    selected = [best]
    for m in modes[1:]:
        x = np.array(m['x'])
        if m['negative_log_density'] - modes[0]['negative_log_density'] <= 5 and all((np.linalg.norm((x - v) / (HIGH - LOW)) > 0.03 for v in selected)):
            selected.append(x)
    initial = np.stack([selected[i % len(selected)] for i in range(4)]) + rng.normal(size=(4, 4)) * 0.0005 * (HIGH - LOW)
    initial = np.clip(initial, LOW + 1e-05 * (HIGH - LOW), HIGH - 1e-05 * (HIGH - LOW))
    start = time.perf_counter()
    common = dict(steps=args.steps, burn_in=args.burn, device='cpu', seed=args.seed, progress=lambda n, total, lp: print(args.station, args.method, n, flush=True))
    if args.method == 'hmc':
        result = hmc_sample(target, initial, LOW, HIGH, leapfrog_steps=8, target_accept=0.9, progress_interval=500, **common)
    else:
        result = dram_sample(target, initial, LOW, HIGH, proposal_scale=0.015, progress_interval=5000, **common)
    elapsed = time.perf_counter() - start
    post = result['posterior'].detach().numpy()
    diag = rank_diagnostics(post)
    np.savez_compressed(outdir / f'{args.station}-{args.method}-chains.npz', chain=result['chain'].numpy(), log_prob=result['log_prob'].numpy(), initial=initial)
    flat = post.reshape(-1, 4)
    take = np.linspace(0, len(flat) - 1, 256, dtype=int)
    pred = []
    test_pred = []
    testf = JointRFForward(data['test_p'], gaussians=[2.5], periods=[0.0], dt=0.2, n_samples=151, t0=-2.0, nfft=512, device='cpu')
    with torch.no_grad():
        for chunk in np.array_split(flat[take], 16):
            pred.append(target.predict(torch.tensor(chunk)).numpy())
            test_pred.append(target.predict(torch.tensor(chunk), testf).numpy())
    pred = np.concatenate(pred)
    test_pred = np.concatenate(test_pred)
    mask = data['fit_mask']
    np.savez_compressed(outdir / f'{args.station}-{args.method}-predictive.npz', time=data['time'], train_mean=pred.mean(0), test_mean=test_pred.mean(0), train_q=np.quantile(pred, [0.025, 0.975], axis=0), test_q=np.quantile(test_pred, [0.025, 0.975], axis=0))
    summary = {'station': args.station, 'method': args.method, 'device': 'cpu', 'threads': torch.get_num_threads(), 'host': platform.node(), 'processor': platform.processor(), 'torch': torch.__version__, 'steps': args.steps, 'burn': args.burn, 'seed': args.seed, 'names': NAMES, 'bounds': [LOW, HIGH], 'map_seconds': map_sec, 'sampling_seconds': elapsed, 'diagnostics': diag, 'acceptance': result.get('acceptance', result.get('acceptance_first', 0) + result.get('acceptance_delayed', 0)), 'mean': flat.mean(0), 'q025': np.quantile(flat, 0.025, axis=0), 'q975': np.quantile(flat, 0.975, axis=0), 'chain_means': post.mean(0), 'initial': initial, 'near_optimal_distinct_candidates': len(selected), 'train_rmse': np.mean((pred.mean(0)[:, mask] - data['rf'][:, mask]) ** 2) ** 0.5, 'holdout_rmse': np.mean((test_pred.mean(0)[:, mask] - data['test_rf'][:, mask]) ** 2) ** 0.5, 'production_divergences': result.get('production_divergences', []), 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'interval_status': 'diagnostics_passed' if diag['converged'] else 'provisional_unconverged'}
    if args.method == 'hmc':
        summary.update(step_size=result['step_size'], gradient_evaluations=result['gradient_evaluations'])
    (outdir / f'{args.station}-{args.method}.json').write_text(json.dumps(plain(summary), indent=2))
    print(json.dumps(plain(summary)), flush=True)
if __name__ == '__main__':
    main()
