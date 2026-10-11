# Our CPU research program; ported imports use the public GRTM RF preview.
"""Sample the training-selected RF model on CPU, with explicit local diagnostics."""
import argparse, hashlib, json, platform, time
import numpy as np
import torch
from scipy.optimize import minimize
from fit_refinement import ROOT, RefinedTarget, load_events, stack, score, plain
from grtm.rf_hmc import hmc_sample, rank_diagnostics

def main():
    p = argparse.ArgumentParser()
    p.add_argument('station')
    p.add_argument('--steps', type=int, default=5000)
    p.add_argument('--burn', type=int, default=2000)
    a = p.parse_args()
    out = ROOT / 'results/fit-refinement-v3/hmc'
    out.mkdir(exist_ok=True, parents=True)
    selection = json.loads((ROOT / 'results/fit-refinement-v3' / f'{a.station}.json').read_text())
    spec = next((c for c in selection['checks'] if c['selected']))
    original, groups = load_events(a.station)
    data = original.copy()
    if spec['model'] != 'legacy':
        data['fit_mask'] = (data['time'] >= -0.601) & (data['time'] <= 26.001)
        data['cov'] = np.stack([stack([e for e in g if e['split'] == 'train'], data['fit_mask'])[2] for g in groups])
    target = RefinedTarget(data, spec['model'], covariance=spec['covariance'])
    lo, hi = (torch.tensor(target.low), torch.tensor(target.high))
    start = time.perf_counter()
    best = np.array(spec['full_candidates'][0]['x'])
    u = np.clip((best - target.low) / (target.high - target.low), 0.01, 0.99)
    q0 = np.log(u / (1 - u))

    def transformed(q):
        s = torch.sigmoid(q)
        logjac = (torch.nn.functional.logsigmoid(q) + torch.nn.functional.logsigmoid(-q)).sum()
        return -target((lo + (hi - lo) * s)[None])[0] - logjac

    def fun(q):
        v = torch.tensor(q, requires_grad=True)
        loss = transformed(v)
        return (float(loss.detach()), torch.autograd.grad(loss, v)[0].detach().numpy())
    opt = minimize(fun, q0, jac=True, method='L-BFGS-B', options=dict(maxiter=400, ftol=1e-10, gtol=1e-05))
    qbest = torch.tensor(opt.x)
    hessian = torch.autograd.functional.hessian(transformed, qbest).detach().numpy()
    eigen, vectors = np.linalg.eigh(0.5 * (hessian + hessian.T))
    metric = vectors / np.maximum(eigen, 0.1) @ vectors.T
    rng = np.random.default_rng(928)
    qinit = opt.x + 0.1 * rng.multivariate_normal(np.zeros(len(lo)), metric, size=4)
    initial = target.low + (target.high - target.low) / (1 + np.exp(-qinit))
    initialization_seconds = time.perf_counter() - start
    start = time.perf_counter()
    result = hmc_sample(target, initial, target.low, target.high, steps=a.steps, burn_in=a.burn, leapfrog_steps=8, target_accept=0.9, inverse_mass=metric, device='cpu', seed=928, progress_interval=500, progress=lambda i, n, lp: print(a.station, 'HMC', i, n, flush=True))
    seconds = time.perf_counter() - start
    post = result['posterior'].detach().numpy()
    diagnostic = rank_diagnostics(post)
    passed = bool(diagnostic['converged'] and (not result['production_divergences'].any()))
    flat = post.reshape(-1, len(lo))
    take = np.linspace(0, len(flat) - 1, 256, dtype=int)
    train, test = ([], [])
    ft = target.make_forward(original['test_p'])
    with torch.no_grad():
        for batch in np.array_split(flat[take], 16):
            x = torch.tensor(batch)
            train.append(target.predict(x).numpy())
            test.append(target.predict(x, ft).numpy())
    train, test = (np.concatenate(train), np.concatenate(test))
    prediction = dict(time=data['time'], train_mean=train.mean(0), test_mean=test.mean(0), train_q=np.quantile(train, [0.025, 0.975], axis=0), test_q=np.quantile(test, [0.025, 0.975], axis=0))
    np.savez_compressed(out / f'{a.station}-chains.npz', chain=result['chain'].numpy(), log_prob=result['log_prob'].numpy(), initial=initial)
    np.savez_compressed(out / f'{a.station}-predictive.npz', **prediction)
    per_chain = []
    with torch.no_grad():
        for chain in range(4):
            x = torch.tensor(post[np.linspace(0, len(post) - 1, 64, dtype=int), chain])
            prediction_chain = target.predict(x, ft).mean(0).numpy()
            per_chain.append(score(prediction_chain, original['test_rf'], original['fit_mask']))
    summary = dict(station=a.station, selected=selection['selected'], names=target.names, bounds=[target.low, target.high], device='cpu', threads=torch.get_num_threads(), host=platform.node(), steps=a.steps, burn=a.burn, seed=928, initialization='local logit stationary point and regularized Hessian; four jittered chains near the selected physical MAP; no global mode guarantee', initialization_seconds=initialization_seconds, transformed_optimizer_success=bool(opt.success), hessian_eigenvalues=eigen, sampling_seconds=seconds, diagnostics=diagnostic, production_divergences=result['production_divergences'], acceptance=result['acceptance'], step_size=result['step_size'], gradient_evaluations=result['gradient_evaluations'], status='local_diagnostics_passed' if passed else 'provisional_unconverged', mean=flat.mean(0), q025=np.quantile(flat, 0.025, axis=0), q975=np.quantile(flat, 0.975, axis=0), chain_means=post.mean(0), train=score(train.mean(0), original['rf'], original['fit_mask']), test=score(test.mean(0), original['test_rf'], original['fit_mask']), per_chain_test=per_chain, covariance=spec['covariance'], composite_power=0.2 if spec['covariance'] == 'diagonal' else 1.0, likelihood_fit_window_s=spec['fit_window_s'], score_window_s=[0.8, 26.0], script_sha256=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest(), target_script_sha256=selection['script_sha256'], note='Conditional on selected model/working likelihood; model-selection uncertainty is omitted and interval coverage is not calibrated. RF fit is not independent geological accuracy.')
    (out / f'{a.station}.json').write_text(json.dumps(plain(summary), indent=2))
    print(json.dumps(plain(summary)), flush=True)
if __name__ == '__main__':
    main()
