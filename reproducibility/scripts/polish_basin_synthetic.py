# Our CPU research program; ported imports use the public GRTM RF preview.
"""Continue budget-limited synthetic optimizations using fitting data only.

The original frozen experiment is preserved. Every candidate target follows
the same continuation rule; outer RFs and true structure never select fits.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import argparse, copy, hashlib, json, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from basin_assessment import OUT, DEPTH, synthetic_data, refit, uncertainty, target_hash, script_hash
from basin_rf import BasinTarget, event_data, choose, profile
from fit_refinement import RefinedTarget, partition_data, plain, score
DEST = OUT / 'synthetic-polished'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def continuation(target, candidates):
    answers = copy.deepcopy(candidates)
    best = answers[0]
    if best['success']:
        return (answers, [])
    lo = torch.tensor(target.low)
    span = torch.tensor(target.high - target.low)

    def objective(u):
        q = torch.tensor(u, requires_grad=True)
        loss = -target((lo + span * q)[None])[0]
        grad = torch.autograd.grad(loss, q)[0]
        if not torch.isfinite(loss) or not torch.isfinite(grad).all():
            raise FloatingPointError('Nonfinite continuation target')
        return (float(loss.detach()), grad.detach().numpy())
    attempts = []
    for block in range(2):
        u = (np.array(best['x']) - target.low) / (target.high - target.low)
        initial = objective(u)[0]
        np.testing.assert_allclose(initial, best['nll'], atol=1e-08, rtol=1e-10)
        started = time.perf_counter()
        opt = minimize(objective, u, jac=True, method='L-BFGS-B', bounds=[(0.0001, 0.9999)] * len(u), options=dict(maxiter=1600, ftol=1e-09, gtol=1e-05))
        attempts.append(dict(block=block, initial_nll=initial, final_nll=float(opt.fun), success=bool(opt.success), message=str(opt.message), iterations=int(opt.nit), seconds=time.perf_counter() - started))
        if opt.fun > initial + 1e-08:
            raise RuntimeError('Continuation increased objective')
        best.update(x=target.low + (target.high - target.low) * opt.x, nll=float(opt.fun), success=bool(opt.success), message=str(opt.message), iterations=int(best['iterations']) + int(opt.nit))
        if best['success']:
            break
    answers.sort(key=lambda c: c['nll'])
    return (answers, attempts)

def make_target(original, groups, c, which):
    if c['key'].startswith('basin_'):
        d, _ = event_data(original, groups, which)
        return BasinTarget(d, c['model'])
    if which == 'inner':
        base, _, _ = partition_data(original, groups)
        full, _ = event_data(original, groups, 'inner')
    else:
        base = original
        full, _ = event_data(original, groups, 'train')
    return RefinedTarget(base if c['model'] == 'legacy' else full, c['model'], covariance=c['covariance'])

def run(case, noise, seed):
    key = f'{case}-noise{noise:.3f}-seed{seed}'
    source = OUT / 'synthetic' / f'{key}.json'
    destination = DEST / f'{key}.json'
    if destination.exists():
        return
    while not source.exists():
        time.sleep(30)
    d = json.loads(source.read_text())
    a = dict(np.load(source.with_suffix('.npz')))
    original, groups, truth, parameters = synthetic_data(case, noise, seed)
    np.testing.assert_array_equal(original['rf'], a['rf_train'])
    valid, _ = event_data(original, groups, 'validation')
    _, oldvalid, _ = partition_data(original, groups)
    log = []
    for c in d['checks']:
        target = make_target(original, groups, c, 'inner')
        c['inner_candidates'], attempts = continuation(target, c['inner_candidates'])
        log.append(dict(key=c['key'], stage='inner', attempts=attempts))
        with torch.no_grad():
            prediction = target.predict(torch.tensor(c['inner_candidates'][0]['x'])[None], target.event_forward(valid) if c['key'].startswith('basin_') else target.make_forward(oldvalid['p']))[0].numpy()
        c['validation'] = score(prediction, valid['rf'], original['fit_mask'])
        print(key, c['key'], 'inner', c['inner_candidates'][0]['success'], c['validation']['rmse'], flush=True)
    previous_selected = d['selected']
    d['selected'] = choose(d['checks'])
    test, _ = event_data(original, groups, 'test')
    actual = profile(truth, DEPTH)
    for name in dict.fromkeys(['legacy_full', 'flex5_diagonal', d['selected']]):
        c = next((c for c in d['checks'] if c['key'] == name))
        if 'full_candidates' not in c:
            refit(original, groups, c)
        target = make_target(original, groups, c, 'train')
        c['full_candidates'], attempts = continuation(target, c['full_candidates'])
        log.append(dict(key=name, stage='full', attempts=attempts))
        x = torch.tensor(c['full_candidates'][0]['x'])[None]
        with torch.no_grad():
            train = target.predict(x)[0].numpy()
            prediction = target.predict(x, target.event_forward(test) if name.startswith('basin_') else target.make_forward(test['p']))[0].numpy()
            layers = target.model(x)[0].numpy()
        c.update(train=score(train, original['rf'], original['fit_mask']), test=score(prediction, test['rf'], original['fit_mask']), layers=layers)
        a.update({name + '_train': train, name + '_test': prediction, name + '_profile': profile(layers, DEPTH)})
        c['Vs_profile_rmse'] = float(np.sqrt(np.mean((profile(layers, DEPTH) - actual) ** 2)))
        shallow = DEPTH < parameters['hs']
        c['shallow_Vs_rmse'] = float(np.sqrt(np.mean((profile(layers, DEPTH[shallow]) - profile(truth, DEPTH[shallow])) ** 2)))
        c['H_abs_error_km'] = float(abs(c['full_candidates'][0]['x'][2] - parameters['H']))
        if name.startswith('basin_'):
            wider = BasinTarget(target.data, c['model'], nfft=2048)
            with torch.no_grad():
                refined = wider.predict(x, wider.event_forward(test))[0].numpy()
            c['nfft1024_2048_max_test_rf_difference'] = float(np.max(np.abs(prediction - refined)))
        print(key, name, 'full', c['full_candidates'][0]['success'], c['test']['rmse'], flush=True)
    d.update(previous_selected=previous_selected, polish_log=log, polish_script_sha256=sha(Path(__file__)), source_record_sha256=sha(source), source_arrays_sha256=sha(source.with_suffix('.npz')), scope='Synthetic optimization continuation; best saved candidate only, same targets/selection, no outer-data or truth tuning')
    np.savez_compressed(destination.with_suffix('.npz'), **a)
    destination.write_text(json.dumps(plain(d), indent=2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case')
    parser.add_argument('--noise', type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    DEST.mkdir(exist_ok=True, parents=True)
    if args.case:
        run(args.case, args.noise, args.seed)
        return
    path = OUT / 'synthetic-polish-protocol.json'
    frozen = dict(created_utc=datetime.now(timezone.utc).isoformat(), target_sha256=target_hash(), script_sha256=script_hash(), polish_script_sha256=sha(Path(__file__)), rule='If the best saved candidate is budget-limited, continue it with at most two 1600-iteration blocks; same ftol1e-9/gtol1e-5. Apply to all five inner targets before reselecting, then to original/compact/selected full-training targets.', note='Training objective only; original results preserved; neither outer amplitudes nor true profiles influence continuation or selection')
    if path.exists():
        assert json.loads(path.read_text())['polish_script_sha256'] == frozen['polish_script_sha256']
    else:
        path.write_text(json.dumps(frozen, indent=2))
    jobs = [(c, n, s) for c in ['shallow', 'deep'] for n in [0.025, 0.06] for s in [200, 201, 202]]

    def launch(job):
        c, n, s = job
        label = f'{c}-noise{n:.3f}-seed{s}'
        with (OUT / f'polish-{label}.log').open('w') as f:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--case', c, '--noise', str(n), '--seed', str(s)], stdout=f, stderr=subprocess.STDOUT)
        print(label, result.returncode, flush=True)
        return dict(label=label, returncode=result.returncode)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        status = list(pool.map(launch, jobs))
    (OUT / 'synthetic-polish-status.json').write_text(json.dumps(status, indent=2))
    assert not any((s['returncode'] for s in status)), status
if __name__ == '__main__':
    main()
