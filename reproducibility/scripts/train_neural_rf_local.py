# Our CPU research program; ported imports use the public GRTM RF preview.
"""Controlled CPU retraining with source-held-out fine perturbations.

Frozen V1, extra easy-pair training, and local-pair training share an encoder.
Both retrained arms start at the same V1 checkpoint, use the same source data,
seed, epochs and optimizer steps. Field/benchmark RFs and structures never
enter training or checkpoint selection. Expanded context is common to both
arms; the easy control separates it from the addition of local supervision.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from grtm import JointRFForward
from neural_rf import ROOT, TIME, MASK, EDGES, context, depth_projection, load_network, auc
from neural_rf_experiment import sample_model, noise_covariance, sha, write_json, plain
from neural_rf_local import SCALES, MODES, perturb_model, local_pair_loss
OUT = ROOT / 'results/neural-rf-local-v3'
torch.set_num_threads(2)

def sync(device):
    pass

def freeze(device):
    OUT.mkdir(parents=True, exist_ok=True)
    p = dict(version=3, train=2048, validation=256, test=512, seeds=dict(train=34001, validation=34002, test=34003, network=34004, evaluate=34005, bootstrap=34006), epochs=160, batch_sources=128, learning_rate=0.0002, weight_decay=0.0002, scales=SCALES, modes=MODES, candidates=6, local_temperature=1.0, local_rank_weight=2.0, local_gap_weight=0.05, local_gap_clip=8.0, context=dict(p_low=[0.039, 0.049, 0.065], p_high=[0.051, 0.067, 0.086], p_sd=[0.0002, 0.008], sigma=[0.015, 0.2], covariance='Original conditioned Gaussian, correlation 0.35 s'), perturbations='One source plus five same-direction scales; four modes balanced by source. Constant-Vp/Vs velocity, differential Vp/Vs, thickness-fraction interfaces at fixed Moho, and 2*scale seconds time shift. Random direction drawn before RF/noise. No perturbation is automatically a negative.', teacher='Exact GRTM CPU predictions; -0.5 sum of squared residuals whitened by the known noise covariance and sigma in the fixed .8--26 s window. All 15 unordered candidate pairs are supervised.', loss='global independent-pair BCE + 2*(soft-preference BCE + .05*bounded physical-gap SmoothL1). Global BCE retained. Local logits are score(first)-score(second).', arms=['frozen_V1', 'easy_retrain', 'local_retrain'], control='Both retrained arms start from the unchanged V1 checkpoint, same sources, expanded context, random seed, 160 epochs and 16 optimizer steps per epoch. Easy arm has four matched/independent pairs per source; local arm has one global pair and six local candidates: both evaluate eight scores per source. Physical teacher generation is additional cost, recorded separately.', selection='Best validation loss for each declared training objective. No test, old diagnostic source, benchmark truth, field RF, HMC candidate, or outer event participates.', split='New independent full source structures/noise/RNG streams. All source-derived candidates remain in one split. Three splits are mutually exclusive by source_id.', checkpoint_sha256=sha(ROOT / 'results/neural-rf-v1/network.pt'), sources={n: sha(ROOT / 'scripts' / n) for n in ('neural_rf.py', 'neural_rf_experiment.py', 'neural_rf_local.py', 'train_neural_rf_local.py')}, runtime=dict(device=device, python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__, host=platform.node(), gpu=None), scope='Exploratory amortized physical-supervised ranking; same 1D GRTM and Gaussian noise family. Not an independent geological prior, calibrated neural posterior, or guaranteed structural accuracy gain. Separate inversion evaluation retains exact physical likelihood.')
    path = OUT / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == plain(p), 'Frozen protocol changed: create a new version'
    else:
        write_json(path, p)
    return p

def cpu_forward(layers, p, p_sd, tau, device='cpu', nfft=1024):
    rays = (np.asarray(p)[:, None] + np.asarray(p_sd)[:, None] * np.sqrt(1.5) * [-1.0, 0.0, 1.0]).ravel()
    f = JointRFForward(rays, gaussians=[2.5], periods=[0.0], dt=0.2, n_samples=len(TIME), t0=-2.0, nfft=nfft, device=device, attenuation='constant_q')
    x = f.tensor(layers)
    spectra = f._evaluate(x, check=False)
    r, z = (spectra['radial_spectrum'], spectra['vertical_spectrum'])
    power = z.abs().square()
    den = torch.maximum(power, f.water_level * power.amax(-1, keepdim=True))
    norm = torch.fft.irfft(power / den * f.filters[0], n=nfft)[..., :1]
    shift = f.tensor(tau).reshape(-1, 1, 1)
    y = torch.fft.irfft(r * z.conj() / den * f.filters[0] / norm * f.shift * torch.exp(-1j * f.omega * shift), n=nfft)[..., :len(TIME)]
    y = y / y[..., 10:11]
    return y.reshape(len(x), 3, 3, len(TIME)).mean(2)

def physical_log_likelihood(predicted, observed, sigma, inverse_cholesky):
    residual = (predicted[..., MASK] - observed[:, None, :, MASK]) / sigma[:, None, :, None]
    white = residual @ inverse_cholesky.T
    return -0.5 * (white * white).sum((-1, -2))

def generate(p, split, device):
    path = OUT / f'{split}.npz'
    if path.exists():
        return
    begin = time.perf_counter()
    rng = np.random.default_rng(p['seeds'][split])
    n = p[split]
    models = []
    tau = []
    modes = []
    for i in range(n):
        m, t = sample_model(rng)
        direction = rng.normal(size=len(m) - 1)
        mode = MODES[i % len(MODES)]
        variants = [(m, t)] + [perturb_model(m, t, s, mode, direction) for s in SCALES]
        models.append(np.stack([v[0] for v in variants]))
        tau.append([v[1] for v in variants])
        modes.append(MODES.index(mode))
    tau = np.asarray(tau)
    lengths = np.array([m.shape[1] for m in models])
    maxk = int(lengths.max())
    nc = 1 + len(SCALES)
    layers = np.zeros((n, nc, maxk, 6))
    layers[..., 4:] = np.inf
    for i, m in enumerate(models):
        layers[i, :, :len(m[0])] = m
    rf = np.empty((n, 3, len(TIME)), np.float32)
    predicted = np.empty((n, nc, 3, len(TIME)), np.float64)
    prof = np.empty((n, nc, 5, len(EDGES) - 1), np.float32)
    pr = np.empty((n, 3))
    ps = np.empty((n, 3))
    sd = np.empty((n, 3))
    cov = noise_covariance()
    eig, U = np.linalg.eigh(cov)
    factor = U @ np.diag(np.sqrt(eig.clip(0)))
    done = 0
    with torch.no_grad():
        for k in np.unique(lengths):
            indices = np.flatnonzero(lengths == k)
            for first in range(0, len(indices), 8):
                ix = indices[first:first + 8]
                pm = rng.uniform(p['context']['p_low'], p['context']['p_high'])
                spread = rng.uniform(*p['context']['p_sd'], 3)
                lm = layers[ix, :, :k].reshape(-1, int(k), 6)
                prediction = cpu_forward(lm, pm, spread, tau[ix].ravel(), device).cpu().numpy().reshape(len(ix), nc, 3, len(TIME))
                sigma = np.exp(rng.uniform(np.log(p['context']['sigma'][0]), np.log(p['context']['sigma'][1]), (len(ix), 3)))
                observed = prediction[:, 0] + rng.normal(size=(len(ix), 3, len(TIME))) @ factor.T * sigma[..., None]
                rf[ix] = observed
                predicted[ix] = prediction
                pr[ix] = pm
                ps[ix] = spread
                sd[ix] = sigma
                prof[ix] = depth_projection(torch.tensor(lm, device=device)).cpu().numpy().reshape(len(ix), nc, 5, -1)
                done += len(ix)
                if done % 256 < 8:
                    print(split, 'sources', done, n, round(time.perf_counter() - begin, 1), 's', flush=True)
    inv = np.linalg.inv(np.linalg.cholesky(cov[np.ix_(MASK, MASK)]))
    lp = physical_log_likelihood(predicted, rf, sd, inv)
    assert np.isfinite(rf).all() and np.isfinite(lp).all() and np.isfinite(prof).all()
    ids = np.arange(n, dtype=np.int64) + p['seeds'][split] * 100000
    np.savez_compressed(path, rf=rf, profile=prof, layers=layers, lengths=lengths, tau=tau, p=pr, p_sd=ps, sigma=sd, source_id=ids, mode=np.array(modes), physical_log_likelihood=lp, predictions=predicted)
    sync(device)
    write_json(OUT / f'{split}-generation.json', dict(count=n, candidate_count=n * nc, seconds=time.perf_counter() - begin, npz_sha256=sha(path), seed=p['seeds'][split], source_beat_perturbation_fraction=(lp[:, 0, None] > lp[:, 1:]).mean(), note='Source model sometimes loses under noisy physical likelihood; labels are not source-versus-perturbation.'))

def tensors(split, device):
    a = dict(np.load(OUT / f'{split}.npz'))
    n, c = a['tau'].shape
    conditions = context(torch.tensor(np.repeat(a['p'], c, axis=0), dtype=torch.float32), np.repeat(a['sigma'], c, axis=0), a['tau'].ravel(), np.repeat(a['p_sd'], c, axis=0)).reshape(n, c, -1)
    return dict(rf=torch.tensor(a['rf'], device=device), profile=torch.tensor(a['profile'], device=device), cond=conditions.to(device), lp=torch.tensor(a['physical_log_likelihood'], dtype=torch.float32, device=device))

def objective(model, d, ix, neg, arm, p):
    rf = d['rf'][ix]
    base = d['profile'][ix, 0]
    cond = d['cond'][ix, 0]
    if arm == 'easy_retrain':
        count = 4
        nr = d['profile'][neg[:, :count], 0]
        nc = cond[:, None].expand(-1, count, -1).clone()
        nc[:, :, 6] = d['cond'][neg[:, :count], 0, 6]
        negative = model(rf[:, None].expand(-1, count, -1, -1).reshape(-1, 3, len(TIME)), nr.reshape(-1, 5, len(EDGES) - 1), nc.reshape(-1, 10)).reshape(-1, count)
        positive = model(rf[:, None].expand(-1, count, -1, -1).reshape(-1, 3, len(TIME)), base[:, None].expand(-1, count, -1, -1).reshape(-1, 5, len(EDGES) - 1), cond[:, None].expand(-1, count, -1).reshape(-1, 10)).reshape(-1, count)
        global_loss = (F.softplus(-positive).mean() + F.softplus(negative).mean()) / 2
        return (global_loss, global_loss, global_loss.new_zeros(()))
    nr = d['profile'][neg[:, 0], 0]
    nc = cond.clone()
    nc[:, 6] = d['cond'][neg[:, 0], 0, 6]
    positive = model(rf, base, cond)
    negative = model(rf, nr, nc)
    global_loss = (F.softplus(-positive).mean() + F.softplus(negative).mean()) / 2
    candidates = d['profile'][ix]
    c = candidates.shape[1]
    scores = model(rf[:, None].expand(-1, c, -1, -1).reshape(-1, 3, len(TIME)), candidates.reshape(-1, 5, len(EDGES) - 1), d['cond'][ix].reshape(-1, 10)).reshape(-1, c)
    local, _, _ = local_pair_loss(scores, d['lp'][ix], p['local_temperature'], p['local_gap_weight'], p['local_gap_clip'])
    return (global_loss + p['local_rank_weight'] * local, global_loss, local)

def train(p, arm, device):
    folder = OUT / arm
    folder.mkdir(exist_ok=True)
    if (folder / 'network.pt').exists():
        return
    torch.manual_seed(p['seeds']['network'])
    model, _ = load_network(dtype=torch.float32)
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    td = tensors('train', device)
    vd = tensors('validation', device)
    opt = torch.optim.AdamW(model.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'])
    generator = torch.Generator(device=device).manual_seed(p['seeds']['network'] + 1)
    vg = torch.Generator(device=device).manual_seed(p['seeds']['network'] + 2)
    vi = torch.arange(len(vd['rf']), device=device)
    vn = torch.randint(len(vi), (len(vi), 4), generator=vg, device=device)
    best = np.inf
    best_state = None
    history = []
    begin = time.perf_counter()
    for epoch in range(1, p['epochs'] + 1):
        model.train()
        order = torch.randperm(len(td['rf']), generator=generator, device=device)
        total = 0.0
        for first in range(0, len(order), p['batch_sources']):
            ix = order[first:first + p['batch_sources']]
            neg = torch.randint(len(order), (len(ix), 4), generator=generator, device=device)
            value, _, _ = objective(model, td, ix, neg, arm, p)
            opt.zero_grad()
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(value.detach()) * len(ix)
        model.eval()
        with torch.no_grad():
            v, g, l = objective(model, vd, vi, vn, arm, p)
        entry = dict(epoch=epoch, train_loss=total / len(order), validation_loss=float(v), validation_global_bce=float(g), validation_local_loss=float(l))
        history.append(entry)
        if float(v) < best:
            best = float(v)
            best_epoch = epoch
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            print(arm, 'epoch', epoch, 'validation', entry['validation_loss'], flush=True)
    sync(device)
    metadata = dict(selected_epoch=best_epoch, best_validation_loss=best, train_seconds=time.perf_counter() - begin, protocol_sha256=sha(OUT / 'protocol.json'), initial_checkpoint_sha256=p['checkpoint_sha256'], data_sha256={s: sha(OUT / f'{s}.npz') for s in ('train', 'validation')}, arm=arm, training_objective='global_BCE_and_physical_local_pairs' if arm == 'local_retrain' else 'global_BCE_only', meaning='Physical-supervised heuristic compatibility ranking; not a calibrated neural ratio or truth probability' if arm == 'local_retrain' else 'Matched versus independent pair score; common expanded-context extra-training control', device=device, torch=str(torch.__version__), host=platform.node())
    torch.save(dict(state_dict=best_state, metadata=metadata), folder / 'network.pt')
    write_json(folder / 'training.json', dict(metadata=metadata, history=history))

def evaluate(p, device):
    if (OUT / 'assessment.json').exists():
        return
    a = dict(np.load(OUT / 'test.npz'))
    d = tensors('test', device)
    rng = np.random.default_rng(p['seeds']['evaluate'])
    other = rng.integers(len(a['rf']), size=len(a['rf']))
    n, c = a['tau'].shape
    arms = {}
    saved = {}
    for arm in p['arms']:
        path = ROOT / 'results/neural-rf-v1/network.pt' if arm == 'frozen_V1' else OUT / arm / 'network.pt'
        model, _ = load_network(path, dtype=torch.float32)
        model.to(device)
        scores = []
        negative = []
        with torch.no_grad():
            for first in range(0, n, 64):
                ix = np.arange(first, min(first + 64, n))
                rf = d['rf'][ix]
                value = model(rf[:, None].expand(-1, c, -1, -1).reshape(-1, 3, len(TIME)), d['profile'][ix].reshape(-1, 5, len(EDGES) - 1), d['cond'][ix].reshape(-1, 10)).reshape(-1, c)
                cond = d['cond'][ix, 0].clone()
                cond[:, 6] = d['cond'][other[ix], 0, 6]
                nv = model(rf, d['profile'][other[ix], 0], cond)
                scores.append(value.cpu().numpy())
                negative.append(nv.cpu().numpy())
        scores = np.concatenate(scores)
        negative = np.concatenate(negative)
        gap = scores[:, 0, None] - scores[:, 1:]
        exact = a['physical_log_likelihood'][:, 0, None] - a['physical_log_likelihood'][:, 1:]
        eligible = abs(exact) > 1e-08
        correct = (gap > 0) == (exact > 0)
        summary = {}
        for j, s in enumerate(SCALES):
            keep = eligible[:, j]
            count = int(keep.sum())
            success = int(correct[keep, j].sum())
            z = 1.96
            f = success / count
            center = (f + z * z / (2 * count)) / (1 + z * z / count)
            half = z * np.sqrt(f * (1 - f) / count + z * z / (4 * count * count)) / (1 + z * z / count)
            summary[str(s)] = dict(agreement=success / count, correct=success, n=count, wilson95=[center - half, center + half], modes={mode: float(correct[(a['mode'] == k) & keep, j].mean()) for k, mode in enumerate(MODES)})
        arms[arm] = dict(checkpoint_sha256=sha(path), random_pair_auc=auc(scores[:, 0], negative), local_scales=summary, source_prefers_fraction=float((gap > 0).mean()), physical_source_prefers_fraction=float((exact > 0).mean()), meaning='Agreement with noisy physical likelihood, not geological truth accuracy')
        saved[arm + '_scores'] = scores
        saved[arm + '_independent_scores'] = negative
        saved[arm + '_correct'] = correct
    rng = np.random.default_rng(p['seeds']['bootstrap'])
    boot = rng.integers(n, size=(2000, n))
    comparisons = {}
    for control in ('frozen_V1', 'easy_retrain'):
        delta = saved['local_retrain_correct'].astype(float) - saved[control + '_correct'].astype(float)
        values = delta[boot].mean(1)
        comparisons[control] = {str(s): dict(agreement_gain=float(delta[:, j].mean()), paired_source_bootstrap95=np.quantile(values[:, j], [0.025, 0.975])) for j, s in enumerate(SCALES)}
    np.savez_compressed(OUT / 'assessment.npz', source_id=a['source_id'], mode=a['mode'], physical_gap=exact, **saved)
    write_json(OUT / 'assessment.json', dict(arms=arms, local_vs_controls=comparisons, test_sha256=sha(OUT / 'test.npz'), prediction_sha256=sha(OUT / 'assessment.npz'), scope='512 fresh source-held-out models, four perturbation types and five repeated scales. Confidence intervals resample full sources; scales are not independent replicates. Test only evaluated after both checkpoints were frozen. Inversion precision is evaluated separately.'))
    print('assessment', json.dumps(plain(arms)), flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['generate', 'train', 'evaluate', 'all'])
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    p = freeze(args.device)
    if args.stage in ('generate', 'all'):
        for split in ('train', 'validation', 'test'):
            generate(p, split, args.device)
    if args.stage in ('train', 'all'):
        for arm in ('easy_retrain', 'local_retrain'):
            train(p, arm, args.device)
    if args.stage in ('evaluate', 'all'):
        evaluate(p, args.device)
    assert sha(ROOT / 'results/neural-rf-v1/network.pt') == p['checkpoint_sha256']
if __name__ == '__main__':
    main()
