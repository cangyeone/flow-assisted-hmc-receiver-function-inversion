# Our CPU research program; ported imports use the public GRTM RF preview.
"""Reproducible CPU simulation, ratio training and frozen out-of-sample tests."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from neural_rf import ROOT, OUT, TIME, MASK, EDGES, RatioNetwork, depth_projection, context, auc, load_network
from grtm import JointRFForward
from grtm.rf_cpu import _surface_torch
torch.set_num_threads(2)

def plain(value):
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value

def write_json(path, value):
    Path(path).write_text(json.dumps(plain(value), indent=2, allow_nan=False) + '\n')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def protocol():
    p = dict(version=1, train=12000, validation=1500, test=1500, seeds=dict(train=31001, validation=31002, test=31003, network=31004), epochs=70, batch=256, learning_rate=0.0008, projection_edges_km=EDGES, rf_time_s=TIME, score_window_s=[0.8, 26.0], nfft=1024, gaussian=2.5, noise_correlation_s=0.35, sediment_layers=[1, 4], crystalline_layers=[1, 8], model_prior='Independent H U(18,72), hs U(.05,10), sediment top Vs U(.3,3.4), compaction fraction U(0,1), kappa_s U(1.6,3.2), rho_s U(1.6,2.7), Qs logU(10,300) with 20% elastic; shared crust kappa U(1.55,2.05), smooth random Vs 2.7--4.3 including 25% LVZ, Dirichlet thicknesses; tau U(-.3,.3). This is a separate simulation prior, not the regional prior.', context='Three slowness-bin means sampled within [.041,.049],[.051,.065],[.067,.083], bin SD U(.0003,.003); stack-noise SD logU(.015,.12) per bin; noise Gaussian conditional on zero at P-normalization sample.', negatives='Independent model/tau from other source draws, keeping RF p/noise metadata. Class counts exactly equal. Local perturbations used only for held-out stress tests.', split='Independent whole-model draws and RNG streams; no base model or its noise/subdivision augmentations crosses splits.', selection='Best validation BCE epoch. Test labels, field RFs and known benchmark structure do not choose checkpoint.', scope='Prototype conditional neural ratio, same 1D constant-Q GRTM physics; not calibrated real-data posterior or new independent physical information.', sources={n: sha(ROOT / 'scripts' / n) for n in ('neural_rf.py', 'neural_rf_experiment.py')})
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        previous = json.loads(path.read_text())
        assert previous == plain(p), 'Protocol changed: use a new experiment directory'
    else:
        write_json(path, p)
    return p

def sample_model(rng):
    hs = rng.uniform(0.05, 10.0)
    H = rng.uniform(18.0, 72.0)
    ns = int(rng.integers(1, 5))
    nc = int(rng.integers(1, 9))
    v0 = rng.uniform(0.3, 3.4)
    g = rng.uniform()
    ks = rng.uniform(1.6, 3.2)
    rho = rng.uniform(1.6, 2.7)
    qs = np.inf if rng.uniform() < 0.2 else np.exp(rng.uniform(np.log(10), np.log(300)))
    vs = v0 + (3.5 - v0) * g * (np.arange(ns) + 0.5) / ns
    sed = np.c_[hs * np.arange(ns) / ns, ks * vs, vs, np.full(ns, rho), np.full(ns, 2 * qs), np.full(ns, qs)]
    frac = rng.dirichlet(np.full(nc, 3.0))
    tops = hs + (H - hs) * np.r_[0.0, frac.cumsum()[:-1]]
    vc = np.clip(rng.uniform(3.0, 3.8) + rng.uniform(-0.2, 0.65) * np.linspace(0, 1, nc) + rng.normal(0, 0.12, nc), 2.7, 4.3)
    if nc >= 3 and rng.uniform() < 0.25:
        vc[nc // 2] = np.clip(vc[nc // 2] - rng.uniform(0.2, 0.6), 2.7, 4.3)
    k = rng.uniform(1.55, 2.05)
    crust = np.c_[tops, vc * k, vc, np.full(nc, 2.8), np.full(nc, np.inf), np.full(nc, np.inf)]
    return (np.vstack((sed, crust, [H, 8.0, 4.5, 3.3, np.inf, np.inf])), rng.uniform(-0.3, 0.3))

def rf_forward(layers, p, tau=0.0, p_sd=None, nfft=1024):
    """Mean of three normalized event rays in each bin, no observed smoothing."""
    p = np.asarray(p, float)
    ps = np.zeros(3) if p_sd is None else np.asarray(p_sd, float)
    rays = (p[:, None] + ps[:, None] * np.sqrt(1.5) * np.array([-1.0, 0.0, 1.0])).ravel()
    f = JointRFForward(rays, gaussians=[2.5], periods=[0.0], dt=0.2, n_samples=len(TIME), t0=-2.0, nfft=nfft, device='cpu', attenuation='constant_q')
    x = f.prepare_layers(torch.as_tensor(layers, dtype=torch.float64), check=True)
    r, z = _surface_torch(x, f.p, f.omega, 'constant_q')
    power = z.abs().square()
    den = torch.maximum(power, f.water_level * power.amax(-1, keepdim=True))
    norm = torch.fft.irfft(power / den * f.filters[0], n=nfft)[..., :1]
    shift = torch.as_tensor(tau, dtype=torch.float64).reshape(-1, 1, 1)
    y = torch.fft.irfft(r * z.conj() / den * f.filters[0] / norm * f.shift * torch.exp(-1j * f.omega * shift), n=nfft)[..., :len(TIME)]
    y = y / y[..., 10:11]
    return y.reshape(len(x), 3, 3, len(TIME)).mean(2)

def noise_covariance():
    cov = 0.85 * np.exp(-0.5 * ((TIME[:, None] - TIME[None, :]) / 0.35) ** 2) + 0.15 * np.eye(len(TIME))
    cov = cov - np.outer(cov[:, 10], cov[10, :]) / cov[10, 10]
    cov[10, :] = 0.0
    cov[:, 10] = 0.0
    return cov

def generate(p, split):
    path = OUT / f'{split}.npz'
    if path.exists():
        return
    count = p[split]
    rng = np.random.default_rng(p['seeds'][split])
    models = []
    tau = []
    for _ in range(count):
        m, t = sample_model(rng)
        models.append(m)
        tau.append(t)
    maxk = max(map(len, models))
    layers = np.zeros((count, maxk, 6))
    layers[..., 4:] = np.inf
    lengths = np.array(list(map(len, models)))
    for i, m in enumerate(models):
        layers[i, :len(m)] = m
    rf = np.empty((count, 3, len(TIME)), np.float32)
    pr = np.empty((count, 3))
    ps = np.empty_like(pr)
    sd = np.empty_like(pr)
    covariance = noise_covariance()
    vals, vecs = np.linalg.eigh(covariance)
    chol = vecs @ np.diag(np.sqrt(vals.clip(0)))
    start = time.perf_counter()
    done = 0
    with torch.no_grad():
        for k in np.unique(lengths):
            indices = np.flatnonzero(lengths == k)
            for first in range(0, len(indices), 32):
                ix = indices[first:first + 32]
                pmean = rng.uniform([0.041, 0.051, 0.067], [0.049, 0.065, 0.083])
                psd = rng.uniform(0.0003, 0.003, 3)
                clean = rf_forward(layers[ix, :k], pmean, np.array(tau)[ix], psd).numpy()
                sigma = np.exp(rng.uniform(np.log(0.015), np.log(0.12), (len(ix), 3)))
                perturb = rng.normal(size=clean.shape) @ chol.T * sigma[..., None]
                rf[ix] = clean + perturb
                pr[ix] = pmean
                ps[ix] = psd
                sd[ix] = sigma
                done += len(ix)
                if done % 512 < 32:
                    print(split, done, count, round(time.perf_counter() - start, 1), 's', flush=True)
    assert np.isfinite(rf).all()
    profile = np.empty((count, 5, len(EDGES) - 1), np.float32)
    for k in np.unique(lengths):
        ix = np.flatnonzero(lengths == k)
        profile[ix] = depth_projection(torch.tensor(layers[ix, :k])).numpy()
    np.savez_compressed(path, rf=rf, profile=profile, layers=layers, lengths=lengths, tau=np.array(tau), p=pr, p_sd=ps, sigma=sd, source_id=np.arange(count) + p['seeds'][split] * 100000)
    write_json(OUT / f'{split}-generation.json', dict(count=count, seconds=time.perf_counter() - start, seed=p['seeds'][split], layer_counts=np.unique(lengths), max_abs_rf=float(np.max(np.abs(rf))), npz_sha256=sha(path)))

def tensors(split):
    a = dict(np.load(OUT / f'{split}.npz'))
    return (torch.tensor(a['rf'], dtype=torch.float32), torch.tensor(a['profile'], dtype=torch.float32), context(torch.tensor(a['p'], dtype=torch.float32), a['sigma'], a['tau'], a['p_sd']).float(), a)

def pair_metrics(network, data, seed=9107):
    rf, profile, cond, _ = data
    generator = torch.Generator().manual_seed(seed)
    negative = torch.randint(len(rf), (len(rf),), generator=generator)
    positive_values = []
    negative_values = []
    with torch.no_grad():
        for i in range(0, len(rf), 256):
            sl = slice(i, i + 256)
            idx = negative[sl]
            nc = cond[sl].clone()
            nc[:, 6] = cond[idx, 6]
            positive_values.append(network(rf[sl], profile[sl], cond[sl]).numpy())
            negative_values.append(network(rf[sl], profile[idx], nc).numpy())
    pos = np.concatenate(positive_values)
    neg = np.concatenate(negative_values)
    bce = float(np.mean(np.logaddexp(0, -pos) + np.logaddexp(0, neg)) / 2)
    prob = 1 / (1 + np.exp(-np.clip(np.r_[pos, neg], -60, 60)))
    labels = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return dict(bce=bce, auc=auc(pos, neg), brier=float(np.mean((prob - labels) ** 2)), positive_logit=pos, negative_logit=neg)

def train(p):
    if (OUT / 'network.pt').exists():
        return
    torch.manual_seed(p['seeds']['network'])
    torch.use_deterministic_algorithms(True)
    model = RatioNetwork().float()
    train_data = tensors('train')
    valid = tensors('validation')
    rf, prof, cond, _ = train_data
    model.rf_center.copy_(rf.mean(0, keepdim=True))
    model.rf_scale.copy_(rf.std(0, keepdim=True).clamp_min(0.05))
    optimizer = torch.optim.AdamW(model.parameters(), lr=p['learning_rate'], weight_decay=0.0002)
    generator = torch.Generator().manual_seed(p['seeds']['network'] + 1)
    best = np.inf
    history = []
    start = time.perf_counter()
    best_state = None
    for epoch in range(p['epochs']):
        model.train()
        order = torch.randperm(len(rf), generator=generator)
        total = 0.0
        for first in range(0, len(order), p['batch']):
            ix = order[first:first + p['batch']]
            neg = torch.randint(len(rf), (len(ix),), generator=generator)
            nc = cond[ix].clone()
            nc[:, 6] = cond[neg, 6]
            positive = model(rf[ix], prof[ix], cond[ix])
            negative = model(rf[ix], prof[neg], nc)
            loss = (F.softplus(-positive).mean() + F.softplus(negative).mean()) / 2
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(ix)
        model.eval()
        v = pair_metrics(model, valid)
        entry = dict(epoch=epoch + 1, train_bce=total / len(rf), validation_bce=v['bce'], validation_auc=v['auc'], seconds=time.perf_counter() - start)
        history.append(entry)
        if v['bce'] < best:
            best = v['bce']
            best_state = {k: value.detach().clone() for k, value in model.state_dict().items()}
            best_epoch = epoch + 1
        print('epoch', epoch + 1, 'train', round(entry['train_bce'], 4), 'valid', round(v['bce'], 4), 'auc', round(v['auc'], 4), flush=True)
    metadata = dict(selected_epoch=best_epoch, best_validation_bce=best, train_seconds=time.perf_counter() - start, threads=torch.get_num_threads(), device='cpu', torch=str(torch.__version__), host=platform.node(), protocol_sha256=sha(OUT / 'protocol.json'), architecture_sha256=sha(ROOT / 'scripts/neural_rf.py'), data_sha256={s: sha(OUT / f'{s}.npz') for s in ('train', 'validation')}, parameter_count=sum((x.numel() for x in model.parameters())), meaning='matched versus independent source pair log ratio; not probability of true geology')
    torch.save(dict(state_dict=best_state, metadata=metadata), OUT / 'network.pt')
    write_json(OUT / 'training.json', dict(metadata=metadata, history=history))

def evaluate():
    model, metadata = load_network(dtype=torch.float32)
    data = tensors('test')
    metrics = pair_metrics(model, data)
    rf, prof, cond, a = data
    rng = np.random.default_rng(31005)
    perturbed = []
    split_errors = []
    count = 256
    modified = []
    tau = []
    indices = []
    for i in range(count):
        layers = a['layers'][i, :a['lengths'][i]].copy()
        new = layers.copy()
        new[:-1, 2] *= np.exp(rng.normal(0, 0.07, len(new) - 1))
        new[:-1, 2] = np.clip(new[:-1, 2], 0.3, 4.3)
        new[:-1, 1] = np.maximum(new[:-1, 1], 1.3 * new[:-1, 2])
        modified.append(depth_projection(torch.tensor(new))[0].float())
        indices.append(i)
        j = int(rng.integers(len(layers) - 1))
        bottom = layers[j + 1, 0]
        row = layers[j].copy()
        row[0] = (layers[j, 0] + bottom) / 2
        split = np.insert(layers, j + 1, row, axis=0)
        with torch.no_grad():
            before = model.score(rf[i], torch.tensor(layers).float(), a['p'][i], a['sigma'][i], a['tau'][i], a['p_sd'][i])
            after = model.score(rf[i], torch.tensor(split).float(), a['p'][i], a['sigma'][i], a['tau'][i], a['p_sd'][i])
        split_errors.append(float((before - after).abs()))
    with torch.no_grad():
        nearby = model(rf[:count], torch.stack(modified), cond[:count]).numpy()
        matched = model(rf[:count], prof[:count], cond[:count]).numpy()
    result = dict(metadata=metadata, checkpoint_sha256=sha(OUT / 'network.pt'), test_npz_sha256=sha(OUT / 'test.npz'), test_pairs=len(rf), bce=metrics['bce'], auc=metrics['auc'], brier=metrics['brier'], local_stress_n=count, matched_above_perturbed_fraction=float(np.mean(matched > nearby)), split_layer_max_logit_difference=max(split_errors), scope='Random-pair discrimination and local stress only; neither is calibrated structural accuracy or coverage')
    np.savez_compressed(OUT / 'classification.npz', positive=metrics['positive_logit'], negative=metrics['negative_logit'], local_matched=matched, local_perturbed=nearby)
    write_json(OUT / 'classification.json', result)
    print(json.dumps(plain(result)), flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['generate', 'train', 'evaluate', 'all'])
    args = ap.parse_args()
    p = protocol()
    if args.stage in ('generate', 'all'):
        for split in ('train', 'validation', 'test'):
            generate(p, split)
    if args.stage in ('train', 'all'):
        train(p)
    if args.stage in ('evaluate', 'all'):
        evaluate()
if __name__ == '__main__':
    main()
