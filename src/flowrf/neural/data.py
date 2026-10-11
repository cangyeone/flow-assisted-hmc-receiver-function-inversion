"""Sharded, resumable exact-GRTM simulation and online RF noise.

One source structure belongs to exactly one split. Three Gaussian widths and
three slowness bins form 9 RF channels; three quadrature rays per bin. Slowness
conditions are shared within each simulation microbatch (recorded cluster).
"""
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from .structure import decode, sample_prior
from .models import NT

GAUSSIANS = [1., 2.5, 4.]
TIME = np.arange(NT) * .2 - 2.


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def exact_forward(layers, p, p_sd, tau, device='cpu'):
    from grtm import JointRFForward
    rays = (np.asarray(p)[:, None] + np.asarray(p_sd)[:, None] * np.sqrt(1.5) * np.array([-1., 0., 1.])).ravel()
    f = JointRFForward(rays, gaussians=GAUSSIANS, periods=[0.], dt=.2,
                       n_samples=NT, t0=-2., nfft=1024, device=device,
                       attenuation='constant_q')
    spectra = f._evaluate(layers, check=True)
    r, z = spectra['radial_spectrum'], spectra['vertical_spectrum']
    power = z.abs().square()
    den = torch.maximum(power, f.water_level * power.amax(-1, keepdim=True))
    ref = power[:, :, None] / den[:, :, None] * f.filters
    norm = torch.fft.irfft(ref, n=f.nfft)[..., :1]
    phase = f.shift * torch.exp(-1j * f.omega * f.tensor(tau).reshape(-1, 1, 1, 1))
    y = torch.fft.irfft(r[:, :, None] * z[:, :, None].conj() / den[:, :, None]
                       * f.filters / norm * phase, n=f.nfft)[..., :NT]
    y = y / y[..., 10:11]
    # channel order: slowness bin first, Gaussian width second.
    return y.reshape(len(y), 3, 3, 3, NT).mean(2).reshape(len(y), 9, NT)


def generate_shard(folder, split, shard, size, config, device):
    dest = Path(folder) / 'data' / split / f'{shard:05d}'
    receipt = dest / 'receipt.json'
    if receipt.exists():
        meta = json.loads(receipt.read_text())
        if meta['config_sha256'] != config['config_sha256'] or meta['count'] != size:
            raise ValueError('Existing shard belongs to another frozen protocol')
        return meta
    dest.mkdir(parents=True, exist_ok=True)
    split_index = {'train': 0, 'validation': 1, 'test': 2, 'pilot': 3}[split]
    rng = np.random.default_rng(np.random.SeedSequence([config['data_seed'], split_index, shard]))
    start = time.perf_counter()
    count = rng.integers(config['min_layers'], config['max_layers'] + 1, size, dtype=np.int64)
    u = sample_prior(rng, count, config['max_layers'])
    layers, tau, _ = decode(torch.as_tensor(u, dtype=torch.float64), torch.as_tensor(count))
    layers, tau = layers.numpy(), tau.numpy()
    clean = np.empty((size, 9, NT), np.float32)
    ps, spreads = np.empty((size, 3), np.float32), np.empty((size, 3), np.float32)
    cluster = np.empty(size, np.int64)
    replacements, ci = 0, 0
    with torch.no_grad():
        for k in np.unique(count):
            indices = np.flatnonzero(count == k)
            for first in range(0, len(indices), config['teacher_batch']):
                take = indices[first:first + config['teacher_batch']]
                p = rng.uniform([.039, .051, .067], [.049, .065, .082])
                sd = rng.uniform(.0002, .003, 3)
                for retry in range(50):
                    y = exact_forward(layers[take, :k + 1], p, sd, tau[take], device).cpu().numpy()
                    ok = np.isfinite(y).all((1, 2)) & (np.abs(y).max((1, 2)) <= config['rf_abs_limit'])
                    clean[take[ok]] = y[ok]
                    if ok.all():
                        break
                    # This acceptance filter explicitly defines a truncated
                    # simulator prior; it is identical in all splits.
                    bad = take[~ok]
                    if retry == 49:
                        raise RuntimeError('Too many RF normalization rejections')
                    u[bad] = sample_prior(rng, count[bad], config['max_layers'])
                    lm, tm, _ = decode(torch.tensor(u[bad], dtype=torch.float64), torch.tensor(count[bad]))
                    layers[bad], tau[bad] = lm.numpy(), tm.numpy()
                    replacements += len(bad)
                    take = bad
                original = indices[first:first + config['teacher_batch']]
                ps[original], spreads[original] = p, sd
                cluster[original] = split_index * 10**12 + shard * 100000 + ci
                ci += 1
    sigma = np.exp(rng.uniform(np.log(.005), np.log(.025), (size, 3))).astype(np.float32)
    source_id = split_index * 10**12 + shard * config['shard_size'] + np.arange(size, dtype=np.int64)
    arrays = dict(u=u, count=count.astype(np.int16), layers=layers.astype(np.float32), tau=tau.astype(np.float32),
                  clean=clean, p=ps, p_sd=spreads, sigma=sigma, source_id=source_id, context_cluster=cluster)
    hashes = {}
    for name, value in arrays.items():
        temp = dest / f'{name}.tmp.npy'
        np.save(temp, value)
        os.replace(temp, dest / f'{name}.npy')
        hashes[name] = sha(dest / f'{name}.npy')
    meta = dict(split=split, shard=shard, count=size, replacements=replacements,
                seconds=time.perf_counter() - start, context_clusters=ci,
                config_sha256=config['config_sha256'], hashes=hashes,
                rf_min=float(clean.min()), rf_max=float(clean.max()),
                count_histogram={str(k): int((count == k).sum()) for k in np.unique(count)})
    save_json(receipt, meta)
    print(json.dumps(dict(stage='generate', **{k: v for k, v in meta.items() if k != 'hashes'})), flush=True)
    return meta


def generate(folder, split, size, config, device):
    receipts = []
    for i, start in enumerate(range(0, size, config['shard_size'])):
        receipts.append(generate_shard(folder, split, i, min(size - start, config['shard_size']), config, device))
        save_json(Path(folder) / 'status.json', dict(stage='generate', split=split,
                  completed=sum(x['count'] for x in receipts), target=size, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    save_json(Path(folder) / f'{split}-data.json', dict(count=size, shards=receipts))


def load_arrays(folder, split, limit=None):
    paths = sorted((Path(folder) / 'data' / split).glob('*/receipt.json'))
    if not paths:
        raise FileNotFoundError(f'No completed shards: {split}')
    keys = ['u', 'count', 'layers', 'tau', 'clean', 'p', 'p_sd', 'sigma', 'source_id', 'context_cluster']
    data = {key: [] for key in keys}
    have = 0
    for path in paths:
        if limit is not None and have >= limit:
            break
        n = json.loads(path.read_text())['count']
        take = n if limit is None else min(n, limit - have)
        for key in keys:
            data[key].append(np.load(path.parent / f'{key}.npy', mmap_mode='r')[:take])
        have += take
    return {key: np.concatenate(value) for key, value in data.items()}


class Noise:
    """Small bounded heteroscedastic, temporally/cross-band correlated noise.

    Sigma is observation metadata. Zero-lag normalization is preserved exactly.
    This is a specified augmentation model, not a real-event noise estimate.
    """
    def __init__(self, device):
        delta = TIME[:, None] - TIME[None]
        cov = .85 * np.exp(-.5 * (delta / .35)**2) + .15 * np.eye(NT)
        cov -= np.outer(cov[:, 10], cov[10]) / cov[10, 10]
        cov[10] = 0
        cov[:, 10] = 0
        eig, vec = np.linalg.eigh(cov)
        self.factor = torch.tensor(vec @ np.diag(np.sqrt(np.maximum(eig, 0))), dtype=torch.float32, device=device)
        self.marginal = torch.tensor(np.diag(cov).clip(0), dtype=torch.float32, device=device).sqrt()

    def draw(self, clean, sigma, generator=None):
        b = len(clean)
        common = torch.randn(b, 3, 1, NT, device=clean.device, generator=generator)
        independent = torch.randn(b, 3, 3, NT, device=clean.device, generator=generator)
        # 80% shared variance across bands, 20% independent. Both have the
        # same fixed time covariance; each marginal is known exactly.
        white = math_sqrt(.8) * common + math_sqrt(.2) * independent
        noise = white.reshape(b, 9, NT) @ self.factor.T
        envelope = .6 + .4 * clean.abs() / (.1 + clean.abs())
        amp = sigma.repeat_interleave(3, -1)[..., None] * envelope
        observed = clean + noise * amp
        std = amp * self.marginal
        return observed, std


def math_sqrt(x):
    return float(np.sqrt(x))
