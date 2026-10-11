# Our CPU research program; ported imports use the public GRTM RF preview.
"""Larger independent-source replication after V3 validation overfitting.

The first experiment is preserved. No benchmark RF or truth trains this
network. A new test stream is reserved for evaluation after both checkpoints
are frozen. This remains exploratory development, not preregistered evidence.
"""
import argparse
import json
import math
import platform
import torch
import train_neural_rf_local as base
from neural_rf import ROOT
from neural_rf_experiment import sha, write_json, plain
OUT = ROOT / 'results/neural-rf-local-v4'

def freeze(device):
    p = json.loads((ROOT / 'results/neural-rf-local-v3/protocol.json').read_text())
    p['sources']['train_neural_rf_local_v4.py'] = sha(__file__)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == plain(p), 'Frozen V4 changed'
    else:
        write_json(path, p)
    return p
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['generate', 'train', 'evaluate', 'all'])
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    p = freeze(args.device)
    base.OUT = OUT
    if args.stage in ('generate', 'all'):
        for split in ('train', 'validation', 'test'):
            base.generate(p, split, args.device)
    if args.stage in ('train', 'all'):
        for arm in ('easy_retrain', 'local_retrain'):
            base.train(p, arm, args.device)
    if args.stage in ('evaluate', 'all'):
        base.evaluate(p, args.device)
    assert sha(ROOT / 'results/neural-rf-v1/network.pt') == p['checkpoint_sha256']
