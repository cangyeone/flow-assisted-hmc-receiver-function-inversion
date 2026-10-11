# Our CPU research program; ported imports use the public GRTM RF preview.
"""Scale-aware local retraining with unchanged V4 source data/easy control."""
import argparse, json, shutil, platform
import torch
import train_neural_rf_local as base
from neural_rf import ROOT
from neural_rf_local_scaled import scaled_local_pair_loss
from neural_rf_experiment import sha, write_json, plain
OUT = ROOT / 'results/neural-rf-local-v5'
PARENT = ROOT / 'results/neural-rf-local-v4'

def freeze(device):
    p = json.loads((PARENT / 'protocol.json').read_text())
    p['seeds'] = dict(p['seeds'])
    p['seeds']['test'] = 36003
    p['seeds']['evaluate'] = 36005
    p['seeds']['bootstrap'] = 36006
    p['sources'].update({n: sha(ROOT / 'scripts' / n) for n in ['neural_rf_local_scaled.py', 'train_neural_rf_local_v5.py']})
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == plain(p), 'Frozen V5 changed'
    else:
        write_json(path, p)
    names = ['train.npz', 'train-generation.json', 'validation.npz', 'validation-generation.json']
    for name in names:
        dst = OUT / name
        if not dst.exists():
            shutil.copy2(PARENT / name, dst)
        assert sha(dst) == sha(PARENT / name)
    control = OUT / 'easy_retrain'
    control.mkdir(exist_ok=True)
    for name in ['network.pt', 'training.json']:
        dst = control / name
        if not dst.exists():
            shutil.copy2(PARENT / 'easy_retrain' / name, dst)
        assert sha(dst) == sha(PARENT / 'easy_retrain' / name)
    write_json(OUT / 'reused-training.json', dict(parent_protocol_sha256=sha(PARENT / 'protocol.json'), files={n: sha(PARENT / n) for n in names}, easy_control_checkpoint_sha256=sha(PARENT / 'easy_retrain/network.pt'), note='Identical train/validation source structures, RFs, local candidates and physical teacher values; only local score-distance normalization changes. No easy retraining repeated.'))
    return p
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['generate', 'train', 'evaluate', 'all'])
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    p = freeze(args.device)
    base.OUT = OUT
    base.local_pair_loss = scaled_local_pair_loss
    if args.stage in ('generate', 'all'):
        base.generate(p, 'test', args.device)
    if args.stage in ('train', 'all'):
        base.train(p, 'local_retrain', args.device)
    if args.stage in ('evaluate', 'all'):
        base.evaluate(p, args.device)
    assert sha(ROOT / 'results/neural-rf-v1/network.pt') == p['checkpoint_sha256']
