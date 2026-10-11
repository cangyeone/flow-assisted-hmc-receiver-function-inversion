# Our CPU-only archived research implementation; experimental, not default inference.
"""Replay the larger-source local discriminator with unchanged controls."""
import argparse, json, platform
import torch
import hmc_dnn_refinement_local as base
from neural_rf_experiment import sha, write_json, plain
OUT = base.ROOT / 'results/hmc-dnn-refine-local-v4'
NETWORKS = base.ROOT / 'results/neural-rf-local-v4'

def freeze(device):
    assert (NETWORKS / 'assessment.json').exists(), 'Finish new network selection and assessment first'
    p = json.loads((base.ROOT / 'results/hmc-dnn-refine-local-v3/protocol.json').read_text())
    p['sources']['hmc_dnn_refinement_local_v4.py'] = sha(__file__)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == plain(p), 'Frozen V4 replay changed'
    else:
        write_json(path, p)
    return p
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', choices=base.CASES)
    ap.add_argument('--arm', choices=base.ARMS)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    p = freeze(args.device)
    base.OUT = OUT
    base.NETWORKS = NETWORKS
    for arm in [args.arm] if args.arm else base.ARMS:
        for case in [args.case] if args.case else base.CASES:
            base.run(arm, case, p, args.device)
