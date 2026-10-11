# Our CPU-only archived research implementation; experimental, not default inference.
"""HMC candidate refinement with the scale-aware local discriminator."""
import argparse, json, platform
import torch
import hmc_dnn_refinement_local as base
from neural_rf_experiment import sha, write_json, plain
OUT = base.ROOT / 'results/hmc-dnn-refine-local-v5'
NETWORKS = base.ROOT / 'results/neural-rf-local-v5'

def freeze(device):
    assert (NETWORKS / 'assessment.json').exists(), 'Freeze both checkpoints and assess fresh sources first'
    p = json.loads((base.ROOT / 'results/hmc-dnn-refine-local-v3/protocol.json').read_text())
    p['sources'].update({n: sha(base.ROOT / 'scripts' / n) for n in ['hmc_dnn_refinement_local_v5.py', 'neural_rf_local_scaled.py']})
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == plain(p), 'Frozen V5 replay changed'
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
