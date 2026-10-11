# Our CPU-only archived research implementation; experimental, not default inference.
"""Two-stage point refinement with the physical energy in its original units.

The V1 cloud-wide energy normalization weakened the likelihood when separate
chains occupied poor modes. This exploratory repair reuses every frozen HMC
draw and event partition; only the second-stage physical scale changes to 1.
The original failed experiment is preserved, and outer RF/truth never choose
weights. This is not a posterior update.
"""
import argparse, json, time, shutil, platform
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
import hmc_dnn_refinement as original
from hmc_dnn_refinement import ROOT, WEIGHTS, CASES, CompatibilityGuide, RefinementObjective, data_target, choose_weight, final_scores, sync
from neural_rf import load_network
from neural_rf_experiment import sha, write_json, plain
from fit_refinement import score
OUT = ROOT / 'results/hmc-dnn-refine-v2'
PARENT = ROOT / 'results/hmc-dnn-refine-v1'

def freeze(device):
    parent = json.loads((PARENT / 'protocol.json').read_text())
    for name, h in parent['sources'].items():
        assert sha(ROOT / 'scripts' / name) == h, name
    assert sha(ROOT / 'results/neural-rf-v1/network.pt') == parent['checkpoint_sha256']
    p = dict(parent)
    p['version'] = 2
    p['parent_protocol_sha256'] = sha(PARENT / 'protocol.json')
    p['sources'] = dict(parent['sources'])
    p['sources']['hmc_dnn_refinement_v2.py'] = sha(__file__)
    p['refinement'] = dict(parent['refinement'])
    p['refinement']['physical_scale'] = '1.0; unscaled physical negative log posterior'
    p['refinement']['objective'] = 'physical_NLL + weight * softplus((median_logit-logit)/neural_scale) + .01*sum(whitened_logit_coordinate^2)'
    p['repair'] = 'Physical scale changed after examining the first V1 synthetic case. Preserved V1 results. Reuse identical physical HMC draws/inputs; reselect weights on internal validation only. Continued exploratory development, not a blind new test.'
    p['device'] = device
    p['refinement_runtime'] = dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__, host=platform.node(), gpu=None)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == plain(p), 'Frozen V2 changed; create a new version'
    else:
        write_json(path, p)
    return p

def fine_stage(target, validation, network, saved, folder, phase, p):
    path = folder / f'{phase}-refinement.json'
    if path.exists():
        return json.loads(path.read_text())
    guide = CompatibilityGuide(network, target)
    geometry = original.refinement_geometry(target, guide, saved, p['hmc']['burn'])
    geometry['original_cloud_energy_spread'] = geometry['physical_scale']
    geometry['physical_scale'] = 1.0
    c = np.array(geometry['center'])
    T = np.array(geometry['transform'])
    inverse = np.linalg.inv(T)
    u = np.clip((saved['best_per_chain'] - target.low) / (target.high - target.low), 1e-12, 1 - 1e-12)
    starts = (np.log(u / (1 - u)) - c) @ inverse.T
    starts = np.clip(starts, -3.0, 3.0)
    device = target.device
    rows = []
    mask = (target.data['time'] >= 0.799) & (target.data['time'] <= 26.001)
    for weight in p['weights']:
        begin = time.perf_counter()
        objective = RefinementObjective(target, guide, torch.tensor(c, device=device), torch.tensor(T, device=device), 1.0, geometry['neural_center'], geometry['neural_scale'], weight)

        def f(y):
            leaf = torch.tensor(y, device=device, requires_grad=True)
            loss = objective(leaf[None])[0]
            g = torch.autograd.grad(loss, leaf)[0]
            if not bool(torch.isfinite(loss) & torch.isfinite(g).all()):
                raise FloatingPointError('Refinement loss/gradient')
            return (float(loss.detach()), g.detach().cpu().numpy())
        candidates = []
        for i, start in enumerate(starts):
            opt = minimize(f, start, jac=True, method='L-BFGS-B', bounds=[(-3.0, 3.0)] * len(c), options=dict(maxiter=p['refinement']['maxiter'], ftol=1e-10, gtol=1e-05))
            with torch.no_grad():
                x = objective.x(torch.tensor(opt.x, device=device)[None])[0].cpu().numpy()
            candidates.append(dict(start=i, x=x, y=opt.x, objective=float(opt.fun), success=bool(opt.success), iterations=int(opt.nit), message=str(opt.message)))
        best = min(candidates, key=lambda a: a['objective'])
        x = torch.tensor(best['x'], device=device)[None]
        with torch.no_grad():
            train = target.predict(x)[0].cpu().numpy()
            valid = validation.predict(x)[0].cpu().numpy()
            nll = float(-target(x)[0])
            logit = float(guide(x)[0])
        sync(device)
        seconds = time.perf_counter() - begin
        rows.append(dict(weight=weight, candidates=candidates, selected_start=best['start'], x=best['x'], y=best['y'], optimizer_success=best['success'], training=score(train, target.data['rf'], mask), validation=score(valid, validation.data['rf'], mask), physical_nll=nll, neural_logit=logit, seconds=seconds))
        print(folder.name, phase, 'V2 fine weight', weight, 'seconds', seconds, 'validation', rows[-1]['validation']['rmse'], flush=True)
    record = dict(phase=phase, geometry=geometry, rows=rows, meaning='Physical energy in original units; heuristic constrained point estimate, not posterior samples')
    write_json(path, record)
    return plain(record)

def run(case, p, device):
    folder = OUT / case
    folder.mkdir(exist_ok=True)
    source = PARENT / case
    if (folder / 'result.json').exists():
        print(case, 'V2 already complete', flush=True)
        return
    assert (source / 'result.json').exists(), 'Complete the original physical HMC experiment first'
    names = ['data-info.json', 'inner-hmc.json', 'inner-hmc.npz', 'train-hmc.json', 'train-hmc.npz'] + [f'data-{phase}.npz' for phase in ('inner', 'validation', 'train', 'test')]
    info = json.loads((source / 'data-info.json').read_text())
    if info['cohort'] == 'synthetic':
        names += ['truth.json', 'truth.npz', 'test-noiseless.npz']
    for name in names:
        path = folder / name
        if not path.exists():
            shutil.copy2(source / name, path)
        assert sha(path) == sha(source / name)
    write_json(folder / 'reused-inputs.json', dict(parent_case_result_sha256=sha(source / 'result.json'), inputs={name: sha(source / name) for name in names}, note='Exact copies of unchanged original HMC draws and event data. No HMC rerun or prior fitted model leakage into internal selection.'))
    network, _ = load_network(dtype=torch.float32)
    network.to(device)
    inner = data_target(folder, 'inner', info['kind'], device)
    valid = data_target(folder, 'validation', info['kind'], device)
    inner_fine = fine_stage(inner, valid, network, dict(np.load(folder / 'inner-hmc.npz')), folder, 'inner', p)
    selected = choose_weight(inner_fine['rows'])
    write_json(folder / 'selection.json', dict(selected_weight=selected, rule=p['selection'], input_sha256=sha(folder / 'inner-refinement.json')))
    del inner, valid
    full = data_target(folder, 'train', info['kind'], device)
    full_hmc = json.loads((folder / 'train-hmc.json').read_text())
    full_fine = fine_stage(full, full, network, dict(np.load(folder / 'train-hmc.npz')), folder, 'train', p)
    test = data_target(folder, 'test', info['kind'], device)
    result = final_scores(folder, info, full, test, full_hmc, full_fine, selected)
    result['repair'] = p['repair']
    write_json(folder / 'result.json', result)
    print(case, 'V2 completed selected weight', selected, flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=CASES)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    p = freeze(args.device)
    for case in [args.case] if args.case else CASES:
        run(case, p, args.device)
