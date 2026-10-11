# Our CPU-only archived research implementation; experimental, not default inference.
"""Replay identical HMC candidates with the local-pair and easy controls.

No chain is rerun or treated as converged. Common zero-weight optimizations
are exactly reused. Network/weight/acceptance selection never sees outer RF
or structural truth. Point estimates are not posterior samples.
"""
import argparse
import json
import shutil
import platform
import time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
import hmc_dnn_refinement as original
from hmc_dnn_refinement import ROOT, CASES, CompatibilityGuide, RefinementObjective, data_target, choose_weight, final_scores, sync
from neural_rf import load_network
from neural_rf_experiment import sha, write_json, plain
from fit_refinement import score
from guard_hmc_dnn_refinement import accept_refinement
OUT = ROOT / 'results/hmc-dnn-refine-local-v3'
PARENT = ROOT / 'results/hmc-dnn-refine-v2'
NETWORKS = ROOT / 'results/neural-rf-local-v3'
ARMS = ['easy_retrain', 'local_retrain']

class ExpandedGuide(CompatibilityGuide):

    def metadata(self):
        meta = super().metadata()
        meta['V1_domain_flags'] = meta.pop('domain_flags')
        p = np.array(meta['p'])
        ps = np.array(meta['p_sd'])
        s = np.array(meta['sigma_proxy'])
        flags = []
        if np.any((ps < 0.0002) | (ps > 0.008)):
            flags.append('ray spread outside local training range')
        if np.any((s < 0.015) | (s > 0.2)):
            flags.append('noise proxy outside local training range')
        if np.any((p < [0.039, 0.049, 0.065]) | (p > [0.051, 0.067, 0.086])):
            flags.append('ray means outside local training range')
        meta['domain_flags'] = flags
        meta['limits'] = 'Expanded marginal context bounds do not validate field covariance, event averaging, normalization, or geological prior match.'
        return meta

def freeze(device):
    OUT.mkdir(parents=True, exist_ok=True)
    assert (NETWORKS / 'assessment.json').exists(), 'Freeze and evaluate both networks first'
    parent = json.loads((PARENT / 'protocol.json').read_text())
    p = dict(version=3, cases=CASES, arms=ARMS, weights=[0.0, 0.3, 3.0], hmc=parent['hmc'], refinement=parent['refinement'], parent_protocol_sha256=sha(PARENT / 'protocol.json'), network_protocol_sha256=sha(NETWORKS / 'protocol.json'), checkpoints={a: sha(NETWORKS / a / 'network.pt') for a in ARMS}, selection=parent['selection'], guard='Accept selected refinement only for >=1% inner-validation RF RMSE improvement over the inner-HMC best. Otherwise retain full-HMC best. Existing operational guard fixed before this run.', sources={n: sha(ROOT / 'scripts' / n) for n in ('hmc_dnn_refinement_local.py', 'hmc_dnn_refinement.py', 'neural_rf.py', 'neural_rf_local.py', 'guard_hmc_dnn_refinement.py', 'rf_physics_surrogate.py')}, device=device, runtime=dict(python=platform.python_version(), torch=str(torch.__version__), host=platform.node(), gpu=None), network_dtype='float64, both newly trained arms; gradients through arbitrary-layer depth projection', reuse='Identical V2 event arrays, physical HMC clouds, parameterization, trust geometry and zero-weight optimization candidates. New positive weights use identical maxiter/starts and physical scale=1.', scope='Exploratory, previously examined 4 synthetic cases/2 field stations. HMC diagnostics unchanged and failed. Neural training used none of these RFs, candidates, or truth models. Marginal context expansion is controlled by easy_retrain. No new geological evidence or calibrated posterior precision claim.')
    path = OUT / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == plain(p), 'Frozen replay changed'
    else:
        write_json(path, p)
    return p

def fine_stage(target, validation, network, saved, folder, source, phase, p):
    path = folder / f'{phase}-refinement.json'
    if path.exists():
        return json.loads(path.read_text())
    guide = ExpandedGuide(network, target)
    geometry = original.refinement_geometry(target, guide, saved, p['hmc']['burn'])
    geometry['original_cloud_energy_spread'] = geometry['physical_scale']
    geometry['physical_scale'] = 1.0
    parent = json.loads((source / f'{phase}-refinement.json').read_text())
    for key in ('center', 'transform'):
        np.testing.assert_allclose(geometry[key], parent['geometry'][key], rtol=0, atol=0)
    rows = [next((r for r in parent['rows'] if r['weight'] == 0.0))]
    c = np.array(geometry['center'])
    T = np.array(geometry['transform'])
    inverse = np.linalg.inv(T)
    u = np.clip((saved['best_per_chain'] - target.low) / (target.high - target.low), 1e-12, 1 - 1e-12)
    starts = np.clip((np.log(u / (1 - u)) - c) @ inverse.T, -3.0, 3.0)
    device = target.device
    mask = (target.data['time'] >= 0.799) & (target.data['time'] <= 26.001)
    for weight in p['weights'][1:]:
        begin = time.perf_counter()
        objective = RefinementObjective(target, guide, torch.tensor(c, device=device), torch.tensor(T, device=device), 1.0, geometry['neural_center'], geometry['neural_scale'], weight)

        def f(y):
            leaf = torch.tensor(y, device=device, requires_grad=True)
            value = objective(leaf[None])[0]
            g = torch.autograd.grad(value, leaf)[0]
            if not bool(torch.isfinite(value) & torch.isfinite(g).all()):
                raise FloatingPointError('Local refinement gradient')
            return (float(value.detach()), g.detach().cpu().numpy())
        candidates = []
        for i, start in enumerate(starts):
            opt = minimize(f, start, jac=True, method='L-BFGS-B', bounds=[(-3.0, 3.0)] * len(c), options=dict(maxiter=p['refinement']['maxiter'], ftol=1e-10, gtol=1e-05))
            with torch.no_grad():
                x = objective.x(torch.tensor(opt.x, device=device)[None])[0].cpu().numpy()
            candidates.append(dict(start=i, x=x, y=opt.x, objective=float(opt.fun), success=bool(opt.success), iterations=int(opt.nit), message=str(opt.message)))
        best = min(candidates, key=lambda r: r['objective'])
        x = torch.tensor(best['x'], device=device)[None]
        with torch.no_grad():
            train = target.predict(x)[0].cpu().numpy()
            valid = validation.predict(x)[0].cpu().numpy()
            nll = float(-target(x)[0])
            logit = float(guide(x)[0])
        sync(device)
        seconds = time.perf_counter() - begin
        rows.append(dict(weight=weight, candidates=candidates, selected_start=best['start'], x=best['x'], y=best['y'], optimizer_success=best['success'], training=score(train, target.data['rf'], mask), validation=score(valid, validation.data['rf'], mask), physical_nll=nll, neural_logit=logit, seconds=seconds))
        print(folder.parent.name, folder.name, phase, 'weight', weight, 'validation', rows[-1]['validation']['rmse'], 'seconds', round(seconds, 1), flush=True)
    record = dict(phase=phase, geometry=geometry, rows=rows, zero_control_reused=True, zero_control_source_sha256=sha(source / f'{phase}-refinement.json'), meaning='Exact physical posterior retained in point objective. Neural score is a heuristic regularizer. No posterior uncertainty update.')
    write_json(path, record)
    return plain(record)

def run(arm, case, p, device):
    folder = OUT / arm / case
    folder.mkdir(parents=True, exist_ok=True)
    source = PARENT / case
    if (folder / 'result.json').exists():
        return
    info = json.loads((source / 'data-info.json').read_text())
    names = ['data-info.json', 'inner-hmc.json', 'inner-hmc.npz', 'train-hmc.json', 'train-hmc.npz'] + [f'data-{s}.npz' for s in ('inner', 'validation', 'train', 'test')]
    if info['cohort'] == 'synthetic':
        names += ['truth.json', 'truth.npz', 'test-noiseless.npz']
    for name in names:
        dst = folder / name
        if not dst.exists():
            shutil.copy2(source / name, dst)
        assert sha(dst) == sha(source / name)
    write_json(folder / 'reused-inputs.json', dict(parent_result_sha256=sha(source / 'result.json'), inputs={n: sha(source / n) for n in names}))
    checkpoint = NETWORKS / arm / 'network.pt'
    assert sha(checkpoint) == p['checkpoints'][arm]
    network, meta = load_network(checkpoint, dtype=torch.float64)
    network.to(device)
    inner = data_target(folder, 'inner', info['kind'], device)
    valid = data_target(folder, 'validation', info['kind'], device)
    inner_saved = dict(np.load(folder / 'inner-hmc.npz'))
    fine = fine_stage(inner, valid, network, inner_saved, folder, source, 'inner', p)
    weight = choose_weight(fine['rows'])
    candidate = next((r for r in fine['rows'] if r['weight'] == weight))
    with torch.no_grad():
        reference = valid.predict(torch.tensor(inner_saved['best'], device=device)[None])[0].cpu().numpy()
    mask = (valid.data['time'] >= 0.799) & (valid.data['time'] <= 26.001)
    ref_rmse = score(reference, valid.data['rf'], mask)['rmse']
    accepted = accept_refinement(ref_rmse, candidate['validation']['rmse'])
    method = f'fine_weight{weight:g}' if accepted else 'HMC_best'
    selection = dict(arm=arm, case=case, selected_weight=weight, accepted_refinement=accepted, method=method, hmc_inner_validation_rmse=ref_rmse, refined_inner_validation_rmse=candidate['validation']['rmse'], weight_rule=p['selection'], guard=p['guard'], input_sha256=sha(folder / 'inner-refinement.json'))
    write_json(folder / 'selection.json', selection)
    del inner, valid
    full = data_target(folder, 'train', info['kind'], device)
    full_fine = fine_stage(full, full, network, dict(np.load(folder / 'train-hmc.npz')), folder, source, 'train', p)
    test = data_target(folder, 'test', info['kind'], device)
    result = final_scores(folder, info, full, test, json.loads((folder / 'train-hmc.json').read_text()), full_fine, weight)
    result.update(arm=arm, guarded_method=method, accepted_refinement=accepted, checkpoint_sha256=sha(checkpoint), selection_sha256=sha(folder / 'selection.json'), checkpoint_meaning=meta['meaning'])
    write_json(folder / 'result.json', result)
    row = next((r for r in result['rows'] if r['method'] == method))
    write_json(folder / 'guarded-result.json', dict(selection=selection, row=row, result_sha256=sha(folder / 'result.json')))
    print(arm, case, 'complete weight', weight, 'accepted', accepted, flush=True)
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', choices=CASES)
    ap.add_argument('--arm', choices=ARMS)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    p = freeze(args.device)
    for arm in [args.arm] if args.arm else ARMS:
        for case in [args.case] if args.case else CASES:
            run(arm, case, p, args.device)
