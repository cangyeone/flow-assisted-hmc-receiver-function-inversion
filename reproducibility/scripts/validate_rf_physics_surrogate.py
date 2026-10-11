# Our CPU-only archived research implementation; experimental, not default inference.
"""Validate frozen GPU experiments and save exact RF prediction diagnostics.

This script never retrains/selects a checkpoint. Unresolved chain diagnostics
remain visible, including in the RF comparisons.
"""
import argparse, json, hashlib
import numpy as np
import torch
from train_rf_physics_surrogate import build_target
from rf_physics_surrogate import ROOT, OUT, DeviceBasinTarget
from basin_rf import event_data
from fit_refinement import load_events, score
from neural_rf_experiment import sha, write_json
from grtm.rf_hmc import rank_diagnostics
torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)

def validate(case, device):
    folder = OUT / case
    p = json.loads((folder / 'protocol.json').read_text())
    for name, value in p['sources'].items():
        assert sha(ROOT / 'scripts' / name) == value, name
    teacher, _, info = build_target(case, device)
    reconstruction_difference = 0.0
    if info != p['info']:
        assert case != 'conditional'
        assert {k: v for k, v in info.items() if k != 'rf_sha256'} == {k: v for k, v in p['info'].items() if k != 'rf_sha256'}
        arrays = np.load(folder / 'teacher-arrays.npz')
        record = json.loads((folder / 'teacher-arrays.json').read_text())
        assert sha(folder / 'teacher-arrays.npz') == record['sha256']
        assert hashlib.sha256(arrays['rf'].tobytes()).hexdigest() == p['info']['rf_sha256']
        np.testing.assert_allclose(teacher.data['rf'], arrays['rf'], rtol=0, atol=1e-13)
        np.testing.assert_allclose(teacher.invchol.cpu().numpy(), arrays['invchol'], rtol=1e-12, atol=1e-12)
        reconstruction_difference = float(abs(teacher.data['rf'] - arrays['rf']).max())
        teacher.y = torch.tensor(arrays['rf'][:, teacher.mask.cpu().numpy()], device=device)
        teacher.invchol = torch.tensor(arrays['invchol'], device=device)
    assessment = json.loads((folder / 'assessment.json').read_text())
    assert sha(folder / 'network.pt') == assessment['checkpoint_sha256']
    assert sha(folder / 'test.npz') == assessment['test_sha256']
    metadata = json.loads((folder / 'training.json').read_text())['metadata']
    for name in ('train', 'validation'):
        assert sha(folder / f'{name}.npz') == metadata[f'{name}_sha256']
    assert sha(folder / 'protocol.json') == metadata['protocol_sha256']
    smoke = json.loads((OUT / f'{case}-smoke.json').read_text())
    assert smoke['status'] == 'passed'
    chains = []
    comparisons = []
    arrays = {}
    time = np.arange(151) * 0.2 - 2.0
    mask = (time >= 0.799) & (time <= 26.001)
    if case == 'conditional':
        observed = np.load(ROOT / 'results/neural-rf-v1/in-distribution/observations.npz')['observed']
        prediction_target = teacher
        with torch.no_grad():
            truth = teacher.predict(torch.tensor(p['info']['truth'], device=device)[None]).cpu().numpy()[0]
        arrays.update(observed=observed, truth=truth)
    else:
        data, groups = load_events(case)
        heldout, _ = event_data(data, groups, 'test')
        prediction_target = DeviceBasinTarget(heldout, teacher.kind, device)
        arrays.update(observed=heldout['rf'], observation_standard_error=np.sqrt(np.diagonal(heldout['cov'], axis1=1, axis2=2)))
        saved = json.loads((ROOT / 'results/basin-assessment-v1/real' / f'{case}.json').read_text())
        selected = next((c for c in saved['checks'] if c['key'] == saved['selected']))
        baseline = np.asarray(selected['full_candidates'][0]['x'])
        with torch.no_grad():
            pred = prediction_target.predict(torch.tensor(baseline, device=device)[None]).cpu().numpy()[0]
        arrays['training_selected_map'] = pred
        comparisons.append(dict(arm='training_selected_map', heldout=score(pred, arrays['observed'], mask), interpretation='Point prediction selected previously using training/internal validation events only'))
    for seed in p['sampling']['seeds']:
        for arm in ('physical', 'quadratic', 'neural'):
            f = folder / f'{arm}-seed{seed}.json'
            summary = json.loads(f.read_text())
            saved = np.load(f.with_suffix('.npz'))
            post = saved['chain'][p['sampling']['burn_in']:]
            flat = post.reshape(-1, post.shape[-1])
            diagnostics = rank_diagnostics(post)
            np.testing.assert_allclose(flat.mean(0), summary['mean'], rtol=0, atol=1e-12)
            np.testing.assert_allclose(flat.std(0), summary['sd'], rtol=0, atol=1e-12)
            for k in ('rhat', 'ess_bulk', 'ess_tail'):
                np.testing.assert_allclose(diagnostics[k], summary['diagnostics'][k], rtol=1e-10, atol=1e-10)
            rng = np.random.default_rng(32130 + seed)
            idx = rng.choice(len(flat), 20, replace=False)
            with torch.no_grad():
                lp = teacher(torch.tensor(flat[idx], device=device)).cpu().numpy()
            cached = saved['log_prob'][p['sampling']['burn_in']:].ravel()[idx]
            np.testing.assert_allclose(lp, cached, rtol=2e-10, atol=2e-08)
            if arm != 'physical':
                assert summary['physical_gradient_evaluations'] == 0
            chains.append(dict(arm=arm, seed=seed, converged=bool(diagnostics['converged']), max_saved_density_difference=float(abs(lp - cached).max()), verified_draws=len(idx)))
            if seed != p['sampling']['seeds'][0]:
                continue
            draw_idx = np.linspace(0, len(post) - 1, 16, dtype=int)
            draws = post[draw_idx].reshape(-1, post.shape[-1])
            predictions = []
            with torch.no_grad():
                for first in range(0, len(draws), 4):
                    predictions.append(prediction_target.predict(torch.tensor(draws[first:first + 4], device=device)).cpu().numpy())
            predictions = np.concatenate(predictions)
            pred = predictions.mean(0)
            arrays[f'{arm}_draw_rf'] = predictions
            arrays[f'{arm}_mean_rf'] = pred
            c = dict(arm=arm, seed=seed, heldout=score(pred, arrays['observed'], mask), converged=bool(diagnostics['converged']), rf_draws=len(draws), interpretation='Mean exact response over fixed systematic chain draws; diagnostic only if unconverged')
            if case == 'conditional':
                c['noiseless'] = score(pred, truth, mask)
            comparisons.append(c)
    arrays['time'] = time
    arrays['mask'] = mask
    np.savez_compressed(folder / 'rf-predictions.npz', **arrays)
    write_json(folder / 'rf-comparison.json', dict(case=case, rows=comparisons, window_seconds=[0.8, 26.0], scope='Conditional known-truth RF check or disjoint-event field RF prediction; no independent geological accuracy validation', prediction_sha256=sha(folder / 'rf-predictions.npz')))
    write_json(folder / 'validation.json', dict(status='passed', frozen_sources=len(p['sources']), teacher_inputs_verified=True, checkpoint_verified=True, diagnostics_recomputed=True, chains=chains, preserved_unresolved_chains=[c for c in chains if not c['converged']], validator_sha256=sha(__file__), cross_runtime_rf_reconstruction_max_difference=reconstruction_difference))
    print(case, 'validation passed', [(c['arm'], c['converged']) for c in chains], flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('case', choices=['conditional', 'V20A', 'X20A'])
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    validate(args.case, args.device)
