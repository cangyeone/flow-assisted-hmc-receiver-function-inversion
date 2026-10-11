# Our CPU-only archived research implementation; experimental, not default inference.
"""Independent label/selection/metric/objective verification for local DNNs."""
import argparse, json
import numpy as np
import torch
from scipy.linalg import solve_triangular
from neural_rf import ROOT, MASK, TIME, load_network, depth_projection
from neural_rf_experiment import noise_covariance, rf_forward, sha, write_json
from neural_rf_local import SCALES
from hmc_dnn_refinement import data_target, RefinementObjective
from hmc_dnn_refinement_local import ExpandedGuide

def verify_sources(p):
    for name, h in p['sources'].items():
        assert sha(ROOT / 'scripts' / name) == h, name

def network_checks(version, device):
    folder = ROOT / f'results/neural-rf-local-{version}'
    p = json.loads((folder / 'protocol.json').read_text())
    verify_sources(p)
    assert sha(ROOT / 'results/neural-rf-v1/network.pt') == p['checkpoint_sha256']
    cov = noise_covariance()[np.ix_(MASK, MASK)]
    chol = np.linalg.cholesky(cov)
    sets = []
    receipts = []
    max_lp = 0.0
    max_rf = 0.0
    max_relative_lp = 0.0
    for split in ('train', 'validation', 'test'):
        a = dict(np.load(folder / f'{split}.npz'))
        gen = json.loads((folder / f'{split}-generation.json').read_text())
        assert sha(folder / f'{split}.npz') == gen['npz_sha256']
        sets.append(set(a['source_id']))
        assert len(sets[-1]) == p[split]
        residual = (a['predictions'][..., MASK] - a['rf'][:, None, :, MASK]) / a['sigma'][:, None, :, None]
        white = solve_triangular(chol, residual.reshape(-1, chol.shape[0]).T, lower=True).T.reshape(residual.shape)
        lp = -0.5 * np.sum(white ** 2, axis=(-1, -2))
        error = float(abs(lp - a['physical_log_likelihood']).max())
        max_lp = max(max_lp, error)
        max_relative_lp = max(max_relative_lp, float(np.max(abs(lp - a['physical_log_likelihood']) / (1 + abs(lp)))))
        np.testing.assert_allclose(lp, a['physical_log_likelihood'], rtol=2e-12, atol=2e-09)
        sampled = []
        for mode in range(4):
            i = int(np.flatnonzero(a['mode'] == mode)[0])
            k = int(a['lengths'][i])
            m = a['layers'][i, :, :k]
            with torch.no_grad():
                pred = rf_forward(m, a['p'][i], a['tau'][i], a['p_sd'][i]).numpy()
            diff = float(abs(pred - a['predictions'][i]).max())
            max_rf = max(max_rf, diff)
            np.testing.assert_allclose(pred, a['predictions'][i], rtol=1e-09, atol=2e-09)
            np.testing.assert_allclose(depth_projection(torch.tensor(m)).numpy(), a['profile'][i], rtol=2e-07, atol=5e-07)
            sampled.append(i)
        receipts.append(dict(split=split, count=len(sets[-1]), all_labels_recomputed=True, CPU_teacher_source_indices=sampled, npz_sha256=sha(folder / f'{split}.npz'), max_label_difference=error))
    assert sets[0].isdisjoint(sets[1]) and sets[0].isdisjoint(sets[2]) and sets[1].isdisjoint(sets[2])
    a = np.load(folder / 'assessment.npz')
    t = np.load(folder / 'test.npz')
    report = json.loads((folder / 'assessment.json').read_text())
    np.testing.assert_array_equal(a['source_id'], t['source_id'])
    exact = t['physical_log_likelihood'][:, 0, None] - t['physical_log_likelihood'][:, 1:]
    np.testing.assert_allclose(exact, a['physical_gap'], atol=0, rtol=0)
    for arm in p['arms']:
        scores = a[arm + '_scores']
        gap = scores[:, 0, None] - scores[:, 1:]
        correct = (gap > 0) == (exact > 0)
        np.testing.assert_array_equal(correct, a[arm + '_correct'])
        for j, s in enumerate(SCALES):
            keep = abs(exact[:, j]) > 1e-08
            r = report['arms'][arm]['local_scales'][str(s)]
            assert int(keep.sum()) == r['n'] and int(correct[keep, j].sum()) == r['correct']
        if arm == 'frozen_V1':
            continue
        training = json.loads((folder / arm / 'training.json').read_text())
        best = min(training['history'], key=lambda r: r['validation_loss'])
        assert best['epoch'] == training['metadata']['selected_epoch']
        assert best['validation_loss'] == training['metadata']['best_validation_loss']
        path = folder / arm / 'network.pt'
        assert sha(path) == report['arms'][arm]['checkpoint_sha256']
        for split in ('train', 'validation'):
            assert sha(folder / f'{split}.npz') == training['metadata']['data_sha256'][split]
    result = dict(status='passed', version=version, splits=receipts, source_disjoint=True, all_test_counts_recomputed=True, max_physical_label_difference=max_lp, max_label_difference_relative_to_1_plus_abs_energy=max_relative_lp, max_CPU_reference_teacher_rf_difference=max_rf, tests='All labels from stored predictions recomputed; 12 sources/72 candidate predictions recomputed with CPU GRTM. This is numerical verification, not an accuracy certificate.', source_sha256=sha(__file__))
    write_json(folder / 'validation.json', result)
    print('network numerical verification', version, result, flush=True)

def replay_case(version, arm, case, device):
    root = ROOT / f'results/hmc-dnn-refine-local-{version}'
    folder = root / arm / case
    p = json.loads((root / 'protocol.json').read_text())
    verify_sources(p)
    info = json.loads((folder / 'data-info.json').read_text())
    receipt = json.loads((folder / 'reused-inputs.json').read_text())
    for name, h in receipt['inputs'].items():
        assert sha(folder / name) == h
    checkpoint = ROOT / f'results/neural-rf-local-{version}' / arm / 'network.pt'
    assert sha(checkpoint) == p['checkpoints'][arm]
    net, _ = load_network(checkpoint, dtype=torch.float64)
    net.to(device)
    max_objective = 0.0
    max_metric = 0.0
    failed = 0
    for phase in ('inner', 'train'):
        target = data_target(folder, phase, info['kind'], device)
        guide = ExpandedGuide(net, target)
        record = json.loads((folder / f'{phase}-refinement.json').read_text())
        g = record['geometry']
        for row in record['rows']:
            objective = RefinementObjective(target, guide, torch.tensor(g['center'], device=device), torch.tensor(g['transform'], device=device), 1.0, g['neural_center'], g['neural_scale'], row['weight'])
            ys = np.array([c['y'] for c in row['candidates']])
            xs = np.array([c['x'] for c in row['candidates']])
            assert np.max(abs(ys)) <= 3.0 + 1e-10
            with torch.no_grad():
                values = objective(torch.tensor(ys, device=device)).cpu().numpy()
                mapped = objective.x(torch.tensor(ys, device=device)).cpu().numpy()
            np.testing.assert_allclose(mapped, xs, rtol=1e-11, atol=1e-11)
            diff = float(abs(values - np.array([c['objective'] for c in row['candidates']])).max())
            max_objective = max(max_objective, diff)
            np.testing.assert_allclose(values, [c['objective'] for c in row['candidates']], rtol=1e-10, atol=2e-08)
            best = min(row['candidates'], key=lambda c: c['objective'])
            assert best['start'] == row['selected_start']
            failed += int(not best['success'])
        if phase == 'inner':
            zero = next((r for r in record['rows'] if r['weight'] == 0.0))
            best = min(record['rows'], key=lambda r: (r['validation']['rmse'], r['weight']))
            expected = best['weight'] if best['weight'] > 0 and best['validation']['rmse'] <= 0.99 * zero['validation']['rmse'] else 0.0
            selection = json.loads((folder / 'selection.json').read_text())
            assert selection['selected_weight'] == expected
            valid = data_target(folder, 'validation', info['kind'], device)
            mask = np.asarray(valid.data['time']) >= 0.799
            mask &= np.asarray(valid.data['time']) <= 26.001
            saved = np.load(folder / 'inner-hmc.npz')
            with torch.no_grad():
                pred = valid.predict(torch.tensor(saved['best'], device=device)[None])[0].cpu().numpy()
            reference = float(np.sqrt(np.mean((pred[:, mask] - valid.data['rf'][:, mask]) ** 2)))
            chosen = next((r for r in record['rows'] if r['weight'] == expected))
            with torch.no_grad():
                pred = valid.predict(torch.tensor(chosen['x'], device=device)[None])[0].cpu().numpy()
            rmse = float(np.sqrt(np.mean((pred[:, mask] - valid.data['rf'][:, mask]) ** 2)))
            np.testing.assert_allclose(rmse, chosen['validation']['rmse'], atol=1e-11, rtol=1e-11)
            accepted = bool(np.isfinite(rmse) and rmse <= 0.99 * reference)
            assert accepted == selection['accepted_refinement']
            y = torch.tensor(record['rows'][1]['candidates'][0]['y'], device=device, requires_grad=True)
            objective = RefinementObjective(target, guide, torch.tensor(g['center'], device=device), torch.tensor(g['transform'], device=device), 1.0, g['neural_center'], g['neural_scale'], 0.3)
            grad = torch.autograd.grad(objective(y[None]).sum(), y)[0].detach().cpu().numpy()
            fd = []
            coords = [0, len(y) - 1]
            with torch.no_grad():
                for j in coords:
                    step = torch.zeros_like(y)
                    step[j] = 1e-05
                    fd.append(float((objective((y + step)[None]) - objective((y - step)[None])) / 2e-05))
            np.testing.assert_allclose(grad[coords], fd, rtol=2e-05, atol=3e-05)
    result = json.loads((folder / 'result.json').read_text())
    arrays = np.load(folder / 'predictions.npz')
    mask = (arrays['time'] >= 0.799) & (arrays['time'] <= 26.001)
    for row in result['rows']:
        for phase in ('train', 'test'):
            value = float(np.sqrt(np.mean((arrays[row['method'] + '_' + phase][:, mask] - arrays['rf_' + phase][:, mask]) ** 2)))
            max_metric = max(max_metric, abs(value - row[phase]['rmse']))
            np.testing.assert_allclose(value, row[phase]['rmse'], atol=1e-12, rtol=1e-12)
        if info['cohort'] == 'synthetic':
            value = float(np.sqrt(np.mean((arrays[row['method'] + '_profile'] - arrays['truth_profile']) ** 2)))
            np.testing.assert_allclose(value, row['Vs_profile_rmse'], atol=1e-12, rtol=1e-12)
    record = dict(status='passed', case=case, arm=arm, max_objective_difference=max_objective, max_stored_metric_difference=max_metric, selected_optimizer_failures_across_all_weight_phases=failed, source_sha256=sha(__file__), scope='All candidate point objectives/mappings/bounds and saved metrics verified; weight/acceptance reconstructed. Two combined-gradient coordinates checked. Does not certify HMC convergence or geological accuracy.')
    write_json(folder / 'validation.json', record)
    print('replay numerical verification', version, arm, case, flush=True)
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', choices=['v3', 'v4', 'v5'], default='v5')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--network-only', action='store_true')
    args = ap.parse_args()
    network_checks(args.version, args.device)
    if not args.network_only:
        p = json.loads((ROOT / f'results/hmc-dnn-refine-local-{args.version}/protocol.json').read_text())
        for arm in p['arms']:
            for case in p['cases']:
                replay_case(args.version, arm, case, args.device)
