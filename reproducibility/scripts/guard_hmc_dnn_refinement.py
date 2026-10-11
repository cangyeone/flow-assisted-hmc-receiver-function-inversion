# Our CPU-only archived research implementation; experimental, not default inference.
"""Internal-event acceptance guard for the exploratory two-stage point fit.

The rule is added after inspecting exploratory failures and is not a blind
new validation. It keeps the original HMC candidate if refinement fails to
improve internal-validation RF RMSE by 1%. It changes no posterior samples,
network, fitting result or original assessment file.
"""
import argparse, json
import numpy as np
import torch
from hmc_dnn_refinement import ROOT, data_target
from neural_rf_experiment import sha, write_json
OUT = ROOT / 'results/hmc-dnn-refine-v2'

def accept_refinement(hmc_rmse, refined_rmse):
    return bool(np.isfinite(refined_rmse) and refined_rmse <= 0.99 * hmc_rmse)

def guard(case):
    folder = OUT / case
    info = json.loads((folder / 'data-info.json').read_text())
    fine = json.loads((folder / 'inner-refinement.json').read_text())
    selection = json.loads((folder / 'selection.json').read_text())
    weight = selection['selected_weight']
    candidate = next((r for r in fine['rows'] if r['weight'] == weight))
    target = data_target(folder, 'validation', info['kind'], 'cpu')
    best = np.load(folder / 'inner-hmc.npz')['best']
    with torch.no_grad():
        prediction = target.predict(torch.tensor(best)[None])[0].numpy()
    mask = (target.data['time'] >= 0.799) & (target.data['time'] <= 26.001)
    reference = float(np.sqrt(np.mean((prediction[:, mask] - target.data['rf'][:, mask]) ** 2)))
    accepted = accept_refinement(reference, candidate['validation']['rmse'])
    method = f'fine_weight{weight:g}' if accepted else 'HMC_best'
    choice = dict(case=case, accepted_refinement=accepted, method=method, selected_DNN_weight=weight, hmc_inner_validation_rmse=reference, refined_inner_validation_rmse=candidate['validation']['rmse'], rule='Accept refinement only if its inner-validation RF RMSE improves at least 1% over the inner-HMC best candidate. Otherwise keep the full-training HMC best candidate.', scope='Post-exploration operational guard; no outer/truth input to the decision. Predictive point estimates only, no validated posterior uncertainty.', source_sha256=sha(__file__), inputs={name: sha(folder / name) for name in ('inner-hmc.npz', 'inner-refinement.json', 'selection.json', 'data-validation.npz')})
    write_json(folder / 'guard-selection.json', choice)
    results = json.loads((folder / 'result.json').read_text())
    row = next((r for r in results['rows'] if r['method'] == method))
    write_json(folder / 'guarded-result.json', dict(selection=choice, row=row, result_sha256=sha(folder / 'result.json'), guard_selection_sha256=sha(folder / 'guard-selection.json'), full_hmc_converged=results['full_hmc_converged']))
    print(case, 'accept refinement', accepted, 'weight', weight, 'inner RF', reference, '->', candidate['validation']['rmse'], flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case')
    args = parser.parse_args()
    p = json.loads((OUT / 'protocol.json').read_text())
    for case in [args.case] if args.case else p['cases']:
        guard(case)
