# Reproduction and archival evidence

This folder is separate from the reusable package. It contains our CPU research programs, retained model weights, processed observations, result arrays, configurations and provenance. It contains **no third-party inversion program**. Original author software must be obtained separately; its paper can always be cited without distributing code.

## 1. Verify the release

```bash
python -m pip install '.[repro,test]'
python reproducibility/scripts/verify_release.py
```

`export-manifest.json` records each original file hash and its distributed hash. CPU imports/output paths and private workstation paths were adapted where necessary, with numeric results preserved. Historical source hashes in archived experiment protocols refer to the original experimental programs, not these CPU ports. Historical hardware/timing records are archival evidence, not measured CPU speed of the new API.

## 2. Reconstruct the real-data figures from frozen arrays

```bash
python reproducibility/scripts/build_field_figures.py
```

No inversion or raw archive download is needed. Outputs go to `reproducibility/outputs/field-figures/`: regional basin/station maps and depth sections, paired RF fits, and diagnostics in PDF/SVG/PNG, plus provenance. The 42-station archive retains 35 passing local diagnostics, 7 unresolved runs, 32 compact-model boundary solutions, all 42 selected models near a boundary, and the observed fit comparison (26 improve, 3 worsen, 13 unchanged). Maps do not interpolate or claim a new geological discovery.

## 3. Run CPU inversion on existing observations

After `flowrf install-grtm`:

```bash
flowrf invert reproducibility/data/processed/V20A.npz --initializer sobol \
  --sampler hmc --steps 3000 --burn-in 1000 --output outputs/V20A
python reproducibility/scripts/cpu_hmc.py V20A --method hmc \
  --steps 3000 --burn 1000 --starts 16 --out outputs/baseline
```

Archived EARS observations are single-band (Gaussian 2.5), not valid 9-channel neural-flow inputs. Use the prior/Sobol baseline for these arrays. Genuine multiband observations are required for the released conditional flow. Do not use an emulator to create missing measured bands.

For flexible crust interfaces, sediment gradients, attenuation and training-only model selection, inspect `scripts/fit_refinement.py`, `basin_rf.py`, `synthetic_fit_check.py` and `basin_assessment.py`. Their original fixed experiment settings are retained, with the public CPU GRTM imports. They write to `reproducibility/results/`; use new result directories or save the frozen archive before a long rerun. Default user API parameterizations are deliberately simpler and are not a bitwise replacement for every historical target.

## 4. Retrieve observational inputs for event-level preprocessing

```bash
python reproducibility/scripts/download_observations.py --station V20A
# Omit --station to retrieve the full declared cohort.
```

The downloader uses the recorded EARS URLs and checks original archive SHA-256. It does not execute downloaded content. Raw archives are deliberately excluded from Git; processed stacks/covariances, event splits and source hashes are included. `prepare_ears.py` documents the geographic selection and processing; `prepare_basin_context.py` documents USGS map context. Live external archive availability is not guaranteed; frozen processed arrays keep figure reproduction independent of it.

## 5. New CPU accuracy, sampling and initialization comparison

```bash
python reproducibility/scripts/benchmark_cpu.py --output reproducibility/outputs/benchmark
```

This makes independent noisy basin and thin-sediment data and compares Sobol multi-start versus flow-assisted multi-start with **the same physical target** using HMC, our DRAM and MWG. It reports parameter/profile RMSE, RF fit, interval widths/coverage indicators, initialization/sampling time and unresolved diagnostics. It runs a new protocol, not the paper's historic original-author comparison. A quick smoke can use `--repeats 1 --steps 18 --burn-in 8 --starts 2 --maxiter 2 --flow-draws 4 --methods hmc`; short results must stay provisional.

For the author's original Bayesian program, obtain it directly from the authors, run it independently with matching data, priors, likelihood, parameterization and budget, then convert its retained samples to an NPZ `samples[draw,chain,parameter]`. Import **only the output**:

```bash
python reproducibility/scripts/import_external_results.py author-results.npz \
  --citation 'doi:10.1029/2025JB032180' --parameter-names hs Vs_s H kappa \
  --output reproducibility/outputs/author-method
```

The importer is our array reader, not an implementation of the author program. `data/flow-initialization/`, `component-ablation/` and `iteration-budget/` preserve historic precision/ablation/budget summaries and predictive arrays. They are not labeled as new CPU speed measurements. An unresolved original comparison remains unresolved; no result is selected or omitted to prove our method superior.

## 6. CPU neural simulation/training

Small complete pipeline:

```bash
python -m flowrf.training --config reproducibility/configs/smoke.json --output reproducibility/outputs/train-smoke --stage generate
python -m flowrf.training --config reproducibility/configs/smoke.json --output reproducibility/outputs/train-smoke --stage pilot
python -m flowrf.training --config reproducibility/configs/smoke.json --output reproducibility/outputs/train-smoke --stage bootstrap
python -m flowrf.training --config reproducibility/configs/smoke.json --output reproducibility/outputs/train-smoke --stage full
```

Use `configs/million.json` for the full declared sample sizes and architecture; run all four stages in a new output folder. CPU million-model generation/training is costly, and no training-time claim is made here. Frozen protocol records include source hashes, physical-module hash and runtime versions; mismatched resumes are rejected. Atomic shards/receipts retain source IDs, source-disjoint splits and RNG seeds; complete-epoch checkpoints retain optimizer/RNG state for resume. `configs/smoke.json` validates execution only and is not a trained scientific model. Million-model generated shards are not committed; they are regenerated from the declared prior, physical dependency, configuration and seeds. Reproduction is numerical/protocol reproduction, not promised bitwise identity across platforms.

The RF emulator and inverse flow train on exact physical RFs; the inverse is not trained on the emulator's predicted RFs. Bounded heteroscedastic losses, noise covariance and local-pair scorer objectives remain available in our source. Experimental scoring models are not used by default inversion. The ratio-network loader is `scripts/neural_rf.py`; the local likelihood loader is `scripts/experimental_likelihood.py`.

[中文说明](README.zh-CN.md) · [model cards](../docs/models.md) · [method references](../docs/references.md)
