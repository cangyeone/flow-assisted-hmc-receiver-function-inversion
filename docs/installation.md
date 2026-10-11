# Installation and dependency boundaries

FlowRF is installed from this repository. Its Python package requires NumPy, SciPy and PyTorch. Plotting uses the `plot` extra; legacy observational preprocessing uses `repro`.

The physical solver is **external** `seismicx-grtm`, community RF preview **0.6.0.dev2**. Stable 0.5.1 and the earlier 0.6.0.dev1 preview lack `JointRFForward`, `hmc_sample` and `dram_sample` in their public wheels. Do not use a private `GRTM_SOURCE` path to make an installation appear reproducible.

## Supported physical preview

| Platform | Python | Wheel |
|---|---|---|
| macOS 14+ Apple Silicon | 3.12 | `cp312-cp312-macosx_11_0_arm64` |
| Linux x86_64, glibc ≥2.28 | 3.13 | `cp313-cp313-manylinux_2_28_x86_64` |

Run `flowrf install-grtm`. The installer selects a supported wheel and verifies its SHA-256 against the hashes committed in this release before invoking pip. It fails clearly on unsupported combinations. Python-only neural tools can be used without this physical dependency.

For a manual install, obtain the matching wheel from the [GRTM release](https://github.com/cangyeone/seismicx-grtm-community/releases/tag/v0.6.0.dev2), compare it with `SHA256SUMS-0.6.0.dev2`, then use `python -m pip install path/to/wheel.whl`. CPU physical imports are available through `from grtm import JointRFForward, hmc_sample, dram_sample`; callers always select `device="cpu"`.

On Linux, install CPU PyTorch before FlowRF:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install '.[plot,repro,test]'
flowrf install-grtm
flowrf doctor
python -m pytest -q
```

A wheel built with `python -m build` includes the production model weights; installed tools do not assume a source-checkout location. Experimental weights and paper arrays remain in the repository's reproduction folder. No program is downloaded from a comparison-method archive by the installer.

## Optional RF QC

[DNN-RFQC community](https://github.com/cangyeone/dnn-rfqc-community) performs RF quality screening; follow its own installation and CPU inference instructions. It has a separate GPL license and is not a FlowRF dependency. Select events using a predeclared QC rule, then estimate stack noise/covariance from the retained training events. Neither QC scores nor our experimental neural scores are automatically extra posterior evidence.
