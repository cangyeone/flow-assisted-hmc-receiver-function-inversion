# Released model cards

All weights are ours. Production weights are bundled in the installed Python wheel; experimental weights are in the reproduction tree. Complete paths and SHA-256 hashes are in `src/flowrf/weights/manifest.json`. Loading uses CPU, `weights_only=True`, strict architecture matching and hash verification.

| Set | Networks | Intended use |
|---|---|---|
| `global-v2` (default) | ForwardTransformer and PosteriorTransformerFlow | global synthetic RF emulator and starting-point proposals |
| `global-v1` | same trained parameters, earlier provenance metadata | archived comparison of full checkpoints |
| `reproducibility/models/transformer-{pilot,bootstrap}` | both networks | training-stage reconstruction, not production claims |
| `neural-rf-v1` / `neural-rf-local-v3,v4,v5` | global and local-pair ratio scorers, easy controls | experimental ranking / negative controls; not default inference |
| `neural-rf-physics-v2` | conditional, V20A and X20A local likelihood networks | observation-specific experiments; not transferable global posterior models |

## Global Transformer / conditional flow

Training uses 1,000,000 source structures, 8,192 source-disjoint validation structures and 8,192 test structures. Architecture: width 96, encoder depth 3; conditional flow has 8 affine coupling blocks. The best checkpoint is selected by validation, never the test set. Historical training used accelerators; the published programs expose CPU execution and do not promise identical training time or bitwise retraining across platforms.

One synthetic-trained checkpoint is used across stations; no regional retraining is performed by the public API. This design is global in its training prior, not proof of universal geological transfer. Support: 2–16 finite layers, depth 12–80 km, Vs .3–4.3 km/s, Vp/Vs lower limit 1.55 with Vp≤8.8, density 1.6–3.2, Qs 10–1000, Qp=2Qs, common shift ±.3 s, mantle fixed to 8/4.5/3.3. Simulator prior includes an 80% sediment-rich mixture and occasional low-velocity structure; RF absolute amplitude >8 rejects/replaces a source, defining a truncated training prior. See `reproducibility/configs/million.json` and `flowrf.neural.structure.sample_prior`.

Input channels: three slowness bins times Gaussian widths 1/2.5/4; .2 s sampling, −2 to 28 s; P normalization to 1. Slowness metadata ranges approximately .039–.049, .051–.065, .067–.082 s/km, spread .0002–.003. Base noise sigma .005–.025; bounded temporal heteroscedasticity, time covariance and 80% shared band variance are declared in `flowrf.neural.data.Noise`. Emulator output std range .001–.03, with zero variance at the normalization point.

Default inverse checkpoint SHA-256: `d4584c37ac1ba8ddcce1c817eedb5f88270ddddf88b037640b68c5457cdb87e2`. The two sets have exactly equal state tensors in both arms; their checkpoint hashes differ because provenance metadata differ. They are not independent training replications. Earlier global-v1 inverse: `728cd0121b04a1844a8cef53ecb933a2820e6c0cc1d76ffa49624951a66ad469`.

**Limits:** conditional-flow candidate density is learned from a specified simulator prior and observation noise. It is not calibrated field uncertainty. Layer-count accuracy outside 2–16 and RF accuracy on unseen layer counts are not established. Neural forward approximation error can exceed the precision required by the physical likelihood; its bounded std is not an error envelope. The public inverter therefore evaluates the physical solver throughout optimization and sampling.

## Experimental weights

The local-pair scorers were trained on fine perturbations using physical likelihood preferences, rather than labeling every perturbation wrong. Their ranking is still heuristic. Archived tests did not establish a reliable accuracy gain, so these models do not constrain default physical inference.

The observation-specific likelihood approximation stores its physical bounds, logit whitening and calibration point. It is tied to the exact observation/model family used during calibration. `reproducibility/scripts/experimental_likelihood.py` supplies its architecture and CPU loader; `neural_rf.py` supplies ratio-network loading. Model files alone do not establish independent evidence, transferable calibration or global mode coverage.
