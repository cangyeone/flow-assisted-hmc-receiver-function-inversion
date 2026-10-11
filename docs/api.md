# API, arrays and physical priors

```python
from flowrf import RFData, forward, NeuralModels, RFInverter, BasinModel, VelocityControls
```

## Physical forward modeling

`forward(layers, p, gaussians=(2.5,), dt=.2, n_samples=151, t0=-2, nfft=1024, tau=0, p_sd=0, attenuation=None)` returns a differentiable float64 CPU tensor `[batch, bin, band, time]`. Four-column rows are `(top, Vp, Vs, density)`; six columns add `(Qp, Qs)`. Units are km, km/s, g/cm³, seconds and s/km. Tops begin at zero and increase strictly; the final row is a halfspace. Require positive Vs/density/Q, Vp > sqrt(4/3) Vs and subcritical rays. Four-column models are elastic; six-column models default to constant-Q attenuation. Infinite Q is elastic.

The solver retains sediment reverberations. Gaussian filtering, water-level deconvolution and time shift are differentiable. `tau>0` delays predicted arrivals. Each ray RF is normalized by its radial P sample at zero, then three quadrature rays are averaged per bin if `p_sd` is supplied. This is a **radial P-normalized RF convention**, not arbitrary event deconvolution conventions; match observed processing before inversion.

## Observations

`RFData(rf, p, gaussians, sigma=.03, p_sd=0, covariance=None, fit_mask=None, dt=.2, t0=-2, nfft=1024)` validates all metadata.

- `rf`: `[bin, band, time]` or flattened `[bin*band, time]`; bin first, then band.
- `p`, `sigma`, `p_sd`: one value per slowness bin; scalar noise/spread broadcasts.
- `fit_mask`: time-axis boolean selection, excluding t=0. Default 0.8–26 s.
- `covariance`: `[channel, selected_time, selected_time]` for independent channels, or `[channel*selected_time, channel*selected_time]` for full channel/time correlation. Both must be positive definite. Correlation across bands is not automatically inferred.
- Without covariance, `sigma` defines independent Gaussian errors; this working assumption must be checked on real data. Normalized stack covariance should reflect normalization and event sampling.

`data.save("data.npz")` and `RFData.load("data.npz")` round-trip metadata. An archived EARS file with keys `rf,p,time,fit_mask,cov,test_rf,test_p` is accepted; the API uses fitting RFs only. Holdout arrays are evaluated separately in reproduction scripts. For minimal manually authored NPZ input include `rf,p,gaussians,time,sigma,nfft`; add `covariance,fit_mask,p_sd` as available.

## Neural candidates and RF emulator

`NeuralModels(model_set="global-v2")` loads integrity-checked CPU checkpoints with `weights_only=True` and strict state-dict loading. `propose(data, draws=256, count=None, seed=728)` returns padded `layers`, `count` (finite layers only), `tau`, invertible coordinates `u`, coordinate Jacobian and learned count probabilities. Use `layers[i,:count[i]+1]` to remove padding. Output is a learned simulator-prior approximation, not a validated physical posterior or calibrated count model.

The released flow requires three slowness bins, Gaussian widths `[1,2.5,4]`, and 151 samples at `-2:.2:28` s with sample 10 equal to 1. It learned 2–16 finite layers. `count=` restricts conditional candidate generation to that learned support; it does not validate a geological layer count.

`generate(layers, p, p_sd=0, sigma=.012, tau=0, draws=128, seed=728)` returns `[batch,9,151]` mean/std and `[batch,draw,9,151]` correlated realizations. It accepts arbitrary input layer sequence lengths; unseen counts/properties do not have established accuracy. Std is capped at .03 RF units and zero at P; it describes specified training noise, not calibrated forward-model error.

## Physical inversion

`RFInverter(model=BasinModel(), threads=1).invert(data, initializer="sobol", sampler="hmc", starts=16, chains=4, steps=3000, burn_in=1000, seed=728)` returns `InversionResult`.

`initializer="flow"` physically scores flow projections, uses up to half the optimizer starts from those candidates, and retains independent Sobol starts. Flow density/score does not enter the production target. All starts use the same L-BFGS-B objective and declared iteration cap; selected distinct near-optimal modes initialize jittered chains. This cannot guarantee finding all posterior modes, and pooling chains with diagnostic failure does not estimate mode weights.

| Model | Coordinates and bounds | Prior |
|---|---|---|
| `BasinModel()` | sediment thickness .2–8 km, sediment Vs 1.5–3.2 km/s, Moho 22–65 km, crust Vp/Vs 1.6–2 | uniform physical box; sediment Vp/Vs=1.8/rho=2.5, crust Vp=6.5/rho=2.8, mantle 8/4.5/3.3 fixed |
| `VelocityControls(n=4, smoothness=1)` | sediment thickness .05–10, Vs .7–3.5, Moho 15–75; n crustal Vs controls 2.7–4.3; kappa 1.55–2.05; shift ±.6 s | box plus Gaussian penalties: shift/.25, (kappa−1.78)/.12, .25 sum((Vs−3.6)/.5)², smoothness sum(ΔVs/.8)²; each included in −.5 penalty |

`VelocityControls` uses equal crustal thickness fractions and piecewise constant velocities, with the same fixed sediment and mantle properties. It is a deliberately explicit public model family; archived paper refinement also assessed flexible interfaces, sediment gradients and attenuation through `reproducibility/scripts/`. Model selection uncertainty is not included in these conditional intervals.

Custom physical families can be passed to `RFInverter(model=your_model)`. Supply finite ordered `lower`/`upper` vectors, `names`, `attenuation`, and CPU differentiable methods `layers(x)`, `shift(x)` and `log_prior(x)`, where x is `[chain,parameter]`. `layers` can return any fixed number of solid layers for that run. For flow initialization also supply `project(layers,tau)` returning a physical-coordinate initial vector within your bounds. The declared prior must be in those physical coordinates; include any parameterization Jacobian in `log_prior` if needed. Ordinary HMC has a fixed dimension; discrete layer counts are not automatically sampled by this physical interface.

Samplers: `hmc` uses an exact box Jacobian, first-order autograd, Metropolis correction and warmup-only step-size/mass adaptation; `dram` is our independent two-stage delayed-rejection implementation with warmup-only adaptation; `mwg` uses symmetric coordinate-wise proposals; `hmc_mwg` composes two frozen-step HMC moves with a coordinate sweep, so it costs more transitions per stored iteration. Compare measured evaluations/time and effective samples, not iteration count alone. None is the third-party authors' original implementation.

`result.samples`: `[production_draw,chain,parameter]`; `chain` includes warmup/rejections; `log_prob` is physical-coordinate log density; `predictive` contains up to 64 deterministic physical predictions from retained states. `save()` writes `summary.json`, `posterior.npz`, and `input.npz`. `plot()` displays observations, physical mean RF and a 95% structural predictive interval (no new observation noise added).

`status="local_diagnostics_passed"` requires rank/folded split R-hat <1.01, bulk/tail ESS≥400 for all parameters and no recorded production divergence. Short/unresolved runs remain `provisional_unconverged`. Both report intervals, but unresolved intervals must not be interpreted as calibrated posterior uncertainty. Use independent holdout RFs, geological constraints and synthetic coverage tests before drawing structural conclusions.
