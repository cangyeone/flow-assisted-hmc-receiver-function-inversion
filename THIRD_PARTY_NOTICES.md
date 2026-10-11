# Third-party notices and publication boundary

Only this project's own programs and trained model weights are published here. There is no `third_party/` or `vendor/` implementation tree, no author Bayesian/LITMOD program, no Fortran/C/C++/CUDA solver source, no third-party software archive and no GRTM implementation binary in this repository.

External dependencies are installed through their own distributions and retain their licenses: NumPy/SciPy, PyTorch, Matplotlib, ObsPy, Requests, Packaging and PyShp. GRTM community retains its research/noncommercial license and associated binary notices. DNN-RFQC community is an optional separate GPL package; its code and model files are not copied here. This repository's MIT license does not change those terms.

Our archived DRAM implementation follows the cited algorithm independently. Result arrays from an externally run original method are research outputs, not a redistribution of that program. Comparison instructions cite papers and accept user-generated outputs; they do not fetch or package author code.

Processed EARS RFs originate from the public EarthScope/IRIS EARS archive, associated Transportable Array stations and its data contributors. The per-station URLs, event members, train/test splits and source SHA-256 values are retained in `reproducibility/data/processed/manifest.json`. Derived stacks/covariances and analysis results were produced by this project. Credit original waveform/RF providers in scientific use. Raw source archive bundles are not included.

Basin outlines originate from the USGS Sedimentary Basins of the United States, Open-File Report 2012–1111: [source](https://pubs.usgs.gov/of/2012/1111/Sedimentary_Basins_of_the_United_States.html). The analysis uses simplified regional polygons and does not claim their boundaries were inferred from RFs. Source attribution and map processing are retained in the basin context records.

The export manifest distinguishes original source hashes from CPU-port/publication hashes. Historical result metadata preserves experimental outcomes and hardware provenance; it must not be used to claim newly measured CPU speed.
