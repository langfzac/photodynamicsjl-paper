## Likelihood Profile
The directory contains the code used to compute the profile likelihood for the 2-planet model, and creates Figures 9 and 10.

Order of operations:
1) Generate the synthetic data in the `inference/` directory.
2) Run `curve_fit.jl` to generate data.
3) Create plots with `plots.jl`.