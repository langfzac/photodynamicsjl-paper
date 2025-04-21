## Inference
This directory contains code to create Figures 9, 10, and 11, and carry out the likelihood profile and MCMC computations.

The synthetic data scripts use a command-line argument parser. The options can be show via:\
`julia --project synthetic_data.jl --help`

To generate the 2-body synthetic data: \
`julia --project synthetic_data.jl synthetic_data.jld2`

To generate the TRAPPIST-1 synthetic data: \
`julia --project synthetic_data_T1.jl synthetic_data_T1.jld2`

To choose the specific planets of TRAPPIST-1 to include, provide a list of indices for the planets. E.g. to get a synthetic dataset that includes planets b,c,e:\
`julia --project synthetic_data_T1.jl synthetic_data_T1_bce.jld2 [2,3,5]` 