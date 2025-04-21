## Comparison to `photodynam`
This directory contains code to create Figures 6, 7, and 8.

Order of operations:
1) Compile `photodynam` via the Makefile.
2) Run `compare_photodynam.jl`
3) Run `plots.jl`

The `photodynam` code is from [github.com/dfm/photodynam](https://github.com/dfm/photodynam). We slightly modify the code to:
1) Increase the precision of the output flux values.
2) Not compute the light-travel-time correction.
