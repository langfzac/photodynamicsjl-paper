using Photodynamics, Plots, DelimitedFiles, JLD2
include("../common.jl")

path = pwd()

# Setup simulation
N = 3

# Stellar parameters (some derived from Carter et al. 2012)
mstar = 1.071 # Solar masses
rstar = 0.0075616597663244675 # AU
k = [0.008378525367832375, 0.020743334339337352] # radius ratios
u_n = [0.3, 0.2] # Arbitrary

# Setup dynamical model
BJD = 0.0
t0_ic = 1.0
tmax = 5000.0
ic = get_default_ICs("kepler-36", t0_ic)
# Convert to mass from ratios
ic.m .*= mstar
ic.elements[:,1] .*= mstar
NbodyGradient.amatrix(ic) # Updates the initial condition hierarchy matrix

# Setup big-float dynamical model
ic_big = begin
    elements = big.(get_default_ICs("kepler-36", t0_ic).elements)
    elements[2:end,3] .-= big(BJD)
    ElementsIC(big(t0_ic), N, elements)
end
# Convert to mass from ratios
ic_big.m .*= mstar
ic_big.elements[:,1] .*= mstar
NbodyGradient.amatrix(ic_big) # Updates the initial condition hierarchy matrix

# Choose exposure times
dt = 2 / 60 / 24; # 2 minute cadence in days
obs_duration = tmax

# Run double-float simulation
lc = begin
    s = State(ic)
    intr = Integrator(ic.elements[2,2]/40, 0.0, tmax)
    tt = TransitTiming(tmax, ic)
    ts = TransitSeries(tmax, ic)
    intr(s, ts, tt, grad=false)
    lc = Lightcurve(dt, obs_duration, u_n, k, rstar)
    compute_lightcurve!(lc, ts, tol=1e-6)
    lc
end

lc_big = begin
    s = State(ic_big)
    intr = Integrator(big(ic.elements[2,2]/40), zero(BigFloat), big(tmax)) # Use double ic to ensure correct stepsize
    tt = TransitTiming(big(tmax), ic_big)
    ts = TransitSeries(big(tmax), ic_big)
    intr(s, ts, tt, grad=false)
    lc_big = Lightcurve(big(dt), big.(lc.tobs), big.(lc.fobs), big.(lc.eobs), big.(u_n), big.(k), big(rstar))
    compute_lightcurve!(lc_big, ts, tol=big(1e-6))
    lc_big
end

jldopen("bigfloat_kepler36.jld2", "w") do file
    file["flux"] = lc.flux
    file["flux_big"] = lc_big.flux
    file["times"] = lc.tobs
end