using Photodynamics, Plots, DelimitedFiles, JLD2
include("../common.jl")

path = pwd()

# Setup simulation
N = 8

# Stellar parameters
mstar = 0.0898 # Solar masses
rstar = 0.00465047 * 0.1192 # Trappist-1 (Rstar/AU)
k = sqrt.([0.7277,0.6940,0.3566,0.4802,0.634,0.764,0.346] .* 0.01)[1:N-1]
u_n = [0.2818125324456011, 0.10675603030060994]

# Setup dynamical model
BJD = 7250.0
t0_ic = 1.0
tmax = 5000.0
ic = get_default_ICs("trappist-1", t0_ic)
#ic.elements[2:end,3] .= 0.0
# Convert to mass from ratios
ic.m .*= mstar # Trappist-1 mass
ic.elements[:,1] .*= mstar
NbodyGradient.amatrix(ic) # Updates the initial condition hierarchy matrix

# Setup big-float dynamical model
ic_big = begin
    elements = big.(get_default_ICs("trappist-1", t0_ic).elements)
    #elements[2:end,3] .-= big(BJD)
    ElementsIC(big(t0_ic), N, elements)
end
#ic_big.elements[2:end,3] .= 0.0
# Convert to mass from ratios
ic_big.m .*= mstar # Trappist-1 mass
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

jldopen("bigfloat_trappist1.jld2", "w") do file
    file["flux"] = lc.flux
    file["flux_big"] = lc_big.flux
    file["times"] = lc.tobs
end