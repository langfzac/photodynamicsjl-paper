using Photodynamics, Plots, DelimitedFiles, JLD2
include("../common.jl")

path = pwd()

# Setup simulation
N = 5

# Stellar parameters (from Mills et al. 2016)
rstar = 0.007998803688854911 # AU
k = [0.01596, 0.01847, 0.02791, 0.02450] # radius ratios
u_n = [0.54, 0.2] # Arbitrary

# Setup dynamical model
BJD = 2455701.5155
t0_ic = 1.0
tmax = 5000.0
a = Elements(m=1.125)
b = Elements(
    m = 2.2226e-5,
    P = 7.38449,
    ecosω = 0.057,
    esinω = 0.052,
    I = π/2,
    t0 = 2455701.5155
)
c = Elements(
    m = 1.5318e-5,
    P = 9.84564,
    ecosω = 0.030,
    esinω = 0.134,
    I = π/2,
    t0 = 2455700.1459
)
d = Elements(
    m = 2.4028e-5,
    P = 14.78869,
    ecosω = 0.020,
    esinω = 0.017,
    I = 87.94 * π/180,
    t0 = 2455704.8504
)
e = Elements(
    m = 1.4417e-5,
    P = 19.72567,
    ecosω = 0.017,
    esinω = 0.045,
    I = 88.0 * π/180,
    t0 = 2455717.5237
)
ic = ElementsIC(t0_ic,5,a,b,c,d,e)
ic.elements[2:end,3] .-= BJD

# Setup big-float dynamical model
ic_big = begin
    elements = big.(ic.elements)
    ElementsIC(big(t0_ic), N, elements)
end

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

jldopen("bigfloat_kepler223.jld2", "w") do file
    file["flux"] = lc.flux
    file["flux_big"] = lc_big.flux
    file["times"] = lc.tobs
end