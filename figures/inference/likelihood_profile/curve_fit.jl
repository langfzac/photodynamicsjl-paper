using Photodynamics, LsqFit
using JLD2, Plots, BenchmarkTools
using Calculus, LinearAlgebra

include("../common.jl")

# Get the data and true parameters
f = jldopen("../synthetic_data.jld2")
const θ_true = f["params"]

# Compute the "true" model
tobs = f["times_thin"]
fobs = f["flux_thin"]
eobs = f["errors_thin"]
dt = 2/60/24 # 2 minute cadence in days
tmax = tobs[end]
lc_model = Lightcurve(dt, tobs, fobs, eobs, θ_true[2:3], θ_true[end-1:end], θ_true[1])
dlc_model = dLightcurve(dt, tobs, fobs, eobs, θ_true[2:3], θ_true[end-1:end], θ_true[1])
lcm = LightcurveModel(3, 0.0, tmax, θ_true[5]/40, lc_model, dlc_model);
flux_true = compute_flux(lcm, θ_true)

# Input parameters, assuming rstar, u_n, mstar, I, and Ω are fixed
θ_inds = [[5:8;]...,11,[12:15;]...,18,19,20]
θ_init = θ_true[θ_inds]

# Scale the parameters so the derivatives are similar magnitude
Pfac = 1e6; t0fac = 1e4; eccfac = 1e6; mfac = 1e9; kfac = 1e3
const pfac = [
        Pfac, t0fac, eccfac, eccfac, mfac, 
        Pfac, t0fac, eccfac, eccfac, mfac,
        kfac, kfac
    ] 

function compute_flux_model(t_in, θ_in)
    # Rescale the input params
    θ_fac = θ_in ./ pfac

    # Assume rstar, u_n, mstar, I, and Ω are fixed 
    θ_total = [
        θ_true[1], θ_true[2], θ_true[3], 1.0,
        θ_fac[1:4]..., π/2, 0.0, θ_fac[5],
        θ_fac[6:9]..., π/2, 0.0, θ_fac[10],
        θ_fac[11], θ_fac[12]
    ]

    return compute_flux(lcm, θ_total)#; tol=1e-11, maxdepth=40)
end

function compute_flux_model_prior(t_in, θ_in)
    # Rescale the input params
    θ_fac = θ_in ./ pfac

    # Assume rstar, u_n, mstar, I, and Ω are fixed 
    θ_total = [
        θ_true[1], θ_true[2], θ_true[3], 1.0,
        θ_fac[1:4]..., π/2, 0.0, θ_fac[5],
        θ_fac[6:9]..., π/2, 0.0, θ_fac[10],
        θ_fac[11], θ_fac[12]
    ]

    return [compute_flux(lcm, θ_total); θ_in]#; tol=1e-11, maxdepth=40)
end

function compute_jac_flux_model(t_in, θ_in)
    # Rescale the input params
    θ_fac = θ_in ./ pfac

    # Assume rstar, u_n, mstar, I, and Ω are fixed 
    θ_total = [
        θ_true[1], θ_true[2], θ_true[3], 1.0,
        θ_fac[1:4]..., π/2, 0.0, θ_fac[5],
        θ_fac[6:9]..., π/2, 0.0, θ_fac[10],
        θ_fac[11], θ_fac[12]
    ]

    _, jac_flux = jac_compute_flux(lcm, θ_total)#; tol=1e-11, maxdepth=40)
    jac_flux = jac_flux[:, θ_inds] # Only the non-fixed parameters

    # Transform the jacobian 
    for i in eachindex(pfac)
        jac_flux[:, i] ./= pfac[i]
    end

    return jac_flux
end

function compute_jac_flux_model_prior(t_in, θ_in)
    # Rescale the input params
    θ_fac = θ_in ./ pfac

    # Assume rstar, u_n, mstar, I, and Ω are fixed 
    θ_total = [
        θ_true[1], θ_true[2], θ_true[3], 1.0,
        θ_fac[1:4]..., π/2, 0.0, θ_fac[5],
        θ_fac[6:9]..., π/2, 0.0, θ_fac[10],
        θ_fac[11], θ_fac[12]
    ]

    _, jac_flux = jac_compute_flux(lcm, θ_total)#; tol=1e-11, maxdepth=40)
    jac_flux = jac_flux[:, θ_inds] # Only the non-fixed parameters

    # Transform the jacobian 
    for i in eachindex(pfac)
        jac_flux[:, i] ./= pfac[i]
    end

    dθdθ = Diagonal(ones(length(θ_in)))
    return vcat(jac_flux, dθdθ)
end

# Inputs for the LM optimization
tdata = f["times_thin"]
ydata = f["flux_thin"]
weight = 1 ./ f["errors_thin"].^2
params_init = θ_init .* pfac

# Time the jacobian computations vs finite differences
compute_num_jac_flux_model(tdata, θ_in) = Calculus.jacobian(x->compute_flux_model(tdata, x))(θ_in)
compute_num_jac_flux_model_prior(tdata, θ_in) = Calculus.jacobian(x->compute_flux_model_prior(tdata, x))(θ_in)
@info "Analytic derivatives:"
@btime compute_jac_flux_model($tdata, $params_init)
@info "Numerical derivatives:"
@btime compute_num_jac_flux_model($tdata, $params_init)

# To figure out correct pfac values
# (sqrt.(weight) .* compute_jac_flux_model(tdata, params_init))'*(sqrt.(weight) .* (compute_flux_model(tdata, params_init) .- ydata))

# Run Levenburg Marquardt
# Bounds on the parameters
lb_global = [0.0, -Inf, -1.0, -1.0, 0.0, 0.0, -Inf, -1.0, -1.0, 0.0, 0.0, 0.0] .* pfac
ub_global = [Inf,  Inf,  1.0,  1.0, Inf, Inf,  Inf,  1.0,  1.0, Inf, 1.0, 1.0] .* pfac

# Time analytic vs. numerical Jacobian
@info "Fit with analytic Jacobian:"
@btime curve_fit(
    $compute_flux_model, $compute_jac_flux_model, 
    $tdata, $ydata, $weight, 
    $params_init, lower=$lb_global, upper=$ub_global);
    
@info "Fit with numerical Jacobian:"
@btime curve_fit(
    $compute_flux_model, 
    $tdata, $ydata, $weight, 
    $params_init, lower=$lb_global, upper=$ub_global);

# Now optimize saving the trace for each method
fit = curve_fit(
        compute_flux_model, compute_jac_flux_model, 
        tdata, ydata, weight, 
        params_init, lower=lb_global, upper=ub_global,
        store_trace=true)

fit_num = curve_fit(
        compute_flux_model,
        tdata, ydata, weight, 
        params_init, lower=lb_global, upper=ub_global,
        store_trace=true)

# Get error estimates for the parameters
param_errors = standard_errors(fit)
param_errors_num = standard_errors(fit_num)
jldopen("likelihood_profile.jld2", "a+") do file
    file["best_fit/mean"] = fit.param
    file["best_fit/std"] = param_errors
    file["best_fit/mean_num"] = fit_num.param
    file["best_fit/std_num"] = param_errors_num
    file["best_fit/pfac"] = pfac
    file["best_fit/trace"] = fit.trace
    file["best_fit/trace_num"] = fit_num.trace
end

# Show difference in chi-squared
chi2_diff = sum(fit.resid.^2) - sum(fit_num.resid.^2)
@info chi2_diff
if chi2_diff < 0.0
    @info "Analytic derivatives found better optimum"
else
    @info "Numerical derivatives found better optimum"
end

# Setup a grid for each parameter
# Ordered: right of MLE, then left from MLE
# Want to 'trace' likelihood from the MLE 
get_grid(μ,σ) = [collect(LinRange(μ, μ+3*σ, 10))..., collect(LinRange(μ, μ-3*σ, 10))...]
param_names = ["P1", "t01", "ecosω1", "esinω1", "m1", "P2", "t02", "ecosω2", "esinω2", "m2", "k1", "k2"]

param_grids = get_grid.(fit.param, param_errors)
param_grids_num = get_grid.(fit_num.param, param_errors_num)

# Loop over each set of parameters
for i in eachindex(param_grids)
    lls = []; 
    parameters = [];
    θ_fit = fit.param;
    nparam = length(θ_fit)
    # Loop over each fixed value 
    dp = diff(param_grids[i])[1] # Grid spacing
    @time for p_val in param_grids[i]
        # Check if we're back at the MLE value
        if p_val == fit.param[i]
            # If so, grad the MLE vector to start
            @info "Restart at MLE"
            θ_next = copy(fit.param)
        else
            # If not, use the last fit params
            θ_next = copy(θ_fit)
        end
        θ_next[i] = p_val # Insert the "fixed" value
        weight_param = zeros(nparam); weight_param[i] = 1e10 # Apply "boundary" to "fix" the parameter

        #@info lb_next
        #@info ub_next
        @info θ_next ./ pfac
        flush(stdout); flush(stderr)

        # Optimize with curve_fit
        fit_next = curve_fit(
            compute_flux_model_prior, compute_jac_flux_model_prior, 
            [tdata;ones(nparam)], [ydata;θ_next], [weight;weight_param], 
            θ_next, lower=lb_global, upper=ub_global, 
            maxIter=10000)

        @info "Converged: " fit_next.converged

        # Save optimum
        θ_fit = copy(fit_next.param)
        push!(parameters, θ_fit)
        
        # Compute and save the loglikelihood
        push!(lls, sum(fit_next.resid.^2))
    end
    # Save the set of loglikelihoods to common file
    jldopen("likelihood_profile.jld2", "a+") do file
        file[param_names[i]*"/lls"] = lls
        file[param_names[i]*"/p_val"] = param_grids[i]
        file[param_names[i]*"/param"] = parameters
    end

    # Now with numerical derivatives
    lls_num = []
    parameters_num = []
    θ_fit_num = fit_num.param
    dp_num = diff(param_grids_num[i])[1] # Grid spacing
    @time for p_val in param_grids_num[i]
        # Check if we're back at the MLE value
        if p_val == fit_num.param[i]
            # If so, grad the MLE vector to start
            @info "Restart at MLE"
            θ_next = copy(fit_num.param)
        else
            # If not, use the last fit params
            θ_next = copy(θ_fit_num)
        end
        θ_next[i] = p_val # Insert the "fixed" value
        weight_param = zeros(nparam); weight_param[i] = 1e10

        #@info lb_next
        #@info ub_next
        @info θ_next ./ pfac
        flush(stdout); flush(stderr)

        # Optimize with curve_fit
        fit_next = curve_fit(
            compute_flux_model_prior,
            [tdata;ones(nparam)], [ydata;θ_next], [weight;weight_param],
            θ_next, lower=lb_global, upper=ub_global, 
            maxIter=10000)

        @info "Converged: " fit_next.converged

        # Save optimum
        θ_fit_num = copy(fit_next.param)
        push!(parameters_num, θ_fit_num)
        
        # Compute and save the loglikelihood
        push!(lls_num, sum(fit_next.resid.^2))
    end
    # Save the set of loglikelihoods to common file
    jldopen("likelihood_profile.jld2", "a+") do file
        file[param_names[i]*"/lls_num"] = lls_num
        file[param_names[i]*"/p_val_num"] = param_grids_num[i]
        file[param_names[i]*"/param_num"] = parameters_num
    end
end
