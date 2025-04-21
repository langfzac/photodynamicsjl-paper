using Photodynamics, AdvancedHMC, LsqFit
using JLD2, LinearAlgebra, KahanSummation
using Random, Distributions, CPUTime
using MCMCChains

LinearAlgebra.BLAS.set_num_threads(8) # Set to the default on an 8 core machine

@info "TRAPPIST-1"

include("../common.jl")

# Get the data and true parameters
f = jldopen("../synthetic_data_T1.jld2", "r")
θ_true = f["params"]
nparam = length(θ_true)  # The length of the full vector should be 4+8*nplanet
# Input parameters which are varied, assuming rstar, u_n, mstar, I, and Ω are fixed
# Order of varied parameters:  P, t0, ecos, esin, mass-ratio for each planet, then radius-ratios.
nplanet = 7
θ_inds = Int64[]
for iplanet in 1:nplanet
    append!(θ_inds, [[5+(iplanet-1)*7:1+iplanet*7;]...,4+iplanet*7])
end
append!(θ_inds, [nparam-nplanet+1:nparam]...)

θ_init = θ_true[θ_inds]

# Set up the model
tobs = f["times_thin"]
fobs = f["flux_thin"]
eobs = f["errors_thin"]
dt = 2/60/24
tmax = tobs[end]
lc_model = Lightcurve(dt, tobs, fobs, eobs, θ_true[2:3], θ_true[end-nplanet+1:end], θ_true[1])
dlc_model = dLightcurve(dt, tobs, fobs, eobs, θ_true[2:3], θ_true[end-nplanet+1:end], θ_true[1])
lcm = LightcurveModel(nplanet+1, 0.0, tmax, θ_true[5]/40, lc_model, dlc_model)

# Scale the parameters so the derivatives are similar magnitude
Pfac = 1e8; t0fac = 1e5; eccfac = 1e5; mfac = 1e9; kfac = 1e5
pfac = [repeat([Pfac, t0fac, eccfac, eccfac, mfac],nplanet);repeat([kfac],nplanet)]

function check_priors(θ)
    ipos = Int64[]; ntheta = length(θ)  # The length of theta should be 6*nplanet
    nplanet = div(ntheta,6)
    for iplanet=1:nplanet
        append!(ipos,[1+(iplanet-1)*5,iplanet*5]);
        if 0.2 < sqrt(θ[3+(iplanet-1)*5]^2 + θ[4+(iplanet-1)*5]^2)/eccfac < 0.0; return true; end # Make sure eccentricities are within bounds
    end
    append!(ipos,[ntheta-nplanet+1:ntheta]...)
    if any(θ[ipos] .< 0); return true; end # Make sure period, mass, and radius ratios are positive
    return false
end

compute_flux_model = let lcm=lcm, pfac=pfac, θ_true=θ_true
    (t_in, θ_in) -> begin
        nplanet = div(length(pfac),6)
        # Rescale the input params
        θ_fac = θ_in ./ pfac

        # Assume rstar, u_n, mstar, I, and Ω are fixed 
        θ_total = [θ_true[1:4]...]
        # Add in planetary dynamical parameters:
        for iplanet=1:nplanet
           append!(θ_total,[θ_fac[1+(iplanet-1)*5:4+(iplanet-1)*5]...,θ_true[7*iplanet+2], 0.0, θ_fac[5*iplanet]])
        end
        # Add in planetary radius-ratios:
        append!(θ_total,θ_fac[end-nplanet+1:end])
    
        return compute_flux(lcm, θ_total)#; tol=1e-11, maxdepth=40)
    end
end

compute_jac_model = let lcm=lcm, pfac=pfac, θ_true=θ_true, θ_inds=θ_inds
    (t_in, θ_in) -> begin
        nplanet = div(length(pfac),6)
        # Rescale the input params
        θ_fac = θ_in ./ pfac

        # Assume rstar, u_n, mstar, I, and Ω are fixed 
        θ_total = [θ_true[1:4]...]
        # Add in planetary dynamical parameters:
        for iplanet=1:nplanet
           append!(θ_total,[θ_fac[1+(iplanet-1)*5:4+(iplanet-1)*5]...,θ_true[7*iplanet+2], 0.0, θ_fac[5*iplanet]])
        end
        # Add in planetary radius-ratios:
        append!(θ_total,θ_fac[end-nplanet+1:end])
    
        _, jac_flux = jac_compute_flux(lcm, θ_total)#; tol=1e-11, maxdepth=40)
        jac_flux = jac_flux[:, θ_inds] # Only the non-fixed parameters
    
        # Transform the jacobian 
        for i in eachindex(pfac)
            jac_flux[:, i] ./= pfac[i]
        end
    
        return jac_flux
    end
end

compute_jac_flux_model = let lcm=lcm, pfac=pfac, θ_true=θ_true, θ_inds=θ_inds
    (t_in, θ_in) -> begin
        nplanet = div(length(pfac),6)
        # Rescale the input params
        θ_fac = θ_in ./ pfac

        # Assume rstar, u_n, mstar, I, and Ω are fixed 
        θ_total = θ_true[1:4]
        # Add in planetary dynamical parameters:
        for iplanet=1:nplanet
           append!(θ_total,[θ_fac[1+(iplanet-1)*5:4+(iplanet-1)*5]...,θ_true[7*iplanet+2], 0.0, θ_fac[5*iplanet]])
        end
        # Add in planetary radius-ratios:
        append!(θ_total,θ_fac[end-nplanet+1:end])
    
        flux, jac_flux_all = jac_compute_flux(lcm, θ_total)#; tol=1e-11, maxdepth=40)
        jac_flux = jac_flux_all[:, θ_inds] # Only the non-fixed parameters
    
        # Transform the jacobian 
        for i in eachindex(pfac)
            jac_flux[:, i] ./= pfac[i]
        end
    
        return flux, jac_flux
    end
end

# Run curve_fit to get a MLE and estimate of the covariance
# inputs for LM optimization
tdata = f["times_thin"]
ydata = f["flux_thin"]
weight = 1 ./ f["errors_thin"].^2
params_init = θ_init .* pfac

# To figure out correct pfac values
# (sqrt.(weight) .* compute_jac_model(tdata, params_init))'*(sqrt.(weight) .* (compute_flux_model(tdata, params_init) .- ydata))

# Bounds on the parameters
lb_global = [repeat([0.0, -Inf, -1.0, -1.0, 0.0],nplanet);repeat([0.0],nplanet)] .* pfac
ub_global = [repeat([Inf,  Inf,  1.0,  1.0, Inf],nplanet);repeat([1.0],nplanet)] .* pfac

fit = curve_fit(
    compute_flux_model, compute_jac_model,
    tdata, ydata, weight,
    params_init, lower=lb_global, upper=ub_global)

# Now iterate to make sure it has converged:
icount = 1
chisq_prior = Inf
chisq_curr = sum(fit.resid.^2)
while abs.(chisq_prior - chisq_curr) > (1e-8*chisq_curr)
    global chisq_prior = chisq_curr
    global fit = curve_fit(
            compute_flux_model, compute_jac_model,
            tdata, ydata, weight,
            fit.param, lower=lb_global, upper=ub_global; x_tol = 1e-12, g_tol = 1e-15)
    global chisq_curr = sum(fit.resid.^2)
    println(icount," ",chisq_curr," ",chisq_prior - chisq_curr)
    global icount += 1
end

# Get the estimate of the covariance matrix (M = inv(C))
fit_covar = Matrix(Hermitian(estimate_covar(fit)))

# Get initial guess of parameters
fit_params = fit.param

# Distribution to sample starting points from
dist = MvNormal(fit_params, fit_covar)

### MCMC sampling ###
# Construct target distribution and gradient for NUTS
compute_loglike = let compute_model = Base.Fix1(compute_flux_model, tdata), ydata=ydata, weight=weight
    θ::Vector{<:Real} -> begin
        if check_priors(θ); return -Inf; end

        f_model = compute_model(θ)
        res = ydata .- f_model
        loglike = -0.5*sum_kbn(res.^2 .* weight)
        return loglike
    end
end

compute_grad_loglike = let compute_jac_model = Base.Fix1(compute_jac_flux_model, tdata), ydata=ydata, weight=weight
    θ::Vector{<:Real} -> begin
        if check_priors(θ); return (-Inf, zeros(eltype(θ), length(θ))); end

        f_model, jac_model = compute_jac_model(θ)
        res = ydata .- f_model
        res_weight = res .* weight
        loglike = -0.5*sum_kbn(res .* res_weight)
        ∇loglike = zeros(length(θ))
        for i in eachindex(θ)
            ∇loglike[i] = sum_kbn(jac_model[:,i] .* res_weight)
        end
        return loglike, ∇loglike
    end
end

# Set up and run NUTS sampling
rng = Random.Xoshiro(0)
initial_mcmc_params = rand(rng, dist)
n_samples, n_adapts = 1000, 100
metric = DenseEuclideanMetric(fit_covar)
hamiltonian = Hamiltonian(metric, compute_loglike, compute_grad_loglike)
initial_ϵ = find_good_stepsize(rng, hamiltonian, initial_mcmc_params)
integrator = Leapfrog(initial_ϵ)
kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn(;max_depth=3)))
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

t0 = time()
CPUtic()
@CPUtime samples, stats = sample(rng, hamiltonian, kernel, initial_mcmc_params, n_samples, adaptor, n_adapts; progress=true)
t_hmc = (time() - t0) / 60 / 60
t_hmc_cpu = CPUtoc() / 60 / 60
@info "NUTS: $(t_hmc) hours; $(t_hmc_cpu) CPU hours"
jldopen("hmc_results_T1.jld2", "a+") do file
    file["total_time"] = t_hmc
    file["total_CPU_time"] = t_hmc_cpu
    samples_nuts = [sample ./ pfac for sample in samples] # Transform back to unscaled model parameters
    file["samples"] = samples_nuts
    file["stats"] = stats

    # Get the MAP flux model
    θ_nuts = mean(hcat(samples_nuts...), dims=2) .* pfac
    file["flux_map"] = compute_flux_model(tdata, θ_nuts)
end