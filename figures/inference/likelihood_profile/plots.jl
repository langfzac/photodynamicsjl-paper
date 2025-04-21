using Plots, LaTeXStrings, Statistics, Printf
using JLD2
pythonplot()

include("../common.jl")

default(
    legendfontsize=8,
    tickfontsize=12,
    xlabelfontsize=12,
    ylabelfontsize=14,
    grid=:off, box=:on,
    palette=:seaborn_colorblind
)

function make_xticks(x, dx)
    xmid = middle(x)
    return [xmid-dx, xmid, xmid+dx]
end

# Scaling constants
Pfac = 1e6; t0fac = 1e4; eccfac = 1e6; mfac = 1e9; kfac = 1e3
pfac = [
    Pfac, t0fac, eccfac, eccfac, mfac, 
    Pfac, t0fac, eccfac, eccfac, mfac,
    kfac, kfac] 

# Read in the likelihood profile results
f = jldopen("likelihood_profile.jld2")
param_names = ["P1", "m1", "P2", "m2"] 

# Get the best fit parameters and variances
param_inds = [1,5,6,10]
param_mean = (f["best_fit"]["mean"] ./ pfac)[param_inds]
param_std = (f["best_fit"]["std"] ./ pfac)[param_inds]
param_mean_num = (f["best_fit"]["mean_num"] ./ pfac)[param_inds] 
param_std_num = (f["best_fit"]["std_num"] ./ pfac)[param_inds]

# Get the true parameters for comparison
f_data = jldopen("../synthetic_data.jld2")
param_true = f_data["params"][[5,11,12,18]]
close(f_data)

# Rescale masses to Jupiter mass
sol2jup = 1047.5655146604772 # computed using UnitfulAstro
param_mean[[2,4]] .*= sol2jup
param_std[[2,4]] .*= sol2jup
param_mean_num[[2,4]] .*= sol2jup
param_std_num[[2,4]] .*= sol2jup
param_true[[2,4]] .*= sol2jup

annotations = [
    "P_b", "m_b",
    "P_c", "m_c"
]

xlabels = repeat(["Period [days]", "Mass ["*latexstring("M_{jup}")*"]"], 2)

# Save all to files
ps = Plots.Plot[]
for i in eachindex(param_names)
    leg = i == 3 ? true : false
    p_vals = (f[param_names[i]]["p_val"] ./ pfac[param_inds][i]); p_inds = sortperm(p_vals); p_vals = p_vals[p_inds]
    any(i .== [2,4]) && (p_vals .*= sol2jup)
    p_grid = minimum(p_vals):0.001*(maximum(p_vals) - minimum(p_vals)):maximum(p_vals)
    lls = f[param_names[i]]["lls"][p_inds]

    p_vals_num = f[param_names[i]]["p_val_num"] ./ pfac[param_inds][i]; p_inds_num = sortperm(p_vals_num); p_vals_num = p_vals_num[p_inds_num]
    any(i .== [2,4]) && (p_vals_num .*= sol2jup)
    p_grid_num = minimum(p_vals_num):0.001*(maximum(p_vals_num) - minimum(p_vals_num)):maximum(p_vals_num)
    lls_num = f[param_names[i]]["lls_num"][p_inds_num]

    p = scatter(
        p_vals, exp.((minimum(lls) .- lls) ./ 2), 
        annotate=((0.05, 0.9), Plots.text(latexstring(annotations[i]), :left)), 
        ms=6, msw=0, label="", color=1,
        yticks=any(i .== [1,2]) ? :auto : :none,
        #xticks=make_xticks(p_vals),
        xlabel=xlabels[i], legend=leg,
        ylims=(-0.001,1.05))
    p = scatter!(p_vals_num, exp.((minimum(lls) .- lls_num) ./ 2), ms=6, msw=0, label="", color=2, alpha=0.5)

    chi2 = exp.(-0.5 .* (p_grid .- param_mean[i]).^2 ./ param_std[i].^2)
    chi2_num = exp.(-0.5 .* (p_grid_num .- param_mean_num[i]).^2 ./ param_std_num[i].^2) .* maximum(exp.((minimum(lls) .- lls_num) ./ 2)) # .* sqrt(2π*param_std_num[i].^2) ./ sqrt(2π*param_std[i].^2)
    p = plot!(p_grid, chi2, label="Likelihood (analytic)",color=1,lw=3)
    p = plot!(p_grid_num, chi2_num, label="Likelihood (numerical)",color=2,lw=3, alpha=0.5)

    p = vline!([param_true[i]], label="True Value", color=5)
    p = vline!(param_mean[i] .+ param_std[i] .* [-1,1], label="1σ (analytic)", color=1, lw=2, linestyle=:dash)
    p = vline!(param_mean_num[i] .+ param_std_num[i] .* [-1,1], label="1σ (numerical)",color=2, lw=2, linestyle=:dash, alpha=0.5)
    push!(ps, p)
end

layout = @layout [
    [grid(2,1)] [grid(2,1)]
]
pf = plot(ps..., size=(1200,800), layout=layout, sharey=true, widen=false, formatter=:plain)
savefig("likelihood_profile.pdf")

# Now make a riverplot for the synthetic data
f_data = jldopen("../synthetic_data.jld2")

# Okay, let's make river plots for each planet.  First fit
# the ephemerides:
tt1 = f_data["transit_times"][f_data["transit_body"] .== 2]; pest1 = median(tt1[2:end] .- tt1[1:end-1])
β1, tlin1, ttv1 = compute_ttvs(tt1,pest1)
tt2 = f_data["transit_times"][f_data["transit_body"] .== 3]; pest2 = median(tt2[2:end] .- tt2[1:end-1])
β2, tlin2, ttv2 = compute_ttvs(tt2,pest2)

# Now make a riverplot:
nt1 = length(tt1)
nt2 = length(tt2)
width = 80
image1 = zeros(nt1,width*2+1)
tobs = f_data["times"]; fobs1 = f_data["flux"] .- f_data["flux_noiseless_z1"]
for i1=1:nt1
    tcurr = tlin1[i1]
    i1min = argmin(abs.(tcurr .- tobs))
    image1[i1,:] .= fobs1[i1min-width:i1min+width]
end

fig,axes = PythonPlot.subplots(1,2)

ax=axes[0]
ax.imshow(image1,aspect="auto",interpolation="nearest",extent = [-width*2/60,width*2/60,nt1-1,0])
ax.set_xlabel("Time [hr]")
ax.set_ylabel("Transit number")
image2 = zeros(nt2,width*2+1)

tobs = f_data["times"]; fobs2 = f_data["flux"] .- f_data["flux_noiseless_z2"]
nobs = length(tobs)
for i2=1:nt2
    tcurr = tlin2[i2]
    i2min = argmin(abs.(tcurr .- tobs))
    if ((i2min - width) > 0)  && ((i2min + width) <= nobs)
      image2[i2,:] .= fobs2[i2min-width:i2min+width]
    end
end
ax=axes[1]
ax.imshow(image2,aspect="auto",interpolation="nearest",extent = [-width*2/60,width*2/60,nt2-1,0])
ax.set_xlabel("Time [hr]")
ax.set_ylabel("Transit number")
fig.tight_layout()
fig.savefig("riverplot.pdf")