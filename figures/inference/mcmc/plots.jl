# Comparison of the sampling efficiency (CPU per
# effective sample) for affine-invariant versus
# NUTS sampler.

using PythonPlot

# Number of planets in each comparison
nplanet = [2,3,4,5]

nparam = nplanet .* 6

cpu_time_ai = [0.1433,0.4267,1.002,1.946] * 60.0 # in minutes for 100 samples from affine_results_T1_N.jld2
nsamples_ai = [2500,2500,5000,5000] # Number of samples in each Affine-invariant chain

# Use MCMCChains ess statistics:
ess_ai = [417.0,558.0,1248.6,1386.2]  # Computed from affine_results_T1_N.jld2
cpu_per_effsamp_ai = cpu_time_ai .* nsamples_ai ./ (100.0 * ess_ai)

# For NUTS, We will assume 1000 effective samples. Timing with BLAS.num_threads(1):
cpu_time_nuts = [7989.57,17083.35,32024.3,52655.85]/60.0 # in minutes
cpu_per_effsamp_nuts = cpu_time_nuts ./ 1000.0

fig, ax1 = PythonPlot.subplots()
ax2 = ax1.twinx()
ax1.plot(nparam,cpu_per_effsamp_ai,label="AInv")
ax1.plot(nparam,cpu_per_effsamp_ai,"o",color="C0")
ax1.plot(nparam,cpu_per_effsamp_nuts,label="NUTS")
ax1.semilogy(nparam,cpu_per_effsamp_nuts,"o",color="C1")
ax2.plot(nparam,cpu_per_effsamp_nuts ./ cpu_per_effsamp_ai,label="NUTS/AInv",color="C2")
ax2.plot(nparam,cpu_per_effsamp_nuts ./ cpu_per_effsamp_ai ,"o",color="C2")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
ax1.set_ylabel("CPU time per effective sample [min]")
ax2.set_ylabel("Ratio of CPU time per effective sample (NUTS/AInv)")
ax1.axis([10.0,32.0,0.1,10.0])
ax2.axis([10.0,32.0,0.2,0.28])
tick_locations=nparam
tick_labels=["2/12","3/18","4/24","5/30"]
ax2.set_xticks(tick_locations,tick_labels)
ax2.xaxis.tick_top()
ax1.set_xlabel("Number of planets/parameters")
fig.tight_layout()
fig.savefig("compare_ai_nuts.png",bbox_inches="tight")
