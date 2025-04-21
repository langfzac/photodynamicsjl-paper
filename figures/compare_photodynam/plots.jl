using Photodynamics, JLD2, DelimitedFiles, Plots
pythonplot()

# Set some defaults
default(
    legendfontsize=8,
    tickfontsize=10,
    xlabelfontsize=12,
    ylabelfontsize=12,
    grid=:off, box=:on,
    palette=:seaborn_colorblind
)

# Load data from Photodynamics
pd_data = load("compare_photodynam.jld2")
lc = pd_data["lc"];
ts = pd_data["ts"];

# Load data from photodynam
jc_data = Float64.(readdlm("photodynam/trappist1/trappist1_output.txt",comments=true, '\t')[:,1:end-1]);

function plot_comparison_good(midpoint, bounds)
    xbounds = (midpoint-bounds, midpoint+bounds)
    residuals = ((lc.flux .+ 1.0) .- jc_data[:, 2]) ./ jc_data[:, 2]
    
    l = @layout [
        a{0.6h};
        b{0.4h}
    ]
    
    p1 = scatter(
        lc.tobs, lc.flux .+ 1, 
        markersize=6, label="Photodynamics.jl")
    p1 = scatter!(
        lc.tobs, jc_data[:,2], 
        markersize=3, label="photodynam",
        xlim=xbounds,
        ylim=(0.985, 1.0021),
        legend=:bottomleft,
        xticks=nothing,
        ylabel="Relative Flux",
        yticks=[0.99, 0.995, 1.0]
    )
    
    p2 = scatter(
        lc.tobs, residuals, 
        xlim=xbounds,
        ylim=(-1.5e-7, 1.5e-7),
        #ylim=(-1e-7,1e-7),
        legend=false,
        ylabel="Flux Difference",
        xlabel="Time [Days]",
        yticks=([-1e-7, 0, 1e-7], ["-10\$^{-7}\$", "0", "10\$^{-7}\$"]),
        markersize=5
    )
    return plot(
        p1, p2, 
        layout=l, msw=0
    )
end

# Generate zoom-in transit plots
midpoint = 1594.855
bounds = 0.05
plot_comparison_good(midpoint, bounds)
savefig("transit_residuals_double.pdf")

midpoint = 1557.265
bounds = 0.12
plot_comparison_good(midpoint, bounds)
savefig("transit_residuals_single.pdf")

# Now generate the residuals plot
flux_diff = jc_data[:,2] .- (lc.flux .+ 1)
mask = (flux_diff .!= 0.0) .& (abs.(flux_diff) .< 1e-6)
plot(lc.tobs[mask], abs.(flux_diff[mask]),
     tickfontsize=14, xlabelfontsize=18, ylabelfontsize=18,
     yscale=:log10, yticks=[1e-16,1e-14,1e-12,1e-10,1e-8,1e-6],
     color=2, leg=false, ylims=(1e-17, 1e-5),
     ylabel="Abs. Flux Difference",
     xlabel="Time [Days]")
hline!([1e-6], linestyle=:dash, color="black")
savefig("residuals.pdf")