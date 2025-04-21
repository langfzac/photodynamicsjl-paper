using Photodynamics, DelimitedFiles, Plots, JLD2, BenchmarkTools

include("../common.jl")

function write_photodynam_input(mstar, rstar, k, u_n, els...; fname="", hfac=40)
    @assert fname != "" "Must supply a file name. (kwarg: fname)"
	
    n = length(els)
	output = Array{Any}(undef,9+n,clamp(n+1,6,20))
	output .= ""
	output[1,1] = n+1; output[1,2] = 1.0
	output[2,1] = els[1].P/hfac; output[2,2] = 1e-16
	output[4,1] = mstar*GNEWT; output[4,2:2+n-1] .= [el.m*GNEWT for el in els] # Get masses in proper units for photodynam
	output[5,1] = rstar; output[5,2:2+n-1] .= rstar.*k # Radius ratios
	output[6,1] = 1.0; output[6,2:2+n-1] .= 0.0
	output[7:8,1] .= u_n; output[7:8, 2:2+n-1] .= 0.0
	for (i,el) in enumerate(els)
		output[10+i-1,1:6] .= [el.a, el.e, -el.I, el.ω, 0.0, compute_mean_anomaly(output[1,2], el)]
	end
	writedlm(fname, output, ' ')
	return output
end

function run_photodynam(input, report, output, path)
    run(pipeline(`$(path)/photodynam $(input) $(report)`, stdout=output))
end

function get_photodynam_coords(posvel, masses, N)
    coords = zeros(N, 7)
    coords[:, 1] .= masses
    coords[:, 2:4] .= permutedims(reshape(posvel[1,3:2+N*3], 3, N))
    coords[:, 5:7] .= permutedims(reshape(posvel[1,3+N*3:end], 3, N))
    return coords
end

function run_photodynamics(output, ic, jc_coords, dt, tmax; hfac=40, der=false)
    N = ic.nbody
    t0_ic = ic.t0
    
    # Setup
    ic_coords = CartesianIC(t0_ic, N, copy(jc_coords))
    
    # Compute the dynamics
    intr = Integrator(ic.elements[2,2]/hfac, 0.0, tmax)
    s = State(ic_coords);
    tt = TransitTiming(intr.tmax, ic) # need elements ic right now, but this doesn't actually effect integration.
    ts = TransitSeries(intr.tmax, ic)
    if der
        intr(s, ts, tt, grad=true)
    else
        intr(s, ts, tt, grad=false)
    end

    tobs = out_data[:,1]
    fobs = zeros(length(tobs))
    eobs = zeros(length(tobs))
    if der
        lc = dLightcurve(dt, tobs, fobs, eobs, u_n, k, rstar)
    else
        lc = Lightcurve(dt, tobs, fobs, eobs, u_n, k, rstar)
    end
    compute_lightcurve!(lc, ts)
    
    # Save to file
    save(output, Dict("lc"=>lc,"ts"=>ts, "ic"=>ic_coords))
    return 
end

if abspath(PROGRAM_FILE) == @__FILE__
    path = pwd()
    input = path*"/photodynam/trappist1/trappist1_input.txt"
    report = path*"/photodynam/trappist1/trappist1_report.txt"
    output = path*"/photodynam/trappist1/trappist1_output.txt"

    N = 8

    # Stellar params
    mstar = 0.0898
    rstar = 0.00465047 * 0.1192 # Trappist-1 (Rstar/AU)
    k = get_radius_ratios_trappist(N)
    u_n = get_limbdark_coeffs_trappist()

    # Setup ICs
    BJD = 7250.0
    t0_ic = 1.0
    tmax = 1600.0
    ic = setup_ICs(N, BJD, t0_ic, fname=path*"/../elements.txt")
    ic.elements[2:end,3] .= 0.0
    # Convert to mass from ratios
    ic.m .*= mstar # Trappist-1 mass
    ic.elements[:,1] .*= mstar
    amatrix(ic) # Updates the initial condition hierarchy matrix
    elems = get_orbital_elements(State(ic), ic)
    write_photodynam_input(mstar, rstar, k, u_n, elems[2:end]...; fname=input)

    # Choose exposure times
    dt = 2 / 60 / 24; # 2 minutes in days
    obs_duration = tmax # days
    tobs = collect(t0_ic:dt:obs_duration)
    writedlm(report, ["t F x v"; ""; tobs])

    # Run photodynam and get initial cartesian coordinates
    @info "Running photodynam..."
    @btime run_photodynam(input, report, output, path*"/photodynam")
    @info "Done."

    # Get the initial conditions in Cartesian coordinates
    # Read in initial conditions
    out_data = Float64.(readdlm(output, comments=true, '\t')[:,1:end-1])
    jc_coords = get_photodynam_coords(out_data, ic.m, N)

    # Now, run Photodynamics.jl using initial cartesian coords
    @info "Running Photodynamics.jl"
    @btime run_photodynamics("compare_photodynam.jld2", ic, jc_coords, 0.0, tmax)
    @info "Done."

    @info "Running integrated Photodynamics.jl"
    @btime run_photodynamics("compare_photodynam_integrated.jld2", ic, jc_coords, dt, tmax)
    @info "Done"

    @info "Running Photodynamics.jl with Derivatives"
    @btime run_photodynamics("compare_photodynam_derivatives.jld2", ic, jc_coords, 0.0, tmax; der=true)
    @info "Done"

    @info "Running integrated Photodynamics.jl with Derivatives"
    @btime run_photodynamics("compare_photodynam_derivatives_integrated.jld2", ic, jc_coords, dt, tmax; der=true)
    @info "Done"
end
