## Generate a realistic synthetic dataset
using Photodynamics
using Unitful, UnitfulAstro
using Distributions, Random
using JLD2
using ArgParse
using DelimitedFiles

function setup_ICs(body_inds, BJD::T, t0::T; fname="elements.txt") where T<:Real
    elements = T.(readdlm(fname, ',')[body_inds, :])
    elements[2:end, 3] .-= BJD # Shift initial transit times
    ic = ElementsIC(t0, length(body_inds), elements)
    return ic
end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "file_name"
            help = "Name of output file"
            arg_type = String
            required = true
        "planet_inds"
            help = "Which planets to include, e.g. [2,3,...]"
            default = "[2,3,4,5,6,7,8]"
        "seed"
            help = "Choose a random seed"
            arg_type = Int
            default = 1234
        "texp"
            help = "Exposure time in minutes"
            arg_type = Float64
            default = 2.0
        "tmax"
            help = "Maximum duration in days"
            arg_type = Float64
            default = 1600.0
        "eobs"
            help = "Noise level (ppm)"
            arg_type = Float64
            default = 55.0
        "fname_elements"
            help = "Path and filename for the elements file"
            arg_type = String
            default = "../elements.txt"
        "--debug"
            help = "Output some things for debugging"
            action = :store_true
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()
    
    Random.seed!(args["seed"])
    @info "Generating synthetic lightcurve..."
    # Use TRAPPIST-1 parameters to simulate with properties
    # like JWST.  Vary the duration, exposure time, noise level
    # and number of planets.  Use the best-fit
    # parameters from Agol et al. (2021; A21).  See how the inference
    # time (number of effective saamples) scales with the number of 
    # parameters/planets (holding the stellar mass/radius/density, 
    # limb-darkening, inclinations and longitudes of ascending node fixed
    # for the time being).

    #### First, specify parameters (mostly best-fit values from A21):
    # Fixed parameters
    mstar = 0.0898 * u"Msun" # Stellar mass from Mann et al. (2019) mass-M_K relation
    rstar = 0.1192/ustrip(mstar)^(1//3) * u"Rsun" # Stellar radius from A21, converted to radius if star were 1.0 M_sun
    I = [89.728, 89.778, 89.896, 89.793, 89.740, 89.742, 89.805] .*(pi/180) # Inclinations from A21
    Ω = [ 0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000] # Longitude of ascending node

    # Draw each of these from the given distributions (make sure to set random seed!)
    u_n = [0.218,0.021] # Quadratic limbdarkening coefficients from A21
    
    # Radius ratios from A21: 
    ks = [0.08590,0.08440,0.06063,0.07079,0.08040,0.08692,0.05809] # radius ratios from A21
 
    tmax = args["tmax"] * u"d"
    planet_inds = eval(Meta.parse(args["planet_inds"]))
    body_inds = [1; planet_inds]
    nplanet = length(body_inds) - 1

    # Get time stamps
    t0 = 0.0 * u"d"
    dt = args["texp"] * u"minute"  # 2 minute cadence
    tobs = [t0:dt:tmax;] .|> u"d" .|> ustrip
    nobs = length(tobs)

    ### Compute the lightcurve with Photodynamics.jl
    # For the dynamical parameters we will use the best-fit values from A21:
    BJD_shift = 0.0
    ic = setup_ICs(body_inds,BJD_shift,ustrip(t0),fname=args["fname_elements"])
    # Copy over the inclinations and longitudes of ascending node:
    ic.elements[2:nplanet+1,6] .= I[planet_inds .- 1]
    ic.elements[2:nplanet+1,7] .= Ω[planet_inds .- 1]
    @info ic
    flush(stderr); flush(stdout)

    @info "Computing Photodynamics.jl model."
    s = State(ic)
    intr = Integrator(ic.elements[2,2]/40, tmax |> u"d" |> ustrip)
    tt = TransitTiming(tmax |> u"d" |> ustrip, ic)
    ts = TransitSeries(tmax |> u"d" |> ustrip, ic)
    intr(s, ts, tt)

    lc = Lightcurve(dt |> u"d" |> ustrip, tmax |> u"d" |> ustrip, u_n, ks[planet_inds .- 1], rstar |> u"AU" |> ustrip)
    lc_i = Lightcurve(dt |> u"d" |> ustrip, tmax |> u"d" |> ustrip, u_n, ks[planet_inds .- 1], rstar |> u"AU" |> ustrip) # Copy for the noiseless lightcurves
    compute_lightcurve!(lc, ts)

    #### Finally, compute the errors for the photometry

    # Use specified error level (in ppm):
    eobs = args["eobs"] .* 1e-6

    # Set the lightcurve errors
    lc.eobs .= eobs

    # Now add noise of the same magnitude to the data
    lc.fobs .= lc.flux .+ rand(MvNormal(zeros(length(lc.flux)), eobs))

    # Now thin the data so that we minimize model evaluations not in transit
    # Setup observation times to avoid bulk of the out-of-transit data
    tmasks = []
    for t in ts.times
        # Make a window equal to 1/30th of orbital period of the inner planet:
        m = (t - ic.elements[2,2]/30) .< lc.tobs .< (t + ic.elements[2,2]/30)
        push!(tmasks, m)
    end
    tmask = sum(tmasks) .> 0
    tobs_thin = lc.tobs[tmask]
    fobs_thin = lc.fobs[tmask]
    eobs_thin = lc.eobs[tmask]
    flux_thin = lc.flux[tmask]

    #### Save data to CSV
    # Save the fake lightcurve data, and true parameters for comparison
    @info "Saving to $(args["file_name"])."
    jldopen(args["file_name"], "w") do file
        file["flux"] = lc.fobs
        file["flux_noiseless"] = lc.flux
        file["times"] = lc.tobs
        file["errors"] = lc.eobs
        file["flux_thin"] = fobs_thin
        file["flux_noiseless_thin"] = flux_thin
        file["times_thin"] = tobs_thin
        file["errors_thin"] = eobs_thin

        # Compute lightcurves for only a single body
        for i in 1:nplanet
            Photodynamics.zero_out!(lc_i)
            compute_lightcurve!(lc_i, ts; body_index=i+1)
            file["flux_noiseless_z$(i)"] = lc_i.flux
        end 

        # Create a vector of parameters, starting with the stellar parameters
        # (with the mass of the star equal to 1.0):
        params_vec = [rstar |> u"AU" |> ustrip, u_n[1],u_n[2],1.0]
        for iplanet=1:nplanet
          #  Parameters are: b.P,b.t0,b.ecosω,b.esinω,b.I,b.Ω,b.m for planet 'b':
          append!(params_vec,ic.elements[iplanet+1,2:7])
          append!(params_vec,ic.elements[iplanet+1,1])
        end
        # Append the radius-ratios:
        append!(params_vec,ks[planet_inds .- 1])
        # Pass these parametrs to the jld2 file:
        file["params"] = params_vec
        file["transit_times"] = ts.times
        file["transit_body"] = ts.bodies
    end

    if args["debug"]
        @info "Output debugging files."
        # Save some processed data, ttvs, etc.
        # Get linear transit times and transit number for each planet
        for j=1:nplanet
          tn, lintt = get_linear_transit_times(ts, j+1) 

          # Compute ttvs for each planet
          ttvs = get_transit_times(ts, j+1) .- lintt

          transit_data = DataFrame(number=tn, time=lintt, Δtime=ttvs)
          CSV.write(string("debug_transit_timing_",nplanet,"_",j,".csv"), transit_data)
        end
    end
    @info "Done."
end

main()