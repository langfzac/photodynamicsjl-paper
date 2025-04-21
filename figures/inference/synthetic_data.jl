## Generate a realistic synthetic dataset
using Photodynamics
using Unitful, UnitfulAstro
using Distributions, Random
using JLD2
using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "file_name"
            help = "Name of output file"
            arg_type = String
            required = true
        "seed"
            help = "Choose a random seed"
            arg_type = Int
            default = 1234
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
    # Make a 3 body system in a 4:3 period resonance.
    # Want the signal to noise to be high enough to be
    # able to model the chopping signal.

    #### First, choose/draw parameters
    # Fixed parameters
    mstar = 0.5 * u"Msun" # Stellar mass 
    rstar = 0.5 * u"Rsun" # Stellar radius
    I = [π/2, π/2] # Inclinations (Coplanar)
    Ω = [0.0, 0.0] # Longitude of ascending node

    # Draw each of these from the given distributions (make sure to set random seed!)
    u_n = rand(Uniform(0.1,0.3), 2) # Quadratic limbdarkening coefficients
    m1,m2 = 3 .* rand(Uniform(1e-6,1e-5),2) * u"Msun"# Mass of planet 1, 2 (two super-ish Earths)
    ecosω1,esinω1,ecosω2,esinω2 = rand(MvNormal(zeros(2), 0.01), 2) # eccentricity vector components
    ks = rand(Uniform(0.08, 0.1), 2) # radius ratios (~1-10 earth radii)

    # Get a random period ratio about the 4:3 period resonance
    # Choose P1 and derive P2
    P21 = 4/3 + 2e-2
    P1 = rand(Uniform(2, 10.0)) * u"d" # Choose bounds so that the ingration time isn't too long
    P2 = P21*P1 

    # Get a first transit time for each
    t01 = rand(Uniform(-P1/2 |> u"d" |> ustrip, P1/2 |> u"d" |> ustrip)) * u"d"
    t02 = rand(Uniform(-P2/2 |> u"d" |> ustrip, P2/2 |> u"d" |> ustrip)) * u"d"

    #### Now, setup photometry parameters
    # Compute time to observe 2 TTV periods
    Pttv = 1 / abs(4/P2 - 3/P1)
    tmax = 2 * Pttv

    # Get time stamps
    t0 = 0.0 * u"d"
    dt = 2.0 * u"minute"  # 2 minute cadence
    tobs = [t0:dt:tmax;] .|> u"d" .|> ustrip
    nobs = length(tobs)

    ### Compute the lightcurve with Photodynamics.jl
    a = Elements(m=1.0)
    b = Elements(
        m = m1/mstar |> NoUnits,
        P = P1 |> u"d" |> ustrip,
        t0 = t01 |> u"d" |> ustrip,
        ecosω = ecosω1,
        esinω = esinω1,
        I = π/2 
    )
    c = Elements(
        m = m2/mstar |> NoUnits,
        P = P2 |> u"d" |> ustrip,
        t0 = t02 |> u"d" |> ustrip,
        ecosω = ecosω2,
        esinω = esinω2,
        I = π/2 
    )
    ic = ElementsIC(t0 |> u"d" |> ustrip, 3, a,b,c)

    @info "Computing Photodynamics.jl model."
    s = State(ic)
    intr = Integrator(P1/40 |> u"d" |> ustrip, tmax |> u"d" |> ustrip)
    tt = TransitTiming(tmax |> u"d" |> ustrip, ic)
    ts = TransitSeries(tmax |> u"d" |> ustrip, ic)
    intr(s, ts, tt)

    lc = Lightcurve(dt |> u"d" |> ustrip, tmax |> u"d" |> ustrip, u_n, ks, rstar |> u"AU" |> ustrip)
    compute_lightcurve!(lc, ts)
    lc_z1 = Lightcurve(dt |> u"d" |> ustrip, tmax |> u"d" |> ustrip, u_n, ks, rstar |> u"AU" |> ustrip)
    compute_lightcurve!(lc_z1, ts; body_index=3)
    lc_z2 = Lightcurve(dt |> u"d" |> ustrip, tmax |> u"d" |> ustrip, u_n, ks, rstar |> u"AU" |> ustrip)
    compute_lightcurve!(lc_z2, ts; body_index=2)

    #### Finally, compute the errors for the photometry
    # Draw a large enough signal to noise
    snr = rand(Uniform(10,20))

    # Compute the standard deviation of the chopping signal for each planet
    μ1, μ2 = m1/mstar, m2/mstar # Mass ratios
    σ_norm = 4.5e-4 / (4/3 - P2/P1)^2 # Analytic approx (from TTVFaster sims (cite/show?))
    σ_chop1 = σ_norm * P1 * μ2 
    σ_chop2 = σ_norm * P2 * μ1

    # Now the timing uncertainty
    # Assuming circular orbits should be good enough here..
    Γ = 1/dt
    R1, R2 = ks * rstar
    ρ1, ρ2 = ks[1], ks[2] # radius ratios
    a1, a2 = @. cbrt(u"G" * mstar * [P1,P2]^2 / 4 / π^2) # orbital radius
    v1, v2 = @. sqrt(u"G" * mstar / [a1,a2]) # Sky velocity mid transit
    te1,te2 =  @. 2*[R1,R2] / [v1,v2] # ingress/egress durations (zero impact parameter)

    # Desired timing uncertainties from snr, σ_chop, and nobs
    σ_t1, σ_t2 = @. sqrt(tt.count[2:end] * [σ_chop1, σ_chop2]^2 / snr^2)

    # Now invert equation from Ford & Gaudi 2006 to get photometric errors
    σ_ph1, σ_ph2 = @. [ρ1,ρ2]^2 * [σ_t1,σ_t2]*sqrt(2*Γ/[te1,te2]) .|> NoUnits # resolve units to get just floats (will complain if not actually unitless)

    # Use minimum so each chopping signal has high snr.
    eobs = min(σ_ph1, σ_ph2)

    # Set the lightcurve errors
    lc.eobs .= eobs

    # Now add noise of the same magnitude to the data
    lc.fobs .= lc.flux .+ rand(MvNormal(zeros(length(lc.flux)), eobs))

    # Now thin the data so that we minimize model evaluations not in transit
    # Setup observation times to avoid bulk of the out-of-transit data
    tmasks = []
    for t in ts.times
        m = (t - c.P/30) .< lc.tobs .< (t + c.P/30)
        push!(tmasks, m)
    end
    tmask = sum(tmasks) .> 0
    tobs_thin = lc.tobs[tmask]
    fobs_thin = lc.fobs[tmask]
    eobs_thin = lc.eobs[tmask]
    flux_thin = lc.flux[tmask]
    flux_z1 = lc_z1.flux
    flux_z2 = lc_z2.flux

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
        file["flux_noiseless_z1"] = flux_z1
        file["flux_noiseless_z2"] = flux_z2
        file["times_thin"] = tobs_thin
        file["errors_thin"] = eobs_thin
        file["flux_noiseless_z1"] = flux_z1
        file["flux_noiseless_z2"] = flux_z2
        file["params"] = [
            rstar |> u"AU" |> ustrip,
            u_n[1],u_n[2],1.0,
            b.P,b.t0,b.ecosω,b.esinω,b.I,b.Ω,b.m,
            c.P,c.t0,c.ecosω,c.esinω,c.I,c.Ω,c.m,
            ks[1],ks[2]
        ]
        file["transit_times"] = ts.times
        file["transit_body"] = ts.bodies
    end

    if args["debug"]
        @info "Output debugging files."
        # Save some processed data, ttvs, etc.
        # Get linear transit times and transit number for each planet
        tn1, lintt1 = get_linear_transit_times(ts, 2) 
        tn2, lintt2 = get_linear_transit_times(ts, 3) 

        # Compute ttvs for each planet
        ttvs1 = get_transit_times(ts, 2) .- lintt1
        ttvs2 = get_transit_times(ts, 3) .- lintt2

        transit_data_1 = DataFrame(number=tn1, time=lintt1, Δtime=ttvs1)
        transit_data_2 = DataFrame(number=tn2, time=lintt2, Δtime=ttvs2)
        CSV.write("debug_transit_timing_1.csv", transit_data_1)
        CSV.write("debug_transit_timing_2.csv", transit_data_2)
    end
    @info "Done."
end

main()
