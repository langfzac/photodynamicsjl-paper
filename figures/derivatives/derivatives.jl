using Photodynamics, JLD2, Plots, Printf, DelimitedFiles, LaTeXStrings, Statistics
include("../common.jl")
pythonplot()

function make_yticks(y; dmin=0.8, dmax=0.8)
    ymin = minimum(y)
    ymax = maximum(y)
    ymid = middle(y)

    if ymax < abs(ymid)
        y1 = dmin*ymin
        y3 = abs(ymid)/2
        y2 = (y3 - abs(y1)) / 2
        return [y1,y2,y3]
    end
    return [dmin*ymin, 0.0, dmax*ymax]
end

if abspath(PROGRAM_FILE) == @__FILE__
    path = pwd()

    # Setup simulation
    N = 8

    # Stellar params
    mstar = 0.0898 # Solar masses
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

    # Choose exposure times
    dt = 30 / 60 / 60 / 24; # 30 seconds in days
    obs_duration = tmax # days

    # Run simulation
    intr = Integrator(ic.elements[2,2]/40, 0.0, tmax)
    s = State(ic)
    tt = TransitTiming(tmax, ic)
    ts = TransitSeries(tmax, ic)
    intr(s, ts, tt; grad=true)
    lc = dLightcurve(dt, obs_duration, u_n, k, rstar)
    compute_lightcurve!(lc, ts)
    transform_to_elements!(s, lc);

    ### Pick out triple transit in light curve to plot ###
    xbounds = (176.75, 176.865)
    mask = xbounds[1] .< lc.tobs .< xbounds[2]

    # All relevant derivatives for plotting
    ders1 = (
        m_a=lc.dfdelements[mask,7],
        r_a=lc.dfdr[mask],
        u_1=lc.dfdu[mask,1],
        u_2=lc.dfdu[mask,2],
        k_b=lc.dfdk[mask,1],
        k_c=lc.dfdk[mask,2],
        k_e=lc.dfdk[mask,4],
        m_b=lc.dfdelements[mask,7*2],
        m_c=lc.dfdelements[mask,7*3],
        m_d=lc.dfdelements[mask,7*4],
        m_e=lc.dfdelements[mask,7*5],
        m_f=lc.dfdelements[mask,7*6],
        m_g=lc.dfdelements[mask,7*7],
        m_h=lc.dfdelements[mask,7*8],
        P_b=lc.dfdelements[mask,1+7],
        P_c=lc.dfdelements[mask,1+7*2],
        P_d=lc.dfdelements[mask,1+7*3],
        P_e=lc.dfdelements[mask,1+7*4],
        P_f=lc.dfdelements[mask,1+7*5],
        P_g=lc.dfdelements[mask,1+7*6],
        P_h=lc.dfdelements[mask,1+7*7],
        t0_b=lc.dfdelements[mask,2+7],
        t0_c=lc.dfdelements[mask,2+7*2],
        t0_d=lc.dfdelements[mask,2+7*3],
        t0_e=lc.dfdelements[mask,2+7*4],
        t0_f=lc.dfdelements[mask,2+7*5],
        t0_g=lc.dfdelements[mask,2+7*6],
        t0_h=lc.dfdelements[mask,2+7*7],
    )
    ders2 = (
        ecosω_b=lc.dfdelements[mask,3+7],
        ecosω_c=lc.dfdelements[mask,3+7*2],
        ecosω_d=lc.dfdelements[mask,3+7*3],
        ecosω_e=lc.dfdelements[mask,3+7*4],
        ecosω_f=lc.dfdelements[mask,3+7*5],
        ecosω_g=lc.dfdelements[mask,3+7*6],
        ecosω_h=lc.dfdelements[mask,3+7*7],
        esinω_b=lc.dfdelements[mask,4+7],
        esinω_c=lc.dfdelements[mask,4+7*2],
        esinω_d=lc.dfdelements[mask,4+7*3],
        esinω_e=lc.dfdelements[mask,4+7*4],
        esinω_f=lc.dfdelements[mask,4+7*5],
        esinω_g=lc.dfdelements[mask,4+7*6],
        esinω_h=lc.dfdelements[mask,4+7*7],
        I_b=lc.dfdelements[mask,5+7],
        I_c=lc.dfdelements[mask,5+7*2],
        I_d=lc.dfdelements[mask,5+7*3],
        I_e=lc.dfdelements[mask,5+7*4],
        I_f=lc.dfdelements[mask,5+7*5],
        I_g=lc.dfdelements[mask,5+7*6],
        I_h=lc.dfdelements[mask,5+7*7],
        Ω_b=lc.dfdelements[mask,6+7],
        Ω_c=lc.dfdelements[mask,6+7*2],
        Ω_d=lc.dfdelements[mask,6+7*3],
        Ω_e=lc.dfdelements[mask,6+7*4],
        Ω_f=lc.dfdelements[mask,6+7*5],
        Ω_g=lc.dfdelements[mask,6+7*6],
        Ω_h=lc.dfdelements[mask,6+7*7],
    );

    # Reset things for the following plot
    default(
        legendfontsize=8,
        tickfontsize=12,
        xlabelfontsize=12,
        ylabelfontsize=14,
        grid=:off, box=:on,
        palette=:seaborn_colorblind
    )

    # Convert tobs to minutes
    # Makes plots look nicer
    dth = dt * 24 * 60 # Cadence in minutes
    bound = length(lc.tobs[mask]) * dth / 2
    tobs = collect(-bound:dth:bound)

    # Make each parameter type a different color
    # ie. masses one color, periods another...
    c1 = vcat([1,8,9,9,10,10,10], [repeat([i], 7) for i in 1:3]...)
    c2 = vcat([repeat([i], 7) for i in 4:7]...)

    # Set locations of the annotations
    locs1 = [
        (0.05, 0.9); (0.05, 0.9);
        (0.85, 0.875); (0.85, 0.875);
        (0.05, 0.9); (0.05, 0.9); (0.05,0.9);
        repeat([(0.05, 0.9)], 7); # Masses
        repeat([(0.85, 0.85)], 7); # Period
        repeat([(0.825, 0.85)], 7); # t0
    ]
    locs2 = [
        repeat([(0.045, 0.875)], 7); # ecosω
        repeat([(0.05, 0.875)], 7); # esinω
        repeat([(0.85, 0.875)], 7); # I
        repeat([(0.05, 0.875)], 7); # Ω
    ]

    # Save the total lightcurve
    flux_tot = lc.flux[mask] .+ 1.0
    tobs_min = lc.tobs[mask] .* 24 .* 60
    tmid = middle(tobs_min)

    # Get the times of transits in minutes
    tts = ts.times[302:304] .* 24 .* 60 .- tmid

    function make_plots(tobs, ders, colors, locs, fname; tts=tts)
        # Hardcode xticks for nice looking plot
        xticks = [-60,-30,0,30,60]
    
        layout = @layout [
            [grid(7,1)] [grid(7,1)] [grid(7,1)] [grid(7,1)]
        ]

        ps = Plots.Plot[]
        for (i, key) in enumerate(keys(ders))
            p = plot(
                tobs, ders[key], label=nothing,
                color=colors[i], linewidth=2.5,
                top_margin=(0,:mm), bottom_margin=(0,:mm),
                yformatter=x->x==0.0 ? "0.0" : @sprintf("%1.e", x),
                xticks=(xticks, [(all(i.!=7*[1,2,3,4,5,6,7,8]) ? "" : xt) for xt in xticks]),
                yticks=make_yticks(ders[key]),
                ylims=(1.4 .* make_yticks(ders[key]))[[1,3]],
                xlabel=all(i.!=7*[1,2,3,4]) ? "" : "Time [minutes]",
                annotate=(locs[i],text(latexstring(key),:left)),
            )
            push!(ps, p)
        end
        pf = plot(ps..., size=(1200,1200), layout=layout, sharex=true, widen=false)
        savefig(pf, fname)
        return
    end

    make_plots(tobs, ders1, c1, locs1, "derivatives_1.pdf")
    make_plots(tobs, ders2, c2, locs2, "derivatives_2.pdf")

    # Now just the flux
    # Reset things for the following plot
    default(
        legendfontsize=10,
        tickfontsize=12,
        xlabelfontsize=12,
        ylabelfontsize=12,
        grid=:off, box=:on,
        palette=:seaborn_colorblind
    )
    # Get each individual transit of b,c,e
    compute_lightcurve!(lc, ts; body_index=2) 
    mask_b = lc.flux[mask] .!= 0.0
    flux_b = lc.flux[mask][mask_b] .+ 1.0
    tobs_b = tobs_min[mask_b]
    compute_lightcurve!(lc, ts; body_index=3)
    mask_c = lc.flux[mask] .!= 0.0
    flux_c = lc.flux[mask][mask_c] .+ 1.0
    tobs_c = tobs_min[mask_c]
    compute_lightcurve!(lc, ts; body_index=5)
    mask_e = lc.flux[mask] .!= 0.0
    flux_e = lc.flux[mask][mask_e] .+ 1.0
    tobs_e = tobs_min[mask_e]

    plot(tobs_min.-tmid, flux_tot, lw=2, c=1,
        label=nothing, xlabel="Time [minutes]", ylabel="Relative Flux", legend=:bottomright
    )
    plot!(tobs_b.-tmid, flux_b, ls=:dash, lab="b", c=2, lw=2)
    plot!(tobs_c.-tmid, flux_c, ls=:dash, lab="c", c=3, lw=2)
    plot!(tobs_e.-tmid, flux_e, ls=:dash, lab="e", c=5, lw=2)
    savefig("transit_triple.pdf")
end