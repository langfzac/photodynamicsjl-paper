using Photodynamics

# Structure to hold model arrays and what not
struct LightcurveModel{T<:Real}
    N::Int
    t0::T
    tmax::T
    h::T
    lc::Lightcurve{T}
    dlc::dLightcurve{T}
end

### NEED TO FIX IN NBG
# The input vector will track the derivatives, but we need to input into
# initial conditions with masses at the beginning. 
function handle_elements_matrix(θ, N)
    @assert (length(θ) % (N-1)) == 1 "Lengths need to agree (N and size of elements matrix)."
    # Assume ordered by planet, then element. Stellar mass at the start
    elements_matrix = zeros(N, 7)

    elements_masses = permutedims(reshape(θ[2:end], 7, N-1))
    elements_matrix[2:end, 1] .= elements_masses[:, end] # Masses
    elements_matrix[2:end, 2:end] .= elements_masses[:, 1:end-1]
    elements_matrix[1] = θ[1]

    return elements_matrix
end

# For now assume we're doing every parameter. We then wrap in a function that fixes the ones we want.
# θ = {rs,u1,u2,ms,P1,t01,ec1,es1,I1,Ω1,m1...N,k1,...kN}
function compute_flux(lcm::LightcurveModel, θ; tol=1e-6, maxdepth=6)
    lc = lcm.lc
    N = lcm.N
    t0 = lcm.t0
    tmax = lcm.tmax
    h = lcm.h
    
    # Replace parameters in lightcurve
    # Start with the lightcurve parameters
    lc.rstar .= θ[1]
    lc.u_n .= θ[2:3]
    lc.k .= θ[end-(N-2):end] # -2 for only planet count

    # Now get new initial conditions
    # Get the elements matrix
    elements = handle_elements_matrix(θ[4:4+7*(N-1)], N)
    ic = ElementsIC(t0, N, elements)

    # Setup integrator and run N-body integrator
    intr = Integrator(h, tmax)
    s = State(ic)
    ts = TransitSeries(tmax, ic)
    tt = TransitTiming(tmax, ic)
    intr(s, ts, tt; grad=false)

    # Compute the photometry
    compute_lightcurve!(lc, ts; tol=tol, maxdepth=maxdepth)
    
    return copy(lc.flux)
end

function jac_compute_flux(lcm::LightcurveModel, θ; tol=1e-6, maxdepth=6)
    lc = lcm.lc
    dlc = lcm.dlc
    N = lcm.N
    t0 = lcm.t0
    tmax = lcm.tmax
    h = lcm.h
    
    # Replace parameters in both lightcurves
    lc.rstar .= θ[1]
    dlc.rstar .= θ[1]
    lc.u_n .= θ[2:3]
    dlc.u_n .= θ[2:3]
    lc.k .= θ[end-(N-2):end] # -2 for only planet count
    dlc.k .= θ[end-(N-2):end]

    # Now get new initial conditions
    # Get the elements matrix
    elements = handle_elements_matrix(θ[4:4+7*(N-1)], N)
    ic = ElementsIC(t0, N, elements)

    # Setup integrator and run N-body integrator
    intr = Integrator(h, tmax)
    s = State(ic)
    ts = TransitSeries(tmax, ic)
    tt = TransitTiming(tmax, ic)
    intr(s, ts, tt; grad=true)

    # Compute the photometry and derivatives
    compute_lightcurve!(dlc, ts; tol=tol, maxdepth=maxdepth)
    lc.flux .= dlc.flux

    # Transform derivatives from wrt Cartesian back to wrt orbital elements
    transform_to_elements!(s, dlc)
    
    # Collect Jacobian arrays
    # θ = {rs,u1,u2,ms,P1,t01,ec1,es1,I1,Ω1,m1...N,k1,...kN}
    jac_flux = hcat(dlc.dfdr, dlc.dfdu, dlc.dfdelements[:, 7:end], dlc.dfdk)

    return copy(dlc.flux), jac_flux
end

function compute_ttvs(transits,p0)
    # First, estimate the epoch:
    epoch = round.(Int64,(transits .- transits[1])./p0)
    ntransit = length(transits)
    # Now, carry out linear regression:
    X = Array{Float64, 2}(hcat(ones(ntransit), epoch))
    β = X \ transits
    tlin = β[1] .+ β[2] .* epoch
    ttv = transits .- tlin
    return β, tlin, ttv
end