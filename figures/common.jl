# Common functions for generating paper figures
using Photodynamics, DelimitedFiles
import Photodynamics.NbodyGradient: set_state!, amatrix, get_orbital_elements, GNEWT

function setup_ICs(N, BJD, t0; fname="elements.txt")
    elements = readdlm(fname, ',')[1:N,:]
    elements[2:end,3] .-= BJD # Shift initial transit times
    ic = ElementsIC(t0, N, elements)
    return ic
end

function compute_mean_anomaly(t, el)
    n = 2π/el.P
    sqrt1me2 = sqrt(1.0 - el.e^2)
    den1 = el.esinω - el.ecosω - el.e
    tp = (el.t0 - sqrt1me2/n*el.ecosω/(1.0-el.esinω)-2/n*atan(sqrt(1-el.e)*(el.esinω+el.ecosω+el.e), sqrt(1+el.e)*den1))
    M = n * (t - tp)
    return M
end

function get_radius_ratios_trappist(n)
    depth = [0.7277,0.6940,0.3566,0.4802,0.634,0.764,0.346] .* 0.01
    return sqrt.(depth)[1:n-1]
end

function get_limbdark_coeffs_trappist()
    q = [0.11235270319764341, 0.42037661035916857]#, 0.352424321959808, 0.2864053200404355]
    return [2*sqrt(q[1])*q[2],2*sqrt(q[1])*(1-2q[2])]
end