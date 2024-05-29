
"""
    SliceSteppingOut(window; max_stepping_out, max_proposals)

Univariate slice sampling by automatically adapting the initial interval through the "stepping-out" procedure (Scheme 3 by Neal[^N2003])

# Arguments
- `window::Real`: Proposal window.

# Keyword Arguments
- `max_stepping_out::Int`: Maximum number of "stepping outs" (default: 32).
- `max_proposals::Int`: Maximum number of proposals allowed until throwing an error (default: `typemax(Int)`).
"""
struct SliceSteppingOut{W <: Real} <: AbstractUnivariateSliceSampling
    window          ::W
    max_stepping_out::Int
    max_proposals   ::Int
end

function SliceSteppingOut(
    window          ::Real;
    max_stepping_out::Int = 32,
    max_proposals   ::Int = typemax(Int),
)
    @assert window > 0
    SliceSteppingOut(window, max_stepping_out, max_proposals)
end

function find_interval(
    rng  ::Random.AbstractRNG,
    alg  ::SliceSteppingOut,
    model,
    w    ::Real,
    ℓy   ::Real,
    θ₀   ::F,
) where {F <: Real}
    m      = alg.max_stepping_out
    u      = rand(rng, F)
    L      = θ₀ - w*u
    R      = L + w
    V      = rand(rng, F)
    J      = floor(Int, m*V)
    K      = (m - 1) - J 
    n_eval = 0

    while J > 0 && ℓy < LogDensityProblems.logdensity(model, L)
        L = L - w
        J = J - 1
        n_eval += 1
    end
    while K > 0 && ℓy < LogDensityProblems.logdensity(model, R)
        R = R + w
        K = K - 1
        n_eval += 1
    end
    L, R, n_eval
end
