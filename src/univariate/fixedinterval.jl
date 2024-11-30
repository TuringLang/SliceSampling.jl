
"""
    Slice(window; max_proposals)

Univariate slice sampling with a fixed initial interval (Scheme 2 by Neal[^N2003])

# Arguments
- `window::Real`: Proposal window.

# Keyword Arguments
- `max_proposals::Int`: Maximum number of proposals allowed until throwing an error (default: `$(DEFAULT_MAX_PROPOSALS)`).
"""
struct Slice{W<:Real} <: AbstractUnivariateSliceSampling
    window        :: W
    max_proposals :: Int
end

function Slice(window::Real; max_proposals::Int=DEFAULT_MAX_PROPOSALS)
    @assert window > 0
    return Slice(window, max_proposals)
end

function find_interval(
    rng::Random.AbstractRNG, ::Slice, ::Any, w::Real, ::Real, θ₀::F
) where {F<:Real}
    u = rand(rng, F)
    L = θ₀ - w * u
    R = L + w
    return L, R, 0
end
