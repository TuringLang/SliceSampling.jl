
"""
    Slice(window; max_proposals)

Univariate slice sampling with a fixed initial interval (Scheme 2 by Neal[^N2003])

# Arguments
- `window::Union{<:Real, <:AbstractVector}`: Proposal window.

# Keyword Arguments
- `max_proposals::Int`: Maximum number of proposals allowed until throwing an error (default: `typemax(Int)`).
"""
struct Slice{W <: Union{<:AbstractVector, <:Real}} <: AbstractGibbsSliceSampling
    window       ::W
    max_proposals::Int
end

function Slice(
    window       ::Union{<:AbstractVector, <:Real};
    max_proposals::Int = typemax(Int), 
)
    Slice(window, max_proposals)
end

function find_interval(
    rng::Random.AbstractRNG,
       ::Slice,
       ::Any,
    w  ::Real,
       ::Real,
    θ₀ ::F,
) where {F <: Real}
    u = rand(rng, F)
    L = θ₀ - w*u
    R = L + w
    L, R, 0
end

accept_slice_proposal(
    ::AbstractSliceSampling,
    ::Any,
    ::Real,
    ::Real,
    ::Real,
    ::Real,
    ::Real,
    ::Real,
) = true

function slice_sampling_univariate(
    rng     ::Random.AbstractRNG,
    alg     ::AbstractSliceSampling,
    model, 
    w       ::Real,
    ℓπ      ::Real,
    θ       ::F,
    max_prop::Int,
) where {F <: Real}
    ℓy          = ℓπ - Random.randexp(rng, F)
    L, R, props = find_interval(rng, alg, model, w, ℓy, θ)

    for _ in 1:max_prop
        U     = rand(rng, F)
        θ′     = L + U*(R - L)
        ℓπ′    = LogDensityProblems.logdensity(model, θ′)
        props += 1
        if (ℓy < ℓπ′) && accept_slice_proposal(alg, model, w, ℓy, θ, θ′, L, R)
            return θ′, ℓπ′, props
        end

        if θ′ < θ
            L = θ′
        else
            R = θ′
        end
    end
    exceeded_max_prop(max_prop)
end

