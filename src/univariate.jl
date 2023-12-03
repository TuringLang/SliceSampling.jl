
struct Slice{W <: Union{<:AbstractVector, <:Real}} <: AbstractGibbsSliceSampling
    window::W
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
    rng  ::Random.AbstractRNG,
    alg  ::AbstractSliceSampling,
    model, 
    w    ::Real,
    ℓπ   ::Real,
    θ    ::F,
) where {F <: Real}
    ℓy          = ℓπ - Random.randexp(rng, F)
    L, R, props = find_interval(rng, alg, model, w, ℓy, θ)
    while true
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
end

