
"""
    Slice(window; max_proposals)

Univariate slice sampling with a fixed initial interval (Scheme 2 by Neal[^N2003])

# Arguments
- `window::Real`: Proposal window.

# Keyword Arguments
- `max_proposals::Int`: Maximum number of proposals allowed until throwing an error (default: `typemax(Int)`).
"""
struct Slice{W <: Real} <: AbstractUnivariateSliceSampling
    window       ::W
    max_proposals::Int
end

function Slice(
    window       ::Real;
    max_proposals::Int = typemax(Int), 
)
    @assert window > 0
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
    rng   ::Random.AbstractRNG,
    alg   ::AbstractSliceSampling,
    model,
    ℓπ    ::Real,
    θ     ::F,
) where {F <: Real}
    w, max_prop = alg.window, alg.max_proposals
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

struct UnivariateSliceState{T <: Transition}
    "Current [`Transition`](@ref)."
    transition::T
end

function AbstractMCMC.step(rng    ::Random.AbstractRNG,
                           model  ::AbstractMCMC.LogDensityModel,
                           sampler::AbstractUnivariateSliceSampling;
                           initial_params = nothing,
                           kwargs...)
    logdensitymodel = model.logdensity
    θ  = isnothing(initial_params) ? initial_sample(rng, logdensitymodel) : initial_params
    @assert length(θ) == 1 "The dimensionality of the parameter should be 1."
    lp = LogDensityProblems.logdensity(logdensitymodel, θ)
    t  = Transition(θ, lp, NamedTuple())
    return t, UnivariateSliceState(t)
end

function AbstractMCMC.step(
    rng    ::Random.AbstractRNG,
    model  ::AbstractMCMC.LogDensityModel, 
    sampler::AbstractUnivariateSliceSampling,
    state  ::UnivariateSliceState;
    kwargs...,
)
    logdensitymodel = model.logdensity
    θ, ℓp = only(state.transition.params), state.transition.lp

    θ, ℓp, props = slice_sampling_univariate(
        rng, sampler, logdensitymodel, ℓp, θ
    )

    t = Transition([θ], ℓp, (num_proposals=props,))
    t, UnivariateSliceState(t)
end
