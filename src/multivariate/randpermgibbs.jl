
"""
    RandPermGibbs(unislice)

Random permutation coordinate-wise Gibbs sampling strategy.
This applies `unislice` coordinate-wise in a random order.

# Arguments
- `unislice::Union{<:AbstractUnivariateSliceSampling,<:AbstractVector{<:AbstractUnivariateSliceSampling}}`: a single or a vector of univariate slice sampling algorithms.

When `unislice` is a vector of samplers, each slice sampler is applied to the corresponding coordinate of the target posterior.
In that case, the `length(unislice)` must match the dimensionality of the posterior.
"""
struct RandPermGibbs{
    S<:Union{
        <:AbstractUnivariateSliceSampling,
        <:AbstractVector{<:AbstractUnivariateSliceSampling},
    },
} <: AbstractMultivariateSliceSampling
    unislice::S
end

struct GibbsState{T<:Transition}
    "Current [`Transition`](@ref)."
    transition::T
end

function AbstractMCMC.setparams!!(
    model::AbstractMCMC.LogDensityModel,
    state::GibbsState,
    params
)
    lp = LogDensityProblems.logdensity(model.logdensity, params)
    return GibbsState(Transition(params, lp, NamedTuple()))
end

struct GibbsTarget{Model,Idx<:Integer,Vec<:AbstractVector}
    model :: Model
    idx   :: Idx
    θ     :: Vec
end

function LogDensityProblems.logdensity(gibbs::GibbsTarget, θi)
    (; model, idx, θ) = gibbs
    return LogDensityProblems.logdensity(model, (@set θ[idx] = θi))
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.LogDensityModel,
    sampler::RandPermGibbs;
    initial_params=nothing,
    kwargs...,
)
    logdensitymodel = model.logdensity
    θ = initial_params === nothing ? initial_sample(rng, logdensitymodel) : initial_params
    d = length(θ)
    if sampler.unislice isa AbstractVector
        @assert length(sampler.unislice) == d "Number of slice samplers does not match the dimensionality of the initial parameter."
    end
    lp = LogDensityProblems.logdensity(logdensitymodel, θ)
    t  = Transition(θ, lp, NamedTuple())
    return t, GibbsState(t)
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.LogDensityModel,
    sampler::RandPermGibbs,
    state::GibbsState;
    kwargs...,
)
    logdensitymodel = model.logdensity
    ℓp              = state.transition.lp
    θ               = copy(state.transition.params)
    d               = length(θ)
    unislices       = if sampler.unislice isa AbstractVector
        sampler.unislice
    else
        Fill(sampler.unislice, d)
    end

    props = zeros(Int, d)
    for i in shuffle(rng, 1:d)
        model_gibbs = GibbsTarget(logdensitymodel, i, θ)
        unislice = unislices[i]
        θ′_coord, ℓp, props_coord = slice_sampling_univariate(
            rng, unislice, model_gibbs, ℓp, θ[i]
        )
        props[i] = props_coord
        θ[i] = θ′_coord
    end
    t = Transition(θ, ℓp, (num_proposals=props,))
    return t, GibbsState(t)
end
