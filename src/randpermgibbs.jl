
"""
    RandPermGibbs(unislice)

Random permutation coordinate-wise Gibbs sampling strategy.
This applies `unislice` coordinate-wise in a random order.

# Arguments
- `unislice`: Univariate slice sampling algorithm.

`unislice` can also be a vector of univeriate slice samplers, where each slice sampler is applied to each corresponding coordinate of the target posterior.
"""
struct RandPermGibbs{
    S <: Union{
        <: AbstractUnivariateSliceSampling,
        <: AbstractVector{<: AbstractUnivariateSliceSampling}
    }
} <: AbstractMultivariateSliceSampling
    unislice::S
end

function AbstractMCMC.step(rng    ::Random.AbstractRNG,
                           model  ::AbstractMCMC.LogDensityModel,
                           sampler::RandPermGibbs;
                           initial_params = nothing,
                           kwargs...)
    logdensitymodel = model.logdensity
    θ  = isnothing(initial_params) ? initial_sample(rng, logdensitymodel) : initial_params
    d  = length(θ)
    if sampler.unislice isa AbstractVector
        @assert length(sampler.unislice) == d "Number of slice samplers does not match the dimensionality of the initial parameter."
    end
    lp = LogDensityProblems.logdensity(logdensitymodel, θ)
    t  = Transition(θ, lp, NamedTuple())
    return t, GibbsState(t)
end

function AbstractMCMC.step(
    rng    ::Random.AbstractRNG,
    model  ::AbstractMCMC.LogDensityModel, 
    sampler::RandPermGibbs,
    state  ::GibbsState;
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
        unislice    = unislices[i]
        θ′_coord, ℓp, props_coord = slice_sampling_univariate(
            rng, unislice, model_gibbs, ℓp, θ[i]
        )
        props[i] = props_coord
        θ[i]     = θ′_coord
    end
    t = Transition(θ, ℓp, (num_proposals=props,))
    t, GibbsState(t)
end
