
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
    lp = LogDensityProblems.logdensity(logdensitymodel, θ)
    t  = Transition(θ, lp, NamedTuple())
    return t, GibbsSliceState(t)
end

function AbstractMCMC.step(
    rng    ::Random.AbstractRNG,
    model  ::AbstractMCMC.LogDensityModel, 
    sampler::RandPermGibbs,
    state  ::GibbsSliceState;
    kwargs...,
)
    logdensitymodel = model.logdensity
    ℓp              = state.transition.lp
    θ               = copy(state.transition.params)
    d               = length(θ)
    unislices       = if sampler.unislice isa AbstractVector
        @assert length(sampler.unislice) == d "Number of slice samplers does not match target posterior dimensionality."
        sampler.unislice
    else
        Fill(sampler.unislice, d)
    end

    n_props = zeros(Int, length(θ))
    for i in shuffle(rng, 1:length(θ))
        model_gibbs = GibbsTarget(logdensitymodel, i, θ)
        unislice    = unislices[i]
        θ′idx, ℓp, props = slice_sampling_univariate(
            rng, unislice, model_gibbs, ℓp, θ[i]
        )
        n_props[i] = props
        θ[i]       = θ′idx
    end
    t = Transition(θ, ℓp, (num_proposals=n_props,))
    t, GibbsSliceState(t)
end
