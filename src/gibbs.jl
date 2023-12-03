
struct GibbsObjective{Model, Idx <: Integer, Vec <: AbstractVector}
    model::Model
    idx  ::Idx
    θ    ::Vec
end

function LogDensityProblems.logdensity(gibbs::GibbsObjective, θi)
    @unpack model, idx, θ = gibbs
    LogDensityProblems.logdensity(model, (@set θ[idx] = θi))
end

struct SliceState{P, L <: Real, I <: NamedTuple}
    params::P
    lp    ::L
    info  ::I
end

function AbstractMCMC.step(rng    ::Random.AbstractRNG,
                           model,
                           sampler::AbstractGibbsSliceSampling;
                           initial_params = nothing,
                           kwargs...)
    θ  = initial_params === nothing ? initial_sample(rng, model) : initial_params
    lp = LogDensityProblems.logdensity(model, θ)
    return θ, SliceState(θ, lp, NamedTuple())
end

function AbstractMCMC.step(
    rng    ::Random.AbstractRNG,
    model, 
    sampler::AbstractGibbsSliceSampling,
    state  ::SliceState,
    kwargs...,
)
    w = if sampler.window isa Real
        Fill(sampler.window, LogDensityProblems.dimension(model))
    else
        sampler.window
    end
    ℓp = state.lp
    θ  = copy(state.params)

    total_props = 0
    for idx in shuffle(rng, 1:length(θ))
        model_gibbs = GibbsObjective(model, idx, θ)
        θ′idx, ℓp, props = slice_sampling_univariate(
            rng, sampler, model_gibbs, w[idx], ℓp, θ[idx]
        )
        total_props += props
        θ[idx] = θ′idx
    end

    θ, SliceState(θ, ℓp, (num_proposals=total_props,))
end
