
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
    "current state of the slice sampling chain"
    params::P

    "log density of the current state"
    lp::L

    "information generated from the sampler"
    info::I
end

function AbstractMCMC.step(rng    ::Random.AbstractRNG,
                           model  ::AbstractMCMC.LogDensityModel,
                           sampler::AbstractGibbsSliceSampling;
                           initial_params = nothing,
                           kwargs...)
    logdensitymodel = model.logdensity
    d  = LogDensityProblems.dimension(logdensitymodel)
    if sampler.window isa AbstractVector
        @assert length(sampler.window) == d
    end
    θ  = isnothing(initial_params) ? initial_sample(rng, logdensitymodel) : initial_params
    lp = LogDensityProblems.logdensity(logdensitymodel, θ)
    return θ, SliceState(θ, lp, NamedTuple())
end

function AbstractMCMC.step(
    rng    ::Random.AbstractRNG,
    model  ::AbstractMCMC.LogDensityModel, 
    sampler::AbstractGibbsSliceSampling,
    state  ::SliceState;
    kwargs...,
)
    logdensitymodel = model.logdensity
    w = if sampler.window isa Real
        Fill(sampler.window, LogDensityProblems.dimension(logdensitymodel))
    else
        sampler.window
    end
    ℓp = state.lp
    θ  = copy(state.params)

    total_props = 0
    for idx in shuffle(rng, 1:length(θ))
        model_gibbs = GibbsObjective(logdensitymodel, idx, θ)
        θ′idx, ℓp, props = slice_sampling_univariate(
            rng, sampler, model_gibbs, w[idx], ℓp, θ[idx]
        )
        total_props += props
        θ[idx] = θ′idx
    end

    θ, SliceState(θ, ℓp, (num_proposals=total_props,))
end
