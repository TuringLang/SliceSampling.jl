
struct GibbsSliceState{T <: Transition}
    "Current [`Transition`](@ref)."
    transition::T
end

struct GibbsObjective{Model, Idx <: Integer, Vec <: AbstractVector}
    model::Model
    idx  ::Idx
    θ    ::Vec
end

function LogDensityProblems.logdensity(gibbs::GibbsObjective, θi)
    @unpack model, idx, θ = gibbs
    LogDensityProblems.logdensity(model, (@set θ[idx] = θi))
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
    t  = Transition(θ, lp, NamedTuple())
    return t, GibbsSliceState(t)
end

function AbstractMCMC.step(
    rng    ::Random.AbstractRNG,
    model  ::AbstractMCMC.LogDensityModel, 
    sampler::AbstractGibbsSliceSampling,
    state  ::GibbsSliceState;
    kwargs...,
)
    max_prop = sampler.max_proposals
    logdensitymodel = model.logdensity
    w = if sampler.window isa Real
        Fill(sampler.window, LogDensityProblems.dimension(logdensitymodel))
    else
        sampler.window
    end
    ℓp = state.transition.lp
    θ  = copy(state.transition.params)
    @assert length(w) == length(θ) "window size does not match parameter size"

    n_props = zeros(Int, length(θ))
    for idx in shuffle(rng, 1:length(θ))
        model_gibbs = GibbsObjective(logdensitymodel, idx, θ)
        θ′idx, ℓp, props = slice_sampling_univariate(
            rng, sampler, model_gibbs, w[idx], ℓp, θ[idx], max_prop,
        )
        n_props[idx] = props
        θ[idx]       = θ′idx
    end
    t = Transition(θ, ℓp, (num_proposals=n_props,))
    t, GibbsSliceState(t)
end
