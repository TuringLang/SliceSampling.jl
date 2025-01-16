
function slice_sampling_univariate(
    rng::Random.AbstractRNG, alg::AbstractSliceSampling, model, ℓπ::Real, θ::F
) where {F<:Real}
    w, max_prop = alg.window, alg.max_proposals
    ℓy          = ℓπ - Random.randexp(rng, F)
    L, R, props = find_interval(rng, alg, model, w, ℓy, θ)

    for _ in 1:max_prop
        U     = rand(rng, F)
        θ′    = L + U * (R - L)
        ℓπ′   = LogDensityProblems.logdensity(model, θ′)
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
    return exceeded_max_prop(max_prop)
end

struct UnivariateSliceState{T<:Transition}
    "Current [`Transition`](@ref)."
    transition::T
end

function AbstractMCMC.setparams!!(
    model::AbstractMCMC.LogDensityModel, state::UnivariateSliceState, params
)
    lp = LogDensityProblems.logdensity(model.logdensity, params)
    return UnivariateSliceState(Transition(params, lp, NamedTuple()))
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.LogDensityModel,
    sampler::AbstractUnivariateSliceSampling;
    initial_params=nothing,
    kwargs...,
)
    logdensitymodel = model.logdensity
    θ = isnothing(initial_params) ? initial_sample(rng, logdensitymodel) : initial_params
    @assert length(θ) == 1 "The dimensionality of the parameter should be 1."
    lp = LogDensityProblems.logdensity(logdensitymodel, θ)
    t  = Transition(θ, lp, NamedTuple())
    return t, UnivariateSliceState(t)
end

struct UnivariateTarget{Model}
    model::Model
end

function LogDensityProblems.logdensity(uni::UnivariateTarget, θi)
    return LogDensityProblems.logdensity(uni.model, [θi])
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.LogDensityModel,
    sampler::AbstractUnivariateSliceSampling,
    state::UnivariateSliceState;
    kwargs...,
)
    logdensitymodel = model.logdensity
    θ, ℓp = only(state.transition.params), state.transition.lp

    θ, ℓp, props = slice_sampling_univariate(
        rng, sampler, UnivariateTarget(logdensitymodel), ℓp, θ
    )

    t = Transition([θ], ℓp, (num_proposals=props,))
    return t, UnivariateSliceState(t)
end
