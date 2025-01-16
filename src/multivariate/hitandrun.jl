
"""
    HitAndRun(unislice)

Hit-and-run sampling strategy[^BRS1993].
This applies `unislice` along a random direction uniform sampled from the sphere.

# Arguments
- `unislice::AbstractUnivariateSliceSampling`: Univariate slice sampling algorithm.
"""
struct HitAndRun{S<:AbstractUnivariateSliceSampling} <: AbstractMultivariateSliceSampling
    unislice::S
end

struct HitAndRunState{T<:Transition}
    "Current [`Transition`](@ref)."
    transition::T
end

function AbstractMCMC.setparams!!(
    model::AbstractMCMC.LogDensityModel,
    state::HitAndRunState,
    params
)
    lp = LogDensityProblems.logdensity(model.logdensity, params)
    return HitAndRunState(Transition(params, lp, NamedTuple()))
end

struct HitAndRunTarget{Model,Vec<:AbstractVector}
    model     :: Model
    direction :: Vec
    reference :: Vec
end

function LogDensityProblems.logdensity(target::HitAndRunTarget, λ)
    (; model, reference, direction) = target
    return LogDensityProblems.logdensity(model, reference + λ * direction)
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.LogDensityModel,
    sampler::HitAndRun;
    initial_params=nothing,
    kwargs...,
)
    logdensitymodel = model.logdensity
    θ = isnothing(initial_params) ? initial_sample(rng, logdensitymodel) : initial_params
    d = length(θ)
    @assert d ≥ 2 "Hit-and-Run works reliably only in dimension ≥2"
    lp = LogDensityProblems.logdensity(logdensitymodel, θ)
    t  = Transition(θ, lp, NamedTuple())
    return t, HitAndRunState(t)
end

function rand_uniform_unit_sphere(rng::Random.AbstractRNG, type::Type, d::Int)
    x = randn(rng, type, d)
    return x / norm(x)
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.LogDensityModel,
    sampler::HitAndRun,
    state::HitAndRunState;
    kwargs...,
)
    logdensitymodel = model.logdensity
    ℓp              = state.transition.lp
    θ               = copy(state.transition.params)
    d               = length(θ)
    unislice        = sampler.unislice

    direction    = rand_uniform_unit_sphere(rng, eltype(θ), d)
    hnrtarget    = HitAndRunTarget(logdensitymodel, direction, θ)
    λ            = zero(eltype(θ))
    λ, ℓp, props = slice_sampling_univariate(rng, unislice, hnrtarget, ℓp, λ)
    θ′           = θ + direction * λ
    t            = Transition(θ′, ℓp, (num_proposals=props,))
    return t, HitAndRunState(t)
end
