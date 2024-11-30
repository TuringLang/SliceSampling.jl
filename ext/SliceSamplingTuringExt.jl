
module SliceSamplingTuringExt

if isdefined(Base, :get_extension)
    using LogDensityProblemsAD
    using Random
    using SliceSampling
    using Turing
    # using Turing: Turing, Experimental
else
    using ..LogDensityProblemsAD
    using ..Random
    using ..SliceSampling
    using ..Turing
    #using ..Turing: Turing, Experimental
end

# Required for using the slice samplers as `externalsampler`s in Turing
# begin
function Turing.Inference.getparams(
    ::Turing.DynamicPPL.Model, sample::SliceSampling.Transition
)
    return sample.params
end
# end

# Required for using the slice samplers as `Experimental.Gibbs` samplers in Turing
# begin
function Turing.Inference.getparams(
    ::Turing.DynamicPPL.Model, state::SliceSampling.UnivariateSliceState
)
    return state.transition.params
end

function Turing.Inference.getparams(
    ::Turing.DynamicPPL.Model, state::SliceSampling.GibbsState
)
    return state.transition.params
end

function Turing.Inference.getparams(
    ::Turing.DynamicPPL.Model, state::SliceSampling.HitAndRunState
)
    return state.transition.params
end

function Turing.Experimental.gibbs_requires_recompute_logprob(
    model_dst,
    ::Turing.DynamicPPL.Sampler{
        <:Turing.Inference.ExternalSampler{<:SliceSampling.AbstractSliceSampling,A,U}
    },
    sampler_src,
    state_dst,
    state_src,
) where {A,U}
    return false
end
# end

function SliceSampling.initial_sample(rng::Random.AbstractRNG, ℓ::Turing.LogDensityFunction)
    model = ℓ.model
    spl   = Turing.SampleFromUniform()
    vi    = Turing.VarInfo(rng, model, spl)
    θ     = vi[spl]

    init_attempt_count = 1
    while !isfinite(θ)
        if init_attempt_count == 10
            @warn "failed to find valid initial parameters in $(init_attempt_count) tries; consider providing explicit initial parameters using the `initial_params` keyword"
        end

        # NOTE: This will sample in the unconstrained space.
        vi = last(DynamicPPL.evaluate!!(model, rng, vi, SampleFromUniform()))
        θ  = vi[spl]

        init_attempt_count += 1
    end
    return θ
end

end
