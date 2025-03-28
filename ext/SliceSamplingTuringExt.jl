
module SliceSamplingTuringExt

if isdefined(Base, :get_extension)
    using LogDensityProblemsAD
    using Random
    using SliceSampling
    using Turing
else
    using ..LogDensityProblemsAD
    using ..Random
    using ..SliceSampling
    using ..Turing
end

# Required for using the slice samplers as `externalsampler`s in Turing
# begin
function Turing.Inference.getparams(
    ::Turing.DynamicPPL.Model, sample::SliceSampling.Transition
)
    return sample.params
end
# end

# Required for using the slice samplers as `Gibbs` samplers in Turing
# begin
Turing.Inference.isgibbscomponent(::SliceSampling.RandPermGibbs) = true
Turing.Inference.isgibbscomponent(::SliceSampling.HitAndRun) = true
Turing.Inference.isgibbscomponent(::SliceSampling.Slice) = true
Turing.Inference.isgibbscomponent(::SliceSampling.SliceSteppingOut) = true
Turing.Inference.isgibbscomponent(::SliceSampling.SliceDoublingOut) = true

function Turing.Inference.getparams(
    ::Turing.DynamicPPL.Model, sample::SliceSampling.UnivariateSliceState
)
    return sample.transition.params
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
