
module SliceSamplingTuringExt

using Random
using SliceSampling
using Turing

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

const SliceSamplingStates = Union{
    SliceSampling.UnivariateSliceState,
    SliceSampling.GibbsState,
    SliceSampling.HitAndRunState,
    SliceSampling.LatentSliceState,
    SliceSampling.GibbsPolarSliceState,
}
function Turing.Inference.getparams(::Turing.DynamicPPL.Model, sample::SliceSamplingStates)
    return sample.transition.params
end
# end

function SliceSampling.initial_sample(rng::Random.AbstractRNG, ℓ::Turing.LogDensityFunction)
    model  = ℓ.model
    vi     = Turing.DynamicPPL.VarInfo(rng, model, Turing.SampleFromUniform())
    vi_spl = last(Turing.DynamicPPL.evaluate_and_sample!!(rng, model, vi, Turing.SampleFromUniform()))
    θ     = vi_spl[:]

    init_attempt_count = 1
    while !all(isfinite.(θ))
        if init_attempt_count == 10
            @warn "failed to find valid initial parameters in $(init_attempt_count) tries; consider providing explicit initial parameters using the `initial_params` keyword"
        end

        # NOTE: This will sample in the unconstrained space.
        vi_spl = last(
            Turing.DynamicPPL.evaluate_and_sample!!(
                rng, model, vi, Turing.SampleFromUniform()
            ),
        )
        θ = vi_spl[:]

        init_attempt_count += 1
    end
    return θ
end

end
