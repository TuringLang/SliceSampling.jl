
module SliceSamplingTuringExt

using LogDensityProblems
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
    n_max_attempts = 1000

    model, vi = ℓ.model, ℓ.varinfo
    vi_spl = last(Turing.DynamicPPL.init!!(rng, model, vi, Turing.DynamicPPL.InitFromUniform()))
    ℓp = Turing.DynamicPPL.getlogjoint_internal(vi_spl)

    init_attempt_count = 1
    for attempts in 1:n_max_attempts
        if attempts == 10
            @warn "Failed to find valid initial parameters after $(init_attempt_count) attempts; consider providing explicit initial parameters using the `initial_params` keyword"
        end

        # NOTE: This will sample in the unconstrained space.
        vi_spl = last(
            Turing.DynamicPPL.init!!(
                rng, model, vi, Turing.InitFromUniform()
            ),
        )
        ℓp = Turing.DynamicPPL.getlogjoint_internal(vi_spl)
        θ = vi_spl[:]

        if all(isfinite.(θ)) && isfinite(ℓp)
            return θ
        end
    end

    @error "Failed to find valid initial parameters after $(n_max_attempts) attempts; consider providing explicit initial parameters using the `initial_params` keyword"
    θ = vi_spl[:]
    return θ
end

end
