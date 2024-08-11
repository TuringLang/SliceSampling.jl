
module SliceSamplingTuringExt

if isdefined(Base, :get_extension)
    using LogDensityProblemsAD
    using Random
    using SliceSampling
    using Turing: Turing
else
    using ..LogDensityProblemsAD
    using ..Random
    using ..SliceSampling
    using ..Turing: Turing
end

Turing.Inference.getparams(
          ::Turing.DynamicPPL.Model,
    sample::SliceSampling.Transition
) = sample.params

function SliceSampling.initial_sample(
    rng::Random.AbstractRNG,
    ℓ  ::Turing.LogDensityFunction
)
    model = ℓ.model
    spl = Turing.SampleFromUniform()
    vi = Turing.VarInfo(rng, model, spl)
    x = vi[spl]
    init_attempt_count = 1
    while !isfinite(x)
        if init_attempt_count == 10
            @warn "failed to find valid initial parameters in $(init_attempt_count) tries; consider providing explicit initial parameters using the `initial_params` keyword"
        end
        
        # NOTE: This will sample in the unconstrained space.
        vi = last(DynamicPPL.evaluate!!(model, rng, vi, SampleFromUniform()))
        x = vi[spl]

        init_attempt_count += 1
    end
    x
end

end
