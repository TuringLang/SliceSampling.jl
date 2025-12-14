
module SliceSamplingTuringExt

using LogDensityProblems
using Random
using SliceSampling
using Turing

function SliceSampling.initial_sample(rng::Random.AbstractRNG, ℓ::Turing.LogDensityFunction)
    n_max_attempts = 1000

    model, vi = ℓ.model, ℓ.varinfo
    vi_spl = last(
        Turing.DynamicPPL.init!!(rng, model, vi, Turing.DynamicPPL.InitFromUniform())
    )
    ℓp = ℓ.getlogdensity(vi_spl)

    init_attempt_count = 1
    for attempts in 1:n_max_attempts
        if attempts == 10
            @warn "Failed to find valid initial parameters after $(init_attempt_count) attempts; consider providing explicit initial parameters using the `initial_params` keyword"
        end

        # NOTE: This will sample in the unconstrained space if ℓ.varinfo is linked
        vi_spl = last(Turing.DynamicPPL.init!!(rng, model, vi, Turing.InitFromUniform()))
        ℓp = ℓ.getlogdensity(vi_spl)
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
