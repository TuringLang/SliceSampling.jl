
using AbstractMCMC
using Accessors
using Distributions
using LogDensityProblems
using MCMCTesting
using Random 
using Test
using StableRNGs

using SliceSampling

struct Model{F <: Real, V <: AbstractVector}
    α::F
    β::F
    y::V
end

function MCMCTesting.sample_joint(
    rng::AbstractRNG, model::Model{F, V}
) where {F <: Real, V <: AbstractVector}
    α, β = model.α, model.β

    μ = rand(rng, Normal(zero(F), one(F)))
    σ = rand(rng, InverseGamma(α, β))
    y = rand(rng, Normal(μ, σ), 10)
    θ = [μ, σ]
    θ, y
end

function MCMCTesting.markovchain_transition(
    rng    ::Random.AbstractRNG,
    model  ::Model,
    sampler::SliceSampling.AbstractSliceSampling,
    θ, y
)
    model′ = @set model.y = y
    _, init_state = AbstractMCMC.step(rng, model′, sampler; initial_params=copy(θ))
    θ′, _ = AbstractMCMC.step(rng, model′, sampler, init_state)
    θ′
end

function LogDensityProblems.logdensity(model::Model{F, V}, θ) where {F <: Real, V}
    α, β, y = model.α, model.β, model.y

    μ = θ[1]
    σ = θ[2]

    if σ ≤ 0
       return typemin(F)
    end

    logpdf(Normal(zero(F), one(F)), μ) +
        logpdf(InverseGamma(α, β), σ) +
        sum(Base.Fix1(logpdf, Normal(μ, σ)), y)
end

function LogDensityProblems.dimension(model::Model)
    2
end

@testset "slice sampling" begin
    model  = Model(1.0, 1.0, [0.0])
    window = fill(1.0, LogDensityProblems.dimension(model))

    @testset for sampler in [
        Slice(window),
        SliceSteppingOut(window),
        SliceDoublingOut(window),
        LatentSlice(5.0),
    ]
        @testset "determinism" begin
            θ, y  = MCMCTesting.sample_joint(Random.default_rng(), model)
            model = @set model.y = y

            rng           = StableRNG(1)
            _, init_state = AbstractMCMC.step(rng, model, sampler; initial_params=copy(θ))
            θ′, _         = AbstractMCMC.step(rng, model, sampler, init_state)

            rng           = StableRNG(1)
            _, init_state = AbstractMCMC.step(rng, model, sampler; initial_params=copy(θ))
            θ′′, _         = AbstractMCMC.step(rng, model, sampler, init_state)
            @test θ′ == θ′′
        end

        @testset "type stability $(type)" for type in [Float32, Float64]
            model = Model(one(type), one(type), [zero(type)])
            θ, y  = MCMCTesting.sample_joint(Random.default_rng(), model)
            model = @set model.y = y
            rng   = Random.default_rng()

            _, init_state = AbstractMCMC.step(rng, model, sampler; initial_params=copy(θ))
            θ′, _ = AbstractMCMC.step(rng, model, sampler, init_state)

            @test typeof(θ) == typeof(θ′)
        end


        @testset "inference" begin
            n_pvalue_samples = 64
            n_samples        = 100
            n_mcmc_steps     = 10
            n_mcmc_thin      = 10
            test             = ExactRankTest(n_samples, n_mcmc_steps, n_mcmc_thin)

            subject = TestSubject(model, sampler)
            @test seqmcmctest(test, subject, 0.001, n_pvalue_samples; show_progress=false)
        end
    end
end
