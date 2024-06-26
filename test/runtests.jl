
using AbstractMCMC
using Accessors
using Distributions
using LogDensityProblems
using MCMCTesting
using Random 
using Test
using Turing
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
    σ = rand(rng, InverseGamma(α, β)) |> F # InverseGamma is not type stable
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
    model′ = AbstractMCMC.LogDensityModel(@set model.y = y)
    _, init_state = AbstractMCMC.step(rng, model′, sampler; initial_params=copy(θ))
    transition, _ = AbstractMCMC.step(rng, model′, sampler, init_state)
    transition.params
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

function SliceSampling.initial_sample(rng::Random.AbstractRNG, model::Model)
    randn(rng, LogDensityProblems.dimension(model))
end

function LogDensityProblems.capabilities(::Type{<:Model})
    LogDensityProblems.LogDensityOrder{0}()
end

function LogDensityProblems.dimension(model::Model)
    2
end

@testset "sampling" begin
    model   = Model(1., 1., [0.])
    @testset for sampler in [
        # Vector-valued windows
        RandPermGibbs(Slice.(fill(1, LogDensityProblems.dimension(model)))),
        RandPermGibbs(SliceSteppingOut.(fill(1, LogDensityProblems.dimension(model)))),
        RandPermGibbs(SliceDoublingOut.(fill(1, LogDensityProblems.dimension(model)))),

        # Scalar-valued windows
        RandPermGibbs(Slice(1)),
        RandPermGibbs(SliceSteppingOut(1)),
        RandPermGibbs(SliceDoublingOut(1)),

        HitAndRun(Slice(1)),
        HitAndRun(SliceSteppingOut(1)),
        HitAndRun(SliceDoublingOut(1)),

        # Latent slice sampling
        LatentSlice(5),

        # Gibbsian polar slice sampling
        GibbsPolarSlice(10),
    ]
        @testset "determinism" begin
            model  = Model(1.0, 1.0, [0.0])
            θ, y  = MCMCTesting.sample_joint(Random.default_rng(), model)
            model′ = AbstractMCMC.LogDensityModel(@set model.y = y)

            rng           = StableRNG(1)
            _, init_state = AbstractMCMC.step(rng, model′, sampler; initial_params=copy(θ))
            transition, _ = AbstractMCMC.step(rng, model′, sampler, init_state)
            θ′            = transition.params

            rng           = StableRNG(1)
            _, init_state = AbstractMCMC.step(rng, model′, sampler; initial_params=copy(θ))
            transition, _ = AbstractMCMC.step(rng, model′, sampler, init_state)
            θ′′            = transition.params
            @test θ′ == θ′′
        end

        @testset "type stability $(type)" for type in [Float32, Float64]
            rng   = Random.default_rng()
            model = Model(one(type), one(type), [zero(type)])
            θ, y  = MCMCTesting.sample_joint(Random.default_rng(), model)
            model′ = AbstractMCMC.LogDensityModel(@set model.y = y)

            @test eltype(θ) == type
            @test eltype(y) == type

            _, init_state = AbstractMCMC.step(rng, model′, sampler; initial_params=copy(θ))
            transition, _ = AbstractMCMC.step(rng, model′, sampler, init_state)
            θ′             = transition.params

            @test eltype(θ′) == type
        end

        @testset "inference" begin
            n_pvalue_samples = 64
            n_samples        = 100
            n_mcmc_steps     = 10
            n_mcmc_thin      = 10
            test             = ExactRankTest(n_samples, n_mcmc_steps, n_mcmc_thin)

            model   = Model(1., 1., [0.])
            subject = TestSubject(model, sampler)
            @test seqmcmctest(test, subject, 0.001, n_pvalue_samples; show_progress=false)
        end
    end
end

struct WrongModel end

LogDensityProblems.logdensity(::WrongModel, θ) = -Inf

function LogDensityProblems.capabilities(::Type{<:WrongModel})
    LogDensityProblems.LogDensityOrder{0}()
end

function LogDensityProblems.dimension(::WrongModel)
    2
end

@testset "error handling" begin
    model = AbstractMCMC.LogDensityModel(WrongModel())
    @testset for sampler in [
        # Univariate slice samplers
        RandPermGibbs(Slice(1; max_proposals=32)),
        RandPermGibbs(SliceSteppingOut(1; max_proposals=32)),
        RandPermGibbs(SliceDoublingOut(1; max_proposals=32)),

        HitAndRun(Slice(1; max_proposals=32)),
        HitAndRun(SliceSteppingOut(1; max_proposals=32)),
        HitAndRun(SliceDoublingOut(1; max_proposals=32)),

        # Latent slice sampling
        LatentSlice(5; max_proposals=32),

        # Gibbs polar slice sampling
        GibbsPolarSlice(5; max_proposals=32),
    ]
        @testset "max proposal error" begin
            rng = Random.default_rng()
            θ   = [1., 1.]
            _, init_state = AbstractMCMC.step(rng, model, sampler; initial_params=copy(θ))

            @test_throws ErrorException begin
                _, _ = AbstractMCMC.step(rng, model, sampler, init_state)
            end
        end
    end
end

@testset "turing compatibility" begin
    @model function demo()
        s ~ InverseGamma(2, 3)
        m ~ Normal(0, sqrt(s))
        1.5 ~ Normal(m, sqrt(s))
        2.0 ~ Normal(m, sqrt(s))
    end

    n_samples = 1000
    model     = demo()

    @testset for sampler in [
        RandPermGibbs(Slice(1)),
        RandPermGibbs(SliceSteppingOut(1)),
        RandPermGibbs(SliceDoublingOut(1)),
        HitAndRun(Slice(1)),
        HitAndRun(SliceSteppingOut(1)),
        HitAndRun(SliceDoublingOut(1)),
        LatentSlice(5),
        GibbsPolarSlice(5),
    ]
        chain = sample(
            model,
            externalsampler(sampler),
            n_samples;
            initial_params=[1.0, 0.1],
            progress=false,
        )
    end
end
