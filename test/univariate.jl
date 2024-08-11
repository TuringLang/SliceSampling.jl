

struct UniModel{V <: AbstractVector}
    y::V
end

function MCMCTesting.sample_joint(
    rng::AbstractRNG, ::UniModel{<:AbstractVector{F}}
) where {F <: Real}
    μ = rand(rng, Normal(zero(F), one(F)))
    y = rand(rng, Normal(μ, one(F)), 10)
    [μ], y
end

function MCMCTesting.markovchain_transition(
    rng    ::Random.AbstractRNG,
    model  ::UniModel,
    sampler::SliceSampling.AbstractSliceSampling,
    θ, y
)
    model′ = AbstractMCMC.LogDensityModel(@set model.y = y)
    _, init_state = AbstractMCMC.step(rng, model′, sampler; initial_params=copy(θ))
    transition, _ = AbstractMCMC.step(rng, model′, sampler, init_state)
    transition.params
end

function LogDensityProblems.logdensity(
    model::UniModel{<:AbstractVector{F}}, θ
) where {F <: Real}
    y = model.y
    μ = only(θ)

    logpdf(Normal(zero(F), one(F)), μ) + sum(Base.Fix1(logpdf, Normal(μ, one(F))), y)
end

function SliceSampling.initial_sample(rng::Random.AbstractRNG, model::UniModel)
    randn(rng, LogDensityProblems.dimension(model))
end

function LogDensityProblems.capabilities(::Type{<:UniModel})
    LogDensityProblems.LogDensityOrder{0}()
end

function LogDensityProblems.dimension(model::UniModel)
    1
end

@testset "multivariate samplers" begin
    model   = UniModel([0.])
    @testset for sampler in [
        Slice(1),
        SliceSteppingOut(1),
        SliceDoublingOut(1),
    ]
        @testset "initialization" begin
            model  = UniModel([0.0])
            θ, y  = MCMCTesting.sample_joint(Random.default_rng(), model)
            model′ = AbstractMCMC.LogDensityModel(@set model.y = y)

            θ0    = [1.0]
            chain = sample(
                model,
                sampler,
                10;
                initial_params=θ0,
                progress=false,
            )
            @test first(chain).params == θ0
        end

        @testset "initial_sample" begin
            rng = StableRNG(1)
            model = UniModel([0.0])
            θ0 = SliceSampling.initial_sample(rng, model)

            rng = StableRNG(1)
            chain = sample(rng, model, sampler, 10; progress=false)
            @test first(chain).params == θ0
        end

        @testset "determinism" begin
            model  = UniModel([0.0])
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
            model = UniModel([zero(type)])
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

            model   = UniModel([0.])
            subject = TestSubject(model, sampler)
            @test seqmcmctest(test, subject, 0.001, n_pvalue_samples; show_progress=false)
        end
    end
end
