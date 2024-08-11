
struct MultiModel{F <: Real, V <: AbstractVector}
    α::F
    β::F
    y::V
end

function MCMCTesting.sample_joint(
    rng::AbstractRNG, model::MultiModel{F, V}
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
    model  ::MultiModel,
    sampler::SliceSampling.AbstractSliceSampling,
    θ, y
)
    model′ = AbstractMCMC.LogDensityModel(@set model.y = y)
    _, init_state = AbstractMCMC.step(rng, model′, sampler; initial_params=copy(θ))
    transition, _ = AbstractMCMC.step(rng, model′, sampler, init_state)
    transition.params
end

function LogDensityProblems.logdensity(model::MultiModel{F, V}, θ) where {F <: Real, V}
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

function SliceSampling.initial_sample(rng::Random.AbstractRNG, model::MultiModel)
    randn(rng, LogDensityProblems.dimension(model))
end

function LogDensityProblems.capabilities(::Type{<:MultiModel})
    LogDensityProblems.LogDensityOrder{0}()
end

function LogDensityProblems.dimension(model::MultiModel)
    2
end

@testset "multivariate samplers" begin
    model   = MultiModel(1., 1., [0.])
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
        @testset "initial_params" begin
            model  = MultiModel(1.0, 1.0, [0.0])
            θ, y  = MCMCTesting.sample_joint(Random.default_rng(), model)
            model′ = AbstractMCMC.LogDensityModel(@set model.y = y)

            θ0    = [1.0, 0.1]
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
            model = MultiModel(1.0, 1.0, [0.0])
            θ0 = SliceSampling.initial_sample(rng, model)

            rng = StableRNG(1)
            chain = sample(rng, model, sampler, 10; progress=false)
            @test first(chain).params == θ0
        end

        @testset "determinism" begin
            model  = MultiModel(1.0, 1.0, [0.0])
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
            model = MultiModel(one(type), one(type), [zero(type)])
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

            model   = MultiModel(1., 1., [0.])
            subject = TestSubject(model, sampler)
            @test seqmcmctest(test, subject, 0.001, n_pvalue_samples; show_progress=false)
        end
    end
end
