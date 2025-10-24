
@testset "turing compatibility" begin
    @model function demo()
        s ~ InverseGamma(2, 3)
        m ~ Normal(0, sqrt(s))
        1.5 ~ Normal(m, sqrt(s))
        2.0 ~ Normal(m, sqrt(s))
        return nothing
    end

    @model function illbehavedmodel()
        @addlogprob! -Inf
        return nothing
    end

    @model function logp_check()
        a ~ Normal()
        return b ~ Normal()
    end

    rng = Random.default_rng()
    @test begin
        init = SliceSampling.initial_sample(rng, LogDensityFunction(demo()))
        all(isfinite.(init))
    end

    @test_warn "Warning: Failed" SliceSampling.initial_sample(
        rng, LogDensityFunction(illbehavedmodel())
    )

    @test_warn "Error: Failed" SliceSampling.initial_sample(
        rng, LogDensityFunction(illbehavedmodel())
    )

    n_samples = 1000
    model = demo()

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
            initial_params=InitFromParams((s=1.0, m=0.1)),
            progress=false,
        )

        chain = sample(model, externalsampler(sampler), n_samples; progress=false)

        chain_logp_check = sample(
            logp_check(), externalsampler(sampler), 100; progress=false
        )
        @test isapprox(
            logpdf.(Normal(), chain_logp_check[:a]) .+
            logpdf.(Normal(), chain_logp_check[:b]),
            chain_logp_check[:lp],
        )
    end

    @testset "gibbs($sampler)" for sampler in [
        RandPermGibbs(Slice(1)),
        RandPermGibbs(SliceSteppingOut(1)),
        RandPermGibbs(SliceDoublingOut(1)),
        Slice(1),
        SliceSteppingOut(1),
        SliceDoublingOut(1),
    ]
        sample(
            model,
            Turing.Gibbs(:s => externalsampler(sampler), :m => externalsampler(sampler)),
            n_samples;
            progress=false,
        )

        chain_logp_check = sample(
            logp_check(),
            Turing.Gibbs(:a => externalsampler(sampler), :b => externalsampler(sampler)),
            100;
            progress=false,
        )
        @test isapprox(
            logpdf.(Normal(), chain_logp_check[:a]) .+
            logpdf.(Normal(), chain_logp_check[:b]),
            chain_logp_check[:lp],
        )
    end
end
