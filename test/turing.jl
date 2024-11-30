
@testset "turing compatibility" begin
    @model function demo()
        s ~ InverseGamma(2, 3)
        m ~ Normal(0, sqrt(s))
        1.5 ~ Normal(m, sqrt(s))
        return 2.0 ~ Normal(m, sqrt(s))
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

        chain = sample(model, externalsampler(sampler), n_samples; progress=false)
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
            Turing.Experimental.Gibbs((
                s=externalsampler(sampler), m=externalsampler(sampler)
            ),),
            n_samples;
            progress=false,
        )
    end
end
