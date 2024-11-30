
struct WrongModel end

LogDensityProblems.logdensity(::WrongModel, θ) = -Inf

function LogDensityProblems.capabilities(::Type{<:WrongModel})
    return LogDensityProblems.LogDensityOrder{0}()
end

function LogDensityProblems.dimension(::WrongModel)
    return 2
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
            θ = [1.0, 1.0]
            _, init_state = AbstractMCMC.step(rng, model, sampler; initial_params=copy(θ))

            @test_throws ErrorException begin
                _, _ = AbstractMCMC.step(rng, model, sampler, init_state)
            end
        end
    end
end
