
struct LatentSlice{B <: Real} <: AbstractSliceSampling
    beta::B
end

struct LatentSliceState{V <: AbstractVector, L <: Real, I <: NamedTuple}
    y   ::V
    s   ::V
    lp  ::L
    info::I
end

function AbstractMCMC.step(rng    ::Random.AbstractRNG,
                           model,
                           sampler::LatentSlice;
                           initial_params = nothing,
                           kwargs...)
    y  = initial_params === nothing ? initial_sample(rng, model) : initial_params
    β  = sampler.beta
    d  = length(y)
    lp = LogDensityProblems.logdensity(model, y)
    s  = rand(rng, Gamma(2, 1/β), d)
    return y, LatentSliceState(y, s, lp, NamedTuple())
end

function AbstractMCMC.step(
    rng    ::Random.AbstractRNG,
    model, 
    sampler::LatentSlice,
    state  ::LatentSliceState,
    kwargs...,
)
    β  = sampler.beta
    ℓp = state.lp
    y  = copy(state.y)
    s  = copy(state.s)
    d  = length(y)
    ℓw = ℓp - Random.randexp(rng, eltype(y))

    u_l = rand(rng, eltype(y), d)
    l   = (y - s/2) + u_l.*s
    a   = l - s/2
    b   = l + s/2

    @assert all(@. a ≤ y ≤ b)
    
    props = 0
    while true
        props += 1

        u_y    = rand(rng, eltype(y), d)
        ystar  = a + u_y.*(b - a)
        ℓpstar = LogDensityProblems.logdensity(model, ystar)

        if ℓw < ℓpstar
            ℓp = ℓpstar
            y  = ystar
            break
        end

        @inbounds for i = 1:d
            if ystar[i] < y[i]
                a[i] = ystar[i]
            else
                b[i] = ystar[i]
            end
        end
    end
    s = β*randexp(rng, eltype(y), d) + 2*abs.(l - y)
    y, LatentSliceState(y, s, ℓp, NamedTuple())
end
