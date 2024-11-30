
"""
    LatentSlice(beta)

Latent slice sampling algorithm by Li and Walker[^LW2023].

# Arguments
- `beta::Real`: Beta parameter of the Gamma distribution of the auxiliary variables.

# Keyword Arguments
- `max_proposals::Int`: Maximum number of proposals allowed until throwing an error (default: `$(DEFAULT_MAX_PROPOSALS)`).
"""
struct LatentSlice{B<:Real} <: AbstractMultivariateSliceSampling
    beta          :: B
    max_proposals :: Int
end

function LatentSlice(beta::Real; max_proposals::Int=DEFAULT_MAX_PROPOSALS)
    @assert beta > 0 "Beta must be strictly positive"
    return LatentSlice(beta, max_proposals)
end

struct LatentSliceState{T<:Transition,S<:AbstractVector}
    "Current [`Transition`](@ref)."
    transition::T

    "Auxiliary variables for adapting the slice window (\$s\$ in the original paper[^LW2023])"
    sliceparams::S
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.LogDensityModel,
    sampler::LatentSlice;
    initial_params=nothing,
    kwargs...,
)
    logdensitymodel = model.logdensity
    y = initial_params === nothing ? initial_sample(rng, logdensitymodel) : initial_params
    β = sampler.beta
    d = length(y)
    lp = LogDensityProblems.logdensity(logdensitymodel, y)
    s = convert(Vector{eltype(y)}, rand(rng, Gamma(2, 1 / β), d))
    t = Transition(y, lp, NamedTuple())
    return t, LatentSliceState(t, s)
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.LogDensityModel,
    sampler::LatentSlice,
    state::LatentSliceState;
    kwargs...,
)
    logdensitymodel = model.logdensity
    max_proposals   = sampler.max_proposals

    β  = sampler.beta
    ℓp = state.transition.lp
    y  = state.transition.params
    s  = state.sliceparams
    d  = length(y)
    ℓw = ℓp - Random.randexp(rng, eltype(y))

    u_l = rand(rng, eltype(y), d)
    l   = (y - s / 2) + u_l .* s
    a   = l - s / 2
    b   = l + s / 2

    props = 0
    while true
        props += 1

        u_y    = rand(rng, eltype(y), d)
        ystar  = a + u_y .* (b - a)
        ℓpstar = LogDensityProblems.logdensity(logdensitymodel, ystar)

        if ℓw < ℓpstar
            ℓp = ℓpstar
            y  = ystar
            break
        end

        if props > max_proposals
            exceeded_max_prop(max_proposals)
        end

        @inbounds for i in 1:d
            if ystar[i] < y[i]
                a[i] = ystar[i]
            else
                b[i] = ystar[i]
            end
        end
    end
    s = β * randexp(rng, eltype(y), d) + 2 * abs.(l - y)
    t = Transition(y, ℓp, (num_proposals=props,))
    return t, LatentSliceState(t, s)
end
