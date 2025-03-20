
"""
    StereographicSlice(invscale; max_proposals)

Stereographic slice sampling algorithm by Bell, Latuszynski, and Roberts[^BLR2024].

# Keyword Arguments
- `invscale::Real`: Inverse scale of the stereographic projection.
- `max_proposals::Int`: Maximum number of proposals allowed until throwing an error (default: `$(100*DEFAULT_MAX_PROPOSALS)`).
"""
struct StereographicSlice{S<:Real} <: AbstractMultivariateSliceSampling
    invscale::S
    max_proposals::Int
end

function StereographicSlice(invscale::Real; max_proposals::Int = 100*DEFAULT_MAX_PROPOSALS)
    @assert invscale > 0
    StereographicSlice{typeof(invscale)}(invscale, max_proposals)
end

function AbstractMCMC.setparams!!(
    model::AbstractMCMC.LogDensityModel, ::SliceSampling.Transition, params
)
    lp = LogDensityProblems.logdensity(model.logdensity, params)
    return Transition(params, lp, NamedTuple())
end

function rand_uniform_sphere_orthogonal_subspace(
    rng::Random.AbstractRNG, subspace_vector::AbstractVector{T}
) where {T<:Real}
    z      = subspace_vector
    d      = length(subspace_vector)
    v      = randn(rng, T, d)
    v_proj = dot(z, v) / sum(abs2, z) * z
    v_orth = v - v_proj
    return v_orth / norm(v_orth)
end

function stereographic_projection(z::AbstractVector{T}, R::T) where {T<:Real}
    d = length(z) - 1
    return R * z[1:d] ./ (1 - z[d + 1])
end

function stereographic_inverse_projection(x::AbstractVector{T}, R::T) where {T<:Real}
    d        = length(x)
    R2       = R*R
    z        = zeros(T, d + 1)
    x_norm2  = sum(abs2, x)
    z[1:d]   = 2 * R * x / (x_norm2 + R2)
    z[d + 1] = (x_norm2 - R2) / (x_norm2 + R2)
    return z
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.LogDensityModel,
    sampler::StereographicSlice;
    initial_params=nothing,
    kwargs...,
)
    logdensitymodel = model.logdensity
    x = initial_params === nothing ? initial_sample(rng, logdensitymodel) : initial_params
    lp = LogDensityProblems.logdensity(logdensitymodel, x)
    t = Transition(x, lp, NamedTuple())
    return t, t
end

function logdensity_sphere(ℓπ::Real, x::AbstractVector{T}, R::T) where {T<:Real}
    d = length(x)
    return ℓπ + d * log(R*R + sum(abs2, x))
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.LogDensityModel,
    sampler::StereographicSlice,
    state::Transition;
    kwargs...,
)
    logdensitymodel = model.logdensity
    max_proposals   = sampler.max_proposals

    ℓp        = state.lp
    x         = state.params
    R         = convert(eltype(x), sampler.invscale)
    z         = stereographic_inverse_projection(x, R)
    v         = rand_uniform_sphere_orthogonal_subspace(rng, z)
    ℓp_sphere = logdensity_sphere(ℓp, x, R)
    ℓw        = ℓp_sphere - Random.randexp(rng, eltype(x))

    θ     = convert(eltype(x), 2π) * rand(rng, eltype(x))
    θ_max = θ
    θ_min = θ - convert(eltype(x), 2π)

    props = 0
    while true
        props += 1

        x_prop         = stereographic_projection(z * cos(θ) + v * sin(θ), R)
        ℓp_prop        = LogDensityProblems.logdensity(logdensitymodel, x_prop)
        ℓp_sphere_prop = logdensity_sphere(ℓp_prop, x_prop, R)

        if ℓw < ℓp_sphere_prop
            ℓp = ℓp_prop
            x  = x_prop
            break
        end

        if props > max_proposals
            exceeded_max_prop(max_proposals)
        end

        if θ < 0
            θ_min = θ
        else
            θ_max = θ
        end
        θ = (θ_max - θ_min) * rand(rng, eltype(x))
    end
    t = Transition(x, ℓp, (num_proposals=props,))
    return t, t
end
