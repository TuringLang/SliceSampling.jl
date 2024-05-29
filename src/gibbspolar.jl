
"""
    GibbsPolarSlice(w; max_proposals)

Gibbsian polar slice sampling algorithm by P. Schär, M. Habeck, and D. Rudolf [^SHR2023].

# Arguments
- `w::Real`: Initial window size for the radius shrinkage procedure.

# Keyword Arguments
- `w::Real`: Initial window size for the radius shrinkage procedure
- `max_proposals::Int`: Maximum number of proposals allowed until throwing an error (default: `typemax(Int)`).
"""
struct GibbsPolarSlice{W <: Real} <: AbstractSliceSampling
    w::W
    max_proposals::Int
end

GibbsPolarSlice(w::Real; max_proposals::Int = typemax(Int)) = GibbsPolarSlice(w, max_proposals)

struct GibbsPolarSliceState{T <: Transition, R <: Real, D <: AbstractVector} 
    "Current [`Transition`](@ref)."
    transition ::T

    "direction (\$\\theta\$ in the original paper[^SHR2023])"
    direction::D

    "radius (\$r\$ in the original paper[^SHR2023])"
    radius::R
end

struct GibbsPolarSliceTarget{M}
    model::M
end

function logdensity(target::GibbsPolarSliceTarget, x)
    d = length(x)
    (d-1)*log(norm(x)) + LogDensityProblems.logdensity(target.model, x)
end

function AbstractMCMC.step(rng    ::Random.AbstractRNG,
                           model  ::AbstractMCMC.LogDensityModel,
                           sampler::GibbsPolarSlice;
                           initial_params = nothing,
                           kwargs...)
    logdensitymodel = model.logdensity
    x  = initial_params === nothing ? initial_sample(rng, model) : initial_params
    d  = length(x)
    @assert d ≥ 2 "Gibbsian polar slice sampling works reliably only in dimension ≥2"
    r  = norm(x)
    if r < 1e-5
        @warn "The norm of initial_params is smaller than 1e-5, which might be result in unstable behavior and the sampler might even get stuck indefinitely. If you are using Turing, this might be due to change of support through Bijectors."
    end
    θ  = x / r
    ℓp = LogDensityProblems.logdensity(logdensitymodel, x)
    t  = Transition(x, ℓp, NamedTuple())
    return t, GibbsPolarSliceState(t, θ, r)
end

function rand_subsphere(rng::Random.AbstractRNG, θ::AbstractVector)
    d  = length(θ)
    V1 = randn(rng, eltype(θ), d)
    V2 = V1 - dot(θ, V1)*θ
    V2 / max(norm(V2), eps(eltype(θ)))
end

function geodesic_shrinkage(
    rng     ::Random.AbstractRNG,
    ϱ1      ::GibbsPolarSliceTarget,
    ℓT      ::F,
    θ       ::AbstractVector{F},
    r       ::F,
    max_prop::Int
) where {F <: Real}
    y     = rand_subsphere(rng, θ)   
    ω_max = convert(F, 2π)*rand(rng, F)
    ω_min = ω_max - convert(F, 2π)

    for n_props in 1:max_prop
        # `Uniform` had a type instability issue:
        # https://github.com/JuliaStats/Distributions.jl/pull/1860
        # ω = rand(rng, Uniform(ω_min, ω_max))
        ω = ω_min + (ω_max - ω_min)*rand(rng, F)
        θ′ = θ*cos(ω) + y*sin(ω)

        if logdensity(ϱ1, r*θ′) > ℓT
            return θ′, n_props
        end

        if ω < 0
            ω_min = ω
        else
            ω_max = ω
        end
    end
    exceeded_max_prop(max_prop)
end

function radius_shrinkage(
    rng     ::Random.AbstractRNG,
    ϱ1      ::GibbsPolarSliceTarget,
    ℓT      ::F,
    θ       ::AbstractVector{F},
    r       ::F,
    w       ::Real,
    max_prop::Int
) where {F <: Real}
    u             = rand(rng, F)
    w             = convert(F, w)
    r_min         = max(r - u*w, 0)
    r_max         = r + (1 - u)*w
    n_props_total = 0

    n_props = 0
    while (r_min > 0) && logdensity(ϱ1, r_min*θ) > ℓT
        r_min    = max(r_min - w, 0)

        n_props += 1
        if n_props > max_prop
            exceeded_max_prop(max_prop)
        end
    end
    n_props_total += n_props

    n_props = 0
    while logdensity(ϱ1, r_max*θ) > ℓT
        r_max = r_max + w

        n_props += 1
        if n_props > max_prop
            exceeded_max_prop(max_prop)
        end
    end
    n_props_total += n_props

    for n_props in 1:max_prop
        # `Uniform` had a type instability issue:
        # https://github.com/JuliaStats/Distributions.jl/pull/1860
        #r′ = rand(rng, Uniform{F}(r_min, r_max))
        r′ = r_min + (r_max - r_min)*rand(rng, F)

        if logdensity(ϱ1, r′*θ) > ℓT
            n_props_total += n_props
            return r′, n_props_total
        end

        if r′ < r      
            r_min = r′
        else
            r_max = r′
        end
    end
    exceeded_max_prop(max_prop)
end

function AbstractMCMC.step(
    rng    ::Random.AbstractRNG,
    model  ::AbstractMCMC.LogDensityModel, 
    sampler::GibbsPolarSlice,
    state  ::GibbsPolarSliceState;
    kwargs...,
)
    logdensitymodel = model.logdensity
    max_prop        = sampler.max_proposals

    x  = state.transition.params
    ℓp = state.transition.lp
    w  = sampler.w
    r  = state.radius
    θ  = state.direction
    ϱ1 = GibbsPolarSliceTarget(logdensitymodel)

    d  = length(x)
    ℓT = ((d-1)*log(norm(x)) + ℓp) - Random.randexp(rng, eltype(ℓp))

    θ, n_props_θ = geodesic_shrinkage(rng, ϱ1, ℓT, θ, r,    max_prop)
    r, n_props_r = radius_shrinkage(  rng, ϱ1, ℓT, θ, r, w, max_prop)
    x            = θ*r

    ℓp = LogDensityProblems.logdensity(logdensitymodel, x)
    t  = Transition(x, ℓp, (
        num_radius_proposals    = n_props_r,
        num_direction_proposals = n_props_θ,
    ))
    t, GibbsPolarSliceState(t, θ, r)
end
