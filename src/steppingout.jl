

"""
    SliceSteppingOut(max_stepping_out, window)
    SliceSteppingOut(window)

Univariate slice sampling by automatically adapting the initial interval through the "stepping-out" procedure (Scheme 3 by Neal[^N2003])

# Fields
- `max_stepping_out`: Maximum number of "stepping outs" (default: 32).
- `window::Union{<:Real, <:AbstractVector}`: Proposal window.
"""
struct SliceSteppingOut{W <: Union{<:AbstractVector, <:Real}} <: AbstractGibbsSliceSampling
    max_stepping_out::Int
    window          ::W
end

SliceSteppingOut(window::Union{<:AbstractVector, <:Real}) = SliceSteppingOut(32, window)

function find_interval(
    rng  ::Random.AbstractRNG,
    alg  ::SliceSteppingOut,
    model,
    w    ::Real,
    ℓy   ::Real,
    θ₀   ::F,
) where {F <: Real}
    m      = alg.max_stepping_out
    u      = rand(rng, F)
    L      = θ₀ - w*u
    R      = L + w
    V      = rand(rng, F)
    J      = floor(Int, m*V)
    K      = (m - 1) - J 
    n_eval = 0

    while J > 0 && ℓy < LogDensityProblems.logdensity(model, L)
        L = L - w
        J = J - 1
        n_eval += 1
    end
    while K > 0 && ℓy < LogDensityProblems.logdensity(model, R)
        R = R + w
        K = K - 1
        n_eval += 1
    end
    L, R, n_eval
end
