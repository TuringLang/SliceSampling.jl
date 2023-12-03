
abstract type AbstractSliceSampling <: AbstractMCMC.AbstractSampler end

struct SliceDoublingOut{Win <: AbstractVector} <: AbstractSliceSampling
    max_doubling_out::Int
    window          ::Win
end

SliceDoublingOut(window::Real) = SliceDoublingOut(8, window)

struct SliceSteppingOut{Win <: AbstractVector} <: AbstractSliceSampling
    max_stepping_out::Int
    window          ::Win
end

SliceSteppingOut(window::Real) = SliceSteppingOut(32, window)

struct Slice{Win <: AbstractVector} <: AbstractSliceSampling
    window::Win
end

function find_interval(
    rng::Random.AbstractRNG,
       ::Slice,
       ::GibbsObjective,
    w  ::Real,
       ::Real,
    θ₀ ::F,
) where {F <: Real}
    u = rand(rng, F)
    L = θ₀ - w*u
    R = L + w
    L, R, 0
end

function find_interval(
    rng  ::Random.AbstractRNG,
    alg  ::SliceDoublingOut,
    model::GibbsObjective,
    w    ::Real,
    ℓy   ::Real,
    θ₀   ::F,
) where {F <: Real}
    #=
        Doubling out procedure for finding a slice
        (An acceptance rate < 1e-4 is treated as a potential infinite loop)

        Radford M. Neal,  
        "Slice Sampling," 
        Annals of Statistics, 2003.
    =##
    p = alg.max_doubling_out

    u = rand(rng, F)
    L = θ₀ - w*u
    R = L + w

    ℓπ_L = LogDensityProblems.logdensity(model, L)
    ℓπ_R = LogDensityProblems.logdensity(model, R)
    K    = 2

    for _ = 1:p
        if ((ℓy ≥ ℓπ_L) && (ℓy ≥ ℓπ_R))
            break
        end
        v = rand(rng, F)
        if v < 0.5
            L    = L - (R - L)
            ℓπ_L = LogDensityProblems.logdensity(model, L)
        else
            R    = R + (R - L)
            ℓπ_R = LogDensityProblems.logdensity(model, R)
        end
        K += 1
    end
    L, R, K
end

function find_interval(
    rng  ::Random.AbstractRNG,
    alg  ::SliceSteppingOut,
    model::GibbsObjective,
    w    ::Real,
    ℓy   ::Real,
    θ₀   ::F,
) where {F <: Real}
    #=
        Stepping out procedure for finding a slice
        (An acceptance rate < 1e-4 is treated as a potential infinite loop)

        Radford M. Neal,  
        "Slice Sampling," 
        Annals of Statistics, 2003.
    =##
    m = alg.max_stepping_out

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

accept_slice_proposal(
    ::AbstractSliceSampling,
    ::GibbsObjective,
    ::Real,
    ::Real,
    ::Real,
    ::Real,
    ::Real,
    ::Real,
) = true

function accept_slice_proposal(
         ::SliceDoublingOut,
    model::GibbsObjective,
    w    ::Real,
    ℓy   ::Real,
    θ₀   ::Real,
    θ₁   ::Real,
    L    ::Real,
    R    ::Real,
) 
    #=
        acceptance rule for the doubling procedure

        Radford M. Neal,  
        "Slice Sampling," 
        Annals of Statistics, 2003.
    =##
    D    = false
    ℓπ_L = LogDensityProblems.logdensity(model, L)
    ℓπ_R = LogDensityProblems.logdensity(model, R)

    while R - L > 1.1*w
        M = (L + R)/2
        if (θ₀ < M && θ₁ ≥ M) || (θ₀ ≥ M && θ₁ < M)
            D = true
        end

        if θ₁ < M
            R    = M
            ℓπ_R = LogDensityProblems.logdensity(model, R)
        else
            L    = M
            ℓπ_L = LogDensityProblems.logdensity(model, L)
        end

        if D && ℓy ≥ ℓπ_L && ℓy ≥ ℓπ_R
            return false
        end
    end
    true
end

function slice_sampling_univariate(
    rng  ::Random.AbstractRNG,
    alg  ::AbstractSliceSampling,
    model::GibbsObjective, 
    w    ::Real,
    ℓπ₀  ::Real,
    θ₀   ::F,
) where {F <: Real}
    #=
        Univariate slice sampling kernel
        (An acceptance rate < 1e-4 is treated as a potential infinite loop)

        Radford M. Neal,  
        "Slice Sampling," 
        Annals of Statistics, 2003.
    =##
    u  = rand(rng, F)
    ℓy = log(u) + ℓπ₀

    L, R, n_prop = find_interval(rng, alg, model, w, ℓy, θ₀)

    while true
        U   = rand(rng, F)
        θ′  = L + U*(R - L)
        ℓπ′ = LogDensityProblems.logdensity(model, θ′)

        n_prop += 1

        if (ℓy < ℓπ′) && accept_slice_proposal(alg, model, w, ℓy, θ₀, θ′, L, R)
            return θ′, ℓπ′, 1/n_prop
        end

        if θ′ < θ₀
            L = θ′
        else
            R = θ′
        end
    end
end

