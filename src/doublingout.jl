
struct SliceDoublingOut{Win <: AbstractVector} <: AbstractGibbsSliceSampling
    max_doubling_out::Int
    window          ::Win
end

Base.show(io::IO, sampler::SliceDoublingOut) =
    print(io, "SliceDoublingOut(max_doubling_out=$(sampler.max_doubling_out))")

SliceDoublingOut(window::AbstractVector) = SliceDoublingOut(8, window)

function find_interval(
    rng  ::Random.AbstractRNG,
    alg  ::SliceDoublingOut,
    model,
    w    ::Real,
    ℓy   ::Real,
    θ₀   ::F,
) where {F <: Real}
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

function accept_slice_proposal(
         ::SliceDoublingOut,
    model,
    w    ::Real,
    ℓy   ::Real,
    θ₀   ::Real,
    θ₁   ::Real,
    L    ::Real,
    R    ::Real,
) 
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
