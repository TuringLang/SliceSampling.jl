
struct GibbsSliceState{T <: Transition}
    "Current [`Transition`](@ref)."
    transition::T
end

struct GibbsTarget{Model, Idx <: Integer, Vec <: AbstractVector}
    model::Model
    idx  ::Idx
    θ    ::Vec
end

function LogDensityProblems.logdensity(gibbs::GibbsTarget, θi)
    @unpack model, idx, θ = gibbs
    LogDensityProblems.logdensity(model, (@set θ[idx] = θi))
end
