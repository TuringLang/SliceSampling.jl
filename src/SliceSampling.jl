
module SliceSampling

using AbstractMCMC
using Accessors
using LogDensityProblems
using SimpleUnPack
using Random

# reexports
using AbstractMCMC: sample, MCMCThreads, MCMCDistributed, MCMCSerial
export sample, MCMCThreads, MCMCDistributed, MCMCSerial

export Slice, SliceSteppingOut, SliceDoublingOut

abstract type AbstractSliceSampling <: AbstractMCMC.AbstractSampler end

abstract type AbstractGibbsSliceSampling <: AbstractSliceSampling end

function accept_slice_proposal end

function find_interval end

include("interface.jl")
include("gibbs.jl")
include("univariate.jl")
include("steppingout.jl")
include("doublingout.jl")


include("latentslice.jl")

end
