
module SliceSampling

using AbstractMCMC
using Accessors
using Distributions
using FillArrays
using LogDensityProblems
using SimpleUnPack
using Random

# reexports
using AbstractMCMC: sample, MCMCThreads, MCMCDistributed, MCMCSerial
export sample, MCMCThreads, MCMCDistributed, MCMCSerial

export Slice, SliceSteppingOut, SliceDoublingOut

abstract type AbstractSliceSampling <: AbstractMCMC.AbstractSampler end

# Univariate Slice Sampling Algorithms

abstract type AbstractGibbsSliceSampling <: AbstractSliceSampling end

function accept_slice_proposal end

function find_interval end

include("interface.jl")
include("gibbs.jl")
include("univariate.jl")
include("steppingout.jl")
include("doublingout.jl")

# Latent Slice Sampling 
export LatentSlice

include("latent.jl")

end
