
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

# Interfaces
abstract type AbstractSliceSampling <: AbstractMCMC.AbstractSampler end

"""
    initial_sample(rng, model)

Return the initial sample for the `model` using the random number generator `rng`.

# Arguments
- `rng::Random.AbstractRNG`: Random number generator.
- `model`: the model of interest.
"""
function initial_sample end

# Univariate Slice Sampling Algorithms
export Slice, SliceSteppingOut, SliceDoublingOut

abstract type AbstractGibbsSliceSampling <: AbstractSliceSampling end

function accept_slice_proposal end

function find_interval end

include("gibbs.jl")
include("univariate.jl")
include("steppingout.jl")
include("doublingout.jl")

# Latent Slice Sampling 
export LatentSlice

include("latent.jl")

end
