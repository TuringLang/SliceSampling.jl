
module SliceSampling

using AbstractMCMC
using LogDensityProblems
using Random

# reexports
using AbstractMCMC: sample, MCMCThreads, MCMCDistributed, MCMCSerial
export sample, MCMCThreads, MCMCDistributed, MCMCSerial

include("interface.jl")
include("slice.jl")
include("latentslice.jl")

end
