
# General Usage

This package implements the `AbstractMCMC` [interface](https://github.com/TuringLang/AbstractMCMC.jl).
`AbstractMCMC` provides a unifying interface for MCMC algorithms applied to [LogDensityProblems](https://github.com/tpapp/LogDensityProblems.jl).

## Drawing Samples From `Turing` Models
`SliceSampling.jl` can be used to sample from [Turing](https://github.com/TuringLang/Turing.jl) models through `Turing`'s `externalsampler` interface.
See the following example of using the [latent slice sampler](@ref latent):

```@example turing
using Distributions
using Turing
using SliceSampling

@model function demo()
    s ~ InverseGamma(3, 3)
    m ~ Normal(0, sqrt(s))
end

sampler   = LatentSlice(2)
n_samples = 10000
model     = demo()
sample(model, externalsampler(sampler), n_samples; initial_params=[1.0, 0.0])
```

## Drawing Samples

For drawing samples using the algorithms provided by `SliceSampling`, the user only needs to call:
```julia
sample([rng,] model, slice, N; initial_params)
```
- `slice::AbstractSliceSampling`: Any slice sampling algorithm provided by `SliceSampling`.
- `model`: A model implementing the `LogDensityProblems` interface.
- `N`: The number of samples

The output is a `SliceSampling.Transition` object, which contains the following:
```@docs
SliceSampling.Transition
```


For the keyword arguments, `SliceSampling` allows:
- `initial_params`: The intial state of the Markov chain (default: `nothing`).

If `initial_params` is `nothing`, the following function can be implemented to provide an initialization:
```@docs
SliceSampling.initial_sample
```

## Performing a Single Transition 
For more fined-grained control, the user can call `AbstractMCMC.step`.
That is, the chain can be initialized by calling:
```julia
param, state = AbstractMCMC.steps([rng,] model, slice; initial_params)
```
and then each MCMC transition on `state` can be performed by calling:
```julia
param, state = AbstractMCMC.steps([rng,] model, slice, state)
```

For more details, refer to the documentation of `AbstractMCMC`.
