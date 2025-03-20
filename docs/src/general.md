
# General Usage

This package implements the `AbstractMCMC` [interface](https://github.com/TuringLang/AbstractMCMC.jl).
`AbstractMCMC` provides a unifying interface for MCMC algorithms applied to [LogDensityProblems](https://github.com/tpapp/LogDensityProblems.jl).

## Examples
### Drawing Samples From a `LogDensityProblems` Through `AbstractMCMC`
`SliceSampling.jl` implements the [`AbstractMCMC`](https://github.com/TuringLang/AbstractMCMC.jl) interface through [`LogDensityProblems`](https://github.com/tpapp/LogDensityProblems.jl).
That is, one simply needs to define a `LogDensityProblems` and pass it to `AbstractMCMC`:

```@example logdensityproblems
using AbstractMCMC
using Distributions
using LinearAlgebra
using LogDensityProblems
using Plots

using SliceSampling

struct Target{D}
	dist::D
end

LogDensityProblems.logdensity(target::Target, x) = logpdf(target.dist, x)

LogDensityProblems.dimension(target::Target) = length(target.distx)

LogDensityProblems.capabilities(::Type{<:Target}) = LogDensityProblems.LogDensityOrder{0}()

sampler         = GibbsPolarSlice(2.0)
n_samples       = 10000
model           = Target(MvTDist(5, zeros(10), Matrix(I, 10, 10)))
logdensitymodel = AbstractMCMC.LogDensityModel(model)

chain   = sample(logdensitymodel, sampler, n_samples; initial_params=randn(10))
samples = hcat([transition.params for transition in chain]...)

plot(samples[1,:], xlabel="Iteration", ylabel="Trace")
savefig("abstractmcmc_demo.svg")
```
![](abstractmcmc_demo.svg)

### Drawing Samples From `Turing` Models
`SliceSampling.jl` can also be used to sample from [Turing](https://github.com/TuringLang/Turing.jl) models through `Turing`'s `externalsampler` interface:

```@example turing
using Distributions
using Turing
using SliceSampling

@model function demo()
    s ~ InverseGamma(3, 3)
    m ~ Normal(0, sqrt(s))
end

sampler   = RandPermGibbs(SliceSteppingOut(2.))
n_samples = 10000
model     = demo()
sample(model, externalsampler(sampler), n_samples)
```

### Conditional sampling in a `Turing.Gibbs` sampler
`SliceSampling.jl` be used as a conditional sampler in `Turing.Gibbs`.

```@example turinggibbs
using Distributions
using Turing
using SliceSampling

@model function simple_choice(xs)
    p ~ Beta(2, 2)
    z ~ Bernoulli(p)
    for i in 1:length(xs)
        if z == 1
            xs[i] ~ Normal(0, 1)
        else
            xs[i] ~ Normal(2, 1)
        end
    end
end

sampler = Turing.Gibbs(
    :p => externalsampler(SliceSteppingOut(2.0)),
    :z => PG(20),
)

n_samples = 1000
model     = simple_choice([1.5, 2.0, 0.3])
sample(model, sampler, n_samples)
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
transition, state = AbstractMCMC.steps([rng,] model, slice; initial_params)
```
and then each MCMC transition on `state` can be performed by calling:
```julia
transition, state = AbstractMCMC.steps([rng,] model, slice, state)
```
For more details, refer to the documentation of `AbstractMCMC`.
