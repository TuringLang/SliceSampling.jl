# Slice Sampling Algorithms in Julia

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://TuringLang.org/SliceSampling.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://TuringLang.org/SliceSampling.jl/dev/)
[![Build Status](https://github.com/TuringLang/SliceSampling.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/TuringLang/SliceSampling.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/TuringLang/SliceSampling.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/TuringLang/SliceSampling.jl)

This package implements slice sampling algorithms accessible through the `AbstractMCMC` [interface](https://github.com/TuringLang/AbstractMCMC.jl).
For general usage, please refer to [here](https://turinglang.org/SliceSampling.jl/dev/general/).

## Implemented Algorithms
### Univariate Slice Sampling Algorithms
- Univariate slice sampling ([Slice](https://turinglang.org/SliceSampling.jl/dev/univariate_slice/)) algorithms by R. Neal [^N2003]:
  - Fixed window (`Slice`)
  - stepping-out window adaptation (`SliceSteppingOut`)
  - doubling-out window adaptation (`SliceDoublingOut`)

### Meta Multivariate Samplers for Augmenting Univariate Samplers
- Random permutation coordinate-wise Gibbs sampling[^GG1984] (`RandPermGibbs`)
- Hit-and-run sampling[^BRS1993] (`HitAndRun`)

### Multivariate Slice Sampling Algorithms
- Latent slice sampling ([LSS](https://turinglang.org/SliceSampling.jl/dev/latent_slice/)) by Li and Walker[^LW2023] (`LatentSlice`)
- Gibbsian polar slice sampling ([GPSS](https://turinglang.org/SliceSampling.jl/dev/gibbs_polar/)) by P. Schär, M. Habeck, and D. Rudolf[^SHR2023] (`GibbsPolarSlice`)

## Example with Turing Models
This package supports the [Turing](https://github.com/TuringLang/Turing.jl) probabilistic programming framework:

```julia
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

The following slice samplers can also be used as a conditional sampler in `Turing.Gibbs` sampler:
* For multidimensional variables: 
  * `RandPermGibbs`
  * `HitAndRun`
* For unidimensional variables: 
  * `Slice`
  * `SliceSteppingOut`
  * `SliceDoublingOut`

See the following example:
```julia
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

[^N2003]: Neal, R. M. (2003). Slice sampling. The annals of statistics, 31(3), 705-767.
[^LW2023]: Li, Y., & Walker, S. G. (2023). A latent slice sampling algorithm. Computational Statistics & Data Analysis, 179, 107652.
[^SHR2023]: Schär, P., Habeck, M., & Rudolf, D. (2023, July). Gibbsian polar slice sampling. In International Conference on Machine Learning.
[^GG1984]: Geman, S., & Geman, D. (1984). Stochastic relaxation, Gibbs distributions, and the Bayesian restoration of images. IEEE Transactions on Pattern Analysis and Machine Intelligence, (6).
[^BRS1993]: Bélisle, C. J., Romeijn, H. E., & Smith, R. L. (1993). Hit-and-run algorithms for generating multivariate distributions. Mathematics of Operations Research, 18(2), 255-266.
