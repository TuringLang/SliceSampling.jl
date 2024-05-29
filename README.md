# Slice Sampling Algorithms in Julia

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://TuringLang.org/SliceSampling.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://TuringLang.org/SliceSampling.jl/dev/)
[![Build Status](https://github.com/TuringLang/SliceSampling.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Red-Portal/SliceSampling.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/TuringLang/SliceSampling.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Red-Portal/SliceSampling.jl)

This package implements slice sampling algorithms accessible through the `AbstractMCMC` [interface](https://github.com/TuringLang/AbstractMCMC.jl).
For general usage, please refer to [here](https://turinglang.org/SliceSampling.jl/dev/general/).

## Implemented Algorithms
### Univariate Slice Sampling Algorithms
- Univariate slice sampling ([Slice](https://turinglang.org/SliceSampling.jl/dev/univariate_slice/)) algorithms with coordinate-wise Gibbs sampling by R. Neal [^N2003].

### Univariate-to-Multivariate Strategies
- Random Permutation coordinate-wise Gibbs sampling
- Hit-and-run sampling

### Multivariate Slice Sampling Algorithms
- Latent slice sampling ([LSS](https://turinglang.org/SliceSampling.jl/dev/latent_slice/)) by Li and Walker[^LW2023]
- Gibbsian polar slice sampling ([GPSS](https://turinglang.org/SliceSampling.jl/dev/gibbs_polar/)) by P. Schär, M. Habeck, and D. Rudolf[^SHR2023].

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
sample(model, externalsampler(sampler), n_samples; initial_params=[1.0, 0.0])
```

[^N2003]: Neal, R. M. (2003). Slice sampling. The annals of statistics, 31(3), 705-767.
[^LW2023]: Li, Y., & Walker, S. G. (2023). A latent slice sampling algorithm. Computational Statistics & Data Analysis, 179, 107652.
[^SHR2023]: Schär, P., Habeck, M., & Rudolf, D. (2023, July). Gibbsian polar slice sampling. In International Conference on Machine Learning.


