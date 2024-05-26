# Implementation of slice sampling algorithms

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://Red-Portal.github.io/SliceSampling.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://Red-Portal.github.io/SliceSampling.jl/dev/)
[![Build Status](https://github.com/Red-Portal/SliceSampling.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Red-Portal/SliceSampling.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/Red-Portal/SliceSampling.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Red-Portal/SliceSampling.jl)

This package implements slice sampling algorithms accessible through the the `AbstractMCMC` [interface](https://github.com/TuringLang/AbstractMCMC.jl).

## Implemented Algorithms
- Univariate slice sampling algorithms with coordinate-wise Gibbs sampling by R. Neal [^N2003].
- Latent slice sampling by Li and Walker[^LW2023]
- Gibbsian polar slice sampling by P. Schär, M. Habeck, and D. Rudolf[^SHR2023].

## Example with Turing Models
This package supports the [Turing](https://github.com/TuringLang/Turing.jl) probabilistic programming framework:

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

[^N2003]: Neal, R. M. (2003). Slice sampling. The annals of statistics, 31(3), 705-767.
[^LW2023]: Li, Y., & Walker, S. G. (2023). A latent slice sampling algorithm. Computational Statistics & Data Analysis, 179, 107652.
[^SHR2023]: Schär, P., Habeck, M., & Rudolf, D. (2023, July). Gibbsian polar slice sampling. In International Conference on Machine Learning.
=======
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://TuringLang.org/SliceSampling.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://TuringLang.org/SliceSampling.jl/dev/)
[![Build Status](https://github.com/TuringLang/SliceSampling.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Red-Portal/SliceSampling.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/TuringLang/SliceSampling.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Red-Portal/SliceSampling.jl)


For a working example, please see [here](https://turinglang.org/SliceSampling.jl/dev/general/).
