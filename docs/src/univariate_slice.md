
# Univariate Slice Sampling Algorithms
## Introduction
These algorithms are the "single-variable" slice sampling algorithms originally described by Neal[^N2003].
Since these algorithms are univariate, one has to incorporate them into a "meta" multivariate sampler, which are discussed in [this section](@ref meta).

## Fixed Initial Interval Slice Sampling 
This is the most basic form of univariate slice sampling, where the proposals are generated within a fixed interval formed by the `window`.

```@docs
Slice
```

## Adaptive Initial Interval Slice Sampling

These algorithms try to adaptively set the initial interval through a simple search procedure.
The "stepping-out" procedure grows the initial window on a linear scale, while the "doubling-out" procedure grows it geometrically.
`window` controls the scale of the increase.

### What Should I Use?
This highly depends on the problem at hand.
In general, the doubling-out procedure tends to be more expensive as it requires additional log-target evaluations to decide whether to accept a proposal.
However, if the scale of the posterior varies drastically, doubling out might work better.
In general, it is recommended to use the stepping-out procedure.

```@docs
SliceSteppingOut
SliceDoublingOut
```

[^N2003]: Neal, R. M. (2003). Slice sampling. The Annals of Statistics, 31(3), 705-767.
