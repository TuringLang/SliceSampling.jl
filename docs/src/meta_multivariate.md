
# [Meta Multivariate Samplers](@id meta)
To use univariate slice sampling strategies on targets with more than on dimension, one has to embed them into a "meta" multivariate sampling scheme that relies on univariate sampling elements.
The two most popular approaches for this are Gibbs sampling[^GG1984] and hit-and-run[^BRS1993].

## Random Permutation Gibbs 
Gibbs sampling[^GG1984] is a strategy where we sample from the posterior one coordinate at a time, conditioned on the values of all other coordinates.
In practice, one can pick the coordinates in any order they want as long as it does not depend on the state of the chain.
It is generally hard to know a-prior which "scan order" is best, but randomly picking coordinates tend to work well in general.
Currently, we only provide random permutation scan, which guarantees that all coordinates are updated at least once after $$d$$ transitions.
At the same time, reversibility is maintained by randomly permuting the order we go through each coordinate:
```@docs
RandPermGibbs
```
Each call to `AbstractMCMC.step` internally performs $$d$$ Gibbs transition so that all coordinates are updated.

For example:
```julia
RandPermGibbs(SliceSteppingOut(2.))
```

If one wants to use a different slice sampler configuration for each coordinate, one can mix-and-match by passing a `Vector` of slice samplers, one for each coordinate.
For instance, for a 2-dimensional target:
```julia
RandPermGibbs([SliceSteppingOut(2.; max_proposals=32), SliceDoublingOut(2.),])
```

## Hit-and-Run 
Hit-and-run is a simple meta algorithm where we sample over a random 1-dimensional projection of the space.
That is, at each iteration, we sample a random direction
```math
    \theta_n \sim \operatorname{Uniform}(\mathbb{S}^{d-1}),
```
and perform a Markov transition along the 1-dimensional subspace
```math
\begin{aligned}
    \lambda_n &\sim p\left(\lambda \mid x_{n-1}, \theta_n \right) \propto \pi\left( x_{n-1} + \lambda \theta_n \right) \\
    x_{n} &= x_{n-1} + \lambda_n \theta_n,
\end{aligned}
```
where $$\pi$$ is the target unnormalized density.
Applying slice sampling for the 1-dimensional subproblem has been popularized by David Mackay[^M2003], and is, technically, also a Gibbs sampler. 
(Or is that Gibbs samplers are hit-and-run samplers?)
Unlike `RandPermGibbs`, which only makes axis-aligned moves, `HitAndRun` can choose arbitrary directions, which could be helpful in some cases.

```@docs
HitAndRun
```

This can be used, for example, as follows:

```julia
HitAndRun(SliceSteppingOut(2.))
```
Unlike `RandPermGibbs`, `HitAndRun` does not provide the option of using a unique `unislice` object for each coordinate.
This is a natural limitation of the hit-and-run sampler: it does not operate on individual coordinates.

[^GG1984]: Geman, S., & Geman, D. (1984). Stochastic relaxation, Gibbs distributions, and the Bayesian restoration of images. IEEE Transactions on Pattern Analysis and Machine Intelligence, (6).
[^BRS1993]: Bélisle, C. J., Romeijn, H. E., & Smith, R. L. (1993). Hit-and-run algorithms for generating multivariate distributions. Mathematics of Operations Research, 18(2), 255-266.
[^M2003]: MacKay, D. J. (2003). Information theory, inference and learning algorithms. Cambridge university press.
