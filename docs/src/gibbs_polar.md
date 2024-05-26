
# [Gibbsian Polar Slice Sampling](@id polar)

## Introduction
Gibbsian polar slice sampling (GPSS) is a recent vector-valued slice sampling algorithm proposed by P. Schär, M. Habeck, and D. Rudolf[^SHR2023].
It is an computationally efficient variant of polar slice sampler previously proposed by Roberts and Rosenthal[^RR2002].
Unlike other slice sampling algorithms, it operates a Gibbs sampler over polar coordinates, reminiscent of the elliptical slice sampler (ESS).
Due to the involvement of polar coordinates, GPSS only works reliably on more than one dimension.
However, unlike ESS, GPSS is applicable to any target distribution.


## Description
For a $$d$$-dimensional target distribution, GPSS utilizes the following augmented target distribution:
```math
\begin{aligned}
    p(x, T)      &= \varrho_{\pi}^{(0)}(x) \varrho_{\pi}^{(1)}(x) \, \operatorname{Uniform}\left(T; 0, \varrho^1(x)\right) \\
    \varrho_{\pi}^{(0)}(x) &= {\lVert x \rVert}^{1 - d} \\
    \varrho_{\pi}^{(1)}(x) &= {\lVert x \rVert}^{d-1} \pi\left(x\right)
\end{aligned}
```
As described in Appendix A of the GPSS paper, sampling from $$\varrho^{(1)}(x)$$ in polar coordinates magically targets the augmented target distribution.

In a high-level view, GPSS operates a Gibbs sampler in the following fashion:
```math
\begin{aligned}
T_n      &\sim \operatorname{Uniform}\left(0, \varrho^{(1)}\left(x_{n-1}\right)\right) \\
\theta_n &\sim \operatorname{Uniform}\left\{ \theta \in \mathbb{S}^{d-1} \mid \varrho^{(1)}\left(r_{n-1} \theta\right) > T_n \right\} \\
r_n      &\sim \operatorname{Uniform}\left\{ r \in \mathbb{R}_{\geq 0} \mid \varrho^{(1)}\left(r \theta_n\right) > T_n \right\} \\
x      &= \theta r,
\end{aligned}
```
where $$T_n$$ is the usual acceptance threshold auxiliary variable, while $$\theta$$ and $$r$$ are the sampler states in polar coordinates.
The Gibbs steps on $$\theta$$ and $$r$$ are implemented through specialized shrinkage procedures.

The only tunable parameter of the algorithm is the size of the search interval (window) of the shrinkage sampler for the radius variable $$r$$.

!!! info
    Since the direction and radius variables are states of the Markov chain, this sampler is **not reversible** with respect to the samples of the log-target $$x$$.
	
## Interface

!!! warning
    By the nature of polar coordinates, GPSS only works reliably for targets with dimension at least $$d \geq 2$$.

!!! warning
    When initializing the chain (*e.g.* the `initial_params` keyword arguments in `AbstractMCMC.sample`), it is necessary to inialize from a point $$x_0$$ that has a sensible norm $$\lVert x_0 \rVert > 0$$, otherwise, the chain will start from a pathologic point in polar coordinates. If it is smaller than `1e-5`, the current implementation automatically sets the initial radius as `1e-5`.


```@docs
GibbsPolarSlice
```

## Demonstration
As illustrated in the original paper, GPSS shows good performance on heavy-tailed targets despite being a multivariate slice sampler:
```@example gpss
using Distributions
using Turing
using SliceSampling
using LinearAlgebra
using Plots

@model function demo()
    x ~ MvTDist(1, zeros(10), Matrix(I,10,10))
end
model = demo()

n_samples = 10000
chain = sample(model, externalsampler(GibbsPolarSlice(10)), n_samples; initial_params=ones(10))
histogram(chain[:,1,:], xlims=[-10,10])
savefig("cauchy_gpss.svg")
```
![](cauchy_gpss.svg)


[^SHR2023]: Schär, P., Habeck, M., & Rudolf, D. (2023, July). Gibbsian polar slice sampling. In International Conference on Machine Learning.
[^RR2002]: Roberts, G. O., & Rosenthal, J. S. (2002). The polar slice sampler. Stochastic Models, 18(2), 257-280.
