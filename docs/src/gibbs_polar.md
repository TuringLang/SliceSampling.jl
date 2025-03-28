
# [Gibbsian Polar Slice Sampling](@id polar)

## Introduction
Gibbsian polar slice sampling (GPSS) is a multivariate slice sampling algorithm proposed by P. Schär, M. Habeck, and D. Rudolf[^SHR2023].
It is an computationally efficient variant of polar slice sampler previously proposed by Roberts and Rosenthal[^RR2002].
Unlike other slice sampling algorithms, it operates a Gibbs sampler over polar coordinates, reminiscent of the elliptical slice sampler (ESS).
Due to the involvement of polar coordinates, GPSS only works reliably on more than one dimension.
However, unlike ESS, GPSS is applicable to any target distribution.

## Description
For a $$d$$-dimensional target distribution $$\pi$$, GPSS utilizes the following augmented target distribution:
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
x_n      &= \theta_n r_n,
\end{aligned}
```
where $$T_n$$ is the usual acceptance threshold auxiliary variable, while $$\theta$$ and $$r$$ are the sampler states in polar coordinates.
The Gibbs steps on $$\theta$$ and $$r$$ are implemented through specialized shrinkage procedures.

The only tunable parameter of the algorithm is the size of the search interval (window) of the shrinkage sampler for the radius variable $$r$$.

!!! warning 
    A limitation of the current implementation of GPSS is that the acceptance rate exhibits a heavy tail. That is, occasionally, a single transition might take an excessive amount of time.

!!! info
    The kernel corresponding to this sampler is defined on an **augmented state space** and cannot directly perform a transition on $$x$$.
    This also means that the corresponding kernel is not reversible with respect to $$x$$.
	
## Interface

```@docs
GibbsPolarSlice
```

## Demonstration
As illustrated in the original paper, GPSS shows good performance on heavy-tailed targets despite being a multivariate slice sampler.
Consider a 10-dimensional Student-$$t$$ target with 1-degree of freedom (this corresponds to a multivariate Cauchy):

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

n_samples = 1000
latent_chain = sample(model, externalsampler(LatentSlice(10)), n_samples; initial_params=ones(10))
polar_chain = sample(model, externalsampler(GibbsPolarSlice(10)), n_samples; initial_params=ones(10))

l = @layout [a; b]
p1 = Plots.plot(1:n_samples, latent_chain[:,1,:], ylims=[-10,10], label="LSS")
p2 = Plots.plot(1:n_samples, polar_chain[:,1,:],  ylims=[-10,10], label="GPSS")
plot(p1, p2, layout = l)
savefig("student_latent_gpss.svg")
```
![](student_latent_gpss.svg)

Clearly, GPSS is better at exploring the deep tails compared to the [latent slice sampler](@ref latent) (LSS) despite having a similar per-iteration cost.


[^SHR2023]: Schär, P., Habeck, M., & Rudolf, D. (2023, July). Gibbsian polar slice sampling. In International Conference on Machine Learning.
[^RR2002]: Roberts, G. O., & Rosenthal, J. S. (2002). The polar slice sampler. Stochastic Models, 18(2), 257-280.
