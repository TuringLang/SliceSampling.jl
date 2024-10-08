
# [Latent Slice Sampling](@id latent)

## Introduction
Latent slice sampling is a recent vector-valued slice sampling algorithm proposed by Li and Walker[^LW2023].
Unlike other slice sampling algorithms, it treats the "search intervals" as auxiliary variables and adapts them along the samples from the log-target in a Gibbs-type scheme.

## Description
Specifically, the extended joint density of the latent slice sampler is as follows:

```math
    p(x, t, s, l) = \pi(x) \, p(s) \, \operatorname{Uniform}\left(t; 0, \pi\left(x\right)\right) \, \operatorname{Uniform}\left(l; \; x - s/2,\, x + s/2\right),
```
where $$y$$ is the parameters of the log-target $$\pi$$, $$s$$ is the width of the search interval and $$l$$ is the centering of the search interval relative to $$y$$.
Naturally, the sampler operates as a blocked-Gibbs sampler 
```math
\begin{aligned}
l_n &\sim \operatorname{Uniform}\left(l; \; x_{n-1} - s_{n-1}/2,\, x_{n-1} + s_{n-1}/2\right) \\
s_n &\sim p(s \mid x_{n-1}, l_{n}) \\
t_n &\sim \operatorname{Uniform}\left(0, \pi\left(x_{n-1}\right)\right) \\
x_n &\sim \operatorname{Uniform}\left\{x \mid \pi\left(x\right) > t_n\right\},
\end{aligned}
```
When $$x_n$$ is updated using the usual shrinkage procedure of Neal[^N2003], $$s_n$$ and $$l_n$$ are used to form the initial search window.
($$s_n$$ is the width of the window and $$l_n$$ is its center point.)
Therefore, the latent slice sampler can be regarded as an automatic tuning mechanism of the "initial interval" of slice samplers.

The only tunable parameter of the algorithm is then the distribution of the width $$p(s)$$.
For this, Li and Walker recommend
```math
    p(s; \beta) = \operatorname{Gamma}(s; 2, \beta),
```
where $$\beta$$ is a tunable parameter.
The use of the gamma distribution is somewhat important since the complete conditional $$p(s \mid y, l)$$ needs to be available in closed-form for efficiency. 
(It is a truncated exponential distribution in case of the gamma.)
Therefore, we only provide control over $$\beta$$.

!!! info
    The kernel corresponding to this sampler is defined on an **augmented state space** and cannot directly perform a transition on $$x$$.
    This also means that the corresponding kernel is not reversible with respect to $$x$$.
	
## Interface


```@docs
LatentSlice
```

[^LW2023]: Li, Y., & Walker, S. G. (2023). A latent slice sampling algorithm. Computational Statistics & Data Analysis, 179, 107652.
[^N2003]: Neal, R. M. (2003). Slice sampling. The annals of statistics, 31(3), 705-767.
