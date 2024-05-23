
# [Latent Slice Sampling](@id latent)

## Introduction
Latent slice sampling is a recent vector-valued slice sampling algorithm proposed by Li and Walker[^LW2023].
Unlike other slice sampling algorithms, it treats the "search intervals" as auxiliary variables and adapts them along the samples from the log-target in a Gibbs-type scheme.

## Description
Specifically, the extended joint density of the latent slice sampler is as follows:

```math
    p(y, s, l) = \pi(y) \, p(s) \, \operatorname{Uniform}\left(l; \; y - s/2,\, y + s/2\right),
```
where $$y$$ is the parameters of the log-target $$\pi$$, $$s$$ is the width of the search interval and $$l$$ is the centering of the search interval relative to $$y$$.
Naturally, the sampler operates as a blocked-Gibbs sampler 
```math
\begin{aligned}
l &\sim \operatorname{Uniform}\left(l; \; y - s/2,\, y + s/2\right) \\
s &\sim p(s \mid y, l) \\
y &\sim \operatorname{slice-sampler}\left(y \mid s, l\right),
\end{aligned}
```
where $$y$$ is updated using the shrinkage procedure by Neal[^N2003] using the initial interval formed by $$(s, l)$$.
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
    Since the search interval variables are states of the Markov chain, this sampler is **not reversible** with respect to the samples of the log-target $$y$$.
	
## Interface


```@docs
LatentSlice
```

[^LW2023]: Li, Y., & Walker, S. G. (2023). A latent slice sampling algorithm. Computational Statistics & Data Analysis, 179, 107652.
[^N2003]: Neal, R. M. (2003). Slice sampling. The annals of statistics, 31(3), 705-767.
