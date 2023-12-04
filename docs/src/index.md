```@meta
CurrentModule = SliceSampling
```

# SliceSampling
This package implements slice sampling algorithms. 
Slice sampling finds its roots in the Swendsen–Wang algorithm for Ising models[^SW1987][^ES1988].
It later came into the interest of the statistical community through Besag and Green[^BG1993], and popularized by Neal [^N2003].
Furthermore, Neal introduced various ways to efficiently implement slice samplers.
This package provides the original slice sampling algorithms by Neal and their later extensions.

[^SW1987]: Swendsen, R. H., & Wang, J. S. (1987). Nonuniversal critical dynamics in Monte Carlo simulations. Physical review letters, 58(2), 86.
[^ES1988]: Edwards, R. G., & Sokal, A. D. (1988). Generalization of the fortuin-kasteleyn-swendsen-wang representation and monte carlo algorithm. Physical review D, 38(6), 2009.
[^BG1993]: Besag, J., & Green, P. J. (1993). Spatial statistics and Bayesian computation. Journal of the Royal Statistical Society Series B: Statistical Methodology, 55(1), 25-37.
[^N2003]: Neal, R. M. (2003). Slice sampling. The annals of statistics, 31(3), 705-767.


```@index
```
