var documenterSearchIndex = {"docs":
[{"location":"latent_slice/#latent","page":"Latent Slice Sampling","title":"Latent Slice Sampling","text":"","category":"section"},{"location":"latent_slice/#Introduction","page":"Latent Slice Sampling","title":"Introduction","text":"","category":"section"},{"location":"latent_slice/","page":"Latent Slice Sampling","title":"Latent Slice Sampling","text":"Latent slice sampling is a recent vector-valued slice sampling algorithm proposed by Li and Walker[LW2023]. Unlike other slice sampling algorithms, it treats the \"search intervals\" as auxiliary variables and adapts them along the samples from the log-target in a Gibbs-type scheme.","category":"page"},{"location":"latent_slice/#Description","page":"Latent Slice Sampling","title":"Description","text":"","category":"section"},{"location":"latent_slice/","page":"Latent Slice Sampling","title":"Latent Slice Sampling","text":"Specifically, the extended joint density of the latent slice sampler is as follows:","category":"page"},{"location":"latent_slice/","page":"Latent Slice Sampling","title":"Latent Slice Sampling","text":"    p(y s l) = pi(y)  p(s)  operatornameUniformleft(l  y - s2 y + s2right)","category":"page"},{"location":"latent_slice/","page":"Latent Slice Sampling","title":"Latent Slice Sampling","text":"where y is the parameters of the log-target pi, s is the width of the search interval and l is the centering of the search interval relative to y. Naturally, the sampler operates as a blocked-Gibbs sampler ","category":"page"},{"location":"latent_slice/","page":"Latent Slice Sampling","title":"Latent Slice Sampling","text":"beginaligned\nl_n sim operatornameUniformleft(l  y_n-1 - s_n-12 y_n-1 + s_n-12right) \ns_n sim p(s mid y_n-1 l_n) \ny_n sim operatornameshrinkageleft(y mid s_n l_nright)\nendaligned","category":"page"},{"location":"latent_slice/","page":"Latent Slice Sampling","title":"Latent Slice Sampling","text":"where y is updated using the shrinkage procedure by Neal[N2003] using the initial interval formed by (s l). Therefore, the latent slice sampler can be regarded as an automatic tuning mechanism of the \"initial interval\" of slice samplers.","category":"page"},{"location":"latent_slice/","page":"Latent Slice Sampling","title":"Latent Slice Sampling","text":"The only tunable parameter of the algorithm is then the distribution of the width p(s). For this, Li and Walker recommend","category":"page"},{"location":"latent_slice/","page":"Latent Slice Sampling","title":"Latent Slice Sampling","text":"    p(s beta) = operatornameGamma(s 2 beta)","category":"page"},{"location":"latent_slice/","page":"Latent Slice Sampling","title":"Latent Slice Sampling","text":"where beta is a tunable parameter. The use of the gamma distribution is somewhat important since the complete conditional p(s mid y l) needs to be available in closed-form for efficiency.  (It is a truncated exponential distribution in case of the gamma.) Therefore, we only provide control over beta.","category":"page"},{"location":"latent_slice/","page":"Latent Slice Sampling","title":"Latent Slice Sampling","text":"info: Info\nSince the search interval variables are states of the Markov chain, this sampler is not reversible with respect to the samples of the log-target y.","category":"page"},{"location":"latent_slice/#Interface","page":"Latent Slice Sampling","title":"Interface","text":"","category":"section"},{"location":"latent_slice/","page":"Latent Slice Sampling","title":"Latent Slice Sampling","text":"LatentSlice","category":"page"},{"location":"latent_slice/#SliceSampling.LatentSlice","page":"Latent Slice Sampling","title":"SliceSampling.LatentSlice","text":"LatentSlice(beta)\n\nLatent slice sampling algorithm by Li and Walker[LW2023].\n\nArguments\n\nbeta::Real: Beta parameter of the Gamma distribution of the auxiliary variables.\n\nKeyword Arguments\n\nmax_proposals::Int: Maximum number of proposals allowed until throwing an error (default: typemax(Int)).\n\n\n\n\n\n","category":"type"},{"location":"latent_slice/","page":"Latent Slice Sampling","title":"Latent Slice Sampling","text":"[LW2023]: Li, Y., & Walker, S. G. (2023). A latent slice sampling algorithm. Computational Statistics & Data Analysis, 179, 107652.","category":"page"},{"location":"latent_slice/","page":"Latent Slice Sampling","title":"Latent Slice Sampling","text":"[N2003]: Neal, R. M. (2003). Slice sampling. The annals of statistics, 31(3), 705-767.","category":"page"},{"location":"gibbs_polar/#polar","page":"Gibbsian Polar Slice Sampling","title":"Gibbsian Polar Slice Sampling","text":"","category":"section"},{"location":"gibbs_polar/#Introduction","page":"Gibbsian Polar Slice Sampling","title":"Introduction","text":"","category":"section"},{"location":"gibbs_polar/","page":"Gibbsian Polar Slice Sampling","title":"Gibbsian Polar Slice Sampling","text":"Gibbsian polar slice sampling (GPSS) is a recent vector-valued slice sampling algorithm proposed by P. Schär, M. Habeck, and D. Rudolf[SHR2023]. It is an computationally efficient variant of polar slice sampler previously proposed by Roberts and Rosenthal[RR2002]. Unlike other slice sampling algorithms, it operates a Gibbs sampler over polar coordinates, reminiscent of the elliptical slice sampler (ESS). Due to the involvement of polar coordinates, GPSS only works reliably on more than one dimension. However, unlike ESS, GPSS is applicable to any target distribution.","category":"page"},{"location":"gibbs_polar/#Description","page":"Gibbsian Polar Slice Sampling","title":"Description","text":"","category":"section"},{"location":"gibbs_polar/","page":"Gibbsian Polar Slice Sampling","title":"Gibbsian Polar Slice Sampling","text":"For a d-dimensional target distribution pi, GPSS utilizes the following augmented target distribution:","category":"page"},{"location":"gibbs_polar/","page":"Gibbsian Polar Slice Sampling","title":"Gibbsian Polar Slice Sampling","text":"beginaligned\n    p(x T)      = varrho_pi^(0)(x) varrho_pi^(1)(x)  operatornameUniformleft(T 0 varrho^1(x)right) \n    varrho_pi^(0)(x) = lVert x rVert^1 - d \n    varrho_pi^(1)(x) = lVert x rVert^d-1 pileft(xright)\nendaligned","category":"page"},{"location":"gibbs_polar/","page":"Gibbsian Polar Slice Sampling","title":"Gibbsian Polar Slice Sampling","text":"As described in Appendix A of the GPSS paper, sampling from varrho^(1)(x) in polar coordinates magically targets the augmented target distribution.","category":"page"},{"location":"gibbs_polar/","page":"Gibbsian Polar Slice Sampling","title":"Gibbsian Polar Slice Sampling","text":"In a high-level view, GPSS operates a Gibbs sampler in the following fashion:","category":"page"},{"location":"gibbs_polar/","page":"Gibbsian Polar Slice Sampling","title":"Gibbsian Polar Slice Sampling","text":"beginaligned\nT_n      sim operatornameUniformleft(0 varrho^(1)left(x_n-1right)right) \ntheta_n sim operatornameUniformleft theta in mathbbS^d-1 mid varrho^(1)left(r_n-1 thetaright)  T_n right \nr_n      sim operatornameUniformleft r in mathbbR_geq 0 mid varrho^(1)left(r theta_nright)  T_n right \nx_n      = theta_n r_n\nendaligned","category":"page"},{"location":"gibbs_polar/","page":"Gibbsian Polar Slice Sampling","title":"Gibbsian Polar Slice Sampling","text":"where T_n is the usual acceptance threshold auxiliary variable, while theta and r are the sampler states in polar coordinates. The Gibbs steps on theta and r are implemented through specialized shrinkage procedures.","category":"page"},{"location":"gibbs_polar/","page":"Gibbsian Polar Slice Sampling","title":"Gibbsian Polar Slice Sampling","text":"The only tunable parameter of the algorithm is the size of the search interval (window) of the shrinkage sampler for the radius variable r.","category":"page"},{"location":"gibbs_polar/","page":"Gibbsian Polar Slice Sampling","title":"Gibbsian Polar Slice Sampling","text":"info: Info\nSince the direction and radius variables are states of the Markov chain, this sampler is not reversible with respect to the samples of the log-target x.","category":"page"},{"location":"gibbs_polar/#Interface","page":"Gibbsian Polar Slice Sampling","title":"Interface","text":"","category":"section"},{"location":"gibbs_polar/","page":"Gibbsian Polar Slice Sampling","title":"Gibbsian Polar Slice Sampling","text":"info: Info\nBy the nature of polar coordinates, GPSS only works reliably for targets with dimension at least d geq 2.","category":"page"},{"location":"gibbs_polar/","page":"Gibbsian Polar Slice Sampling","title":"Gibbsian Polar Slice Sampling","text":"GibbsPolarSlice","category":"page"},{"location":"gibbs_polar/#SliceSampling.GibbsPolarSlice","page":"Gibbsian Polar Slice Sampling","title":"SliceSampling.GibbsPolarSlice","text":"GibbsPolarSlice(w; max_proposals)\n\nGibbsian polar slice sampling algorithm by P. Schär, M. Habeck, and D. Rudolf [SHR2023].\n\nArguments\n\nw::Real: Initial window size for the radius shrinkage procedure.\n\nKeyword Arguments\n\nw::Real: Initial window size for the radius shrinkage procedure\nmax_proposals::Int: Maximum number of proposals allowed until throwing an error (default: typemax(Int)).\n\n\n\n\n\n","category":"type"},{"location":"gibbs_polar/","page":"Gibbsian Polar Slice Sampling","title":"Gibbsian Polar Slice Sampling","text":"warning: Warning\nWhen initializing the chain (e.g. the initial_params keyword arguments in AbstractMCMC.sample), it is necessary to inialize from a point x_0 that has a sensible norm lVert x_0 rVert  0, otherwise, the chain will start from a pathologic point in polar coordinates. This might even result in the sampler getting stuck in an infinite loop. (This can be prevented by setting max_proposals.) If lVert x_0 rVert leq 10^-5, the current implementation will display a warning. ","category":"page"},{"location":"gibbs_polar/","page":"Gibbsian Polar Slice Sampling","title":"Gibbsian Polar Slice Sampling","text":"info: Info\nFor Turing users: Turing might change initial_params to match the support of the posterior. This might lead to lVert x_0 rVert being small, even though the vector you passed toinitial_params has a sufficiently large norm. If this is suspected, simply try a different initialization value.","category":"page"},{"location":"gibbs_polar/#Demonstration","page":"Gibbsian Polar Slice Sampling","title":"Demonstration","text":"","category":"section"},{"location":"gibbs_polar/","page":"Gibbsian Polar Slice Sampling","title":"Gibbsian Polar Slice Sampling","text":"As illustrated in the original paper, GPSS shows good performance on heavy-tailed targets despite being a multivariate slice sampler. Consider a 10-dimensional Student-t target with 1-degree of freedom (this corresponds to a multivariate Cauchy):","category":"page"},{"location":"gibbs_polar/","page":"Gibbsian Polar Slice Sampling","title":"Gibbsian Polar Slice Sampling","text":"using Distributions\nusing Turing\nusing SliceSampling\nusing LinearAlgebra\nusing Plots\n\n@model function demo()\n    x ~ MvTDist(1, zeros(10), Matrix(I,10,10))\nend\nmodel = demo()\n\nn_samples = 1000\nlatent_chain = sample(model, externalsampler(LatentSlice(10)), n_samples; initial_params=ones(10))\npolar_chain = sample(model, externalsampler(GibbsPolarSlice(10)), n_samples; initial_params=ones(10))\n\nl = @layout [a; b]\np1 = Plots.plot(1:n_samples, latent_chain[:,1,:], ylims=[-10,10], label=\"LSS\")\np2 = Plots.plot(1:n_samples, polar_chain[:,1,:],  ylims=[-10,10], label=\"GPSS\")\nplot(p1, p2, layout = l)\nsavefig(\"student_latent_gpss.svg\")","category":"page"},{"location":"gibbs_polar/","page":"Gibbsian Polar Slice Sampling","title":"Gibbsian Polar Slice Sampling","text":"(Image: )","category":"page"},{"location":"gibbs_polar/","page":"Gibbsian Polar Slice Sampling","title":"Gibbsian Polar Slice Sampling","text":"Clearly, GPSS is better at exploring the deep tails compared to the latent slice sampler (LSS) despite having a similar per-iteration cost.","category":"page"},{"location":"gibbs_polar/","page":"Gibbsian Polar Slice Sampling","title":"Gibbsian Polar Slice Sampling","text":"[SHR2023]: Schär, P., Habeck, M., & Rudolf, D. (2023, July). Gibbsian polar slice sampling. In International Conference on Machine Learning.","category":"page"},{"location":"gibbs_polar/","page":"Gibbsian Polar Slice Sampling","title":"Gibbsian Polar Slice Sampling","text":"[RR2002]: Roberts, G. O., & Rosenthal, J. S. (2002). The polar slice sampler. Stochastic Models, 18(2), 257-280.","category":"page"},{"location":"univariate_slice/#Univariate-Slice-Sampling-Algorithms","page":"Univariate Slice Sampling","title":"Univariate Slice Sampling Algorithms","text":"","category":"section"},{"location":"univariate_slice/#Introduction","page":"Univariate Slice Sampling","title":"Introduction","text":"","category":"section"},{"location":"univariate_slice/","page":"Univariate Slice Sampling","title":"Univariate Slice Sampling","text":"These algorithms are the \"single-variable\" slice sampling algorithms originally described by Neal[N2003]. Since these algorithms are univariate, one has to combine them with univariate-to-multivariate strategies, which are discussed in this section.","category":"page"},{"location":"univariate_slice/#Fixed-Initial-Interval-Slice-Sampling","page":"Univariate Slice Sampling","title":"Fixed Initial Interval Slice Sampling","text":"","category":"section"},{"location":"univariate_slice/","page":"Univariate Slice Sampling","title":"Univariate Slice Sampling","text":"This is the most basic form of univariate slice sampling, where the proposals are generated within a fixed interval formed by the window.","category":"page"},{"location":"univariate_slice/","page":"Univariate Slice Sampling","title":"Univariate Slice Sampling","text":"Slice","category":"page"},{"location":"univariate_slice/#SliceSampling.Slice","page":"Univariate Slice Sampling","title":"SliceSampling.Slice","text":"Slice(window; max_proposals)\n\nUnivariate slice sampling with a fixed initial interval (Scheme 2 by Neal[N2003])\n\nArguments\n\nwindow::Real: Proposal window.\n\nKeyword Arguments\n\nmax_proposals::Int: Maximum number of proposals allowed until throwing an error (default: typemax(Int)).\n\n\n\n\n\n","category":"type"},{"location":"univariate_slice/#Adaptive-Initial-Interval-Slice-Sampling","page":"Univariate Slice Sampling","title":"Adaptive Initial Interval Slice Sampling","text":"","category":"section"},{"location":"univariate_slice/","page":"Univariate Slice Sampling","title":"Univariate Slice Sampling","text":"These algorithms try to adaptively set the initial interval through a simple search procedure. The \"stepping-out\" procedure grows the initial window on a linear scale, while the \"doubling-out\" procedure grows it geometrically. window controls the scale of the increase.","category":"page"},{"location":"univariate_slice/#What-Should-I-Use?","page":"Univariate Slice Sampling","title":"What Should I Use?","text":"","category":"section"},{"location":"univariate_slice/","page":"Univariate Slice Sampling","title":"Univariate Slice Sampling","text":"This highly depends on the problem at hand. In general, the doubling-out procedure tends to be more expensive as it requires additional log-target evaluations to decide whether to accept a proposal. However, if the scale of the posterior varies drastically, doubling out might work better. In general, it is recommended to use the stepping-out procedure.","category":"page"},{"location":"univariate_slice/","page":"Univariate Slice Sampling","title":"Univariate Slice Sampling","text":"SliceSteppingOut\nSliceDoublingOut","category":"page"},{"location":"univariate_slice/#SliceSampling.SliceSteppingOut","page":"Univariate Slice Sampling","title":"SliceSampling.SliceSteppingOut","text":"SliceSteppingOut(window; max_stepping_out, max_proposals)\n\nUnivariate slice sampling by automatically adapting the initial interval through the \"stepping-out\" procedure (Scheme 3 by Neal[N2003])\n\nArguments\n\nwindow::Real: Proposal window.\n\nKeyword Arguments\n\nmax_stepping_out::Int: Maximum number of \"stepping outs\" (default: 32).\nmax_proposals::Int: Maximum number of proposals allowed until throwing an error (default: typemax(Int)).\n\n\n\n\n\n","category":"type"},{"location":"univariate_slice/#SliceSampling.SliceDoublingOut","page":"Univariate Slice Sampling","title":"SliceSampling.SliceDoublingOut","text":"SliceDoublingOut(window; max_doubling_out, max_proposals)\n\nUnivariate slice sampling by automatically adapting the initial interval through the \"doubling-out\" procedure (Scheme 4 by Neal[N2003])\n\nArguments\n\nwindow::Real: Proposal window.\n\nKeyword Arguments\n\nmax_doubling_out: Maximum number of \"doubling outs\" (default: 8).\nmax_proposals::Int: Maximum number of proposals allowed until throwing an error (default: typemax(Int)).\n\n\n\n\n\n","category":"type"},{"location":"univariate_slice/","page":"Univariate Slice Sampling","title":"Univariate Slice Sampling","text":"[N2003]: Neal, R. M. (2003). Slice sampling. The Annals of Statistics, 31(3), 705-767.","category":"page"},{"location":"uni_to_multi/#unitomulti","page":"Univariate to Multivariate","title":"Univariate-to-Multivariate Strategies","text":"","category":"section"},{"location":"uni_to_multi/","page":"Univariate to Multivariate","title":"Univariate to Multivariate","text":"To use univariate slice sampling strategies on targets with more than on dimension, one has to embed them into a multivariate sampling scheme that relies on univariate sampling elements. The two most popular approaches for this are Gibbs sampling[GG1984] and hit-and-run[BRS1993].","category":"page"},{"location":"uni_to_multi/#Random-Permutation-Gibbs-Strategy","page":"Univariate to Multivariate","title":"Random Permutation Gibbs Strategy","text":"","category":"section"},{"location":"uni_to_multi/","page":"Univariate to Multivariate","title":"Univariate to Multivariate","text":"Gibbs sampling[GG1984] is a strategy where we sample from the posterior one coordinate at a time, conditioned on the values of all other coordinates. In practice, one can pick the coordinates in any order they want as long as it does not depend on the state of the chain. It is generally hard to know a-prior which \"scan order\" is best, but randomly picking coordinates tend to work well in general. Currently, we only provide random permutation scan, which guarantees that all coordinates are updated at least once after d transitions. At the same time, reversibility is maintained by randomly permuting the order we go through each coordinate:","category":"page"},{"location":"uni_to_multi/","page":"Univariate to Multivariate","title":"Univariate to Multivariate","text":"RandPermGibbs","category":"page"},{"location":"uni_to_multi/#SliceSampling.RandPermGibbs","page":"Univariate to Multivariate","title":"SliceSampling.RandPermGibbs","text":"RandPermGibbs(unislice)\n\nRandom permutation coordinate-wise Gibbs sampling strategy. This applies unislice coordinate-wise in a random order.\n\nArguments\n\nunislice::Union{<:AbstractUnivariateSliceSampling,<:AbstractVector{<:AbstractUnivariateSliceSampling}}: a single or a vector of univariate slice sampling algorithms.\n\nWhen unislice is a vector of samplers, each slice sampler is applied to the corresponding coordinate of the target posterior. Furthermore, the length(unislice) must match the dimensionality of the posterior.\n\n\n\n\n\n","category":"type"},{"location":"uni_to_multi/","page":"Univariate to Multivariate","title":"Univariate to Multivariate","text":"Each call to AbstractMCMC.step internally performs d Gibbs transition so that all coordinates are updated.","category":"page"},{"location":"uni_to_multi/","page":"Univariate to Multivariate","title":"Univariate to Multivariate","text":"For example:","category":"page"},{"location":"uni_to_multi/","page":"Univariate to Multivariate","title":"Univariate to Multivariate","text":"RandPermGibbs(SliceSteppingOut(2.))","category":"page"},{"location":"uni_to_multi/","page":"Univariate to Multivariate","title":"Univariate to Multivariate","text":"If one wants to use a different slice sampler configuration for each coordinate, one can mix-and-match by passing a Vector of slice samplers, one for each coordinate. For instance, for a 2-dimensional target:","category":"page"},{"location":"uni_to_multi/","page":"Univariate to Multivariate","title":"Univariate to Multivariate","text":"RandPermGibbs([SliceSteppingOut(2.; max_proposals=32), SliceDoublingOut(2.),])","category":"page"},{"location":"uni_to_multi/#Hit-and-Run-Strategy","page":"Univariate to Multivariate","title":"Hit-and-Run Strategy","text":"","category":"section"},{"location":"uni_to_multi/","page":"Univariate to Multivariate","title":"Univariate to Multivariate","text":"Hit-and-run is a simple \"meta\" algorithm, where we sample over a random 1-dimensional projection of the space. That is, at each iteration, we sample a random direction","category":"page"},{"location":"uni_to_multi/","page":"Univariate to Multivariate","title":"Univariate to Multivariate","text":"    theta_n sim operatornameUniform(mathbbS^d-1)","category":"page"},{"location":"uni_to_multi/","page":"Univariate to Multivariate","title":"Univariate to Multivariate","text":"and perform a Markov transition along the 1-dimensional subspace","category":"page"},{"location":"uni_to_multi/","page":"Univariate to Multivariate","title":"Univariate to Multivariate","text":"beginaligned\n    lambda_n sim pleft(lambda mid x_n-1 theta_n right) propto pileft( x_n-1 + lambda theta_n right) \n    x_n = x_n-1 + lambda_n theta_n\nendaligned","category":"page"},{"location":"uni_to_multi/","page":"Univariate to Multivariate","title":"Univariate to Multivariate","text":"where pi is the target unnormalized density. Applying slice sampling for the 1-dimensional subproblem has been popularized by David Mackay[M2003], and is, technically, also a Gibbs sampler.  (Or is that Gibbs samplers are hit-and-run samplers?) Unlike RandPermGibbs, which only makes axis-aligned moves, HitAndRun can choose arbitrary directions, which could be helpful in some cases.","category":"page"},{"location":"uni_to_multi/","page":"Univariate to Multivariate","title":"Univariate to Multivariate","text":"HitAndRun","category":"page"},{"location":"uni_to_multi/#SliceSampling.HitAndRun","page":"Univariate to Multivariate","title":"SliceSampling.HitAndRun","text":"HitAndRun(unislice)\n\nHit-and-run sampling strategy[BRS1993]. This applies unislice along a random direction uniform sampled from the sphere.\n\nArguments\n\nunislice::AbstractUnivariateSliceSampling: Univariate slice sampling algorithm.\n\nunislice can also be a vector of univeriate slice samplers, where each slice sampler is applied to each corresponding coordinate of the target posterior.\n\n\n\n\n\n","category":"type"},{"location":"uni_to_multi/","page":"Univariate to Multivariate","title":"Univariate to Multivariate","text":"This can be used, for example, as follows:","category":"page"},{"location":"uni_to_multi/","page":"Univariate to Multivariate","title":"Univariate to Multivariate","text":"HitAndRun(SliceSteppingOut(2.))","category":"page"},{"location":"uni_to_multi/","page":"Univariate to Multivariate","title":"Univariate to Multivariate","text":"Unlike RandPermGibbs, HitAndRun does not provide the option of using a unique unislice object for each coordinate. This is a natural limitation of the hit-and-run sampler: it does not operate on individual coordinates.","category":"page"},{"location":"uni_to_multi/","page":"Univariate to Multivariate","title":"Univariate to Multivariate","text":"[GG1984]: Geman, S., & Geman, D. (1984). Stochastic relaxation, Gibbs distributions, and the Bayesian restoration of images. IEEE Transactions on Pattern Analysis and Machine Intelligence, (6).","category":"page"},{"location":"uni_to_multi/","page":"Univariate to Multivariate","title":"Univariate to Multivariate","text":"[BRS1993]: Bélisle, C. J., Romeijn, H. E., & Smith, R. L. (1993). Hit-and-run algorithms for generating multivariate distributions. Mathematics of Operations Research, 18(2), 255-266.","category":"page"},{"location":"uni_to_multi/","page":"Univariate to Multivariate","title":"Univariate to Multivariate","text":"[M2003]: MacKay, D. J. (2003). Information theory, inference and learning algorithms. Cambridge university press.","category":"page"},{"location":"general/#General-Usage","page":"General Usage","title":"General Usage","text":"","category":"section"},{"location":"general/","page":"General Usage","title":"General Usage","text":"This package implements the AbstractMCMC interface. AbstractMCMC provides a unifying interface for MCMC algorithms applied to LogDensityProblems.","category":"page"},{"location":"general/#Examples","page":"General Usage","title":"Examples","text":"","category":"section"},{"location":"general/#Drawing-Samples-From-a-LogDensityProblems-Through-AbstractMCMC","page":"General Usage","title":"Drawing Samples From a LogDensityProblems Through AbstractMCMC","text":"","category":"section"},{"location":"general/","page":"General Usage","title":"General Usage","text":"SliceSampling.jl implements the AbstractMCMC interface through LogDensityProblems. That is, one simply needs to define a LogDensityProblems and pass it to AbstractMCMC:","category":"page"},{"location":"general/","page":"General Usage","title":"General Usage","text":"using AbstractMCMC\nusing Distributions\nusing LinearAlgebra\nusing LogDensityProblems\nusing Plots\n\nusing SliceSampling\n\nstruct Target{D}\n\tdist::D\nend\n\nLogDensityProblems.logdensity(target::Target, x) = logpdf(target.dist, x)\n\nLogDensityProblems.dimension(target::Target) = length(target.distx)\n\nLogDensityProblems.capabilities(::Type{<:Target}) = LogDensityProblems.LogDensityOrder{0}()\n\nsampler         = GibbsPolarSlice(2.0)\nn_samples       = 10000\nmodel           = Target(MvTDist(5, zeros(10), Matrix(I, 10, 10)))\nlogdensitymodel = AbstractMCMC.LogDensityModel(model)\n\nchain   = sample(logdensitymodel, sampler, n_samples; initial_params=randn(10))\nsamples = hcat([transition.params for transition in chain]...)\n\nplot(samples[1,:], xlabel=\"Iteration\", ylabel=\"Trace\")\nsavefig(\"abstractmcmc_demo.svg\")","category":"page"},{"location":"general/","page":"General Usage","title":"General Usage","text":"(Image: )","category":"page"},{"location":"general/#Drawing-Samples-From-Turing-Models","page":"General Usage","title":"Drawing Samples From Turing Models","text":"","category":"section"},{"location":"general/","page":"General Usage","title":"General Usage","text":"SliceSampling.jl can also be used to sample from Turing models through Turing's externalsampler interface:","category":"page"},{"location":"general/","page":"General Usage","title":"General Usage","text":"using Distributions\nusing Turing\nusing SliceSampling\n\n@model function demo()\n    s ~ InverseGamma(3, 3)\n    m ~ Normal(0, sqrt(s))\nend\n\nsampler   = RandPermGibbs(SliceSteppingOut(2.))\nn_samples = 10000\nmodel     = demo()\nsample(model, externalsampler(sampler), n_samples; initial_params=[exp(1.0), 0.0])","category":"page"},{"location":"general/#Drawing-Samples","page":"General Usage","title":"Drawing Samples","text":"","category":"section"},{"location":"general/","page":"General Usage","title":"General Usage","text":"For drawing samples using the algorithms provided by SliceSampling, the user only needs to call:","category":"page"},{"location":"general/","page":"General Usage","title":"General Usage","text":"sample([rng,] model, slice, N; initial_params)","category":"page"},{"location":"general/","page":"General Usage","title":"General Usage","text":"slice::AbstractSliceSampling: Any slice sampling algorithm provided by SliceSampling.\nmodel: A model implementing the LogDensityProblems interface.\nN: The number of samples","category":"page"},{"location":"general/","page":"General Usage","title":"General Usage","text":"The output is a SliceSampling.Transition object, which contains the following:","category":"page"},{"location":"general/","page":"General Usage","title":"General Usage","text":"SliceSampling.Transition","category":"page"},{"location":"general/#SliceSampling.Transition","page":"General Usage","title":"SliceSampling.Transition","text":"struct Transition\n\nStruct containing the results of the transition.\n\nFields\n\nparams: Samples generated by the transition.\nlp::Real: Log-target density of the samples.\ninfo::NamedTuple: Named tuple containing information about the transition. \n\n\n\n\n\n","category":"type"},{"location":"general/","page":"General Usage","title":"General Usage","text":"For the keyword arguments, SliceSampling allows:","category":"page"},{"location":"general/","page":"General Usage","title":"General Usage","text":"initial_params: The intial state of the Markov chain (default: nothing).","category":"page"},{"location":"general/","page":"General Usage","title":"General Usage","text":"If initial_params is nothing, the following function can be implemented to provide an initialization:","category":"page"},{"location":"general/","page":"General Usage","title":"General Usage","text":"SliceSampling.initial_sample","category":"page"},{"location":"general/#SliceSampling.initial_sample","page":"General Usage","title":"SliceSampling.initial_sample","text":"initial_sample(rng, model)\n\nReturn the initial sample for the model using the random number generator rng.\n\nArguments\n\nrng::Random.AbstractRNG: Random number generator.\nmodel: the model of interest.\n\n\n\n\n\n","category":"function"},{"location":"general/#Performing-a-Single-Transition","page":"General Usage","title":"Performing a Single Transition","text":"","category":"section"},{"location":"general/","page":"General Usage","title":"General Usage","text":"For more fined-grained control, the user can call AbstractMCMC.step. That is, the chain can be initialized by calling:","category":"page"},{"location":"general/","page":"General Usage","title":"General Usage","text":"transition, state = AbstractMCMC.steps([rng,] model, slice; initial_params)","category":"page"},{"location":"general/","page":"General Usage","title":"General Usage","text":"and then each MCMC transition on state can be performed by calling:","category":"page"},{"location":"general/","page":"General Usage","title":"General Usage","text":"transition, state = AbstractMCMC.steps([rng,] model, slice, state)","category":"page"},{"location":"general/","page":"General Usage","title":"General Usage","text":"For more details, refer to the documentation of AbstractMCMC.","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = SliceSampling","category":"page"},{"location":"#SliceSampling","page":"Home","title":"SliceSampling","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This package implements slice sampling algorithms.  Slice sampling finds its roots in the Swendsen–Wang algorithm for Ising models[SW1987][ES1988]. It later came into the interest of the statistical community through Besag and Green[BG1993], and popularized by Neal [N2003]. Furthermore, Neal introduced various ways to efficiently implement slice samplers. This package provides the original slice sampling algorithms by Neal and their later extensions.","category":"page"},{"location":"","page":"Home","title":"Home","text":"[SW1987]: Swendsen, R. H., & Wang, J. S. (1987). Nonuniversal critical dynamics in Monte Carlo simulations. Physical review letters, 58(2), 86.","category":"page"},{"location":"","page":"Home","title":"Home","text":"[ES1988]: Edwards, R. G., & Sokal, A. D. (1988). Generalization of the fortuin-kasteleyn-swendsen-wang representation and monte carlo algorithm. Physical review D, 38(6), 2009.","category":"page"},{"location":"","page":"Home","title":"Home","text":"[BG1993]: Besag, J., & Green, P. J. (1993). Spatial statistics and Bayesian computation. Journal of the Royal Statistical Society Series B: Statistical Methodology, 55(1), 25-37.","category":"page"},{"location":"","page":"Home","title":"Home","text":"[N2003]: Neal, R. M. (2003). Slice sampling. The annals of statistics, 31(3), 705-767.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"}]
}
