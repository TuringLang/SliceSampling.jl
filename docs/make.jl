using SliceSampling
using Documenter

DocMeta.setdocmeta!(SliceSampling, :DocTestSetup, :(using SliceSampling); recursive=true)

makedocs(;
    modules=[SliceSampling],
    authors="Kyurae Kim <kyrkim@seas.upenn.edu> and contributors",
    repo="https://github.com/TuringLang/SliceSampling.jl/blob/{commit}{path}#{line}",
    sitename="SliceSampling.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://TuringLang.org/SliceSampling.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home"                          => "index.md",
        "General Usage"                 => "general.md",
        "Univariate Slice Sampling"     => "univariate_slice.md",
        "Univariate to Multivariate"    => "uni_to_multi.md",
        "Latent Slice Sampling"         => "latent_slice.md",
        "Gibbsian Polar Slice Sampling" => "gibbs_polar.md"
    ],
)

deploydocs(;
    repo="github.com/TuringLang/SliceSampling.jl",
    push_preview=true
)
