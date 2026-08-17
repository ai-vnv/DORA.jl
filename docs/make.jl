using Documenter
using DORASolvers

makedocs(
    sitename = "DORASolvers.jl",
    modules = [DORASolvers],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://ai-vnv.github.io/DORASolvers.jl/stable/",
    ),
    pages = [
        "Home" => "index.md",
        "Solver Guide" => "guide.md",
        "Examples" => "examples.md",
        "API Reference" => "api.md",
    ],
    checkdocs = :none,
    warnonly = [:missing_docs, :cross_references, :docs_block],
)

deploydocs(
    repo = "github.com/ai-vnv/DORASolvers.jl",
    devbranch = "main",
    push_preview = false,
)
