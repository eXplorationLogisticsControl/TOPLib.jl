"""
Make documentation with Documenter.jl
"""

import Pkg

Pkg.activate(@__DIR__)
Pkg.instantiate()

using Documenter

include(joinpath(dirname(@__FILE__), "../src/TOPLib.jl"))

makedocs(
    clean = true,
    build = joinpath(@__DIR__, "build"),
    modules = [TOPLib],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        edit_link = "main",
        canonical = "https://github.com/eXplorationLogisticsControl/TOPLib.jl",
    ),
    sitename = "TOPLib.jl",
    pages = [
        "Home" => "index.md",
    ],
)
