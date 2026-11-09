"""
Make documentation with Documenter.jl
"""

import Pkg

Pkg.activate(@__DIR__)
Pkg.instantiate()

using Documenter
using Literate
using TOPLib

exampledir = joinpath(@__DIR__, "src", "examples")
for name in ("msfwd_cr3bp", "fwdbck_cr3bp")
    Literate.markdown(
        joinpath(exampledir, name * ".jl"),
        exampledir;
        documenter = true,
        codefence = "```julia" => "```",
    )
end

makedocs(
    clean = true,
    build = joinpath(@__DIR__, "build"),
    modules = [TOPLib],
    checkdocs = :exports,
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        edit_link = "main",
        canonical = "https://github.com/eXplorationLogisticsControl/TOPLib.jl",
    ),
    sitename = "TOPLib.jl",
    pages = [
        "Home" => "index.md",
        "Transcriptions" => Any[
            "Multiple Shooting" => "transcriptions/multipleshooting.md",
            "Forward-Backward Shooting" => "transcriptions/forwardbackward.md",
        ],
        "Examples" => Any[
            "Multiple shooting" => "examples/msfwd_cr3bp.md",
            "Forward-backward shooting" => "examples/fwdbck_cr3bp.md",
        ],
    ],
)
