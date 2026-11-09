"""Run tests"""

using Test

include(joinpath(@__DIR__, "../src/TOPLib.jl"))
include(joinpath(@__DIR__, "problems.jl"))

@testset "Utils" begin
    include("test_utils.jl")
end

@testset "Forward shooting" begin
    include("test_forwardshooting_problem.jl")
    include("test_forwardshooting_objective.jl")
    include("test_forwardshooting_defects.jl")
    include("test_forwardshooting_fitness.jl")
    include("test_forwardshooting_solve.jl")
end

@testset "Impulsive forward shooting" begin
    include("test_impulsive_forwardshooting.jl")
end
