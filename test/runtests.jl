"""Run tests"""

using Test

include(joinpath(@__DIR__, "../src/TOPLib.jl"))
include(joinpath(@__DIR__, "problems.jl"))

@testset "Utils" begin
    include("test_utils.jl")
end

@testset "Forward shooting" begin
    include("test_msfwd_problem.jl")
    include("test_msfwd_objective.jl")
    include("test_msfwd_defects.jl")
    include("test_msfwd_fitness.jl")
    include("test_msfwd_solve.jl")
end

@testset "Impulsive forward shooting" begin
    include("test_impulsive_msfwd.jl")
end

@testset "Forward-backward shooting" begin
    include("test_fwdbck.jl")
end
