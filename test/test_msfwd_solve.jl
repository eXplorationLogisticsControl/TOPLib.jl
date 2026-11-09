"""Solve a stay-put double-integrator transfer"""

using Test

if !@isdefined TOPLib
    include(joinpath(@__DIR__, "../src/TOPLib.jl"))
end

if !@isdefined test_problem
    include(joinpath(@__DIR__, "problems.jl"))
end


function test_solve_stay_put_transfer()
    prob = test_problem(; N=3, tf=1.0, umax=0.5)
    x0 = zeros(6, 3)
    u0 = zeros(4, 2)
    u0[1, :] .= 1.0
    result = TOPLib.solve(prob; algo=:ipopt, x0=x0, u0=u0, print_level=0, tol=1e-8, max_iter=100)
    @test result.status in (:Solve_Succeeded, :Solved_To_Acceptable_Level)
    @test result.objective ≈ 0 atol=1e-6
    @test size(result.x) == size(x0)
    @test size(result.u) == size(u0)
    @test all(abs.(result.x) .< 1e-5)
    @test all(abs.(result.u[4, :]) .< 1e-5)
    @test_throws ErrorException TOPLib.solve(prob; algo=:missing, x0=x0, u0=u0)
end


@testset "solve a stay-put transfer" begin
    test_solve_stay_put_transfer()
end
