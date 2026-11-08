"""Problem size, decision bounds, and boundary conditions"""

using Test

if !@isdefined TOPLib
    include(joinpath(@__DIR__, "../src/TOPLib.jl"))
end

if !@isdefined test_problem
    include(joinpath(@__DIR__, "problems.jl"))
end


function test_problem_size_and_bounds()
    prob = test_problem(; N=4, umax=0.3)
    @test prob.N == 4
    @test prob.ode_reltol == 1e-12
    @test prob.ode_abstol == 1e-12
    z_lb, z_ub = TOPLib.decision_bounds(prob, -10.0, 10.0)
    @test length(z_lb) == 6 * 4 + 4 * 3
    @test all(z_lb[1:6] .== -10.0)
    @test all(z_ub[1:6] .== 10.0)
    # Last segment's control occupies the tail of z: direction, then throttle.
    @test z_lb[end-3:end] == [-1.0, -1.0, -1.0, 0.0]
    @test z_ub[end-3:end] == [1.0, 1.0, 1.0, 0.3]

    z_lb2, z_ub2 = TOPLib.decision_bounds(prob, fill(-2.0, 6), [1.0, 2, 3, 4, 5, 6])
    @test z_lb2[1:6] == fill(-2.0, 6)
    @test z_ub2[7:12] == [1.0, 2, 3, 4, 5, 6]
    @test_throws ArgumentError TOPLib.decision_bounds(prob, -1.0, -2.0)
    @test_throws ArgumentError TOPLib.decision_bounds(prob, zeros(5), 10.0)
end


function test_boundary_conditions()
    x1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.5]
    xN = [2.0, 0.0, 0.0, 0.0, 0.0, -0.25]
    both = test_problem(; x_initial=x1 .- 0.5, x_final=xN .+ 0.25)
    ceq = TOPLib.get_boundary_conditions(both, x1, xN)
    @test ceq == vcat(x1 .- both.x_initial, xN .- both.x_final)

    initial_only = test_problem(; x_initial=zeros(6), x_final=nothing)
    ceq_i = TOPLib.get_boundary_conditions(initial_only, x1, xN)
    @test ceq_i == x1
    @test length(ceq_i) == 6

    final_only = test_problem(; x_initial=nothing, x_final=xN)
    ceq_f = TOPLib.get_boundary_conditions(final_only, x1, xN)
    @test ceq_f == zeros(6)

    free = test_problem(; x_initial=nothing, x_final=nothing)
    @test isempty(TOPLib.get_boundary_conditions(free, x1, xN))
end


@testset "problem size and bounds" begin
    test_problem_size_and_bounds()
end

@testset "boundary conditions" begin
    test_boundary_conditions()
end
