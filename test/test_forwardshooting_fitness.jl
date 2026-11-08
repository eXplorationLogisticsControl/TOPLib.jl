"""Path constraints assembled into the NLP fitness"""

using Test

if !@isdefined TOPLib
    include(joinpath(@__DIR__, "../src/TOPLib.jl"))
end

if !@isdefined test_problem
    include(joinpath(@__DIR__, "problems.jl"))
end


function test_path_constraints_in_the_fitness()
    ceq_path(x, u) = [x[1] - u[4]]
    cineq_path(x, u) = [u[4] - 0.2]
    prob = test_problem(;
        N=3,
        x_initial=zeros(6),
        x_final=zeros(6),
        ceq_path=ceq_path,
        cineq_path=cineq_path,
    )
    x = zeros(6, 3)
    x[1, :] .= [0.1, 0.4, 0.0]
    u = zeros(4, 2)
    u[1, :] .= 1.0
    u[4, :] .= [0.05, 0.5]
    n_ceq, n_cineq = TOPLib.constraint_counts(prob, x, u)
    @test n_ceq == 6 * 2 + 6 + 6 + 2
    @test n_cineq == 2 + 2
    obj, ceq, cineq = TOPLib.get_fitness(prob)(TOPLib.stack_variables(x, u))
    @test obj ≈ 0.55
    @test length(ceq) == n_ceq
    @test length(cineq) == n_cineq
    @test ceq[end-1:end] ≈ [0.1 - 0.05, 0.4 - 0.5]
    @test cineq[1:2] ≈ [-1.0 + 1.0, -1.0 + 1.0]
    @test cineq[3:4] ≈ [0.05 - 0.2, 0.5 - 0.2]
end


@testset "path constraints in the fitness" begin
    test_path_constraints_in_the_fitness()
end
