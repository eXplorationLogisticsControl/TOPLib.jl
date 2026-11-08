"""Path constraints assembled into the NLP fitness"""

using ForwardDiff
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


function test_trajectory_constraints_and_optional_direction()
    ceq(x, u) = diff(u[5, :])
    cineq(x, u) = [u[4, 1] - 0.5]
    objective(x, u) = -x[1, end]
    u_lb = [-1.0, -1.0, -1.0, 0.0, 2.0]
    u_ub = [1.0, 1.0, 1.0, 1.0, 4.0]
    prob = test_problem(;
        N=4,
        nu=5,
        objective=objective,
        x_initial=zeros(6),
        x_final=[1.0, 0.0, 0.0],
        x_final_idx=[1, 2, 3],
        u_lb=u_lb,
        u_ub=u_ub,
        ceq=ceq,
        cineq=cineq,
    )
    x = zeros(6, 4)
    x[1, :] .= 0.2
    u = zeros(5, 3)
    u[1, :] .= 1.0
    u[4, :] .= 0.1
    u[5, :] .= [2.0, 3.0, 4.0]
    n_ceq, n_cineq = TOPLib.constraint_counts(prob, x, u)
    @test n_ceq == 6 * 3 + 6 + 3 + 2
    @test n_cineq == 3 + 1
    obj, ceq_val, cineq_val = TOPLib.get_fitness(prob)(TOPLib.stack_variables(x, u))
    @test obj ≈ -0.2
    @test length(ceq_val) == n_ceq
    @test length(cineq_val) == n_cineq
    @test ceq_val[end-1:end] ≈ [1.0, 1.0]
    @test cineq_val[end] ≈ 0.1 - 0.5

    bare = test_problem(;
        N=3,
        direction_constraint=false,
        x_initial=nothing,
        x_final=nothing,
    )
    x_bare = zeros(6, 3)
    u_bare = zeros(4, 2)
    u_bare[1, :] .= 1.0
    n_ceq_bare, n_cineq_bare = TOPLib.constraint_counts(bare, x_bare, u_bare)
    @test n_ceq_bare == 6 * 2
    @test n_cineq_bare == 0
    _, _, cineq_bare = TOPLib.get_fitness(bare)(TOPLib.stack_variables(x_bare, u_bare))
    @test isempty(cineq_bare)

    z = TOPLib.stack_variables(x, u)
    J = ForwardDiff.jacobian(z -> TOPLib.get_fitness(prob)(z)[2], z)
    @test size(J) == (n_ceq, length(z))
    @test all(isfinite, J)
    g = ForwardDiff.gradient(z -> TOPLib.get_fitness(prob)(z)[1], z)
    @test length(g) == length(z)
    @test all(isfinite, g)
    @test g[(4 - 1) * 6 + 1] ≈ -1.0
end


@testset "path constraints in the fitness" begin
    test_path_constraints_in_the_fitness()
end

@testset "trajectory constraints" begin
    test_trajectory_constraints_and_optional_direction()
end
