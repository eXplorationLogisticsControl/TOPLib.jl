using ForwardDiff
using LinearAlgebra
using Test
using TOPLib

function double_integrator!(dx, x, pu, t)
    u = pu.u
    dx[1:3] = x[4:6]
    dx[4:6] = u[4] * u[1:3]
    return nothing
end

function test_problem(;
    N::Int=3,
    tf::Float64=1.0,
    x_initial=zeros(6),
    x_final=zeros(6),
    umax::Float64=1.0,
    ceq_path=nothing,
    cineq_path=nothing,
)
    dynamics = TOPLib.ContinuousControlDynamics(6, 4, double_integrator!, nothing)
    times = collect(range(0.0, tf; length=N))
    return TOPLib.ForwardShootingProblem(
        dynamics,
        times,
        :fuel,
        x_initial,
        x_final;
        umax=umax,
        ceq_path=ceq_path,
        cineq_path=cineq_path,
    )
end

@testset "TOPLib" begin
@testset "stack and unpack" begin
    x = randn(6, 4)
    u = randn(4, 3)
    z = TOPLib.stack_variables(x, u)
    @test length(z) == length(x) + length(u)
    x2, u2 = TOPLib.unpack_variables(z, 6, 4, 4)
    @test x2 == x
    @test u2 == u
    @test_throws DimensionMismatch TOPLib.unpack_variables(z[1:end-1], 6, 4, 4)
end

@testset "problem size and bounds" begin
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

@testset "boundary conditions" begin
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

@testset "fuel objective and direction inequality" begin
    # Columns: interior direction, unit direction, infeasible direction.
    u = [
        0.0 1.0 1.0
        0.0 0.0 1.0
        0.0 0.0 0.0
        0.2 0.0 0.4
    ]
    @test TOPLib.get_objective(:fuel, zeros(6, 4), u) ≈ 0.6
    @test_throws ErrorException TOPLib.get_objective(:energy, zeros(6, 4), u)
    cineq = TOPLib.fuel_direction_inequalities(u)
    @test cineq ≈ [-1.0, 0.0, 1.0]
    @test cineq[1] < 0
    @test cineq[2] == 0
    @test cineq[3] > 0
end

@testset "dynamics defects" begin
    prob = test_problem(; N=2, tf=1.0, x_initial=nothing, x_final=nothing)
    dt = 1.0
    a = [0.0, 0.0, 0.5]
    x1 = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    x2 = vcat(x1[1:3] .+ x1[4:6] .* dt .+ 0.5 .* a .* dt^2, x1[4:6] .+ a .* dt)
    u = reshape([0.0, 0.0, 1.0, 0.5], 4, 1)
    x = hcat(x1, x2)
    defects = TOPLib.shooting_defects(
        prob.dynamics, prob.times, x, u;
        ode_method=prob.ode_method,
        ode_reltol=prob.ode_reltol,
        ode_abstol=prob.ode_abstol,
    )
    @test defects ≈ zeros(6, 1) atol=1e-8

    x_bad = copy(x)
    x_bad[1, 2] += 0.1
    defects_bad = TOPLib.shooting_defects(
        prob.dynamics, prob.times, x_bad, u;
        ode_method=prob.ode_method,
        ode_reltol=prob.ode_reltol,
        ode_abstol=prob.ode_abstol,
    )
    @test defects_bad[:, 1] ≈ [0.1, 0.0, 0.0, 0.0, 0.0, 0.0] atol=1e-8

    z = TOPLib.stack_variables(x, u)
    defects_vec(ξ) = vec(TOPLib.shooting_defects(
        prob.dynamics, prob.times,
        TOPLib.unpack_variables(ξ, 6, 4, 2)...;
        ode_method=prob.ode_method,
        ode_reltol=prob.ode_reltol,
        ode_abstol=prob.ode_abstol,
    ))
    J = ForwardDiff.jacobian(defects_vec, z)
    @test all(isfinite, J)
    # Defect is x[:, 2] - flow(x[:, 1], u), so the x[:, 2] block is the identity.
    @test J[:, 7:12] ≈ I(6) atol=1e-8
    h = 1e-6
    zp = copy(z)
    zm = copy(z)
    zp[end] += h
    zm[end] -= h
    fd = (defects_vec(zp) - defects_vec(zm)) / (2h)
    @test J[:, end] ≈ fd atol=1e-4
end

@testset "path constraints in the fitness" begin
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

@testset "solve a stay-put transfer" begin
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
end
