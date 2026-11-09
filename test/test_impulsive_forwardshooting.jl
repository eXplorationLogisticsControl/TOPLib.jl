"""Impulsive multiple shooting: defects, boundaries, and a rest-to-rest solve."""

using ForwardDiff
using LinearAlgebra
using Test

if !@isdefined TOPLib
    include(joinpath(@__DIR__, "../src/TOPLib.jl"))
end


function coast_double_integrator!(dx, x, pu, t)
    dx[1:3] = x[4:6]
    dx[4:6] .= 0
    return nothing
end


# Adds thrust if the coast is handed a nonzero control. Impulsive coasts must not.
function thrusting_coast!(dx, x, pu, t)
    dx[1:3] = x[4:6]
    dx[4:6] = pu.u[4] * pu.u[1:3]
    return nothing
end


function mutating_jump!(x_post, x, u)
    x_post .= x
    x_post[4:6] .+= u[4] .* u[1:3]
    x[1] += 10
    u[4] += 10
    return nothing
end


function scalar_coast!(dx, x, pu, t)
    dx[1] = 0
    return nothing
end


function scalar_jump!(x_post, x, u)
    x_post[1] = x[1] + u[1]
    return nothing
end


function impulsive_dynamics(; eom! = coast_double_integrator!, jump! = TOPLib.default_impulsive_jump!)
    return TOPLib.ImpulsiveControlDynamics(6, 4, eom!, nothing; jump! = jump!)
end


function impulsive_problem(;
    N::Int = 2,
    tf::Float64 = 1.0,
    eom! = coast_double_integrator!,
    jump! = TOPLib.default_impulsive_jump!,
    objective = :fuel,
    x_initial = zeros(6),
    x_final = zeros(6),
    umax::Float64 = 1.0,
    x_final_idx = nothing,
    direction_constraint::Bool = true,
    ceq_path = nothing,
)
    dynamics = impulsive_dynamics(; eom! = eom!, jump! = jump!)
    times = collect(range(0.0, tf; length = N))
    return TOPLib.MultipleShootingForwardProblem(
        dynamics,
        times,
        objective,
        x_initial,
        x_final;
        umax = umax,
        x_final_idx = x_final_idx,
        direction_constraint = direction_constraint,
        ceq_path = ceq_path,
    )
end


function test_jump_then_coast_defects()
    prob = impulsive_problem(; N = 2, tf = 1.0, x_initial = nothing, x_final = nothing)
    x1 = zeros(6)
    # Unit z-direction, throttle 0.5: Δv = [0, 0, 0.5]. Coast for 1 second.
    x2 = [0.0, 0.0, 0.5, 0.0, 0.0, 0.5]
    u = zeros(4, 2)
    u[:, 1] = [0.0, 0.0, 1.0, 0.5]
    u[:, 2] = [0.0, 0.0, -1.0, 0.4]
    x = hcat(x1, x2)
    defects = TOPLib.shooting_defects(
        prob.dynamics, prob.times, x, u;
        ode_method = prob.ode_method,
        ode_reltol = prob.ode_reltol,
        ode_abstol = prob.ode_abstol,
    )
    @test defects ≈ zeros(6, 1) atol = 1e-8

    # The terminal burn is not integrated, so it cannot open a defect.
    u_terminal = copy(u)
    u_terminal[:, 2] = [1.0, 0.0, 0.0, 0.9]
    defects_terminal = TOPLib.shooting_defects(
        prob.dynamics, prob.times, x, u_terminal;
        ode_method = prob.ode_method,
        ode_reltol = prob.ode_reltol,
        ode_abstol = prob.ode_abstol,
    )
    @test defects_terminal ≈ defects atol = 1e-8

    x_bad = copy(x)
    x_bad[1, 2] += 0.1
    defects_bad = TOPLib.shooting_defects(
        prob.dynamics, prob.times, x_bad, u;
        ode_method = prob.ode_method,
        ode_reltol = prob.ode_reltol,
        ode_abstol = prob.ode_abstol,
    )
    @test defects_bad[:, 1] ≈ [0.1, 0.0, 0.0, 0.0, 0.0, 0.0] atol = 1e-8

    @test_throws DimensionMismatch TOPLib.shooting_defects(
        prob.dynamics, prob.times, x, u[:, 1:1];
        ode_method = prob.ode_method,
        ode_reltol = prob.ode_reltol,
        ode_abstol = prob.ode_abstol,
    )

    z = TOPLib.stack_variables(x, u)
    defects_vec(ξ) = vec(TOPLib.shooting_defects(
        prob.dynamics, prob.times,
        TOPLib.unpack_variables(ξ, 6, 4, 2; n_controls = 2)...;
        ode_method = prob.ode_method,
        ode_reltol = prob.ode_reltol,
        ode_abstol = prob.ode_abstol,
    ))
    J = ForwardDiff.jacobian(defects_vec, z)
    @test all(isfinite, J)
    # Defect is x[:, 2] - flow(jump(x[:, 1], u[:, 1])), so the x[:, 2] block is I.
    @test J[:, 7:12] ≈ I(6) atol = 1e-8
    # Columns 17:20 are the terminal burn.
    @test J[:, 17:20] ≈ zeros(6, 4) atol = 1e-8
    h = 1e-6
    zp = copy(z)
    zm = copy(z)
    zp[16] += h
    zm[16] -= h
    fd = (defects_vec(zp) - defects_vec(zm)) / (2h)
    @test J[:, 16] ≈ fd atol = 1e-4
end


function test_coast_does_not_see_the_burn()
    prob = impulsive_problem(;
        N = 2, tf = 1.0, eom! = thrusting_coast!, x_initial = nothing, x_final = nothing,
    )
    x1 = zeros(6)
    x2 = [0.0, 0.0, 0.5, 0.0, 0.0, 0.5]
    u = zeros(4, 2)
    u[:, 1] = [0.0, 0.0, 1.0, 0.5]
    u[:, 2] = [1.0, 0.0, 0.0, 0.2]
    defects = TOPLib.shooting_defects(
        prob.dynamics, prob.times, hcat(x1, x2), u;
        ode_method = prob.ode_method,
        ode_reltol = prob.ode_reltol,
        ode_abstol = prob.ode_abstol,
    )
    @test defects ≈ zeros(6, 1) atol = 1e-8
end


function test_jump_does_not_mutate_the_decision()
    prob = impulsive_problem(; N = 2, tf = 1.0, jump! = mutating_jump!)
    x = zeros(6, 2)
    x[4, 1] = 0.2
    u = zeros(4, 2)
    u[1, :] .= 1.0
    u[4, :] .= 0.3
    x_before = copy(x)
    u_before = copy(u)
    TOPLib.shooting_defects(
        prob.dynamics, prob.times, x, u;
        ode_method = prob.ode_method,
        ode_reltol = prob.ode_reltol,
        ode_abstol = prob.ode_abstol,
    )
    @test x == x_before
    @test u == u_before

    z = TOPLib.stack_variables(x, u)
    z_before = copy(z)
    obj, ceq, _ = TOPLib.get_fitness(prob)(z)
    @test z == z_before
    # Initial boundary is the pre-impulse node. A jump that adds 10 to x[1] must not leak in.
    @test obj ≈ 0.6
    @test ceq[7:12] ≈ x[:, 1]
end


function test_post_impulse_boundary_and_counts()
    x1 = [0.2, 0.0, 0.0, 0.2, 0.0, 0.0]
    x2 = [1.0, 9.0, -4.0, 0.3, 0.4, -0.25]
    u = zeros(4, 2)
    u[:, 1] = [1.0, 0.0, 0.0, 0.7]
    u[:, 2] = [1.0, 0.0, 0.0, 0.5]
    x_post = copy(x2)
    x_post[4] += 0.5

    full = impulsive_problem(; N = 2, x_initial = zeros(6), x_final = x_post)
    x = hcat(x1, x2)
    n_ceq, n_cineq = TOPLib.constraint_counts(full, x, u)
    @test n_ceq == 6 * 1 + 6 + 6
    @test n_cineq == 2
    obj, ceq, cineq = TOPLib.get_fitness(full)(TOPLib.stack_variables(x, u))
    @test obj ≈ sum(u[4, :])
    @test length(ceq) == n_ceq
    @test length(cineq) == n_cineq
    # Pre-impulse initial state, not the state after the first burn (Δvx = 0.7).
    @test ceq[7:12] ≈ x1
    @test ceq[10] ≈ 0.2
    # Final target is the post-impulse state. The node itself still has vx = 0.3.
    @test ceq[13:18] ≈ zeros(6) atol = 1e-8
    @test x2[4] != x_post[4]
    @test cineq ≈ [0.0, 0.0]

    partial = impulsive_problem(;
        N = 2,
        x_initial = x1,
        x_final = [1.0, 0.8],
        x_final_idx = [1, 4],
    )
    _, ceq_p, _ = TOPLib.get_fitness(partial)(TOPLib.stack_variables(x, u))
    @test ceq_p[7:12] ≈ zeros(6)
    @test ceq_p[13:14] ≈ zeros(2) atol = 1e-8

    ceq_path(xk, uk) = [xk[1] - uk[4]]
    pathed = impulsive_problem(;
        N = 2, x_initial = zeros(6), x_final = x_post, ceq_path = ceq_path,
    )
    n_ceq_p, _ = TOPLib.constraint_counts(pathed, x, u)
    @test n_ceq_p == 6 * 1 + 6 + 6 + 2
    _, ceq_path_vals, _ = TOPLib.get_fitness(pathed)(TOPLib.stack_variables(x, u))
    @test ceq_path_vals[end - 1:end] ≈ [x1[1] - u[4, 1], x2[1] - u[4, 2]]

    z_lb, z_ub = TOPLib.decision_bounds(full, -10.0, 10.0)
    @test length(z_lb) == 6 * 2 + 4 * 2
    @test z_lb[end - 3:end] == [-1.0, -1.0, -1.0, 0.0]
    @test z_ub[end - 3:end] == [1.0, 1.0, 1.0, 1.0]

    x_round, u_round = TOPLib.unpack_variables(TOPLib.stack_variables(x, u), 6, 4, 2; n_controls = 2)
    @test x_round == x
    @test u_round == u
end


function test_constructor_rejects_bad_impulses()
    @test_throws ArgumentError TOPLib.ImpulsiveControlDynamics(5, 4, coast_double_integrator!, nothing)
    @test_throws ArgumentError TOPLib.ImpulsiveControlDynamics(6, 3, coast_double_integrator!, nothing)
    @test_throws ArgumentError TOPLib.ImpulsiveControlDynamics(0, 4, coast_double_integrator!, nothing)

    prob = impulsive_problem(; N = 2)
    x0 = zeros(6, 2)
    u_short = zeros(4, 1)
    u_short[1, 1] = 1.0
    @test_throws ArgumentError TOPLib.solve(prob; algo = :ipopt, x0 = x0, u0 = u_short)
end


function test_custom_jump_is_used()
    dynamics = TOPLib.ImpulsiveControlDynamics(1, 1, scalar_coast!, nothing; jump! = scalar_jump!)
    times = [0.0, 1.0]
    prob = TOPLib.MultipleShootingForwardProblem(
        dynamics, times, (x, u) -> sum(u), [0.0], [1.5];
        u_lb = [-10.0], u_ub = [10.0], direction_constraint = false,
    )
    x = [0.0 1.5]
    u = [1.5 4.0]
    defects = TOPLib.shooting_defects(
        dynamics, times, x, u;
        ode_method = prob.ode_method,
        ode_reltol = prob.ode_reltol,
        ode_abstol = prob.ode_abstol,
    )
    @test defects ≈ zeros(1, 1) atol = 1e-8
    _, ceq, cineq = TOPLib.get_fitness(prob)(TOPLib.stack_variables(x, u))
    @test isempty(cineq)
    # Final boundary is post-jump: 1.5 + 4, not the pre-impulse node 1.5.
    @test ceq[end] ≈ (1.5 + 4.0) - 1.5
end


function test_solve_rest_to_rest()
    x_final = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    prob = impulsive_problem(; N = 2, tf = 2.0, x_initial = zeros(6), x_final = x_final, umax = 2.0)
    x0 = zeros(6, 2)
    x0[:, 2] = [0.8, 0.0, 0.0, 0.4, 0.0, 0.0]
    u0 = zeros(4, 2)
    u0[:, 1] = [1.0, 0.0, 0.0, 0.4]
    u0[:, 2] = [-1.0, 0.0, 0.0, 0.4]
    result = TOPLib.solve(prob; algo = :ipopt, x0 = x0, u0 = u0, print_level = 0, tol = 1e-8, max_iter = 200)
    @test result.status in (:Solve_Succeeded, :Solved_To_Acceptable_Level)
    @test result.objective ≈ 1 atol = 1e-5
    @test size(result.x) == size(x0)
    @test size(result.u) == size(u0)
    @test result.x[:, 1] ≈ zeros(6) atol = 1e-5
    defects = TOPLib.shooting_defects(
        prob.dynamics, prob.times, result.x, result.u;
        ode_method = prob.ode_method,
        ode_reltol = prob.ode_reltol,
        ode_abstol = prob.ode_abstol,
    )
    @test defects ≈ zeros(6, 1) atol = 1e-6
    x_post = copy(result.x[:, 2])
    x_post[4:6] .+= result.u[4, 2] .* result.u[1:3, 2]
    @test x_post ≈ x_final atol = 1e-5
    @test abs(result.u[4, 1]) ≈ 0.5 atol = 1e-4
    @test abs(result.u[4, 2]) ≈ 0.5 atol = 1e-4
end


@testset "impulsive defects" begin
    test_jump_then_coast_defects()
    test_coast_does_not_see_the_burn()
    test_jump_does_not_mutate_the_decision()
    test_custom_jump_is_used()
end

@testset "impulsive boundaries" begin
    test_post_impulse_boundary_and_counts()
    test_constructor_rejects_bad_impulses()
end

@testset "solve an impulsive rest-to-rest transfer" begin
    test_solve_rest_to_rest()
end
