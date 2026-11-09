"""Forward-backward ZOHα shooting."""

using ForwardDiff
using LinearAlgebra
using OrdinaryDiffEq
using Test

if !@isdefined TOPLib
    include(joinpath(@__DIR__, "../src/TOPLib.jl"))
end

if !@isdefined double_integrator!
    include(joinpath(@__DIR__, "problems.jl"))
end


function fwdbck_problem(;
    N::Int=3,
    tf::Float64=1.0,
    nx::Int=6,
    nu::Int=4,
    alpha::Real=0.5,
    objective=:fuel,
    x_initial=zeros(nx),
    x_final=zeros(nx),
    umax::Union{Real,Nothing}=nothing,
    u_lb=nothing,
    u_ub=nothing,
    x_initial_idx=nothing,
    x_final_idx=nothing,
    direction_constraint::Bool=true,
    ceq_path=nothing,
    cineq_path=nothing,
    ceq=nothing,
    cineq=nothing,
)
    dynamics = TOPLib.ContinuousControlDynamics(nx, nu, double_integrator!, nothing)
    times = collect(range(0.0, tf; length=N))
    if u_lb === nothing && umax === nothing
        umax = 1.0
    end
    bounds = u_lb === nothing ? (; umax) : (; u_lb, u_ub)
    return TOPLib.ForwardBackwardShootingProblem(
        dynamics,
        times,
        objective,
        x_initial,
        x_final;
        alpha=alpha,
        bounds...,
        x_initial_idx=x_initial_idx,
        x_final_idx=x_final_idx,
        direction_constraint=direction_constraint,
        ceq_path=ceq_path,
        cineq_path=cineq_path,
        ceq=ceq,
        cineq=cineq,
    )
end


function flow_segment(x0, u, t0, tf)
    sol = solve(
        ODEProblem(double_integrator!, collect(x0), (t0, tf), (; params=nothing, u=collect(u))),
        Tsit5();
        reltol=1e-12,
        abstol=1e-12,
        save_everystep=false,
        verbose=TOPLib.ode_verbose(),
    )
    return sol.u[end]
end


function flow_forward(times, x_init, u)
    x = collect(x_init)
    nodes = [x]
    for k in 1:(length(times) - 1)
        x = flow_segment(x, u[:, k], times[k], times[k + 1])
        push!(nodes, x)
    end
    return nodes
end


function test_problem_shape()
    mid = fwdbck_problem(; N=5, alpha=0.5)
    @test mid.alpha == 0.5
    @test mid.n_fwd == 2
    @test fwdbck_problem(; N=5, alpha=0).n_fwd == 0
    @test fwdbck_problem(; N=5, alpha=1).n_fwd == 4
    @test fwdbck_problem(; N=4, alpha=0.5).n_fwd == 1
    @test fwdbck_problem().ode_reltol == 1e-12
    @test fwdbck_problem().ode_abstol == 1e-12

    @test_throws ArgumentError fwdbck_problem(; alpha=-0.1)
    @test_throws ArgumentError fwdbck_problem(; alpha=1.1)

    impulsive = TOPLib.ImpulsiveControlDynamics(6, 4, double_integrator!, nothing)
    @test_throws ArgumentError TOPLib.ForwardBackwardShootingProblem(
        impulsive, [0.0, 1.0], :fuel, zeros(6), zeros(6); umax=1.0,
    )

    partial = fwdbck_problem(;
        N=3,
        x_initial=zeros(6),
        x_final=zeros(3),
        x_final_idx=[1, 2, 3],
    )
    @test partial.free_initial_idx == Int[]
    @test partial.free_final_idx == [4, 5, 6]
    z_lb, z_ub = TOPLib.decision_bounds(partial, fill(-2.0, 6), [1.0, 2, 3, 4, 5, 6])
    @test length(z_lb) == 3 + 4 * 2
    @test z_lb[1:3] == fill(-2.0, 3)
    @test z_ub[1:3] == [4.0, 5.0, 6.0]
    @test z_lb[end - 3:end] == [-1.0, -1.0, -1.0, 0.0]
    @test z_ub[end - 3:end] == [1.0, 1.0, 1.0, 1.0]

    open_ends = fwdbck_problem(; N=3, x_initial=nothing, x_final=nothing)
    @test open_ends.free_initial_idx == collect(1:6)
    @test open_ends.free_final_idx == collect(1:6)
    z_open, _ = TOPLib.decision_bounds(open_ends, -10.0, 10.0)
    @test length(z_open) == 6 + 6 + 4 * 2

    x0 = ones(6, 2)
    x0[4:6, 2] .= [0.1, 0.2, 0.3]
    u0 = zeros(4, 2)
    u0[1, :] .= 1.0
    z = TOPLib.decision_vector(partial, x0, u0)
    @test z[1:3] == [0.1, 0.2, 0.3]
    @test z[4:end] == vec(u0)
end


function test_mismatch_and_node_storage()
    N = 4
    times = collect(range(0.0, 1.0; length=N))
    u = zeros(4, N - 1)
    for k in 1:(N - 1)
        u[:, k] = [1.0, 0.0, 0.0, 0.1 * k]
    end
    x_init = zeros(6)
    nodes = flow_forward(times, x_init, u)
    x_final = nodes[end]

    for alpha in (0.0, 0.5, 1.0)
        prob = fwdbck_problem(; N=N, alpha=alpha, x_initial=x_init, x_final=x_final)
        x0 = hcat(x_init, x_final)
        _, ceq, _ = TOPLib.get_fitness(prob)(TOPLib.decision_vector(prob, x0, u))
        @test ceq[1:6] ≈ zeros(6) atol=1e-8
    end

    # alpha = 1: mismatch is the prescribed final state minus the forward arrival.
    forward = fwdbck_problem(; N=N, alpha=1, x_initial=x_init, x_final=x_final .+ [0.1, 0, 0, 0, 0, 0])
    x_bad = hcat(x_init, x_final)
    _, ceq_fwd, _ = TOPLib.get_fitness(forward)(TOPLib.decision_vector(forward, x_bad, u))
    @test ceq_fwd[1:6] ≈ [0.1, 0, 0, 0, 0, 0] atol=1e-8

    # Interior cut stores the backward arrival. A wrong final state keeps the two arrivals apart.
    cut = fwdbck_problem(; N=N, alpha=0.5, x_initial=x_init, x_final=x_final .+ [0.2, 0, 0, 0, 0, 0])
    x_nodes, mismatch = TOPLib.propagate_nodes(cut, x_init, cut.x_final, u)
    @test x_nodes[:, 1] ≈ x_init atol=1e-12
    @test x_nodes[:, N] ≈ cut.x_final atol=1e-12
    x_at_3 = flow_segment(cut.x_final, u[:, 3], times[4], times[3])
    x_at_2 = flow_segment(x_at_3, u[:, 2], times[3], times[2])
    @test x_nodes[:, 3] ≈ x_at_3 atol=1e-8
    @test x_nodes[:, 2] ≈ x_at_2 atol=1e-8
    @test mismatch ≈ x_at_2 .- nodes[2] atol=1e-8
    @test norm(mismatch) > 1e-2

    pure_back = fwdbck_problem(; N=N, alpha=0, x_initial=x_init, x_final=x_final)
    back_nodes, back_mismatch = TOPLib.propagate_nodes(pure_back, x_init, x_final, u)
    @test back_mismatch ≈ zeros(6) atol=1e-8
    @test back_nodes[:, 1] ≈ x_init atol=1e-12
    @test back_nodes[:, 2] ≈ nodes[2] atol=1e-8
end


function test_fitness_layout()
    ceq_path(x, u) = [x[1] - u[4]]
    cineq_path(x, u) = [u[4] - 0.2]
    ceq(x, u) = diff(u[5, :])
    cineq(x, u) = [u[4, 1] - 0.5]
    u_lb = [-1.0, -1.0, -1.0, 0.0, 2.0]
    u_ub = [1.0, 1.0, 1.0, 1.0, 4.0]
    prob = fwdbck_problem(;
        N=4,
        nu=5,
        alpha=0.5,
        objective=:fuel,
        x_initial=zeros(6),
        x_final=zeros(6),
        u_lb=u_lb,
        u_ub=u_ub,
        ceq_path=ceq_path,
        cineq_path=cineq_path,
        ceq=ceq,
        cineq=cineq,
    )
    x0 = zeros(6, 2)
    u = zeros(5, 3)
    u[1, :] .= 1.0
    u[4, :] .= [0.05, 0.5, 0.1]
    u[5, :] .= [2.0, 3.0, 4.0]
    n_ceq, n_cineq = TOPLib.constraint_counts(prob, x0, u)
    @test n_ceq == 6 + 3 + 2
    @test n_cineq == 3 + 3 + 1
    obj, ceq_val, cineq_val = TOPLib.get_fitness(prob)(TOPLib.decision_vector(prob, x0, u))
    @test obj ≈ sum(u[4, :])
    @test length(ceq_val) == n_ceq
    @test length(cineq_val) == n_cineq
    x_nodes, mismatch = TOPLib.propagate_nodes(prob, zeros(6), zeros(6), u)
    @test ceq_val[1:6] ≈ mismatch atol=1e-8
    @test ceq_val[7:9] ≈ [x_nodes[1, k] - u[4, k] for k in 1:3] atol=1e-8
    @test ceq_val[10:11] ≈ [1.0, 1.0]
    @test cineq_val[1:3] ≈ zeros(3) atol=1e-12
    @test cineq_val[4:6] ≈ u[4, :] .- 0.2
    @test cineq_val[end] ≈ 0.05 - 0.5

    bare = fwdbck_problem(;
        N=3,
        alpha=1,
        direction_constraint=false,
        x_initial=nothing,
        x_final=nothing,
        objective=(x, u) -> -x[1, end],
    )
    x_bare = zeros(6, 2)
    x_bare[1, 2] = 0.4
    u_bare = zeros(4, 2)
    u_bare[1, :] .= 1.0
    n_ceq_bare, n_cineq_bare = TOPLib.constraint_counts(bare, x_bare, u_bare)
    @test n_ceq_bare == 6
    @test n_cineq_bare == 0
    obj_bare, _, cineq_bare = TOPLib.get_fitness(bare)(TOPLib.decision_vector(bare, x_bare, u_bare))
    @test isempty(cineq_bare)
    @test obj_bare ≈ -0.4
end


function test_mismatch_jacobian()
    prob = fwdbck_problem(;
        N=3,
        alpha=0.5,
        objective=(x, u) -> (x[1, end] - 0.25)^2,
        x_initial=zeros(6),
        x_final=zeros(5),
        x_final_idx=[2, 3, 4, 5, 6],
    )
    x0 = zeros(6, 2)
    x0[1, 2] = 0.1
    u0 = zeros(4, 2)
    u0[:, 1] = [1.0, 0.0, 0.0, 0.4]
    u0[:, 2] = [-1.0, 0.0, 0.0, 0.2]
    z = TOPLib.decision_vector(prob, x0, u0)
    mismatch(ξ) = TOPLib.get_fitness(prob)(ξ)[2][1:6]
    J = ForwardDiff.jacobian(mismatch, z)
    @test size(J) == (6, length(z))
    @test all(isfinite, J)
    # A final-position offset is invariant for the double integrator, and the
    # forward arc does not see it, so the match residual moves one-for-one.
    @test J[1, 1] ≈ 1.0 atol=1e-6

    h = 1e-6
    throttle = 1 + 4
    zp = copy(z)
    zm = copy(z)
    zp[throttle] += h
    zm[throttle] -= h
    fd = (mismatch(zp) - mismatch(zm)) / (2h)
    @test J[:, throttle] ≈ fd atol=1e-4
end


function test_solve()
    prob = fwdbck_problem(; N=3, tf=1.0, alpha=0.5, umax=0.5)
    x0 = zeros(6, 2)
    u0 = zeros(4, 2)
    u0[1, :] .= 1.0
    result = TOPLib.solve(prob; algo=:ipopt, x0=x0, u0=u0, print_level=0, tol=1e-8, max_iter=100)
    @test result.status in (:Solve_Succeeded, :Solved_To_Acceptable_Level)
    @test result.objective ≈ 0 atol=1e-6
    @test size(result.x) == (6, 3)
    @test size(result.u) == size(u0)
    @test all(abs.(result.x) .< 1e-5)
    @test all(abs.(result.u[4, :]) .< 1e-5)
    @test_throws ErrorException TOPLib.solve(prob; algo=:missing, x0=x0, u0=u0)
    @test_throws ArgumentError TOPLib.solve(prob; algo=:ipopt, x0=zeros(6, 3), u0=u0)

    # Rest-to-rest hop. Two equal segments, acceleration +1 then -1, land at x = 0.25.
    hop = fwdbck_problem(;
        N=3,
        alpha=0.5,
        umax=1.0,
        objective=(x, u) -> (x[1, end] - 0.25)^2,
        x_initial=zeros(6),
        x_final=zeros(5),
        x_final_idx=[2, 3, 4, 5, 6],
    )
    x_hop = zeros(6, 2)
    x_hop[1, 2] = 0.25
    u_hop = zeros(4, 2)
    u_hop[:, 1] = [1.0, 0.0, 0.0, 1.0]
    u_hop[:, 2] = [-1.0, 0.0, 0.0, 1.0]
    hopped = TOPLib.solve(
        hop; algo=:ipopt, x0=x_hop, u0=u_hop, print_level=0, tol=1e-8, max_iter=100,
    )
    @test hopped.status in (:Solve_Succeeded, :Solved_To_Acceptable_Level)
    @test hopped.x[1, end] ≈ 0.25 atol=1e-5
    @test hopped.objective ≈ 0 atol=1e-8
    z_hop = TOPLib.decision_vector(hop, hcat(hopped.x[:, 1], hopped.x[:, end]), hopped.u)
    _, ceq_hop, _ = TOPLib.get_fitness(hop)(z_hop)
    @test norm(ceq_hop[1:6]) < 1e-6
end


@testset "forward-backward problem" begin
    test_problem_shape()
end

@testset "forward-backward mismatch" begin
    test_mismatch_and_node_storage()
end

@testset "forward-backward fitness" begin
    test_fitness_layout()
    test_mismatch_jacobian()
end

@testset "forward-backward solve" begin
    test_solve()
end
