"""Shooting defects and their Jacobian"""

using ForwardDiff
using LinearAlgebra
using Test

if !@isdefined TOPLib
    include(joinpath(@__DIR__, "../src/TOPLib.jl"))
end

if !@isdefined test_problem
    include(joinpath(@__DIR__, "problems.jl"))
end


function test_dynamics_defects()
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


"""Double-integrator step with constant acceleration `u[4] * u[1:3]`."""
function double_integrator_step(x, u, dt)
    a = u[4] .* u[1:3]
    return vcat(x[1:3] .+ x[4:6] .* dt .+ 0.5 .* a .* dt^2, x[4:6] .+ a .* dt)
end


function test_unequal_segment_defects()
    prob = test_problem(; N=4, x_initial=nothing, x_final=nothing)
    times = [0.0, 0.2, 1.0, 1.7]
    u = [
        0.0 1.0 0.0
        0.0 0.0 1.0
        1.0 0.0 0.0
        0.3 0.4 0.2
    ]
    x1 = [0.1, -0.2, 0.3, 0.4, -0.1, 0.2]
    x2 = double_integrator_step(x1, u[:, 1], times[2] - times[1])
    x3 = double_integrator_step(x2, u[:, 2], times[3] - times[2])
    x4 = double_integrator_step(x3, u[:, 3], times[4] - times[3])
    x = hcat(x1, x2, x3, x4)
    defects = TOPLib.shooting_defects(
        prob.dynamics, times, x, u;
        ode_method=prob.ode_method,
        ode_reltol=prob.ode_reltol,
        ode_abstol=prob.ode_abstol,
    )
    @test size(defects) == (6, 3)
    @test defects ≈ zeros(6, 3) atol=1e-8

    # Node 2 is the target of segment 1 and the initial state of segment 2.
    x_bad = copy(x)
    x_bad[1, 2] += 0.1
    defects_bad = TOPLib.shooting_defects(
        prob.dynamics, times, x_bad, u;
        ode_method=prob.ode_method,
        ode_reltol=prob.ode_reltol,
        ode_abstol=prob.ode_abstol,
    )
    @test defects_bad[:, 1] ≈ [0.1, 0.0, 0.0, 0.0, 0.0, 0.0] atol=1e-8
    @test defects_bad[:, 2] ≈ [-0.1, 0.0, 0.0, 0.0, 0.0, 0.0] atol=1e-8
    @test defects_bad[:, 3] ≈ zeros(6) atol=1e-8
end


@testset "dynamics defects" begin
    test_dynamics_defects()
    test_unequal_segment_defects()
end
