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


@testset "dynamics defects" begin
    test_dynamics_defects()
end
