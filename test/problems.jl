"""Shared double-integrator fixture for the forward-shooting tests."""

function double_integrator!(dx, x, pu, t)
    u = pu.u
    dx[1:3] = x[4:6]
    dx[4:6] = u[4] * u[1:3]
    return nothing
end

function test_problem(;
    N::Int=3,
    tf::Float64=1.0,
    nx::Int=6,
    nu::Int=4,
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
    return TOPLib.ForwardShootingProblem(
        dynamics,
        times,
        objective,
        x_initial,
        x_final;
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
