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
