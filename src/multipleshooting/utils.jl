"""Stack and integrate multiple-shooting variables."""


"""Bundle constant parameters with one segment's control, `(; params, u)`."""
dynamics_input(params, u::AbstractVector) = (; params, u)


"""Silent integrator output. OrdinaryDiffEq v7 rejects `verbose=false`."""
function ode_verbose()
    if isdefined(OrdinaryDiffEq, :DEVerbosity)
        return OrdinaryDiffEq.DEVerbosity(OrdinaryDiffEq.SciMLLogging.None())
    else
        return false
    end
end


"""
    stack_variables(x, u) -> z

Column-major stack `z = [vec(x); vec(u)]`.

`x` is `nx × N` (one column per node) and `u` is `nu × (N - 1)` (one column per
zero-order-hold segment).
"""
function stack_variables(x::AbstractMatrix, u::AbstractMatrix)
    return vcat(vec(x), vec(u))
end


"""
    unpack_variables(z, nx, nu, N) -> (x, u)

Inverse of [`stack_variables`](@ref). The returned arrays view `z`.
"""
function unpack_variables(z::AbstractVector, nx::Integer, nu::Integer, N::Integer)
    nx_len = Int(nx) * Int(N)
    u_len = Int(nu) * (Int(N) - 1)
    if length(z) != nx_len + u_len
        throw(DimensionMismatch(
            "z has length $(length(z)), expected $(nx_len + u_len) for nx=$nx, nu=$nu, N=$N",
        ))
    end
    x = reshape(view(z, 1:nx_len), Int(nx), Int(N))
    u = reshape(view(z, nx_len+1:nx_len+u_len), Int(nu), Int(N) - 1)
    return x, u
end


"""
    shooting_defects(dynamics, times, x, u; ode_method, ode_reltol, ode_abstol)

Multiple-shooting defects for zero-order-hold controls.

Segment `k` is integrated from `x[:, k]` over `[times[k], times[k+1]]` with the
constant control `u[:, k]`. The defect column is `x[:, k+1] - x(t_{k+1})`.
Controls are copied before the integrator sees them, so an in-place solver
cannot alias the decision vector.
"""
function shooting_defects(
    dynamics::ContinuousControlDynamics,
    times::AbstractVector,
    x::AbstractMatrix,
    u::AbstractMatrix;
    ode_method,
    ode_reltol::Real,
    ode_abstol::Real,
)
    nx = dynamics.nx
    N = length(times)
    if size(x) != (nx, N)
        throw(DimensionMismatch("x must be $(nx) × $(N), got $(size(x))"))
    end
    if size(u) != (dynamics.nu, N - 1)
        throw(DimensionMismatch("u must be $(dynamics.nu) × $(N - 1), got $(size(u))"))
    end

    T = promote_type(eltype(x), eltype(u))
    defects = zeros(T, nx, N - 1)
    for k in 1:(N - 1)
        # Copies: the ODE solver may store `u0` and `p` for the duration of the step.
        xk = collect(x[:, k])
        uk = collect(u[:, k])
        ode_problem = OrdinaryDiffEq.ODEProblem(
            dynamics.eom!,
            xk,
            (times[k], times[k + 1]),
            dynamics_input(dynamics.params, uk),
        )
        sol = OrdinaryDiffEq.solve(
            ode_problem,
            ode_method;
            reltol = ode_reltol,
            abstol = ode_abstol,
            save_everystep = false,
            verbose = ode_verbose(),
        )
        t_end = sol.t[end]
        reached = OrdinaryDiffEq.successful_retcode(sol.retcode) &&
            abs(t_end - times[k + 1]) <= 1e-8 * (1 + abs(times[k + 1]))
        if !reached
            error("ODE integration failed on segment $k with retcode $(sol.retcode) at t=$t_end")
        end
        defects[:, k] = x[:, k + 1] .- sol.u[end][1:nx]
    end
    return defects
end
