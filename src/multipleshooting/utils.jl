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

`x` is `nx × N` (one column per node). `u` is `nu × (N - 1)` for a continuous
segment, or `nu × N` for one impulse per node.
"""
function stack_variables(x::AbstractMatrix, u::AbstractMatrix)
    return vcat(vec(x), vec(u))
end


"""
    unpack_variables(z, nx, nu, N; n_controls = N - 1) -> (x, u)

Inverse of [`stack_variables`](@ref). The returned arrays view `z`.

`n_controls` is `N - 1` for a continuous segment and `N` for one impulse per node.
"""
function unpack_variables(
    z::AbstractVector,
    nx::Integer,
    nu::Integer,
    N::Integer;
    n_controls::Integer=Int(N) - 1,
)
    nx_len = Int(nx) * Int(N)
    Nu = Int(n_controls)
    u_len = Int(nu) * Nu
    if length(z) != nx_len + u_len
        throw(DimensionMismatch(
            "z has length $(length(z)), expected $(nx_len + u_len) for nx=$nx, nu=$nu, N=$N, n_controls=$Nu",
        ))
    end
    x = reshape(view(z, 1:nx_len), Int(nx), Int(N))
    u = reshape(view(z, nx_len+1:nx_len+u_len), Int(nu), Nu)
    return x, u
end


"""
    propagate_segments(eom!, u0s, tspans, ps; ode_method, ode_reltol, ode_abstol, ensemble_method)

Integrate independent segments together as one ensemble.

`ensemble_method` is the ensemble scheduler, for example `EnsembleThreads()`.

`u0s[k]`, `tspans[k]`, and `ps[k]` belong to segment `k`. Each initial state and
each parameter's control vector is owned by that trajectory. `eom!` may write
only `dx`. Parameter data shared across `ps` is read by every segment.

Returns the copied end state of each segment, in segment order. An empty input
returns an empty vector.
"""
function propagate_segments(
    eom!,
    u0s::AbstractVector,
    tspans::AbstractVector,
    ps::AbstractVector;
    ode_method,
    ode_reltol::Real,
    ode_abstol::Real,
    ensemble_method,
)
    nseg = length(u0s)
    if length(tspans) != nseg || length(ps) != nseg
        throw(DimensionMismatch(
            "u0s, tspans, and ps must have the same length, got $nseg, $(length(tspans)), $(length(ps))",
        ))
    end
    nseg == 0 && return eltype(u0s)[]

    template = OrdinaryDiffEq.ODEProblem(eom!, u0s[1], tspans[1], ps[1])
    prob_func = function (prob, ctx)
        k = ctx.sim_id
        return OrdinaryDiffEq.remake(prob; u0 = u0s[k], tspan = tspans[k], p = ps[k])
    end
    # `(end state, t, retcode)` so the success check can run after the ensemble.
    output_func = function (sol, ctx)
        return ((copy(sol.u[end]), sol.t[end], sol.retcode), false)
    end
    ensemble = OrdinaryDiffEq.EnsembleProblem(
        template;
        prob_func = prob_func,
        output_func = output_func,
        safetycopy = false,
    )
    sim = OrdinaryDiffEq.solve(
        ensemble,
        ode_method,
        ensemble_method;
        trajectories = nseg,
        reltol=ode_reltol,
        abstol=ode_abstol,
        save_everystep=false,
        verbose=ode_verbose(),
    )

    ends = Vector{typeof(sim.u[1][1])}(undef, nseg)
    for k in 1:nseg
        u_end, t_end, retcode = sim.u[k]
        t_target = tspans[k][2]
        reached = OrdinaryDiffEq.successful_retcode(retcode) &&
            abs(t_end - t_target) <= 1e-8 * (1 + abs(t_target))
        if !reached
            error("ODE integration failed on segment $k with retcode $(retcode) at t=$t_end")
        end
        ends[k] = u_end
    end
    return ends
end


"""
    shooting_defects(dynamics, times, x, u; ode_method, ode_reltol, ode_abstol, ensemble_method)

Multiple-shooting defects for zero-order-hold controls.

Segment `k` is integrated from `x[:, k]` over `[times[k], times[k+1]]` with the
constant control `u[:, k]`. The defect column is `x[:, k+1] - x(t_{k+1})`.
Segments propagate together as one ensemble, scheduled by `ensemble_method`.
`eom!` may write only `dx`. Controls are copied before the integrator sees
them, so an in-place solver cannot alias the decision vector.
"""
function shooting_defects(
    dynamics::ContinuousControlDynamics,
    times::AbstractVector,
    x::AbstractMatrix,
    u::AbstractMatrix;
    ode_method,
    ode_reltol::Real,
    ode_abstol::Real,
    ensemble_method,
)
    nx = dynamics.nx
    N = length(times)
    if size(x) != (nx, N)
        throw(DimensionMismatch("x must be $(nx) × $(N), got $(size(x))"))
    end
    if size(u) != (dynamics.nu, N - 1)
        throw(DimensionMismatch("u must be $(dynamics.nu) × $(N - 1), got $(size(u))"))
    end

    nseg = N - 1
    T = promote_type(eltype(x), eltype(u))
    defects = zeros(T, nx, nseg)
    nseg == 0 && return defects

    # Copies: each trajectory owns its initial state and control.
    u0s = map(k -> collect(x[:, k]), 1:nseg)
    tspans = map(k -> (times[k], times[k + 1]), 1:nseg)
    ps = map(k -> dynamics_input(dynamics.params, collect(u[:, k])), 1:nseg)
    ends = propagate_segments(
        dynamics.eom!,
        u0s,
        tspans,
        ps;
        ode_method = ode_method,
        ode_reltol = ode_reltol,
        ode_abstol = ode_abstol,
        ensemble_method = ensemble_method,
    )
    for k in 1:nseg
        defects[:, k] = x[:, k + 1] .- ends[k][1:nx]
    end
    return defects
end
