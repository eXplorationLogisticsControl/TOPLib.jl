"""Forward-backward shooting problem (ZOHα transcription)."""


"""
    ForwardBackwardShootingProblem(dynamics, times, objective, x_initial, x_final; kwargs...)

Zero-order-hold forward–backward shooting on a fixed node grid.

`times` has one entry per node (`N = length(times)`, `nseg = N - 1`). Segment
`k` holds one constant control. `alpha` ∈ [0, 1] sets the cut

    n_fwd = floor(Int, alpha * nseg),    match = n_fwd + 1.

Segments `1:n_fwd` are integrated forward from the initial state. Segments
`nseg:-1:n_fwd+1` are integrated backward from the final state. The only
dynamics equality is the mismatch `x_bck[match] - x_fwd[match]`. `alpha = 1`
is pure forward single shooting and `alpha = 0` is pure backward. The default
is `0.5`.

A boundary value passed in full is the integration initial condition, not an
extra equality. `x_initial_idx` and `x_final_idx` fix only those components.
Every other component is a decision variable, so a partial target still gives
the propagator a complete state. `nothing` leaves that whole endpoint free.

`objective` is `:fuel` or a function `(x, u) -> Real`. Inside the fitness, `x`
is the `nx × N` node matrix and `u` is `nu × (N - 1)`:

- column 1 is the initial boundary state and column `N` is the final one;
- columns `2:n_fwd` are forward-propagated;
- columns `n_fwd+1:N-1` are backward-propagated, so an interior match holds
  the backward arrival.

For `:fuel` the cost is `sum(u[4, :])`. Pass `umax` only for the historical
4-vector control `[ux, uy, uz, u]`. Otherwise pass `u_lb` and `u_ub`, each of
length `nu`. With `direction_constraint` (the default), every control column
satisfies `ux^2 + uy^2 + uz^2 ≤ 1`.

`ceq_path(x_k, u_k)` and `cineq_path(x_k, u_k)` see the forward-time state at
the start of each segment. `ceq(x, u)` and `cineq(x, u)` see the full matrices.
Inequalities are feasible when every entry is `≤ 0`.

Continuous dynamics only.
"""
mutable struct ForwardBackwardShootingProblem
    dynamics::ContinuousControlDynamics
    times::Vector{Float64}
    objective::Union{Symbol,Function}
    N::Int
    alpha::Float64
    n_fwd::Int
    u_lb::Vector{Float64}
    u_ub::Vector{Float64}
    direction_constraint::Bool
    x_initial::Union{Vector{Float64},Nothing}
    x_final::Union{Vector{Float64},Nothing}
    x_initial_idx::Union{Vector{Int},Nothing}
    x_final_idx::Union{Vector{Int},Nothing}
    free_initial_idx::Vector{Int}
    free_final_idx::Vector{Int}
    ceq_path::Union{Function,Nothing}
    cineq_path::Union{Function,Nothing}
    ceq::Union{Function,Nothing}
    cineq::Union{Function,Nothing}
    ode_method
    ode_reltol::Float64
    ode_abstol::Float64

    function ForwardBackwardShootingProblem(
        dynamics::ShootingDynamics,
        times::AbstractVector,
        objective::Union{Symbol,Function},
        x_initial::Union{AbstractVector,Nothing},
        x_final::Union{AbstractVector,Nothing};
        alpha::Real=0.5,
        umax::Union{Real,Nothing}=nothing,
        u_lb::Union{AbstractVector,Nothing}=nothing,
        u_ub::Union{AbstractVector,Nothing}=nothing,
        x_initial_idx::Union{AbstractVector,Nothing}=nothing,
        x_final_idx::Union{AbstractVector,Nothing}=nothing,
        direction_constraint::Bool=true,
        ceq_path::Union{Function,Nothing}=nothing,
        cineq_path::Union{Function,Nothing}=nothing,
        ceq::Union{Function,Nothing}=nothing,
        cineq::Union{Function,Nothing}=nothing,
        ode_method=OrdinaryDiffEq.Tsit5(),
        ode_reltol::Real=1e-12,
        ode_abstol::Real=1e-12,
    )
        if !(dynamics isa ContinuousControlDynamics)
            throw(ArgumentError(
                "forward-backward shooting requires ContinuousControlDynamics, got $(typeof(dynamics))",
            ))
        end
        if objective isa Symbol && objective != :fuel
            error("Objective type $objective not supported")
        end
        if objective === :fuel && dynamics.nu < 4
            throw(ArgumentError("minimum-fuel control must have nu >= 4, got $(dynamics.nu)"))
        end
        if direction_constraint && dynamics.nu < 3
            throw(ArgumentError("direction constraint needs nu >= 3, got $(dynamics.nu)"))
        end
        if !(0 <= alpha <= 1)
            throw(ArgumentError("alpha must lie in [0, 1], got $alpha"))
        end
        ode_reltol > 0 || throw(ArgumentError("ode_reltol must be positive"))
        ode_abstol > 0 || throw(ArgumentError("ode_abstol must be positive"))

        times = collect(Float64, times)
        N = length(times)
        N >= 2 || throw(ArgumentError("times must contain at least two nodes"))
        all(diff(times) .> 0) || throw(ArgumentError("times must be strictly increasing"))

        u_lb, u_ub = _control_bounds(dynamics.nu, umax, u_lb, u_ub)
        x_initial, x_initial_idx = _boundary_pair(x_initial, x_initial_idx, dynamics.nx, "x_initial")
        x_final, x_final_idx = _boundary_pair(x_final, x_final_idx, dynamics.nx, "x_final")
        n_fwd = floor(Int, Float64(alpha) * (N - 1))
        return new(
            dynamics,
            times,
            objective,
            N,
            Float64(alpha),
            n_fwd,
            u_lb,
            u_ub,
            direction_constraint,
            x_initial,
            x_final,
            x_initial_idx,
            x_final_idx,
            _free_boundary_indices(x_initial, x_initial_idx, dynamics.nx),
            _free_boundary_indices(x_final, x_final_idx, dynamics.nx),
            ceq_path,
            cineq_path,
            ceq,
            cineq,
            ode_method,
            Float64(ode_reltol),
            Float64(ode_abstol),
        )
    end
end


"""State components that are decision variables. A missing target frees every component."""
function _free_boundary_indices(target, idx, nx::Integer)
    target === nothing && return collect(1:Int(nx))
    idx === nothing && return Int[]
    return setdiff(1:Int(nx), idx)
end


"""Boundary state from fixed values plus the free decision components."""
function _assemble_boundary(nx, target, idx, free_idx, free_values)
    T = eltype(free_values)
    x = zeros(T, nx)
    if target !== nothing
        if idx === nothing
            x .= target
        else
            x[idx] .= target
        end
    end
    if !isempty(free_idx)
        x[free_idx] .= free_values
    end
    return x
end


"""Integrate one segment. `t_end < t_start` integrates backward."""
function _integrate_segment(prob::ForwardBackwardShootingProblem, x0, u, t_start, t_end)
    dynamics = prob.dynamics
    ode_problem = OrdinaryDiffEq.ODEProblem(
        dynamics.eom!,
        collect(x0),
        (t_start, t_end),
        dynamics_input(dynamics.params, collect(u)),
    )
    sol = OrdinaryDiffEq.solve(
        ode_problem,
        prob.ode_method;
        reltol=prob.ode_reltol,
        abstol=prob.ode_abstol,
        save_everystep=false,
        verbose=ode_verbose(),
    )
    t_sol = sol.t[end]
    reached = OrdinaryDiffEq.successful_retcode(sol.retcode) &&
        abs(t_sol - t_end) <= 1e-8 * (1 + abs(t_end))
    if !reached
        error("ODE integration failed from t=$t_start to t=$t_end with retcode $(sol.retcode) at t=$t_sol")
    end
    return collect(sol.u[end][1:dynamics.nx])
end


"""
    propagate_nodes(prob, x_init, x_final, u) -> (x, mismatch)

Node matrix used by the objective and the path constraints, plus the ZOHα
mismatch `x_bck[match] - x_fwd[match]`.

Column 1 of `x` is `x_init` and column `N` is `x_final`. Columns `2:n_fwd` come
from the forward arc. Columns `n_fwd+1:N-1` come from the backward arc.
"""
function propagate_nodes(prob::ForwardBackwardShootingProblem, x_init, x_final, u)
    nx = prob.dynamics.nx
    N = prob.N
    nseg = N - 1
    n_fwd = prob.n_fwd
    times = prob.times
    T = promote_type(eltype(x_init), eltype(x_final), eltype(u))
    x = zeros(T, nx, N)
    x[:, 1] .= x_init
    x[:, N] .= x_final

    x_fwd = x_init
    for k in 1:n_fwd
        x_fwd = _integrate_segment(prob, x_fwd, view(u, :, k), times[k], times[k + 1])
        node = k + 1
        if 1 < node <= n_fwd
            x[:, node] .= x_fwd
        end
    end

    x_bck = x_final
    for k in nseg:-1:(n_fwd + 1)
        x_bck = _integrate_segment(prob, x_bck, view(u, :, k), times[k + 1], times[k])
        if 1 < k < N
            x[:, k] .= x_bck
        end
    end
    return x, x_bck .- x_fwd
end


"""`nx × 2` state guess and `nu × (N - 1)` control guess."""
function _check_boundary_guess(prob::ForwardBackwardShootingProblem, x0, u0)
    nx = prob.dynamics.nx
    nu = prob.dynamics.nu
    nseg = prob.N - 1
    if size(x0) != (nx, 2)
        throw(ArgumentError("x0 must be $(nx) × 2, got $(size(x0))"))
    end
    if size(u0) != (nu, nseg)
        throw(ArgumentError("u0 must be $(nu) × $(nseg), got $(size(u0))"))
    end
    return nothing
end


"""
    decision_vector(prob, x0, u0) -> z

Pack `[free x_initial; free x_final; vec(u)]`.

`x0` is `nx × 2` (initial column, then final column). Fixed boundary entries in
`x0` are ignored.
"""
function decision_vector(prob::ForwardBackwardShootingProblem, x0::AbstractMatrix, u0::AbstractMatrix)
    _check_boundary_guess(prob, x0, u0)
    n0 = length(prob.free_initial_idx)
    nf = length(prob.free_final_idx)
    z = Vector{eltype(x0)}(undef, n0 + nf + length(u0))
    for (i, idx) in enumerate(prob.free_initial_idx)
        z[i] = x0[idx, 1]
    end
    for (i, idx) in enumerate(prob.free_final_idx)
        z[n0 + i] = x0[idx, 2]
    end
    z[n0 + nf + 1:end] .= vec(u0)
    return z
end


"""Unpack a decision vector into the two boundary states and the control matrix."""
function _unpack_decision(prob::ForwardBackwardShootingProblem, z::AbstractVector)
    nx = prob.dynamics.nx
    nu = prob.dynamics.nu
    nseg = prob.N - 1
    n0 = length(prob.free_initial_idx)
    nf = length(prob.free_final_idx)
    expected = n0 + nf + nu * nseg
    if length(z) != expected
        throw(DimensionMismatch(
            "z has length $(length(z)), expected $expected for nx=$nx, nu=$nu, N=$(prob.N)",
        ))
    end
    T = eltype(z)
    free0 = isempty(prob.free_initial_idx) ? zeros(T, 0) : z[1:n0]
    freef = isempty(prob.free_final_idx) ? zeros(T, 0) : z[n0 + 1:n0 + nf]
    u = reshape(view(z, n0 + nf + 1:expected), nu, nseg)
    x_init = _assemble_boundary(nx, prob.x_initial, prob.x_initial_idx, prob.free_initial_idx, free0)
    x_final = _assemble_boundary(nx, prob.x_final, prob.x_final_idx, prob.free_final_idx, freef)
    return x_init, x_final, u
end


"""Node matrix of the right shape for probing callback lengths."""
function _probe_nodes(prob::ForwardBackwardShootingProblem, x0::AbstractMatrix)
    nx = prob.dynamics.nx
    T = eltype(x0)
    free0 = x0[prob.free_initial_idx, 1]
    freef = x0[prob.free_final_idx, 2]
    x_init = _assemble_boundary(nx, prob.x_initial, prob.x_initial_idx, prob.free_initial_idx, free0)
    x_final = _assemble_boundary(nx, prob.x_final, prob.x_final_idx, prob.free_final_idx, freef)
    x = zeros(T, nx, prob.N)
    x[:, 1] .= x_init
    x[:, prob.N] .= x_final
    return x
end


"""
Equality and inequality counts passed to NLPInterface.

Equalities are the mismatch, then `ceq_path`, then `ceq`. Inequalities are the
direction constraints, then `cineq_path`, then `cineq`.
"""
function constraint_counts(prob::ForwardBackwardShootingProblem, x0::AbstractMatrix, u::AbstractMatrix)
    _check_boundary_guess(prob, x0, u)
    nseg = prob.N - 1
    x = _probe_nodes(prob, x0)
    n_dir = prob.direction_constraint ? nseg : 0
    n_ceq = prob.dynamics.nx +
        nseg * path_block_length(prob.ceq_path, x, u) +
        trajectory_block_length(prob.ceq, x, u)
    n_cineq = n_dir +
        nseg * path_block_length(prob.cineq_path, x, u) +
        trajectory_block_length(prob.cineq, x, u)
    return n_ceq, n_cineq
end


"""
    get_fitness(prob) -> fitness

`fitness(z) -> (obj, ceq, cineq)` for NLPInterface.

`z` is `[free x_initial; free x_final; vec(u)]` and is copied on entry. `ceq`
is `[mismatch; ceq_path; ceq]` and `cineq` is `[direction; cineq_path; cineq]`.
"""
function get_fitness(prob::ForwardBackwardShootingProblem)
    function fitness(z)
        zc = copy(z)
        x_init, x_final, u = _unpack_decision(prob, zc)
        x, mismatch = propagate_nodes(prob, x_init, x_final, u)
        obj = get_objective(prob.objective, x, u)
        ceq = mismatch
        if prob.ceq_path !== nothing
            ceq = vcat(ceq, path_constraints(prob.ceq_path, x, u))
        end
        if prob.ceq !== nothing
            ceq = vcat(ceq, trajectory_constraints(prob.ceq, x, u))
        end
        cineq = prob.direction_constraint ? fuel_direction_inequalities(u) : similar(u, 0)
        if prob.cineq_path !== nothing
            cineq = vcat(cineq, path_constraints(prob.cineq_path, x, u))
        end
        if prob.cineq !== nothing
            cineq = vcat(cineq, trajectory_constraints(prob.cineq, x, u))
        end
        return obj, ceq, cineq
    end
    return fitness
end


"""
Variable bounds on the stacked decision vector.

Each free boundary component uses the matching entry of `x_lb` / `x_ub`.
Each control column is boxed by `u_lb` / `u_ub`.
"""
function decision_bounds(prob::ForwardBackwardShootingProblem, x_lb, x_ub)
    nx = prob.dynamics.nx
    nu = prob.dynamics.nu
    nseg = prob.N - 1
    lb_x = state_bound_vector(x_lb, nx, "x_lb")
    ub_x = state_bound_vector(x_ub, nx, "x_ub")
    if any(lb_x .> ub_x)
        throw(ArgumentError("x_lb must be componentwise <= x_ub"))
    end

    n0 = length(prob.free_initial_idx)
    nf = length(prob.free_final_idx)
    z_lb = Vector{Float64}(undef, n0 + nf + nu * nseg)
    z_ub = similar(z_lb)
    for (i, idx) in enumerate(prob.free_initial_idx)
        z_lb[i] = lb_x[idx]
        z_ub[i] = ub_x[idx]
    end
    for (i, idx) in enumerate(prob.free_final_idx)
        z_lb[n0 + i] = lb_x[idx]
        z_ub[n0 + i] = ub_x[idx]
    end
    u_offset = n0 + nf
    for k in 1:nseg
        i1 = u_offset + (k - 1) * nu + 1
        z_lb[i1:i1 + nu - 1] .= prob.u_lb
        z_ub[i1:i1 + nu - 1] .= prob.u_ub
    end
    return z_lb, z_ub
end


"""
    solve(prob; algo=:ipopt, x0, u0, x_lb=-10, x_ub=10, kwargs...)

Solve a [`ForwardBackwardShootingProblem`](@ref) with Ipopt (`:ipopt`) or SNOPT (`:snopt`).

`x0` is `nx × 2`: the initial state, then the final state. Only free boundary
entries are read. `u0` is `nu × (N - 1)`. Remaining keywords are solver options
(`tol`, `max_iter`, `deriv`, ...).

Returns `(x, u, objective, status)`. `x` is the propagated `nx × N` node matrix
from [`propagate_nodes`](@ref). `u` is `nu × (N - 1)`.
"""
function solve(
    prob::ForwardBackwardShootingProblem;
    algo::Symbol=:ipopt,
    x0::AbstractMatrix,
    u0::AbstractMatrix,
    x_lb::Union{Real,AbstractVector}=-DEFAULT_STATE_BOUND,
    x_ub::Union{Real,AbstractVector}=DEFAULT_STATE_BOUND,
    kwargs...,
)
    x0 = convert(Matrix{Float64}, x0)
    u0 = convert(Matrix{Float64}, u0)
    z0 = decision_vector(prob, x0, u0)
    z_lb, z_ub = decision_bounds(prob, x_lb, x_ub)
    if any(z0 .< z_lb) || any(z0 .> z_ub)
        throw(ArgumentError("initial guess is outside the decision bounds"))
    end
    n_ceq, n_cineq = constraint_counts(prob, x0, u0)
    fitness = get_fitness(prob)

    if algo == :ipopt
        solve_fun = NLPInterface.solve_ipopt
    elseif algo == :snopt
        solve_fun = NLPInterface.solve_snopt
    else
        error("Algorithm $algo not supported")
    end

    zopt, fopt, info = solve_fun(
        fitness,
        z0,
        z_lb,
        z_ub,
        n_ceq,
        n_cineq;
        kwargs...,
    )
    x_init, x_final, uopt = _unpack_decision(prob, zopt)
    xopt, _ = propagate_nodes(prob, x_init, x_final, uopt)
    return (x=Matrix(xopt), u=Matrix(uopt), objective=fopt, status=info)
end
