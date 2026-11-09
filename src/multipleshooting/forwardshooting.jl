"""Multiple shooting transcribed as a nonlinear program."""


"""Default box on every state component, passed to the NLP as variable bounds."""
const DEFAULT_STATE_BOUND = 10.0

const ShootingDynamics = Union{ContinuousControlDynamics, ImpulsiveControlDynamics}


"""
    MultipleForwardShootingProblem(dynamics, times, objective, x_initial, x_final; kwargs...)

Multiple-shooting problem on a fixed node grid.

`times` has one entry per node (`N = length(times)`). A missing endpoint
(`nothing`) leaves that boundary free. `x_initial_idx` and `x_final_idx` fix
only those state components; omit them to fix the whole vector.

For [`ContinuousControlDynamics`](@ref), each segment holds one control
(`nu × (N - 1)`), constant along the integrated arc. For
[`ImpulsiveControlDynamics`](@ref), each node holds one burn (`nu × N`).
Burns `1:N-1` jump the node state, then the arc coasts with zero control.
Burn `N` is not integrated. `x_initial` constrains the pre-impulse state at
node 1. `x_final` constrains the post-impulse state at node N.

`objective` is `:fuel` or a function `(x, u) -> Real`. For `:fuel` the cost is
`sum(u[4, :])`. Pass `umax` only for the historical 4-vector control
`[ux, uy, uz, u]`, with throttle `u ∈ [0, umax]` and box bounds `±1` on the
direction. Otherwise pass `u_lb` and `u_ub`, each of length `nu`.

With `direction_constraint` (the default), every control column satisfies
`ux^2 + uy^2 + uz^2 ≤ 1`. This needs `nu ≥ 3`. `:fuel` needs `nu ≥ 4`.

`ceq_path(x_k, u_k)` and `cineq_path(x_k, u_k)` are evaluated once per control
column. Each returns a vector of fixed length. `ceq(x, u)` and `cineq(x, u)`
are called once on the full state and control matrices. Inequalities are
feasible when every entry is `≤ 0`.
"""
mutable struct MultipleForwardShootingProblem
    dynamics::ShootingDynamics
    times::Vector{Float64}
    objective::Union{Symbol,Function}
    N::Int
    u_lb::Vector{Float64}
    u_ub::Vector{Float64}
    direction_constraint::Bool
    x_initial::Union{Vector{Float64},Nothing}
    x_final::Union{Vector{Float64},Nothing}
    x_initial_idx::Union{Vector{Int},Nothing}
    x_final_idx::Union{Vector{Int},Nothing}
    ceq_path::Union{Function,Nothing}
    cineq_path::Union{Function,Nothing}
    ceq::Union{Function,Nothing}
    cineq::Union{Function,Nothing}
    ode_method
    ode_reltol::Float64
    ode_abstol::Float64

    function MultipleForwardShootingProblem(
        dynamics::ShootingDynamics,
        times::AbstractVector,
        objective::Union{Symbol,Function},
        x_initial::Union{AbstractVector,Nothing},
        x_final::Union{AbstractVector,Nothing};
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
        if objective isa Symbol && objective != :fuel
            error("Objective type $objective not supported")
        end
        if objective === :fuel && dynamics.nu < 4
            throw(ArgumentError("minimum-fuel control must have nu >= 4, got $(dynamics.nu)"))
        end
        if direction_constraint && dynamics.nu < 3
            throw(ArgumentError("direction constraint needs nu >= 3, got $(dynamics.nu)"))
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
        return new(
            dynamics,
            times,
            objective,
            N,
            u_lb,
            u_ub,
            direction_constraint,
            x_initial,
            x_final,
            x_initial_idx,
            x_final_idx,
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


"""Control box from `umax` (nu = 4 only) or from an explicit pair of vectors."""
function _control_bounds(nu::Integer, umax, u_lb, u_ub)
    has_box = u_lb !== nothing || u_ub !== nothing
    if has_box && umax !== nothing
        throw(ArgumentError("pass either umax or u_lb and u_ub, not both"))
    end
    if has_box
        if u_lb === nothing || u_ub === nothing
            throw(ArgumentError("u_lb and u_ub must be provided together"))
        end
        lb = collect(Float64, u_lb)
        ub = collect(Float64, u_ub)
        if length(lb) != nu || length(ub) != nu
            throw(ArgumentError("u_lb and u_ub must have length nu=$nu"))
        end
        if any(lb .> ub)
            throw(ArgumentError("u_lb must be componentwise <= u_ub"))
        end
        return lb, ub
    end
    if umax === nothing
        throw(ArgumentError("pass umax, or u_lb and u_ub"))
    end
    umax >= 0 || throw(ArgumentError("umax must be non-negative"))
    if nu != 4
        throw(ArgumentError("umax applies only when nu = 4; pass u_lb and u_ub for nu = $nu"))
    end
    return [-1.0, -1.0, -1.0, 0.0], [1.0, 1.0, 1.0, Float64(umax)]
end


"""Target values and the state indices they constrain. `nothing` fixes every component."""
function _boundary_pair(x::Nothing, idx, ::Integer, name::AbstractString)
    if idx !== nothing
        throw(ArgumentError("$(name)_idx requires $name"))
    end
    return nothing, nothing
end
function _boundary_pair(x::AbstractVector, idx, nx::Integer, name::AbstractString)
    values = collect(Float64, x)
    if idx === nothing
        if length(values) != nx
            throw(ArgumentError("$name must have length nx=$nx"))
        end
        return values, nothing
    end
    if !all(i -> i isa Integer, idx)
        throw(ArgumentError("$(name)_idx must contain integers"))
    end
    indices = collect(Int, idx)
    if length(indices) != length(values)
        throw(ArgumentError("$(name)_idx must have the same length as $name"))
    end
    if isempty(indices) || !allunique(indices) || any(i -> i < 1 || i > nx, indices)
        throw(ArgumentError("$(name)_idx must be unique indices in 1:$nx"))
    end
    return values, indices
end


"""Number of equality rows contributed by one boundary target."""
function _boundary_length(target, idx)
    target === nothing && return 0
    return idx === nothing ? length(target) : length(idx)
end


"""
    get_boundary_conditions(prob, x1, xN) -> ceq

Flattened boundary equalities. A full target contributes `x - target`.
An index vector contributes `x[idx] - target`. A missing endpoint is omitted.
"""
function get_boundary_conditions(prob::MultipleForwardShootingProblem, x1, xN)
    T = promote_type(eltype(x1), eltype(xN))
    ceq = T[]
    _append_boundary!(ceq, x1, prob.x_initial, prob.x_initial_idx)
    _append_boundary!(ceq, xN, prob.x_final, prob.x_final_idx)
    return ceq
end

function _append_boundary!(ceq, ::Any, ::Nothing, ::Nothing)
    return ceq
end
function _append_boundary!(ceq, x, target::AbstractVector, ::Nothing)
    append!(ceq, x .- target)
    return ceq
end
function _append_boundary!(ceq, x, target::AbstractVector, idx::AbstractVector)
    append!(ceq, x[idx] .- target)
    return ceq
end


"""Minimum-fuel cost `sum(u[4, :])`, or the value of a user objective `(x, u) -> Real`."""
function get_objective(objective::Symbol, x, u)
    if objective == :fuel
        return sum(u[4, :])
    else
        error("Objective type $objective not supported")
    end
end
function get_objective(objective::Function, x, u)
    value = objective(x, u)
    if !(value isa Real)
        throw(ArgumentError("objective must return a real number, got $(typeof(value))"))
    end
    return value
end


"""One inequality per segment: `ux^2 + uy^2 + uz^2 - 1 ≤ 0`."""
function fuel_direction_inequalities(u::AbstractMatrix)
    nseg = size(u, 2)
    cineq = Vector{eltype(u)}(undef, nseg)
    for k in 1:nseg
        cineq[k] = u[1, k]^2 + u[2, k]^2 + u[3, k]^2 - 1
    end
    return cineq
end


"""Evaluate a path callback at every segment and concatenate the results."""
function path_constraints(fun::Function, x::AbstractMatrix, u::AbstractMatrix)
    T = promote_type(eltype(x), eltype(u))
    out = T[]
    for k in 1:size(u, 2)
        # Copies so a callback cannot mutate the decision vector.
        block = fun(x[:, k], u[:, k])
        if !(block isa AbstractVector)
            throw(ArgumentError("path constraints must return an AbstractVector, got $(typeof(block))"))
        end
        append!(out, block)
    end
    return out
end


"""Length of one segment's path block, probed at the first node."""
function path_block_length(fun::Union{Function,Nothing}, x::AbstractMatrix, u::AbstractMatrix)
    fun === nothing && return 0
    block = fun(x[:, 1], u[:, 1])
    if !(block isa AbstractVector)
        throw(ArgumentError("path constraints must return an AbstractVector, got $(typeof(block))"))
    end
    return length(block)
end


"""Evaluate a full-trajectory constraint once. The callback receives copies."""
function trajectory_constraints(fun::Function, x::AbstractMatrix, u::AbstractMatrix)
    block = fun(collect(x), collect(u))
    if !(block isa AbstractVector)
        throw(ArgumentError("trajectory constraints must return an AbstractVector, got $(typeof(block))"))
    end
    return block
end


"""Control columns: one per segment for a coast with thrust, one per node for impulses."""
function n_controls(::ContinuousControlDynamics, N::Integer)
    return Int(N) - 1
end
function n_controls(::ImpulsiveControlDynamics, N::Integer)
    return Int(N)
end


"""State seen by the final boundary condition. An impulsive node jumps first."""
terminal_state(::ContinuousControlDynamics, xN, ::AbstractMatrix) = xN
function terminal_state(dynamics::ImpulsiveControlDynamics, xN, u::AbstractMatrix)
    return apply_impulsive_jump(dynamics, xN, view(u, :, size(u, 2)))
end


"""Length of a full-trajectory constraint block. `nothing` contributes nothing."""
function trajectory_block_length(fun::Union{Function,Nothing}, x::AbstractMatrix, u::AbstractMatrix)
    fun === nothing && return 0
    return length(trajectory_constraints(fun, x, u))
end


"""
Equality and inequality counts passed to NLPInterface.

Equalities are dynamics defects, then boundary conditions, then `ceq_path`,
then `ceq`. Inequalities are the direction constraints, then `cineq_path`,
then `cineq`.
"""
function constraint_counts(prob::MultipleForwardShootingProblem, x::AbstractMatrix, u::AbstractMatrix)
    nx = prob.dynamics.nx
    nseg = prob.N - 1
    n_ctrl = size(u, 2)
    n_bc = _boundary_length(prob.x_initial, prob.x_initial_idx) +
        _boundary_length(prob.x_final, prob.x_final_idx)
    n_dir = prob.direction_constraint ? n_ctrl : 0
    n_ceq = nx * nseg + n_bc +
        n_ctrl * path_block_length(prob.ceq_path, x, u) +
        trajectory_block_length(prob.ceq, x, u)
    n_cineq = n_dir +
        n_ctrl * path_block_length(prob.cineq_path, x, u) +
        trajectory_block_length(prob.cineq, x, u)
    return n_ceq, n_cineq
end


"""
    get_fitness(prob) -> fitness

`fitness(z) -> (obj, ceq, cineq)` for NLPInterface.

`z` is copied on entry because the solver reuses the input array. `ceq` is
`[vec(defects); boundary; ceq_path; ceq]` and `cineq` is
`[direction; cineq_path; cineq]`. The direction block is omitted when
`direction_constraint` is false. For impulsive dynamics the final boundary
uses the post-impulse state; the initial boundary uses the pre-impulse state.
"""
function get_fitness(prob::MultipleForwardShootingProblem)
    nx = prob.dynamics.nx
    nu = prob.dynamics.nu
    N = prob.N
    Nu = n_controls(prob.dynamics, N)
    function fitness(z)
        zc = copy(z)
        x, u = unpack_variables(zc, nx, nu, N; n_controls=Nu)
        obj = get_objective(prob.objective, x, u)
        defects = shooting_defects(
            prob.dynamics,
            prob.times,
            x,
            u;
            ode_method=prob.ode_method,
            ode_reltol=prob.ode_reltol,
            ode_abstol=prob.ode_abstol,
        )
        xN = terminal_state(prob.dynamics, x[:, N], u)
        ceq = vcat(vec(defects), get_boundary_conditions(prob, x[:, 1], xN))
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


"""Expand a scalar or length-`nx` state bound to a vector of length `nx`."""
function state_bound_vector(bound, nx::Integer, name::AbstractString)
    if bound isa Number
        return fill(Float64(bound), nx)
    end
    values = collect(Float64, bound)
    if length(values) != nx
        throw(ArgumentError("$name must be a scalar or a vector of length nx=$nx"))
    end
    return values
end


"""
Variable bounds on the stacked decision vector.

Every state component uses `x_lb` / `x_ub` (scalars apply to all components).
Each control column is boxed by `u_lb` / `u_ub`.
"""
function decision_bounds(prob::MultipleForwardShootingProblem, x_lb, x_ub)
    nx = prob.dynamics.nx
    nu = prob.dynamics.nu
    N = prob.N
    lb_x = state_bound_vector(x_lb, nx, "x_lb")
    ub_x = state_bound_vector(x_ub, nx, "x_ub")
    if any(lb_x .> ub_x)
        throw(ArgumentError("x_lb must be componentwise <= x_ub"))
    end

    Nu = n_controls(prob.dynamics, N)
    z_lb = Vector{Float64}(undef, nx * N + nu * Nu)
    z_ub = similar(z_lb)
    for k in 1:N
        i1 = (k - 1) * nx + 1
        z_lb[i1:i1+nx-1] .= lb_x
        z_ub[i1:i1+nx-1] .= ub_x
    end
    u_offset = nx * N
    for k in 1:Nu
        i1 = u_offset + (k - 1) * nu + 1
        z_lb[i1:i1+nu-1] .= prob.u_lb
        z_ub[i1:i1+nu-1] .= prob.u_ub
    end
    return z_lb, z_ub
end


"""
    solve(prob; algo=:ipopt, x0, u0, x_lb=-10, x_ub=10, kwargs...)

Solve a [`MultipleForwardShootingProblem`](@ref) with Ipopt (`:ipopt`) or SNOPT (`:snopt`).

`x0` is `nx × N`. `u0` is `nu × (N - 1)` for continuous dynamics and `nu × N`
for impulsive dynamics. Remaining keywords are solver options (`tol`,
`max_iter`, `deriv`, ...).

Returns the named tuple `(x, u, objective, status)`.
"""
function solve(
    prob::MultipleForwardShootingProblem;
    algo::Symbol=:ipopt,
    x0::AbstractMatrix,
    u0::AbstractMatrix,
    x_lb::Union{Real,AbstractVector}=-DEFAULT_STATE_BOUND,
    x_ub::Union{Real,AbstractVector}=DEFAULT_STATE_BOUND,
    kwargs...,
)
    nx = prob.dynamics.nx
    nu = prob.dynamics.nu
    N = prob.N
    Nu = n_controls(prob.dynamics, N)
    x0 = convert(Matrix{Float64}, x0)
    u0 = convert(Matrix{Float64}, u0)
    if size(x0) != (nx, N)
        throw(ArgumentError("x0 must be $(nx) × $(N), got $(size(x0))"))
    end
    if size(u0) != (nu, Nu)
        throw(ArgumentError("u0 must be $(nu) × $(Nu), got $(size(u0))"))
    end

    z0 = stack_variables(x0, u0)
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
    xopt, uopt = unpack_variables(zopt, nx, nu, N; n_controls=Nu)
    return (x=Matrix(xopt), u=Matrix(uopt), objective=fopt, status=info)
end
