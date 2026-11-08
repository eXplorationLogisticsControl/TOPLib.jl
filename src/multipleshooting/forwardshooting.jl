"""Multiple shooting transcribed as a nonlinear program."""


"""Default box on every state component, passed to the NLP as variable bounds."""
const DEFAULT_STATE_BOUND = 10.0


"""
    ForwardShootingProblem(dynamics, times, objective_type, x_initial, x_final; kwargs...)

Fixed-time multiple-shooting problem.

`times` has one entry per node (`N = length(times)`). A missing endpoint
(`nothing`) leaves that boundary free. For `:fuel` the control is the
4-vector `[ux, uy, uz, u]`: the equations of motion must apply the thrust
`u * [ux, uy, uz]`, the throttle satisfies `u ∈ [0, umax]`, and the direction
satisfies `ux^2 + uy^2 + uz^2 ≤ 1`.

Optional `ceq_path(x_k, u_k)` and `cineq_path(x_k, u_k)` are equality and
inequality path constraints on each of the `N - 1` segments. Each returns a
vector of fixed length. Inequalities are feasible when every entry is `≤ 0`.
"""
mutable struct ForwardShootingProblem
    dynamics::ContinuousControlDynamics
    times::Vector{Float64}
    objective_type::Symbol
    N::Int
    umax::Float64
    x_initial::Union{Vector{Float64},Nothing}
    x_final::Union{Vector{Float64},Nothing}
    ceq_path::Union{Function,Nothing}
    cineq_path::Union{Function,Nothing}
    ode_method
    ode_reltol::Float64
    ode_abstol::Float64

    function ForwardShootingProblem(
        dynamics::ContinuousControlDynamics,
        times::AbstractVector,
        objective_type::Symbol,
        x_initial::Union{AbstractVector,Nothing},
        x_final::Union{AbstractVector,Nothing};
        umax::Real,
        ceq_path::Union{Function,Nothing}=nothing,
        cineq_path::Union{Function,Nothing}=nothing,
        ode_method=OrdinaryDiffEq.Tsit5(),
        ode_reltol::Real=1e-12,
        ode_abstol::Real=1e-12,
    )
        objective_type == :fuel ||
            error("Objective type $objective_type not supported")
        dynamics.nu == 4 ||
            throw(ArgumentError("minimum-fuel control must have nu = 4, got $(dynamics.nu)"))
        umax >= 0 || throw(ArgumentError("umax must be non-negative"))
        ode_reltol > 0 || throw(ArgumentError("ode_reltol must be positive"))
        ode_abstol > 0 || throw(ArgumentError("ode_abstol must be positive"))

        times = collect(Float64, times)
        N = length(times)
        N >= 2 || throw(ArgumentError("times must contain at least two nodes"))
        all(diff(times) .> 0) || throw(ArgumentError("times must be strictly increasing"))

        x_initial = _boundary_vector(x_initial, dynamics.nx, "x_initial")
        x_final = _boundary_vector(x_final, dynamics.nx, "x_final")
        return new(
            dynamics,
            times,
            objective_type,
            N,
            Float64(umax),
            x_initial,
            x_final,
            ceq_path,
            cineq_path,
            ode_method,
            Float64(ode_reltol),
            Float64(ode_abstol),
        )
    end
end


function _boundary_vector(x::Nothing, ::Integer, ::AbstractString)
    return nothing
end
function _boundary_vector(x::AbstractVector, nx::Integer, name::AbstractString)
    if length(x) != nx
        throw(ArgumentError("$name must have length nx=$nx"))
    end
    return collect(Float64, x)
end


"""
    get_boundary_conditions(prob, x1, xN) -> ceq

Flattened boundary equalities `x1 - x_initial` and then `xN - x_final`.
An endpoint that is `nothing` is omitted.
"""
function get_boundary_conditions(prob::ForwardShootingProblem, x1, xN)
    T = promote_type(eltype(x1), eltype(xN))
    ceq = T[]
    if prob.x_initial isa AbstractVector
        append!(ceq, x1 .- prob.x_initial)
    end
    if prob.x_final isa AbstractVector
        append!(ceq, xN .- prob.x_final)
    end
    return ceq
end


"""Minimum-fuel cost `sum(u[4, :])`, the sum of segment throttles."""
function get_objective(objective_type::Symbol, x, u)
    if objective_type == :fuel
        return sum(u[4, :])
    else
        error("Objective type $objective_type not supported")
    end
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


"""
Equality and inequality counts passed to NLPInterface.

Equalities are dynamics defects, then boundary conditions, then `ceq_path`.
Inequalities are the fuel direction constraints, then `cineq_path`.
"""
function constraint_counts(prob::ForwardShootingProblem, x::AbstractMatrix, u::AbstractMatrix)
    nx = prob.dynamics.nx
    nseg = prob.N - 1
    n_bc = (prob.x_initial isa AbstractVector ? nx : 0) +
        (prob.x_final isa AbstractVector ? nx : 0)
    n_ceq = nx * nseg + n_bc + nseg * path_block_length(prob.ceq_path, x, u)
    n_cineq = nseg + nseg * path_block_length(prob.cineq_path, x, u)
    return n_ceq, n_cineq
end


"""
    get_fitness(prob) -> fitness

`fitness(z) -> (obj, ceq, cineq)` for NLPInterface.

`z` is copied on entry because the solver reuses the input array. `ceq` is
`[vec(defects); boundary; ceq_path]` and `cineq` is `[direction; cineq_path]`.
"""
function get_fitness(prob::ForwardShootingProblem)
    nx = prob.dynamics.nx
    nu = prob.dynamics.nu
    N = prob.N
    function fitness(z)
        zc = copy(z)
        x, u = unpack_variables(zc, nx, nu, N)
        obj = get_objective(prob.objective_type, x, u)
        defects = shooting_defects(
            prob.dynamics,
            prob.times,
            x,
            u;
            ode_method=prob.ode_method,
            ode_reltol=prob.ode_reltol,
            ode_abstol=prob.ode_abstol,
        )
        ceq = vcat(vec(defects), get_boundary_conditions(prob, x[:, 1], x[:, N]))
        if prob.ceq_path !== nothing
            ceq = vcat(ceq, path_constraints(prob.ceq_path, x, u))
        end
        cineq = fuel_direction_inequalities(u)
        if prob.cineq_path !== nothing
            cineq = vcat(cineq, path_constraints(prob.cineq_path, x, u))
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
Direction components lie in `[-1, 1]` and the throttle lies in `[0, umax]`.
"""
function decision_bounds(prob::ForwardShootingProblem, x_lb, x_ub)
    nx = prob.dynamics.nx
    nu = prob.dynamics.nu
    N = prob.N
    lb_x = state_bound_vector(x_lb, nx, "x_lb")
    ub_x = state_bound_vector(x_ub, nx, "x_ub")
    if any(lb_x .> ub_x)
        throw(ArgumentError("x_lb must be componentwise <= x_ub"))
    end

    nseg = N - 1
    z_lb = Vector{Float64}(undef, nx * N + nu * nseg)
    z_ub = similar(z_lb)
    for k in 1:N
        i1 = (k - 1) * nx + 1
        z_lb[i1:i1+nx-1] .= lb_x
        z_ub[i1:i1+nx-1] .= ub_x
    end
    u_offset = nx * N
    for k in 1:nseg
        i1 = u_offset + (k - 1) * nu + 1
        z_lb[i1:i1+2] .= -1.0
        z_ub[i1:i1+2] .= 1.0
        z_lb[i1+3] = 0.0
        z_ub[i1+3] = prob.umax
    end
    return z_lb, z_ub
end


"""
    solve(prob; algo=:ipopt, x0, u0, x_lb=-10, x_ub=10, kwargs...)

Solve a [`ForwardShootingProblem`](@ref) with Ipopt (`:ipopt`) or SNOPT (`:snopt`).

`x0` is `nx × N` and `u0` is `nu × (N - 1)`. Remaining keywords are solver
options (`tol`, `max_iter`, `deriv`, ...).

Returns the named tuple `(x, u, objective, status)`.
"""
function solve(
    prob::ForwardShootingProblem;
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
    x0 = convert(Matrix{Float64}, x0)
    u0 = convert(Matrix{Float64}, u0)
    if size(x0) != (nx, N)
        throw(ArgumentError("x0 must be $(nx) × $(N), got $(size(x0))"))
    end
    if size(u0) != (nu, N - 1)
        throw(ArgumentError("u0 must be $(nu) × $(N - 1), got $(size(u0))"))
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
    xopt, uopt = unpack_variables(zopt, nx, nu, N)
    return (x=Matrix(xopt), u=Matrix(uopt), objective=fopt, status=info)
end
