"""Default jump: copy `x`, then `x_post[4:6] += u[4] * u[1:3]`."""
function default_impulsive_jump!(x_post, x, u)
    x_post .= x
    x_post[4:6] .+= u[4] .* u[1:3]
    return nothing
end


"""
    ImpulsiveControlDynamics(nx, nu, eom!, params; jump! = default_impulsive_jump!)

Coast dynamics plus an instantaneous jump at each node.

`eom!(dx, x, (; params, u), t)` is the coast. The transcriber integrates it with
a zero control, so a coast that reads `u` does not see the burn. `jump!(x_post, x, u)`
writes every component of the post-impulse state and must not rely on `x_post`
aliasing `x`.

The default jump copies `x` and adds `u[4] * u[1:3]` to velocity components
`4:6`. That default needs `nx ≥ 6` and `nu ≥ 4`. Pass `jump!` for any other layout.
"""
mutable struct ImpulsiveControlDynamics
    nx::Int
    nu::Int
    eom!::Function
    params
    jump!::Function

    function ImpulsiveControlDynamics(
        nx::Int,
        nu::Int,
        eom!::Function,
        params;
        jump!::Function=default_impulsive_jump!,
    )
        nx > 0 || throw(ArgumentError("nx must be positive"))
        nu > 0 || throw(ArgumentError("nu must be positive"))
        if jump! === default_impulsive_jump!
            if nx < 6
                throw(ArgumentError(
                    "default impulsive jump needs nx >= 6 (velocity in components 4:6), got $nx",
                ))
            end
            if nu < 4
                throw(ArgumentError(
                    "default impulsive jump needs nu >= 4 (direction and throttle), got $nu",
                ))
            end
        end
        return new(nx, nu, eom!, params, jump!)
    end
end


"""
Post-impulse state. `x` and `u` are copied before `jump!`, so a jump cannot
alias the decision vector.
"""
function apply_impulsive_jump(dynamics::ImpulsiveControlDynamics, x, u)
    T = promote_type(eltype(x), eltype(u))
    xk = Vector{T}(undef, dynamics.nx)
    uk = Vector{T}(undef, dynamics.nu)
    xk .= x
    uk .= u
    x_post = Vector{T}(undef, dynamics.nx)
    dynamics.jump!(x_post, xk, uk)
    return x_post
end


"""
    shooting_defects(dynamics::ImpulsiveControlDynamics, times, x, u; kwargs...)

Defects for one impulse per node.

`u` is `nu × N`. Segment `k` jumps from `x[:, k]` with `u[:, k]`, then coasts
to `times[k + 1]` with a zero control. The defect column is
`x[:, k + 1] - x(t_{k + 1})`. Column `N` is the terminal burn and is not
integrated. Jumps run on the calling thread. The coasts then propagate together
as one ensemble, scheduled by `ensemble_method`. `eom!` may write only `dx`.
"""
function shooting_defects(
    dynamics::ImpulsiveControlDynamics,
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
    if size(u) != (dynamics.nu, N)
        throw(DimensionMismatch("u must be $(dynamics.nu) × $(N), got $(size(u))"))
    end

    nseg = N - 1
    T = promote_type(eltype(x), eltype(u))
    defects = zeros(T, nx, nseg)
    nseg == 0 && return defects

    # Jumps stay on this thread. Each coast owns its post-jump state and a zero control.
    u0s = map(k -> apply_impulsive_jump(dynamics, view(x, :, k), view(u, :, k)), 1:nseg)
    tspans = map(k -> (times[k], times[k + 1]), 1:nseg)
    ps = map(u0s) do xk
        uk = fill(zero(eltype(xk)), dynamics.nu)
        return dynamics_input(dynamics.params, uk)
    end
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
