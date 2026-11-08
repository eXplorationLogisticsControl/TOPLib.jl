"""
    ContinuousControlDynamics(nx, nu, eom!, params)

Continuous dynamics for a controlled ODE.

`eom!(dx, x, (; params, u), t)` is in-place. `params` holds data that does not
change from one shooting segment to the next. The transcriber passes the
segment control as `u`.
"""
mutable struct ContinuousControlDynamics
    nx::Int
    nu::Int
    eom!::Function
    params

    function ContinuousControlDynamics(
        nx::Int,
        nu::Int,
        eom!::Function,
        params,
    )
        nx > 0 || throw(ArgumentError("nx must be positive"))
        nu > 0 || throw(ArgumentError("nu must be positive"))
        return new(nx, nu, eom!, params)
    end
end
