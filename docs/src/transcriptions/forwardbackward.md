```@meta
CurrentModule = TOPLib
```

# Forward-backward shooting

Forward-backward shooting is a zero-order-hold ZOHα transcription on a fixed node grid. `times` holds one entry per node, with `N = length(times)`, `nseg = N − 1`, and `N ≥ 2`. The dynamics are [`ContinuousControlDynamics`](@ref). Segment `k` holds one control, constant on that segment, so `u` is `nu × nseg`.

The cut is set by `alpha ∈ [0, 1]`:

```math
n_{\mathrm{fwd}} = \lfloor \alpha \, (N - 1) \rfloor.
```

The default is `alpha = 0.5`. Segments `1:n_fwd` are integrated forward from the initial state. Segments `nseg, nseg−1, …, n_fwd+1` are integrated backward from the final state. The dynamics equality is the mismatch where the two integrations meet:

```math
x_{\mathrm{bck}}[n_{\mathrm{fwd}} + 1] - x_{\mathrm{fwd}}[n_{\mathrm{fwd}} + 1] = 0.
```

`alpha = 1` integrates every segment forward. `alpha = 0` integrates every segment backward.

A boundary value passed in full is the initial condition of that integration. `x_initial_idx` and `x_final_idx` fix only those components. Every other component is a decision variable, and the propagator still receives a complete state. Passing `nothing` for an endpoint makes every component of that endpoint a decision variable.

Inside the objective and the constraints, `x` is the `nx × N` node matrix:

- column 1 is the initial boundary state and column `N` is the final one,
- columns `2:n_fwd` come from the forward arc,
- columns `n_fwd+1:N−1` come from the backward arc, so the interior match holds the backward arrival.

## Constraints and cost

`ceq_path(x_k, u_k)` and `cineq_path(x_k, u_k)` see the forward-time state at the start of each segment. `ceq(x, u)` and `cineq(x, u)` see the full matrices. A constraint is feasible when every entry is `≤ 0`.

The equality vector is the mismatch, then `ceq_path`, then `ceq`. The inequality vector is the direction constraint (the default) `u_1^2 + u_2^2 + u_3^2 - 1 ≤ 0` on every control column, then `cineq_path`, then `cineq`. The direction constraint requires `nu ≥ 3`.

The objective is `:fuel` or a function `(x, u) -> Real`. `:fuel` costs `sum(u[4, :])` and requires `nu ≥ 4`. Pass `umax` only for the 4-vector control `[u_x, u_y, u_z, u]`. Otherwise pass `u_lb` and `u_ub`, each of length `nu`.

## Building a problem

The decision vector is `[free x_initial; free x_final; vec(u)]`. `x0` is `nx × 2`: the initial state, then the final state. Entries that the boundary targets already fix are ignored. `u0` is `nu × (N − 1)`. [`solve`](@ref) propagates the optimal boundaries and returns `(x, u, objective, status)`, where `x` is the `nx × N` node matrix.

```julia
dynamics = ContinuousControlDynamics(nx, nu, eom!, params)

prob = ForwardBackwardShootingProblem(
    dynamics,
    times,
    objective,
    x_initial,
    x_final;
    alpha = 0.5,
    u_lb = u_lb,
    u_ub = u_ub,
    x_final_idx = x_final_idx,
    ceq = ceq,
)

result = solve(
    prob;
    algo = :ipopt,
    x0 = x0,
    u0 = u0,
    x_lb = x_lb,
    x_ub = x_ub,
)
```

`eom!(dx, x, (; params, u), t)` has the same signature as in multiple shooting. `algo` is `:ipopt` or `:snopt`, and further keywords are solver options.

```@docs
ForwardBackwardShootingProblem
propagate_nodes
solve(::ForwardBackwardShootingProblem)
```

The [Forward-backward free-final-time transfer](@ref) builds this problem for an Earth–Moon transfer with mass and a shared time dilation.
