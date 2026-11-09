```@meta
CurrentModule = TOPLib
```

# Multiple shooting

Multiple shooting transcribes a trajectory on a fixed node grid. `times` holds one entry per node, with `N = length(times)` and `N ≥ 2`, and the entries are strictly increasing. The decision variables are the node states `x`, stored `nx × N`, and the controls `u`.

[`ContinuousControlDynamics`](@ref) places one control on each segment, so `u` is `nu × (N − 1)`. The control is constant on `[t_k, t_{k+1})`. Segment `k` is integrated from `x_k` with that control, and the defect is the mismatch with the next node:

```math
\begin{aligned}
x(t_{k+1}) &= x_k + \int_{t_k}^{t_{k+1}} f(x(t), u_k, t) \, \mathrm{d}t, \\
x_{k+1} - x(t_{k+1}) &= 0, \qquad k = 1, \ldots, N-1.
\end{aligned}
```

[`ImpulsiveControlDynamics`](@ref) places one burn at each node, so `u` is `nu × N`. Burn `k < N` jumps the node state, and the following arc coasts with a zero control. The defect compares `x_{k+1}` with the end of that coast. Burn `N` is the terminal impulse and is left out of the integration. `x_initial` constrains the pre-impulse state at node 1. `x_final` constrains the post-impulse state at node `N`.

## Boundaries, constraints, and cost

A boundary value passed in full fixes every state component. `x_initial_idx` and `x_final_idx` fix only those components, and the target vector then has one entry per index. Passing `nothing` for an endpoint leaves that endpoint free.

The equality vector is assembled in this order:

1. dynamics defects, one `nx`-vector per segment,
2. boundary conditions,
3. `ceq_path(x_k, u_k)`, evaluated once per control column and concatenated,
4. `ceq(x, u)`, evaluated once on the full state and control matrices.

The inequality vector is assembled in this order. A constraint is feasible when every entry is `≤ 0`.

1. With `direction_constraint` (the default), one row per control column, `u_1^2 + u_2^2 + u_3^2 - 1 ≤ 0`. This requires `nu ≥ 3`.
2. `cineq_path(x_k, u_k)`, once per control column.
3. `cineq(x, u)`, once on the full matrices.

The objective is the symbol `:fuel` or a function `(x, u) -> Real`. `:fuel` costs `sum(u[4, :])` and requires `nu ≥ 4`.

## Building a problem

The dynamics are an in-place residual `eom!(dx, x, (; params, u), t)`. `params` holds the data that stays constant from one segment to the next, and the transcriber passes the segment control as `u`.

Pass `umax` only for the 4-vector control `[u_x, u_y, u_z, u]`, which boxes the direction in `[-1, 1]^3` and the throttle in `[0, umax]`. For any other control dimension, pass `u_lb` and `u_ub`, each of length `nu`.

`x0` is `nx × N`. `u0` is `nu × (N − 1)` for continuous dynamics and `nu × N` for impulsive dynamics. `x_lb` and `x_ub` are scalars, applied to every state component, or vectors of length `nx`. [`solve`](@ref) returns the named tuple `(x, u, objective, status)`.

```julia
dynamics = ContinuousControlDynamics(nx, nu, eom!, params)

prob = MultipleShootingForwardProblem(
    dynamics,
    times,
    objective,
    x_initial,
    x_final;
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

`algo` is `:ipopt` or `:snopt`. Further keywords are solver options (`tol`, `max_iter`, `deriv`, and so on).

```@docs
ContinuousControlDynamics
ImpulsiveControlDynamics
MultipleShootingForwardProblem
solve(::MultipleShootingForwardProblem)
```

The [Multiple-shooting free-final-time transfer](@ref) builds this problem for an Earth–Moon transfer with mass and a shared time dilation.
