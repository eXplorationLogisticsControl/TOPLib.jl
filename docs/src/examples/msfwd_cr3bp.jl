# # Multiple-shooting free-final-time transfer
#
# This page builds the same Earth–Moon transfer as
# `examples/cr3bp/msfwd_continuous_mass_freetf.jl`.
# The transcription is a [`MultipleShootingForwardProblem`](@ref):
# every node carries a state, and every segment carries an integrated defect.
# The cost is the negative of the final mass, so the solve maximizes the mass
# that arrives. The final time is free.
#
# Run it from the repository root with the examples project:
#
# ```
# julia -t auto --project=examples docs/src/examples/msfwd_cr3bp.jl
# ```

using CairoMakie
using OrdinaryDiffEq
using TOPLib: TOPLib, ContinuousControlDynamics, MultipleShootingForwardProblem

# ## Dynamics
#
# The state is `[r, v, mass, time]` and the control is `[u_x, u_y, u_z, Γ, s]`.
# The independent variable is `τ ∈ [0, 1]`. On each segment the physical
# derivatives are scaled by the dilation `s`:
#
# ```math
# \frac{\mathrm{d}}{\mathrm{d}\tau}
# \begin{bmatrix} r \\ v \\ m \\ t \end{bmatrix}
# =
# s
# \begin{bmatrix}
# v \\
# a_{\mathrm{CR3BP}}(r, v) + \Gamma \, u_{1:3} \, c_1 / m \\
# -c_2 \, \Gamma \\
# 1
# \end{bmatrix}.
# ```
#
# `c_1` and `c_2` are the thrust and mass-flow scales. A trajectory equality
# forces `s` to take the same value on every segment. With `t(0) = 0`, that
# shared value is the physical final time.

struct ControlParams
    μ::Float64
    c1::Float64
    c2::Float64
end

μ = 1.215058560962404e-02
c1 = 0.1
c2 = 0.1
params = ControlParams(μ, c1, c2)

function eom!(drv, rvm, pu, t)
    (; params, u) = pu
    x, y, z = rvm[1:3]
    vx, vy, vz = rvm[4:6]
    r1 = sqrt((x + params.μ)^2 + y^2 + z^2)
    r2 = sqrt((x - 1 + params.μ)^2 + y^2 + z^2)
    drv[1:3] = rvm[4:6]
    drv[4] = 2 * vy + x - ((1 - params.μ) / r1^3) * (params.μ + x) + (params.μ / r2^3) * (1 - params.μ - x)
    drv[5] = -2 * vx + y - ((1 - params.μ) / r1^3) * y - (params.μ / r2^3) * y
    drv[6] = -((1 - params.μ) / r1^3) * z - (params.μ / r2^3) * z
    drv[4:6] += u[4] * u[1:3] * params.c1 / rvm[7]
    drv[7] = -u[4] * params.c2
    drv[8] = 1.0
    drv[1:8] *= u[5]
    return nothing
end

# ## Boundary orbits
#
# The departure and arrival states are periodic orbits in the Earth–Moon
# system. Position and velocity are fixed at both ends. The departure mass is
# `1` and the departure time is `0`. The arrival mass and the arrival time
# stay free, because the final target passed below lists only components `1:6`.

rv0 = [
    1.0809931218390707E+00,
    0.0000000000000000E+00,
    -2.0235953267405354E-01,
    1.0157158264396639E-14,
    -1.9895001215078018E-01,
    7.2218178975912707E-15,
]
period_0 = 2.3538670417546639E+00

rvf = [
    1.1648780946517576,
    0.0,
    -1.1145303634437023E-1,
    0.0,
    -2.0191923237095796E-1,
    0.0,
]
period_f = 3.3031221822879884

# ## Free final time
#
# `times` is the normalized grid `τ`. `uniform_dilation` is the trajectory
# equality `ceq`: it returns the successive differences of the dilation, so a
# feasible control uses one dilation on every segment. The objective reads the
# mass at the last node.

N = 30
nx = 8
nu = 5
times = collect(LinRange(0.0, 1.0, N))
s_span = (2.0, 4.0)
tf_guess = 3.0

function uniform_dilation(x, u)
    return diff(u[5, :])
end

objective(x, u) = -x[7, end]

# A coast of one normalized period uses zero throttle and a dilation equal to
# the physical period. Those coasts are the reference orbits for the guess and
# for the plot.

function coast_orbit(rv, period)
    u_orbit = zeros(nu)
    u_orbit[5] = period
    return solve(
        ODEProblem(eom!, [rv; 1.0; 0.0], (0.0, 1.0), (; params, u=u_orbit)),
        Tsit5(); reltol=1e-12, abstol=1e-12, verbose=DEVerbosity(SciMLLogging.None()),
    )
end

sol_lpo0 = coast_orbit(rv0, period_0)
sol_lpof = coast_orbit(rvf, period_f)

# ## Initial guess
#
# Sample each coast on the normalized grid and blend position and velocity
# from the departure orbit to the arrival orbit. Mass stays at `1`. Time runs
# linearly from `0` to the guess `tf_guess`. The control guess is a small
# throttle with the same dilation on every segment, so the dilation equality
# already holds.
#
# ```math
# x_{1:6,i} = (1 - \alpha_i)\, x^{\mathrm{dep}}_i + \alpha_i\, x^{\mathrm{arr}}_i,
# \qquad \alpha_i \in [0, 1].
# ```

x_along_lpo0 = sol_lpo0(LinRange(0.0, 1.0, N))
x_along_lpof = sol_lpof(LinRange(0.0, 1.0, N))

x0 = zeros(nx, N)
for (i, alpha) in enumerate(LinRange(0.0, 1.0, N))
    x0[1:6, i] = (1 - alpha) * x_along_lpo0[1:6, i] + alpha * x_along_lpof[1:6, i]
end
x0[7, :] .= 1.0
x0[8, :] = LinRange(0.0, tf_guess, N)
u0 = 0.1 * ones(nu, N - 1)
u0[5, :] .= tf_guess

# ## Problem
#
# The direction box is `[-1, 1]^3`, the throttle `Γ` lies in `[0, 1]`, and the
# dilation lies in `s_span`. The direction constraint `u_x^2 + u_y^2 + u_z^2 ≤ 1`
# is on by default. State bounds keep the mass away from zero.

dynamics = ContinuousControlDynamics(nx, nu, eom!, params)
prob = MultipleShootingForwardProblem(
    dynamics,
    times,
    objective,
    [rv0; 1.0; 0.0],
    rvf;
    u_lb=[-1.0, -1.0, -1.0, 0.0, s_span[1]],
    u_ub=[1.0, 1.0, 1.0, 1.0, s_span[2]],
    x_final_idx=1:6,
    ceq=uniform_dilation,
)

x_lb = [-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, 0.2, -1.0]
x_ub = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 1.5, 10.0]

# ## Solve
#
# `x0` is `8 × 30` and `u0` is `5 × 29`. The defect check propagates the
# segments together and compares each end state with the next node.

solver = :ipopt
options = Dict(
    :tol => 1e-6,
    :constr_viol_tol => 1e-8,
    :print_level => 5,
    :max_iter => 500,
)

result = TOPLib.solve(
    prob;
    algo=solver,
    x0=x0,
    u0=u0,
    x_lb=x_lb,
    x_ub=x_ub,
    options...,
)

defects = TOPLib.shooting_defects(
    prob.dynamics, prob.times, result.x, result.u;
    ode_method=prob.ode_method,
    ode_reltol=prob.ode_reltol,
    ode_abstol=prob.ode_abstol,
    ensemble_method=prob.ensemble_method,
)
println("status       : ", result.status)
println("objective    : ", result.objective)
println("final mass   : ", result.x[7, end])
println("final time   : ", result.x[8, end])
println("dilation     : ", result.u[5, 1])
println("max |Δs|     : ", maximum(abs, diff(result.u[5, :])))
println("max |defect| : ", maximum(abs, defects))

# ## Trajectory
#
# Propagate the zero-order-hold segments together on Julia threads so the plot
# follows the continuous arc. Thrusting segments are red and coast segments are
# black. The blue and green curves are the departure and arrival orbits.

nseg = N - 1
u0s = [copy(result.x[:, k]) for k in 1:nseg]
tspans = [(times[k], times[k + 1]) for k in 1:nseg]
ps = [(; params, u=copy(result.u[:, k])) for k in 1:nseg]
template = ODEProblem(eom!, u0s[1], tspans[1], ps[1])
prob_func = (odeprob, ctx) -> remake(odeprob; u0=u0s[ctx.sim_id], tspan=tspans[ctx.sim_id], p=ps[ctx.sim_id])
ensemble = EnsembleProblem(template; prob_func, safetycopy=false)
sols_opt = solve(
    ensemble,
    prob.ode_method,
    SciMLBase.EnsembleThreads();
    trajectories=nseg,
    reltol=prob.ode_reltol,
    abstol=prob.ode_abstol,
    verbose=DEVerbosity(SciMLLogging.None()),
).u

fig = Figure(size=(1000, 800))
ax3d = Axis3(fig[1, 1]; aspect=:data)
rowsize!(fig.layout, 1, Relative(0.62))
lines!(ax3d, Array(sol_lpo0)[1, :], Array(sol_lpo0)[2, :], Array(sol_lpo0)[3, :], color=:blue)
lines!(ax3d, Array(sol_lpof)[1, :], Array(sol_lpof)[2, :], Array(sol_lpof)[3, :], color=:green)
arc_colors = [result.u[4, i] > 1e-6 ? :red : :black for i in 1:(N - 1)]
for (i, sol) in enumerate(sols_opt)
    traj = Array(sol)
    lines!(ax3d, traj[1, :], traj[2, :], traj[3, :], color=arc_colors[i])
end

ax_u = Axis(fig[1, 2]; xlabel="Time", ylabel="Control")
hlines!(ax_u, 1.0, color=:red, linestyle=:dash)
t_nodes = result.x[8, :]
for i in 1:3
    stairs!(
        ax_u,
        t_nodes,
        [result.u[4, :]; 0.0] .* [result.u[i, :]; 0.0],
        label="Γ u[$i]",
        step=:post,
        linewidth=1.0,
    )
end
stairs!(ax_u, t_nodes, [result.u[4, :]; 0.0], label="Γ", step=:post, linewidth=2.0, color=:black, linestyle=:dash)
axislegend(ax_u, position=:cc)

ax_m = Axis(fig[2, 1:2]; xlabel="Time", ylabel="Mass")
for (i, sol) in enumerate(sols_opt)
    traj = Array(sol)
    lines!(ax_m, traj[8, :], traj[7, :], color=arc_colors[i])
end

fig
