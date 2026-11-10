# # Forward-backward free-final-time transfer
#
# This page builds the same Earth–Moon transfer as
# `examples/cr3bp/fwdback_continuous_mass_freetf.jl`.
# The transcription is a [`ForwardBackwardShootingProblem`](@ref) with
# `alpha = 0.5`. The first half of the segments is integrated forward from the
# departure state, the second half backward from the arrival state, and the
# dynamics equality is the mismatch at the cut. Interior nodes are values of
# that propagation. The cost is the negative of the final mass, and the final
# time is free.
#
# Run it from the repository root with the examples project:
#
# ```
# julia -t auto --project=examples docs/src/examples/fwdbck_cr3bp.jl
# ```

using CairoMakie
using OrdinaryDiffEq
using TOPLib: TOPLib, ContinuousControlDynamics, ForwardBackwardShootingProblem

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
# system. The forward integration starts from `[rv0; 1; 0]`. The backward
# integration starts from a final state whose position and velocity equal
# `rvf`. Components `7` and `8` of that final state, the arrival mass and the
# arrival time, are decision variables.

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
# `times` is the normalized grid `τ`. `uniform_dilation` returns the successive
# differences of the dilation, so a feasible control uses one dilation on every
# segment. The objective reads the mass in the last column of the propagated
# node matrix.

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

# The coasts below are the orbits drawn on the figure. The forward-backward
# decision vector does not contain their interior samples.

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
# `x0` has two columns: the departure state, then a final state with the
# arrival position and velocity, unit mass, and time `tf_guess`. The fixed
# final position and velocity in this guess are ignored, because
# `x_final_idx = 1:6` already supplies them. The free entries that the solver
# reads are the final mass and the final time. Every segment starts with a
# small throttle and the same dilation.

x0 = hcat([rv0; 1.0; 0.0], [rvf; 1.0; tf_guess])
u0 = 0.1 * ones(nu, N - 1)
u0[5, :] .= tf_guess

# ## Problem
#
# `alpha = 0.5` puts the cut at the middle segment. The direction box is
# `[-1, 1]^3`, the throttle `Γ` lies in `[0, 1]`, and the dilation lies in
# `s_span`. State bounds keep the mass away from zero. Those bounds apply to
# the free boundary components, here the final mass and the final time.

dynamics = ContinuousControlDynamics(nx, nu, eom!, params)
prob = ForwardBackwardShootingProblem(
    dynamics,
    times,
    objective,
    [rv0; 1.0; 0.0],
    rvf;
    alpha=0.5,
    u_lb=[-1.0, -1.0, -1.0, 0.0, s_span[1]],
    u_ub=[1.0, 1.0, 1.0, 1.0, s_span[2]],
    x_final_idx=1:6,
    ceq=uniform_dilation,
)

x_lb = [-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, 0.2, -1.0]
x_ub = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 1.5, 10.0]

# ## Solve
#
# The mismatch is the first `nx` entries of the equality residual. It compares
# the forward and backward states at the cut.

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

zopt = TOPLib.decision_vector(prob, hcat(result.x[:, 1], result.x[:, end]), result.u)
_, ceq, _ = TOPLib.get_fitness(prob)(zopt)
println("status        : ", result.status)
println("objective     : ", result.objective)
println("final mass    : ", result.x[7, end])
println("final time    : ", result.x[8, end])
println("dilation      : ", result.u[5, 1])
println("max |Δs|      : ", maximum(abs, diff(result.u[5, :])))
println("max |mismatch|: ", maximum(abs, ceq[1:nx]))

# ## Trajectory
#
# `result.x` is already the propagated node matrix. Propagate the segments
# again, together on Julia threads, to fill in the continuous arc between those
# nodes. Thrusting segments are red and coast segments are black. The blue and
# green curves are the departure and arrival orbits.

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
