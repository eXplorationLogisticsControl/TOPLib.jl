"""Max-final-mass transfer between Earth-Moon LPOs with free final time.

The independent variable is τ ∈ [0, 1]. Thrust uses the unit-direction throttle
Γ * direction * c1 / m, mass flows as ṁ = -c2 * Γ, and every derivative is
multiplied by a per-segment time dilation. A trajectory equality keeps that
dilation the same on every segment, so the physical final time is the shared
value. The cost is minus the final mass.

Run from the examples environment:

    julia --project=.. cr3bp_multipleshooting_mass_freetf.jl
"""

using CairoMakie
using OrdinaryDiffEq
using TOPLib: TOPLib, ContinuousControlDynamics, MultipleShootingForwardProblem


struct ControlParams
    μ::Float64
    c1::Float64
    c2::Float64
end

μ = 1.215058560962404e-02
c1 = 0.1
c2 = 0.1
params = ControlParams(μ, c1, c2)

# State is [r, v, mass, time]. Control is [ux, uy, uz, Γ, dilation].
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

# Coast one normalized period: dilation equals the physical period, throttle is zero.
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
x_along_lpo0 = sol_lpo0(LinRange(0.0, 1.0, N))
x_along_lpof = sol_lpof(LinRange(0.0, 1.0, N))

x0 = zeros(nx, N)
for (i, alpha) in enumerate(LinRange(0.0, 1.0, N))
    x0[1:6, i] = (1 - alpha) * x_along_lpo0[1:6, i] + alpha * x_along_lpof[1:6, i]
end
x0[7, :] .= 1.0
x0[8, :] = LinRange(0.0, tf_guess, N)
u0 = 0.1 * ones(nu, N - 1)              # make control initial guess non-zero
u0[5, :] .= tf_guess                    # overwrite with TOF initial guess

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

# options per solver
solver = :ipopt 
if solver == :ipopt
    options = Dict(
        :tol => 1e-6,                 # bounds on scaled NLP error
        :constr_viol_tol => 1e-8,
        :print_level => 5,
        :max_iter => 500,
    )
elseif solver == :snopt
    options = Dict(
        :Major_optimality_tolerance => 1e-6,
        :Major_feasibility_tolerance => 1e-8,
        :Major_print_level => 1,
        :Major_iterations_limit => 500,
    )
else
    error("Invalid solver: $solver")
end


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
)
println("status       : ", result.status)
println("objective    : ", result.objective)
println("final mass   : ", result.x[7, end])
println("final time   : ", result.x[8, end])
println("dilation     : ", result.u[5, 1])
println("max |Δs|     : ", maximum(abs, diff(result.u[5, :])))
println("max |defect| : ", maximum(abs, defects))

sols_opt = map(1:(N - 1)) do k
    solve(
        ODEProblem(
            eom!,
            result.x[:, k],
            (times[k], times[k + 1]),
            (; params, u=result.u[:, k]),
        ),
        prob.ode_method;
        reltol=prob.ode_reltol,
        abstol=prob.ode_abstol,
        verbose=DEVerbosity(SciMLLogging.None()),
    )
end

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

plots_dir = joinpath(@__DIR__, "plots")
mkpath(plots_dir)
plot_path = joinpath(plots_dir, "cr3bp_traj_multipleshooting_mass_freetf.png")
save(plot_path, fig; px_per_unit=3)
println("Saved plot to ", plot_path)
