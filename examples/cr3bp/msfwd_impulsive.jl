"""Multiple-shooting minimum-fuel impulsive transfer between Earth-Moon LPOs.

Burns are instantaneous: each node carries a direction-throttle impulse
`Δv = throttle * direction`, and the arcs between nodes are unthrusted coasts.
The final boundary condition is applied after the terminal burn.

Same transfer as SCPLib's impulsive dynamics-only problem: N = 60, tf = 2.6,
departure and arrival states fixed, cost `sum(throttle)`.

Run from this directory so the examples environment is used:

    julia --project=.. cr3bp_multipleshooting_impulsive.jl
"""

using CairoMakie
using OrdinaryDiffEq
using TOPLib: TOPLib, ImpulsiveControlDynamics, MultipleForwardShootingProblem


struct ControlParams
    μ::Float64
end

μ = 1.215058560962404e-02
DU = 389703     # km
TU = 382981     # sec
MU = 500.0      # kg
VU = DU / TU    # km/s
params = ControlParams(μ)

# Coast only. The impulse is the default jump, Δv = u[4] * u[1:3] on velocity.
function eom!(drv, rv, pu, t)
    (; params, u) = pu
    x, y, z = rv[1:3]
    vx, vy, vz = rv[4:6]
    r1 = sqrt((x + params.μ)^2 + y^2 + z^2)
    r2 = sqrt((x - 1 + params.μ)^2 + y^2 + z^2)
    drv[1:3] = rv[4:6]
    drv[4] = 2 * vy + x - ((1 - params.μ) / r1^3) * (params.μ + x) + (params.μ / r2^3) * (1 - params.μ - x)
    drv[5] = -2 * vx + y - ((1 - params.μ) / r1^3) * y - (params.μ / r2^3) * y
    drv[6] = -((1 - params.μ) / r1^3) * z - (params.μ / r2^3) * z
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

N = 60
nx = 6
nu = 4                              # [ux, uy, uz, throttle]
tf = 2.6
times = collect(LinRange(0.0, tf, N))

thrust = 0.35    # N
umax = thrust / MU / 1e3 / (VU / TU)

sol_lpo0 = solve(
    ODEProblem(eom!, rv0, [0.0, period_0], (; params, u = zeros(nu))),
    Tsit5(); reltol = 1e-12, abstol = 1e-12, verbose = DEVerbosity(SciMLLogging.None()),
)
sol_lpof = solve(
    ODEProblem(eom!, rvf, [0.0, period_f], (; params, u = zeros(nu))),
    Tsit5(); reltol = 1e-12, abstol = 1e-12, verbose = DEVerbosity(SciMLLogging.None()),
)

x_along_lpo0 = sol_lpo0(LinRange(0.0, period_0, N))
x_along_lpof = sol_lpof(LinRange(0.0, period_f, N))
x0 = zeros(nx, N)
alphas = LinRange(0, 1, N)
for (i, alpha) in enumerate(alphas)
    x0[:, i] = (1 - alpha) * x_along_lpo0[:, i] + alpha * x_along_lpof[:, i]
end
# Unit direction so the direction constraint is not degenerate, throttle still zero.
u0 = zeros(nu, N)
u0[1, :] .= 1.0

dynamics = ImpulsiveControlDynamics(nx, nu, eom!, params)
prob = MultipleForwardShootingProblem(dynamics, times, :fuel, rv0, rvf; umax = umax)

result = TOPLib.solve(
    prob;
    algo = :snopt,
    x0 = x0,
    u0 = u0,
    max_iter = 500,
    tol = 1e-6,
    print_level = 5,
)

defects = TOPLib.shooting_defects(
    prob.dynamics, prob.times, result.x, result.u;
    ode_method = prob.ode_method,
    ode_reltol = prob.ode_reltol,
    ode_abstol = prob.ode_abstol,
)
println("status       : ", result.status)
println("objective    : ", result.objective)
println("max |defect| : ", maximum(abs, defects))

# Jump, then coast with zero thrust, so the plotted arcs are the integrated segments.
sols_opt = map(1:(N - 1)) do k
    xk = copy(result.x[:, k])
    xk[4:6] .+= result.u[4, k] .* result.u[1:3, k]
    solve(
        ODEProblem(eom!, xk, (times[k], times[k + 1]), (; params, u = zeros(nu))),
        prob.ode_method;
        reltol = prob.ode_reltol,
        abstol = prob.ode_abstol,
        verbose = DEVerbosity(SciMLLogging.None()),
    )
end

fig = Figure(size = (1000, 700))
ax3d = Axis3(fig[1, 1]; aspect = :data)
rowsize!(fig.layout, 1, Relative(0.58))
lines!(ax3d, Array(sol_lpo0)[1, :], Array(sol_lpo0)[2, :], Array(sol_lpo0)[3, :], color = :blue)
lines!(ax3d, Array(sol_lpof)[1, :], Array(sol_lpof)[2, :], Array(sol_lpof)[3, :], color = :green)
for sol in sols_opt
    traj = Array(sol)
    lines!(ax3d, traj[1, :], traj[2, :], traj[3, :], color = :black)
end
burn = result.u[4, :] .> 1e-6
scatter!(ax3d, result.x[1, burn], result.x[2, burn], result.x[3, burn], color = :red)

Δv = result.u[1:3, :] .* reshape(result.u[4, :], 1, :)
ax_u = Axis(fig[1, 2]; xlabel = "Time", ylabel = "Impulse")
hlines!(ax_u, umax, color = :red, linestyle = :dash)
for i in 1:3
    scatterlines!(ax_u, times, Δv[i, :], label = "Δv[$i]", linewidth = 1.0)
end
stem!(ax_u, times, result.u[4, :], label = "throttle", color = :black)
axislegend(ax_u, position = :cc)

plots_dir = joinpath(@__DIR__, "plots")
mkpath(plots_dir)
save(joinpath(plots_dir, "cr3bp_traj_multipleshooting_impulsive.png"), fig; px_per_unit = 3)
println("Saved plot to ", joinpath(plots_dir, "cr3bp_traj_multipleshooting_impulsive.png"))
