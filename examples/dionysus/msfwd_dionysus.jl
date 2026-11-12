"""Max-final-mass rendezvous with asteroid Dionysus at a fixed time of flight.

Forward multiple shooting. Every node carries a state, and each segment is
integrated forward under a zero-order-hold control. The dynamics equalities are
the defects between those integrations and the next node. The departure state
is fixed. Arrival equinoctial elements are free; a Cartesian equality places
that arrival on the asteroid. The cost is minus the final mass.

Thrust is `Γ * direction * c1 / m` in the radial–transverse–normal frame, and
the mass flow is `ṁ = -c2 * Γ`. The direction lies in the unit ball and
`Γ ∈ [0, 1]`. State bounds apply at every node: `p ≥ 0.8`, `m ∈ [0.1, 1]`, and
true longitude runs from the departure value to the 5-revolution arrival guess.

Run from this directory:

    julia -t auto --project=.. msfwd_dionysus.jl
"""

using CairoMakie
using LinearAlgebra
using OrdinaryDiffEq

import AstrodynamicsCore: Planet, eph, kep2rv, ma2ta, mee2rv, rv2mee

using TOPLib: TOPLib, ContinuousControlDynamics, MultipleShootingForwardProblem


struct ControlParams
    μ::Float64
    c1::Float64
    c2::Float64
end

const MU_SUN = 132712440018.0
const G0 = 9.8065
const DU = 149.6e6                 # km
const VU = sqrt(MU_SUN / DU)       # km/s
const TU = DU / VU                 # s
const MASS = 4000.0                # kg
const THRUST = 0.32                # N
const ISP = 3000.0                 # s

μ = MU_SUN / (VU^2 * DU)
c1 = THRUST / 1e3 / (MASS * DU / TU^2)
c2 = THRUST / 1e3 / (ISP * G0 / 1e3) / (MASS / TU)
params = ControlParams(μ, c1, c2)

# State is [p, f, g, h, k, L, mass]. Control is [u_r, u_θ, u_n, Γ].
function eom!(drvm, rvm, pu, t)
    (; params, u) = pu
    p, f, g, h, k, L, mass = rvm[1:7]
    cosL = cos(L)
    sinL = sin(L)
    s2 = 1 + h^2 + k^2
    w = 1 + f * cosL + g * sinL
    hsinL_kcosL = h * sinL - k * cosL
    B_mee = sqrt(p / params.μ) * [
        0     2p / w                    0;
        sinL  ((1 + w) * cosL + f) / w  -g / w * hsinL_kcosL;
        -cosL ((1 + w) * sinL + g) / w   f / w * hsinL_kcosL;
        0     0                          s2 / (2w) * cosL;
        0     0                          s2 / (2w) * sinL;
        0     0                          hsinL_kcosL / w
    ]
    D = [0.0, 0.0, 0.0, 0.0, 0.0, sqrt(params.μ / p^3) * w^2]
    drvm[1:6] = B_mee * (params.c1 / mass) * (u[4] * u[1:3]) + D
    drvm[7] = -u[4] * params.c2
    return nothing
end

tof = 3534 * 86400 / TU
N_rev = 5

R0 = [-3637871.081; 147099798.784; -2261.44] / DU
V0 = [-30.265097; -0.8486854; 0.0000505] / VU
RV0 = [R0; V0]
orbit_init = Planet(params.μ, 0.0, RV0, "initial")
mee0 = rv2mee(RV0, params.μ)

M0 = deg2rad(114.4232)
TA0 = ma2ta(M0, 0.542)
kepf = [2.2, 0.542, deg2rad(13.6), deg2rad(82.2), deg2rad(204.2), TA0]
RVf0 = kep2rv(kepf, params.μ)
orbit_final = Planet(params.μ, 0.0, RVf0, "final")
RVf = eph(orbit_final, tof + (56284 - 53400) * 86400 / TU)
meef = rv2mee(RVf, params.μ)
meef[6] += 2π * N_rev

initial_orbit_rvs = hcat([eph(orbit_init, t) for t in LinRange(0.0, orbit_init.period, 300)]...)
final_orbit_rvs = hcat([eph(orbit_final, t) for t in LinRange(0.0, orbit_final.period, 300)]...)

N = 41
nx = 7
nu = 4
times = collect(LinRange(0.0, tof, N))

objective(x, u) = -x[7, end]

function rendezvous(x, u)
    return mee2rv(collect(x[1:6, end]), params.μ) .- RVf
end

function state_bounds(xk, uk)
    return [0.8 - xk[1], 0.1 - xk[7]]
end

function monotone_longitude(x, u)
    return x[6, 1:(end - 1)] .- x[6, 2:end]
end

x_initial = [mee0; 1.0]
x0 = zeros(nx, N)
x0[1:6, :] = hcat(LinRange.(mee0, meef, N)...)'
x0[7, :] = LinRange(1.0, 0.4, N)
u0 = zeros(nu, N - 1)
u0[2, :] .= 0.1
u0[4, :] .= 0.1

x_lb = [0.8, -1.0, -1.0, -1.0, -1.0, mee0[6] - 2π, 0.1]
x_ub = [5.0, 1.0, 1.0, 1.0, 1.0, meef[6] + 2π, 1.0]

dynamics = ContinuousControlDynamics(nx, nu, eom!, params)
prob = MultipleShootingForwardProblem(
    dynamics,
    times,
    objective,
    x_initial,
    nothing;
    u_lb=[-1.0, -1.0, -1.0, 0.0],
    u_ub=[1.0, 1.0, 1.0, 1.0],
    ceq=rendezvous,
    cineq_path=state_bounds,
    cineq=monotone_longitude,
    ode_method=Vern7(),
)

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
rv_err = mee2rv(result.x[1:6, end], params.μ) .- RVf
println("status           : ", result.status)
println("objective        : ", result.objective)
println("final mass       : ", result.x[7, end] * MASS, " kg")
println("max |defect|     : ", maximum(abs, defects))
println("max |rendezvous| : ", maximum(abs, rv_err))

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
ax3d = Axis3(fig[1, 1]; aspect=:data, xlabel="x, AU", ylabel="y, AU", zlabel="z, AU")
rowsize!(fig.layout, 1, Relative(0.62))
scatter!(ax3d, RV0[1], RV0[2], RV0[3], color=:limegreen, markersize=10)
scatter!(ax3d, RVf[1], RVf[2], RVf[3], color=:blue, markersize=10)
lines!(ax3d, initial_orbit_rvs[1, :], initial_orbit_rvs[2, :], initial_orbit_rvs[3, :], color=:limegreen, linewidth=0.8)
lines!(ax3d, final_orbit_rvs[1, :], final_orbit_rvs[2, :], final_orbit_rvs[3, :], color=:blue, linewidth=0.8)
arc_colors = [result.u[4, i] > 1e-2 ? :red : :black for i in 1:nseg]
for (i, sol) in enumerate(sols_opt)
    mee = Array(sol)
    rvs = hcat([mee2rv(mee[1:6, j], params.μ) for j in axes(mee, 2)]...)
    lines!(ax3d, rvs[1, :], rvs[2, :], rvs[3, :], color=arc_colors[i])
end

ax_u = Axis(fig[1, 2]; xlabel="Time, day", ylabel="Control")
hlines!(ax_u, 1.0, color=:red, linestyle=:dash)
t_days = times .* TU ./ 86400
control_labels = ["u_r", "u_θ", "u_n"]
for i in 1:3
    stairs!(ax_u, t_days, [result.u[4, :] .* result.u[i, :]; 0.0], label=control_labels[i], step=:post, linewidth=1.0)
end
stairs!(ax_u, t_days, [result.u[4, :]; 0.0], label="Γ", step=:post, linewidth=1.0, color=:black)
Legend(fig[1, 3], ax_u)

ax_m = Axis(fig[2, 1:3]; xlabel="Time, day", ylabel="Mass, kg")
for (i, sol) in enumerate(sols_opt)
    mee = Array(sol)
    lines!(ax_m, sol.t .* TU ./ 86400, mee[7, :] .* MASS, color=arc_colors[i])
end

plots_dir = joinpath(@__DIR__, "plots")
mkpath(plots_dir)
plot_path = joinpath(plots_dir, "dionysus_msfwd.png")
save(plot_path, fig; px_per_unit=3)
println("Saved plot to ", plot_path)
