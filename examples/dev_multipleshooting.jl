"""Multiple-shooting minimum-fuel transfer between Earth-Moon LPOs."""

using OrdinaryDiffEq
using TOPLib: TOPLib, ContinuousControlDynamics, ForwardShootingProblem


# define dynamics
struct ControlParams
    μ::Float64
end

μ = 1.215058560962404e-02
DU = 389703     # km
TU = 382981     # sec
MU = 500.0      # kg
VU = DU / TU    # km/s
params = ControlParams(μ)

# 4-parameter thrust: acceleration is throttle * direction, u[4] * u[1:3].
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
    drv[4:6] += u[4] * u[1:3]
    return nothing
end

# boundary conditions
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

# -------------------- create problem -------------------- #
N = 30
nx = 6
nu = 4                              # [ux, uy, uz, throttle]
tf = 2.6
times = collect(LinRange(0.0, tf, N))

thrust = 0.35    # N
umax = thrust / MU / 1e3 / (VU / TU)

# initial & final LPO, coasted with zero thrust (same construction as x_ref in SCPLib)
sol_lpo0 = solve(
    ODEProblem(eom!, rv0, [0.0, period_0], (; params, u=zeros(nu))),
    Tsit5(); reltol=1e-12, abstol=1e-12, verbose=DEVerbosity(SciMLLogging.None()),
)
sol_lpof = solve(
    ODEProblem(eom!, rvf, [0.0, period_f], (; params, u=zeros(nu))),
    Tsit5(); reltol=1e-12, abstol=1e-12, verbose=DEVerbosity(SciMLLogging.None()),
)

x_along_lpo0 = sol_lpo0(LinRange(0.0, period_0, N))
x_along_lpof = sol_lpof(LinRange(0.0, period_f, N))
x0 = zeros(nx, N)
alphas = LinRange(0, 1, N)
for (i, alpha) in enumerate(alphas)
    x0[:, i] = (1 - alpha) * x_along_lpo0[:, i] + alpha * x_along_lpof[:, i]
end
# Unit direction so the direction constraint is not degenerate, throttle still zero.
u0 = zeros(nu, N - 1)
u0[1, :] .= 1.0

dynamics = ContinuousControlDynamics(nx, nu, eom!, params)
prob = ForwardShootingProblem(dynamics, times, :fuel, rv0, rvf; umax=umax)

result = TOPLib.solve(
    prob;
    algo=:ipopt,
    x0=x0,
    u0=u0,
    max_iter=500,
    tol=1e-6,
    print_level=5,
)

defects = TOPLib.shooting_defects(
    prob.dynamics, prob.times, result.x, result.u;
    ode_method=prob.ode_method,
    ode_reltol=prob.ode_reltol,
    ode_abstol=prob.ode_abstol,
)
println("status       : ", result.status)
println("objective    : ", result.objective)
println("max |defect| : ", maximum(abs, defects))
