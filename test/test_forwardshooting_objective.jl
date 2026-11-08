"""Minimum-fuel objective and thrust-direction inequalities"""

using Test

if !@isdefined TOPLib
    include(joinpath(@__DIR__, "../src/TOPLib.jl"))
end


function test_fuel_objective_and_direction_inequality()
    # Columns: interior direction, unit direction, infeasible direction.
    u = [
        0.0 1.0 1.0
        0.0 0.0 1.0
        0.0 0.0 0.0
        0.2 0.0 0.4
    ]
    @test TOPLib.get_objective(:fuel, zeros(6, 4), u) ≈ 0.6
    @test_throws ErrorException TOPLib.get_objective(:energy, zeros(6, 4), u)
    cineq = TOPLib.fuel_direction_inequalities(u)
    @test cineq ≈ [-1.0, 0.0, 1.0]
    @test cineq[1] < 0
    @test cineq[2] == 0
    @test cineq[3] > 0
end


@testset "fuel objective and direction inequality" begin
    test_fuel_objective_and_direction_inequality()
end
