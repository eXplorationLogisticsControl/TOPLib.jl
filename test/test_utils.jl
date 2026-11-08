"""Unit tests for stacking and unpacking decision variables"""

using Test

if !@isdefined TOPLib
    include(joinpath(@__DIR__, "../src/TOPLib.jl"))
end


function test_stack_and_unpack()
    x = randn(6, 4)
    u = randn(4, 3)
    z = TOPLib.stack_variables(x, u)
    @test length(z) == length(x) + length(u)
    x2, u2 = TOPLib.unpack_variables(z, 6, 4, 4)
    @test x2 == x
    @test u2 == u
    @test_throws DimensionMismatch TOPLib.unpack_variables(z[1:end-1], 6, 4, 4)
end


@testset "stack and unpack" begin
    test_stack_and_unpack()
end
