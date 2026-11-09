module TOPLib

import OrdinaryDiffEq
using NLPInterface

include("continuous_control.jl")
include("multipleshooting/utils.jl")
include("impulsive_control.jl")
include("multipleshooting/forwardshooting.jl")

export ContinuousControlDynamics, ImpulsiveControlDynamics, MultipleForwardShootingProblem, solve

end # module TOPLib
