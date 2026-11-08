module TOPLib

import OrdinaryDiffEq
using NLPInterface

include("continuous_control.jl")
include("multipleshooting/utils.jl")
include("multipleshooting/forwardshooting.jl")

export ContinuousControlDynamics, ForwardShootingProblem, solve

end # module TOPLib
