```@meta
CurrentModule = TOPLib
```

# `TOPLib.jl`

`TOPLib.jl` transcribes trajectory optimization problems as nonlinear programs. Each transcription integrates the controlled dynamics and passes the resulting equality and inequality residuals to Ipopt or SNOPT.

- [Multiple shooting](@ref) keeps a state at every node and enforces an integrated defect on every segment. Continuous segments carry a zero-order-hold control. Impulsive nodes carry a burn, then a coast.
- [Forward-backward shooting](@ref) integrates from the two endpoints and enforces one mismatch at the cut.
- [Multiple-shooting free-final-time transfer](@ref) and [Forward-backward free-final-time transfer](@ref) solve the same Earth–Moon transfer, maximizing final mass with a free final time.
