# AdaBeliefOptimization.jl

Simple implementation of the AdaBelief optimization algorithm.

Primary method `optimize` is called as:

```julia
optimize(loss, grad, p0::AbstractArray{<:Real}, extra=nothing; miniters::Int=0, maxiters::Int, β1::Real=0.99, β2::Real=0.999, α::Real=0.001, ϵ::Real=1E-8, ftol::Real=1E-6, verbose::Bool=false)
```