export optimize

mutable struct AdaBeliefState{T, M, S, F}
    β1::Float64
    β2::Float64
    α::Float64
    ϵ::Float64
    t::Int
    θ::T
    m::M
    s::S
    f::F
    g::T
end

function optimize(loss, grad, p0::AbstractArray{<:Real}, extra=nothing; miniters::Int=0, maxiters::Int, lb::AbstractArray{<:Real}, ub::AbstractArray{<:Real}, β1::Real=0.99, β2::Real=0.999, α::Real=0.001, ϵ::Real=1E-8, ftol::Real=1E-6, verbose::Bool=false)
    m0 = zero.(similar(p0))
    s0 = zero.(similar(p0))
    state = AdaBeliefState(β1, β2, α, ϵ, 0, deepcopy(p0), m0, s0, loss_wrapper(loss, p0, extra), similar(p0))
    while true
        f = state.f
        step!(state, loss, grad, extra, lb, ub)
        fnew = state.f
        if verbose
            print_verbose(f, fnew, state.t)
        end
        conv = check_convergence(f, fnew, ftol, state.t, miniters, maxiters)
        if conv
            break
        end
    end
    result = (;pbest=state.θ, fbest=state.f, iterations=state.t)
    return result
end

function step!(state::AdaBeliefState, loss, grad, extra, lb, ub)
    state.t += 1
    state.g .= grad(state.θ, extra)
    fix_nans!(state.g)
    state.m .= state.β1 .* state.m .+ (1 - state.β1) .* state.g
    state.s .= state.β2 .* state.s .+ (1 - state.β2) .* (state.g .- state.m).^2 .+ state.ϵ
    mhat = state.m ./ (1 - state.β1^state.t)
    shat = state.s ./ (1 - state.β2^state.t)
    state.θ .-= state.α .* (mhat ./ (sqrt.(shat) .+ state.ϵ))
    state.f = loss_wrapper(loss, state.θ, extra)
    return state
end

function loss_wrapper(loss, θ, extra=nothing)
    if !isnothing(extra)
        return loss(θ, extra)
    else
        return loss(θ)
    end
end

function check_convergence(f1, f2, ftol, iter, miniters, maxiters)
    if iter >= maxiters
        return true
    elseif iter < miniters
        return false
    else
        δf = compute_df(f1, f2)
        fconv = δf < ftol
        return fconv
    end
end

function compute_df(f1, f2)
    favg = (f1 + f2) / 2
    δf = abs(f1 - f2) / favg
    return δf
end

function print_verbose(f1, f2, k)
    δf = compute_df(f1, f2)
    println("[Iteration $k] Loss = $f2, δf (rel) = $δf")
end

function fix_nans!(x, v=0)
    for i in eachindex(x)
        if !isfinite(x[i])
            x[i] = v
        end
    end
    return x
end