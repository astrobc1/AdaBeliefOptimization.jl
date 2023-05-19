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

function optimize(loss, grad, p0, extra=nothing; miniters=0, maxiters, lb=nothing, ub=nothing, β1=0.99, β2=0.999, α=0.001, ϵ=1E-8, ftol=1E-6, verbose=false)
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

# function Base.clamp!(state::AdaBeliefState, lo, hi, n_hit_bounds)
#     if !isnothing(lb)
#         for i in eachindex(state.θ)
#             if state.θ[i] < lb[i]
#                 θ[i] = lo[i]
#                 state.n_clamped[i] += 1
#                 if state.n_clamped[i] >= n_hit_bounds
#                     state.fix[i] = true
#                 end
#             end
#         end
#     end
#     if !isnothing(ub)
#         for i in eachindex(θ)
#             if state.θ[i] > ub[i]
#                 θ[i] = ub[i]
#                 state.n_clamped[i] += 1
#                 if state.n_clamped[i] >= n_hit_bounds
#                     state.fix[i] = true
#                 end
#             end
#         end
#     end
#     return state
# end

function fix_nans!(x)
    for i in eachindex(x)
        if !isfinite(x[i])
            x[i] = 0.0
        end
    end
    return x
end


# function transform_parameters(θ, lo, hi)
#     has_low = !isnothing(lo)
#     has_high = !isnothing(hi)
#     if has_low && has_high
#         θn = @. asin(2 * (θ - lo) / (hi - lo) - 1)
#     elseif has_low && !has_high
#         θn = @. sqrt((hi - θ + 1)^2 - 1)
#     elseif !has_low && has_high
#         θn = @. sqrt((θ - lo + 1)^2 - 1)
#     else
#         return copy(θ)
#     end
# end

# function reverse_transform_parameters(θn, lo, hi)
#     has_low = !isnothing(lo)
#     has_high = !isnothing(hi)
#     if has_low && has_high
#         θ = @. lo + (sin(θ) + 1) * (hi - lo) / 2
#     elseif has_low && !has_high
#         θ = @. hi + 1 - sqrt(θn^2 + 1)
#     elseif !has_low && has_high
#         θ = @. lo - 1 + sqrt(θn^2 + 1)
#     else
#         return copy(θn)
#     end
# end