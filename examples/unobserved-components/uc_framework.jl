using StateSpaceInference
using SSMProblems
using Distributions
using Random
using LinearAlgebra, MatrixEquations

## STOCHASTIC VOLATILITY ######################################################

struct StochasticVolatility{ΣT<:Real, ΓT<:Real}
    γ::ΓT
    p::Float64

    function StochasticVolatility{ΣT}(γ::ΓT, p::Float64) where {ΓT<:Real, ΣT<:Real}
        @assert (p > 0.0 && p < 1.0)
        return new{ΣT, ΓT}(γ, p)
    end
end

Base.eltype(::StochasticVolatility{ΣT, ΓT}) where {ΣT, ΓT} = ΣT

function StochasticVolatility(γ::ΓT, p::Float64) where ΓT<:Real
    return StochasticVolatility{Float64}(γ, p)
end

# sample from the transition process prior density
function initial_draw(
    rng::AbstractRNG,
    ::StochasticVolatility{ΣT}
) where ΣT <: Real
    return 1.e1*randn(rng, ΣT)
end

# sample from the state transition density
@inline function transition(
    rng::AbstractRNG,
    vol::StochasticVolatility{ΣT},
    logσ::ΣT
) where ΣT <: Real
    if rand(rng, Bernoulli(vol.p))
        return logσ + vol.γ*randn(rng, ΣT)
    else
        return logσ
    end
end

## STOCHASTIC CYCLE ###########################################################

struct StochasticCycle{XT<:Real, ΦT<:AbstractMatrix, ΣT<:Union{Real, Missing}}
    ϕ::ΦT
    σ::ΣT
    n::Int64

    function StochasticCycle{XT}(
        n::Int64, ρ::ΡT, λ::ΛT, σκ::ΣT = missing
    ) where {ΡT<:Real, ΛT<:Real, ΣT, XT}

        cosλ = cospi(λ)
        sinλ = sinpi(λ)
        
        ϕ  = kron(I(n), ρ*[cosλ sinλ;-sinλ cosλ])
        ϕ += kron(diagm(1 => ones(n-1)), I(2))

        return new{XT, typeof(ϕ), ΣT}(ϕ, σκ, n)
    end
end

Base.eltype(::StochasticCycle{XT, ΦT, ΣT}) where {XT, ΦT, ΣT} = XT

# define for linear Gaussian process
function StochasticCycle(
    n::Int64, ρ::ΡT, λ::ΛT, σκ²::ΣT
) where {ΡT<:Real, ΛT<:Real, ΣT<:Real}
    XT = promote_type(ΡT, ΛT, ΣT)
    return StochasticCycle{XT}(n, ρ, λ, sqrt(σκ²))
end

function StochasticCycle(
    n::Int64, ρ::ΡT, λ::ΛT
) where {ΡT<:Real, ΛT<:Real}
    XT = promote_type(ΡT, ΛT)
    return StochasticCycle{XT}(n, ρ, λ)
end

# sample from the transition process prior density
function initial_draw(
    rng::AbstractRNG,
    cycle::StochasticCycle{ΨT},
    σ::ΣT
) where {ΨT<:Real, ΣT<:Real}
    dim = 2*cycle.n
    Σψ = zeros(ΣT ,dim, dim)

    σκ² = σ^2
    Σψ[end-1,end-1] = σκ²
    Σψ[end,end] = σκ²

    init_dist = convert(MvNormal{ΨT}, MvNormal(lyapd(cycle.ϕ, Σψ)))
    return rand(rng, init_dist)
end

function initial_draw(
        rng::AbstractRNG,
        cycle::StochasticCycle{ΨT, ΦT, ΣT}
    ) where {ΨT<:Real, ΣT<:Real, ΦT}
    return initial_draw(rng, cycle, cycle.σ)
end

@inline function transition(
    rng::AbstractRNG,
    cycle::StochasticCycle{ΨT},
    ψ::AbstractArray{ΨT},
    σ::ΣT
) where {ΨT<:Real, ΣT<:Real}
    ψ = cycle.ϕ*ψ
    ψ[end-1:end] += σ*randn(rng, 2)
    return convert(AbstractArray{ΨT}, ψ)
end

function transition(
    rng::AbstractRNG,
    cycle::StochasticCycle{ΨT, ΦT, ΣT},
    ψ::AbstractArray{ΨT}
) where {ΨT<:Real, ΣT<:Real, ΦT}
    return transition(rng, cycle, ψ, cycle.σ)
end

## STOCHASTIC TREND ###########################################################

struct StochasticTrend{XT<:Real,ΣT<:Union{Real, Missing}}
    ϕ::Matrix{Float64}
    σ::ΣT
    m::Int64

    init_state::Vector{XT}

    function StochasticTrend{XT}(
        m::Int64, σε::ΣT = missing, init_obs::YT = 0.0
    ) where {YT<:Real, ΣT, XT}
        ϕ = UpperTriangular(ones(m, m))
        
        init_state = zeros(XT, m)
        init_state[1] = convert(XT, init_obs)
        
        return new{XT, ΣT}(ϕ, σε, m, init_state)
    end
end

Base.eltype(::StochasticTrend{XT, ΣT}) where {XT, ΣT} = XT

# outer constructor for unspecified state types
function StochasticTrend(
    m::Int64, σε²::ΣT, init_obs::YT = 0.0
) where {ΣT<:Real, YT<:Real}
    return StochasticTrend{YT}(m, sqrt(σε²), init_obs)
end

function StochasticTrend(
    m::Int64, init_obs::YT = 0.0
) where YT <: Real
    return StochasticTrend{YT}(m, missing, init_obs)
end

# sample from the transition process prior density
function initial_draw(
    rng::AbstractRNG,
    trend::StochasticTrend{XT}
) where XT <: Real
    return trend.init_state + rand(rng, MvNormal(1.e1*I(trend.m)))
end

# sample from the state transition density
@inline function transition(
    rng::AbstractRNG,
    trend::StochasticTrend{XT},
    x::AbstractArray{XT},
    σ::ΣT
) where {XT<:Real, ΣT<:Real}
    x = trend.ϕ*x
    x[end] += σ*randn(rng)
    return convert(AbstractArray{XT}, x)
end

function transition(
    rng::AbstractRNG,
    trend::StochasticTrend{XT, ΣT},
    x::AbstractArray{XT}
) where {XT<:Real, ΣT<:Real}
    return transition(rng, trend, x, trend.σ)
end