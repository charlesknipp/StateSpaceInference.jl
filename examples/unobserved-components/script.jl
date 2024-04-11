using StateSpaceInference
using SSMProblems
using Distributions
using Random
using LinearAlgebra, MatrixEquations

## UC PROCESSES ###############################################################

struct StochasticCycle{XT<:AbstractArray,ΦT<:AbstractMatrix,ΣT<:Real}
    ϕ::ΦT
    σ::ΣT
    θ::NamedTuple{(:λ, :ρ, :σκ²)}
    n::Int64

    function StochasticCycle{XT}(
        n::Int64, ρ::ΡT, λ::ΛT, σκ²::ΣT
    ) where {ΡT<:Real,ΛT<:Real,ΣT<:Real,XT}
        params = (λ=λ, ρ=ρ, σκ²=σκ²)

        cosλ = cospi(λ)
        sinλ = sinpi(λ)
        
        ϕ  = kron(I(n), ρ*[cosλ sinλ;-sinλ cosλ])
        ϕ += kron(diagm(1 => ones(n-1)), I(2))

        return new{XT,typeof(ϕ),ΣT}(ϕ, sqrt(σκ²), params, n)
    end
end

# outer constructor for unspecified state types
function StochasticCycle(
    n::Int64, ρ::ΡT, λ::ΛT, σκ²::ΣT
) where {ΡT<:Real,ΛT<:Real,ΣT<:Real}
    XT = promote_type(ΡT,ΛT,ΣT)
    return StochasticCycle{Vector{XT}}(n, ρ, λ, σκ²)
end

# sample from the transition process prior density
function initial_draw(
    rng::AbstractRNG,
    cycle::StochasticCycle{ΨT}
)::ΨT where ΨT <: AbstractArray
    dim = 2*cycle.n
    Σψ = zeros(dim,dim)

    σκ² = cycle.σ^2
    Σψ[end-1,end-1] = σκ²
    Σψ[end,end] = σκ²

    # get eltype of ::ΨT and convert the distribution
    init_dist = convert(
        MvNormal{ΨT.parameters[1]},
        MvNormal(lyapd(cycle.ϕ,Σψ))
    )

    return rand(rng,init_dist)
end

# sample from the state transition density
@inline function transition(
    rng::AbstractRNG,
    cycle::StochasticCycle{ΨT},
    ψ::ΨT
) where ΨT <: AbstractArray
    ΦT = eltype(ψ)
    ψ = cycle.ϕ*ψ
    ψ[end-1:end] += cycle.σ*randn(rng, ΦT, 2)
    return convert(ΨT,ψ)
end

struct StochasticTrend{XT<:AbstractArray,ΣT<:Real}
    ϕ::Matrix{Float64}
    σ::ΣT
    θ::NamedTuple{(:σε²,)}
    m::Int64

    init_state::XT

    function StochasticTrend{XT}(
        m::Int64, σε²::ΣT, init_obs::YT = 0.0
    ) where {ΣT<:Real,YT<:Real,XT}
        params = (σε²=σε²,)
        ϕ = UpperTriangular(ones(m, m))
        
        init_state = zeros(YT,m)
        init_state[1] = init_obs
        
        return new{XT,ΣT}(ϕ, sqrt(σε²), params, m, convert(XT,init_state))
    end
end

# outer constructor for unspecified state types
function StochasticTrend(
    m::Int64, σε²::ΣT, init_obs::YT = 0.0
) where {ΣT<:Real,YT<:Real}
    return StochasticTrend{Vector{YT}}(m, σε², init_obs)
end

# sample from the transition process prior density
function initial_draw(
    rng::AbstractRNG,
    trend::StochasticTrend{XT}
)::XT where XT <: AbstractArray
    return trend.init_state + rand(rng,MvNormal(1.e1*I(trend.m)))
end

# sample from the state transition density
@inline function transition(
    rng::AbstractRNG,
    trend::StochasticTrend,
    x::XT
) where XT
    ΦT = eltype(x)
    x = trend.ϕ*x
    x[end] += trend.σ*randn(rng,ΦT)
    return convert(XT,x)
end

## HARVEY-TRIMBUR #############################################################

# could be replaced with ComponentArrays or something more robust...
struct UnobservedComponents{XT,ΨT}
    trend::XT
    cycle::ΨT
end

Base.adjoint(uc::UnobservedComponents{XT,ΨT}) where {XT,ΨT} = [uc.trend...,uc.cycle...]'

mutable struct HarveyTrimburSSM{XT,ΨT,ΘT<:Real} <: SSMProblems.AbstractStateSpaceModel
    X::Vector{UnobservedComponents{XT,ΨT}}
    observations::Vector{Float64}
    θ::NamedTuple{(:σε², :λ, :ρ, :σκ², :ση²), NTuple{5, ΘT}}

    trend_process::StochasticTrend{XT,ΘT}
    cycle_process::StochasticCycle{ΨT,Matrix{ΘT},ΘT}

    function HarveyTrimburSSM(
        θ::NamedTuple{(:σε², :λ, :ρ, :σκ², :ση²), NTuple{5, ΘT}},
        n::Int64,
        m::Int64,
        init_state::Float64 = 0.0
    ) where ΘT <: Real
        XT = Vector{Float64}
        ΨT = Vector{Float64}

        trend = StochasticTrend(m,θ.σε²,init_state)
        cycle = StochasticCycle(n,θ.ρ,θ.λ,θ.σκ²)
        return new{XT,ΨT,ΘT}(
            Vector{UnobservedComponents{XT,ΨT}}(),
            Vector{Float64}(),
            θ,
            trend,
            cycle,
        )
    end

    function HarveyTrimburSSM(
        trend::StochasticTrend,
        cycle::StochasticCycle,
        ση²::Float64
    )
        XT = Vector{Float64}
        ΨT = Vector{Float64}
        return new{XT,ΨT,Float64}(
            Vector{UnobservedComponents{XT,ΨT}}(),
            Vector{Float64}(),
            merge(trend.θ,cycle.θ,(ση²=ση²,)),
            trend,
            cycle
        )
    end

    function HarveyTrimburSSM(
        observations::Vector{Float64},
        trend::StochasticTrend,
        cycle::StochasticCycle,
        ση²::Float64
    )
        XT = Vector{Float64}
        ΨT = Vector{Float64}
        return new{XT,ΨT,Float64}(
            Vector{UnobservedComponents{XT,ΨT}}(),
            observations,
            merge(trend.θ,cycle.θ,(ση²=ση²,)),
            trend,
            cycle
        )
    end
end

function SSMProblems.transition!!(
    rng::AbstractRNG,
    model::HarveyTrimburSSM{XT,ΨT}
) where {XT,ΨT}
    x = initial_draw(rng,model.trend_process)
    ψ = initial_draw(rng,model.cycle_process)
    return UnobservedComponents{XT,ΨT}(x,ψ)
end

## FOR COMPATIBILITY WITH StateSpaceInference.jl ##############################

function SSMProblems.transition!!(
    rng::AbstractRNG,
    model::HarveyTrimburSSM,
    state::UnobservedComponents{XT,ΨT}
) where {XT,ΨT}
    trend = transition(rng,model.trend_process,state.trend)
    cycle = transition(rng,model.cycle_process,state.cycle)
    return UnobservedComponents{XT,ΨT}(trend,cycle)
end

function SSMProblems.emission_logdensity(
    model::HarveyTrimburSSM,
    state::UnobservedComponents,
    observation::Float64
)
    η = observation - (state.trend[1]+state.cycle[1])
    return logpdf(Normal(0.0,sqrt(model.θ.ση²)),η)
end

## READING DATA ###############################################################

using CSV, DataFrames

# for demonstration, I queried quarterly PCE (index) from FRED
fred_data = CSV.read("data/fred_data.csv",DataFrame)
gdp_data = fred_data.gdp

## TESTING ####################################################################

using BenchmarkTools

# use the convenience constructor for now
rng = Random.MersenneTwister(1234)
model = HarveyTrimburSSM(
    StochasticTrend(2,1.29,816.542),
    StochasticCycle(3,0.714,0.352,2.54),
    12.9
)

# 109.684 ms (1,598,205 allocations: 148.00 MiB)
@btime sample($rng,$model,$gdp_data,$(PF(1024,1.0)))

# ensure that this has NO type instability
@code_warntype HarveyTrimburSSM(
    (model.θ), 3, 2, 816.542
)

@profview sample(rng,model,gdp_data,PF(1024,1.0))

## SMC ########################################################################

function harvey_trimbur(
    m::Int64,
    n::Int64,
    params::AbstractVector{<:Real},
    init_state::Float64
)
    θ = NamedTuple{(:σε², :λ, :ρ, :σκ², :ση²)}(params)
    return HarveyTrimburSSM(θ, n, m, init_state)
end

prior = product_distribution(
    LogNormal(),
    Beta(2.6377, 15.0577),
    Uniform(0.0, 0.99),
    LogNormal(),
    LogNormal(),
)

ht_smc(smc,data) = begin
    rng = Random.MersenneTwister(1234)
    return batch_tempered_smc(
        rng,
        smc,
        data,
        θ -> harvey_trimbur(2, 3, θ, data[1]),
        prior
    )
end

# run over the entire set
particles = ht_smc(SMC(256, PF(512, 1.0)), fred_data.gdp)