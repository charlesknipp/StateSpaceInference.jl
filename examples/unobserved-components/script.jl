using StateSpaceInference
using SSMProblems
using Distributions
using Random
using LinearAlgebra, MatrixEquations

## UC PROCESSES ###############################################################

struct StochasticCycle{XT,ΣT}
    T::Matrix{XT}
    σ::ΣT
    θ::NamedTuple{(:λ, :ρ, :σκ²)}
    n::Int64

    function StochasticCycle(
        n::Int64, ρ::Float64, λ::ΛT, σκ²::ΣT
    ) where {ΛT<:Number,ΣT<:Number}
        params = (λ=λ, ρ=ρ, σκ²=σκ²)

        cosλ = cospi(λ)
        sinλ = sinpi(λ)
        
        Tψ  = kron(I(n), ρ*[cosλ sinλ;-sinλ cosλ])
        Tψ += kron(diagm(1 => ones(n-1)), I(2))

        new{eltype(Tψ),ΣT}(Tψ, sqrt(σκ²), params, n)
    end
end

# this could be calculated only once...
function initial_draw(
    rng::AbstractRNG,
    cycle::StochasticCycle
)
    dim = 2*cycle.n
    Σψ = zeros(dim,dim)

    Σψ[end-1,end-1] = cycle.θ.σκ²
    Σψ[end,end] = cycle.θ.σκ²

    return rand(rng,MvNormal(lyapd(cycle.T,Σψ)))
end

@inline function transition(
    rng::AbstractRNG,
    cycle::StochasticCycle,
    ψ::AbstractArray{ΨT}
) where ΨT <: Number
    ψ = cycle.T*ψ
    ψ[end-1:end] += cycle.σ*randn(rng, ΨT, 2)
    return ψ
end

struct StochasticTrend{XT,ΣT}
    T::Matrix{XT}
    σ::ΣT
    θ::NamedTuple{(:σε²,)}
    m::Int64

    init_state::Float64

    function StochasticTrend(
        m::Int64, σε²::ΣT, init_state::Float64 = 0.0
    ) where ΣT<:Number
        params = (σε²=σε²,)
        Tx = UpperTriangular(ones(m, m))
        new{eltype(Tx),ΣT}(Tx, sqrt(σε²), params, m, init_state)
    end
end

function initial_draw(
    rng::AbstractRNG,
    trend::StochasticTrend
)
    init_state = rand(rng,MvNormal(1.e1*I(trend.m)))
    init_state[1] += trend.init_state
    return init_state
end

@inline function transition(
    rng::AbstractRNG,
    trend::StochasticTrend,
    x::AbstractArray{XT}
) where XT <: Number
    x = trend.T*x
    x[end] += trend.σ*randn(rng,XT)
    return x
end

## HARVEY-TRIMBUR #############################################################

Parameters = @NamedTuple begin
    ση²::Float64
    σε²::Float64
    σκ²::Float64
    ρ::Float64
    λ::Float64
end

struct UnobservedComponents{XT,ΨT}
    trend::XT
    cycle::ΨT
end

Base.adjoint(uc::UnobservedComponents{XT,ΨT}) where {XT,ΨT} = [uc.trend...,uc.cycle...]'

mutable struct HarveyTrimburSSM{XT,ΨT} <: SSMProblems.AbstractStateSpaceModel
    X::Vector{UnobservedComponents{XT,ΨT}}
    observations::Vector{Float64}
    θ::Parameters

    trend_process::StochasticTrend{Float64,Float64}
    cycle_process::StochasticCycle{Float64,Float64}

    function HarveyTrimburSSM(
        θ::NamedTuple{(:ση², :σε², :σκ², :ρ, :λ)},
        n::Int64,
        m::Int64,
        init_state::Float64 = 0.0
    )
        XT = Vector{Float64}
        ΨT = Vector{Float64}

        trend = StochasticTrend(m,θ.σε²,init_state)
        cycle = StochasticCycle(n,θ.ρ,θ.λ,θ.σκ²)
        return new{XT,ΨT}(
            Vector{UnobservedComponents{XT,ΨT}}(),
            Vector{Float64}(),
            Parameters(θ),
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
        return new{XT,ΨT}(
            Vector{UnobservedComponents{XT,ΨT}}(),
            Vector{Float64}(),
            Parameters((trend.θ...,cycle.θ...,ση²=ση²)),
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
        return new{XT,ΨT}(
            Vector{UnobservedComponents{XT,ΨT}}(),
            observations,
            Parameters((trend.θ...,cycle.θ...,ση²=ση²)),
            trend,
            cycle
        )
    end
end

function SSMProblems.transition!!(
    rng::AbstractRNG,
    model::HarveyTrimburSSM
)
    x = initial_draw(rng,model.trend_process)
    ψ = initial_draw(rng,model.cycle_process)
    return UnobservedComponents(x,ψ)
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

demo_model = HarveyTrimburSSM(
    StochasticTrend(2,1.29,816.542),
    StochasticCycle(3,0.714,0.352,2.54),
    12.9
)

rng = Random.MersenneTwister(1234)

# 105.894 ms (1,599,229 allocations: 147.95 MiB)
@btime sample($rng,$demo_model,$gdp_data,$(PF(1024,1.0)))

## SMC ########################################################################

function harvey_trimbur(
    m::Int64,
    n::Int64,
    params::AbstractVector{<:Real},
    init_state::Float64
)
    θ = NamedTuple{(:ση², :σε², :σκ², :ρ, :λ)}(params)
    return HarveyTrimburSSM(θ, n, m, init_state)
end

prior = product_distribution(
    LogNormal(),
    LogNormal(),
    LogNormal(),
    Uniform(0.0, 0.99),
    Beta(2.6377, 15.0577)
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
particles = ht_smc(SMC(256, PF(2048, 1.0)), fred_data.gdp)