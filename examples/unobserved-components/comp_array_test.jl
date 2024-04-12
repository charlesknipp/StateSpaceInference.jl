include("uc_framework.jl")

using ComponentArrays

mutable struct HarveyTrimburSSM{XT, ΨT, ΘT<:Real} <: SSMProblems.AbstractStateSpaceModel
    θ::NamedTuple{(:σε², :λ, :ρ, :σκ², :ση²), NTuple{5, ΘT}}

    trend_process::StochasticTrend{XT, ΘT}
    cycle_process::StochasticCycle{ΨT, Matrix{ΘT}, ΘT}

    function HarveyTrimburSSM(
        θ::NamedTuple{(:σε², :λ, :ρ, :σκ², :ση²), NTuple{5, ΘT}},
        n::Int64,
        m::Int64;
        init_state::Float64 = 0.0
    ) where ΘT <: Real
        trend = StochasticTrend(m, θ.σε², init_state)
        cycle = StochasticCycle(n, θ.ρ, θ.λ, θ.σκ²)
        return new{eltype(trend), eltype(cycle), ΘT}(
            θ,
            trend,
            cycle,
        )
    end
end

function SSMProblems.transition!!(
    rng::AbstractRNG,
    model::HarveyTrimburSSM{XT, ΨT}
) where {XT, ΨT}
    T = promote_type(XT, ΨT)
    trend = initial_draw(rng, model.trend_process)
    cycle = initial_draw(rng, model.cycle_process)
    return ComponentVector{T}(x=trend, ψ=cycle)
end

# this is the only major bottleneck
function SSMProblems.transition!!(
    rng::AbstractRNG,
    model::HarveyTrimburSSM,
    state::ComponentVector{T}
) where T
    trend = transition(rng, model.trend_process, state[:x])
    cycle = transition(rng, model.cycle_process, state[:ψ])
    return ComponentVector{T}(x=trend, ψ=cycle)
end

function SSMProblems.emission_logdensity(
    model::HarveyTrimburSSM,
    state::ComponentVector,
    observation::Float64
)
    η = observation - (state.x[1]+state.ψ[1])
    return logpdf(Normal(0.0, sqrt(model.θ.ση²)), η)
end

## READING DATA ###############################################################

using CSV, DataFrames

# for demonstration, I queried quarterly PCE (index) from FRED
fred_data = CSV.read("data/fred_data.csv", DataFrame)
gdp_data = fred_data.gdp

## TESTING ####################################################################

using BenchmarkTools

params = (σε²=1.46, λ=0.352, ρ=0.714, σκ²=2.54, ση²=12.9)
model = HarveyTrimburSSM(params, 2, 2; init_state=816.542)

@code_warntype HarveyTrimburSSM(params, 2, 2; init_state=816.542)

rng = Random.MersenneTwister(1234)
sample(rng, model, gdp_data, PF(1024, 1.0))

# 815.391 ms (9,719,551 allocations: 559.65 MiB)
@btime sample($rng,$model,$gdp_data,$(PF(1024,1.0)))

## SMC ########################################################################

function construct_model(
    m::Int64,
    n::Int64,
    params::AbstractVector{<:Real},
    init_state::Float64
)
    θ = NamedTuple{(:σε², :λ, :ρ, :σκ², :ση²)}(params)
    return HarveyTrimburSSM(θ, n, m; init_state = init_state)
end

prior = product_distribution(
    LogNormal(),
    Beta(2.6377, 15.0577),
    Uniform(0.0, 0.99),
    LogNormal(),
    LogNormal(),
)

smc_alg(smc, data) = begin
    rng = Random.MersenneTwister(1234)
    return batch_tempered_smc(
        rng,
        smc,
        data,
        θ -> construct_model(2, 3, θ, data[1]),
        prior
    )
end

# run over the entire set
particles = ht_smc(SMC(256, PF(512, 1.0)), gdp_data)