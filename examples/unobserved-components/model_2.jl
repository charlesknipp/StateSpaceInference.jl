include("uc_framework.jl")

# could be replaced with ComponentArrays or something more robust...
struct UnobservedComponents{XT, ΨT, ΣT}
    trend::Vector{XT}
    cycle::Vector{ΨT}
    vol::ΣT
end

Base.adjoint(uc::UnobservedComponents{XT, ΨT, ΣT}) where {XT, ΨT, ΣT} = begin
    [uc.trend..., uc.cycle..., uc.vol]'
end

mutable struct HarveyTrimburSSM{XT, ΨT, ΣT, ΘT<:Real} <: SSMProblems.AbstractStateSpaceModel
    θ::NamedTuple{(:σε², :λ, :ρ, :σκ², :γ), NTuple{5, ΘT}}

    trend_process::StochasticTrend{XT, ΘT}
    cycle_process::StochasticCycle{ΨT, Matrix{ΘT}, ΘT}
    vol_process::StochasticVolatility{ΣT, ΘT}

    function HarveyTrimburSSM(
        θ::NamedTuple{(:σε², :λ, :ρ, :σκ², :γ), NTuple{5, ΘT}},
        n::Int64,
        m::Int64;
        prob::Float64 = 0.99,
        init_state::Float64 = 0.0
    ) where ΘT <: Real
        trend = StochasticTrend(m, θ.σε², init_state)
        cycle = StochasticCycle(n, θ.ρ, θ.λ, θ.σκ²)
        η_vol = StochasticVolatility(θ.γ, prob)

        return new{eltype(trend), eltype(cycle), eltype(η_vol), ΘT}(
            θ, trend, cycle, η_vol
        )
    end
end

function SSMProblems.transition!!(
    rng::AbstractRNG,
    model::HarveyTrimburSSM{XT, ΨT, ΣT}
) where {XT, ΨT, ΣT}
    logση = initial_draw(rng, model.vol_process)
    trend = initial_draw(rng, model.trend_process)
    cycle = initial_draw(rng, model.cycle_process)
    return UnobservedComponents{XT, ΨT, ΣT}(trend, cycle, logση)
end

function SSMProblems.transition!!(
    rng::AbstractRNG,
    model::HarveyTrimburSSM{XT, ΨT, ΣT},
    state::UnobservedComponents{XT, ΨT, ΣT}
) where {XT, ΨT, ΣT}
    logση = transition(rng, model.vol_process, state.vol)
    trend = transition(rng, model.trend_process, state.trend)
    cycle = transition(rng, model.cycle_process, state.cycle)
    return UnobservedComponents{XT, ΨT, ΣT}(trend, cycle, logση)
end

function SSMProblems.emission_logdensity(
    model::HarveyTrimburSSM,
    state::UnobservedComponents,
    observation::Float64
)
    η = observation - (state.trend[1]+state.cycle[1])
    ση² = 0.5*exp(state.vol)
    return logpdf(Normal(0.0, ση²), η)
end

## READING DATA ###############################################################

using CSV, DataFrames

# for demonstration, I queried quarterly PCE (index) from FRED
fred_data = CSV.read("data/fred_data.csv", DataFrame)
gdp_data = fred_data.gdp

## TESTING ####################################################################

using BenchmarkTools

params = (σε²=1.46, λ=0.352, ρ=0.714, σκ²=2.54, γ=1.5)
model = HarveyTrimburSSM(params, 2, 2; init_state = 816.542)

rng = Random.MersenneTwister(1234)
sample(rng, model, gdp_data, PF(1024, 1.0))

# 108.009 ms (1,598,205 allocations: 143.89 MiB)
@btime sample($rng,$model,$gdp_data,$(PF(1024,1.0)))

## SMC ########################################################################

function construct_model(
    m::Int64,
    n::Int64,
    params::AbstractVector{<:Real},
    init_state::Float64
)
    θ = NamedTuple{(:σε², :λ, :ρ, :σκ², :γ)}(params)
    return HarveyTrimburSSM(θ, n, m; init_state = init_state)
end

prior = product_distribution(
    LogNormal(),
    Beta(2.6377, 15.0577),
    Uniform(0.0, 0.99),
    LogNormal(),
    LogNormal(),
)

smc_run(smc, data) = begin
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
particles = smc_run(SMC(256, PF(512, 1.0)), gdp_data)