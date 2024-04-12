include("uc_framework.jl")

# could be replaced with ComponentArrays or something more robust...
struct UnobservedComponents{XT, ΨT, ΣT}
    trend::Vector{XT}
    cycle::Vector{ΨT}
    vol::Vector{ΣT}
end

Base.adjoint(uc::UnobservedComponents{XT, ΨT, ΣT}) where {XT, ΨT, ΣT} = begin
    [uc.trend..., uc.cycle..., uc.vol...]'
end

mutable struct HarveyTrimburSSM{XT, ΨT, ΣT, ΘT<:Real} <: SSMProblems.AbstractStateSpaceModel
    θ::NamedTuple{(:λ, :ρ, :σκ², :γ), NTuple{4, ΘT}}

    trend_process::StochasticTrend{XT, Missing}
    cycle_process::StochasticCycle{ΨT, Matrix{ΘT}, ΘT}
    
    meas_volatility::StochasticVolatility{ΣT, ΘT}
    trend_volatility::StochasticVolatility{ΣT, ΘT}

    function HarveyTrimburSSM(
        θ::NamedTuple{(:λ, :ρ, :σκ², :γ), NTuple{4, ΘT}},
        n::Int64,
        m::Int64;
        prob::Float64 = 0.99,
        init_state::Float64 = 0.0
    ) where ΘT <: Real
        trend = StochasticTrend(m, init_state)
        cycle = StochasticCycle(n, θ.ρ, θ.λ, θ.σκ²)

        η_vol = StochasticVolatility(θ.γ, prob)
        ε_vol = StochasticVolatility(θ.γ, prob)

        ΣT = Base.promote_eltype(η_vol, ε_vol)
        return new{eltype(trend), eltype(cycle), ΣT, ΘT}(
            θ, trend, cycle, η_vol, ε_vol)
    end
end

function SSMProblems.transition!!(
    rng::AbstractRNG,
    model::HarveyTrimburSSM{XT, ΨT, ΣT}
) where {XT, ΨT, ΣT}
    ε_vol = initial_draw(rng, model.trend_volatility)
    η_vol = initial_draw(rng, model.meas_volatility)

    trend = initial_draw(rng, model.trend_process)
    cycle = initial_draw(rng, model.cycle_process)
    
    return UnobservedComponents{XT, ΨT, ΣT}(trend, cycle, [ε_vol, η_vol])
end

function SSMProblems.transition!!(
    rng::AbstractRNG,
    model::HarveyTrimburSSM,
    state::UnobservedComponents{XT, ΨT, ΣT}
) where {XT, ΨT, ΣT}
    ε_vol = transition(rng, model.trend_volatility, state.vol[1])
    η_vol = transition(rng, model.meas_volatility, state.vol[2])
    
    trend = transition(rng, model.trend_process, state.trend, ε_vol)
    cycle = transition(rng, model.cycle_process, state.cycle)
    
    return UnobservedComponents{XT, ΨT, ΣT}(trend, cycle, [ε_vol, η_vol])
end

function SSMProblems.emission_logdensity(
    model::HarveyTrimburSSM,
    state::UnobservedComponents,
    observation::Float64
)
    η = observation - (state.trend[1]+state.cycle[1])
    ση² = 0.5*exp(state.vol[2])
    return logpdf(Normal(0.0, ση²), η)
end

## READING DATA ###############################################################

using CSV, DataFrames

# for demonstration, I queried quarterly PCE (index) from FRED
fred_data = CSV.read("data/fred_data.csv", DataFrame)
gdp_data = fred_data.gdp

## TESTING ####################################################################

using BenchmarkTools

params = (λ=0.352, ρ=0.714, σκ²=2.54, γ=1.5)
model = HarveyTrimburSSM(params, 2, 2; init_state = 816.542)

rng = Random.MersenneTwister(1234)
sample(rng, model, gdp_data, PF(4096, 1.0))

# 114.630 ms (1,860,349 allocations: 163.89 MiB)
@btime sample($rng,$model,$gdp_data,$(PF(1024,1.0)))

## SMC ########################################################################

function construct_model(
    m::Int64,
    n::Int64,
    params::AbstractVector{<:Real},
    init_state::Float64
)
    θ = NamedTuple{(:λ, :ρ, :σκ², :γ)}(params)
    return HarveyTrimburSSM(θ, n, m; init_state = init_state)
end

prior = product_distribution(
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