using StateSpaceInference
using LinearAlgebra
using StableRNGs
using Distributions

function LocalLevel(
        params::NamedTuple{(:ση², :σε²)};
        initial_observation::XT = 0.0,
        initial_covariance::ΣT = 1e4
    ) where {XT<:Number, ΣT<:Number}

    return LinearGaussianStateSpaceModel(
        [initial_observation],
        [initial_covariance;;],
        [1.0;;],
        zeros(1),
        [params.σε²;;],
        [1.0;;],
        [params.ση²;;]
    )
end

function local_level(
        params::AbstractVector;
        kwargs...
    )
    θ = NamedTuple{(:ση², :σε²)}(params)
    return LocalLevel(θ; kwargs...)
end

prior = product_distribution(
    LogNormal(),
    LogNormal()
)

## READING DATA ###############################################################

using CSV, DataFrames

# for demonstration, I queried quarterly PCE (index) from FRED
fred_data = CSV.read("data/fred_data.csv",DataFrame)

## PMMH #######################################################################

rng = StableRNG(1234)
true_params = rand(rng, prior)
x, ys = generate_data(rng, local_level(true_params; initial_covariance=1.0), 100)
y = vcat(ys...)

rng = StableRNG(1234)
ll_pmmh = PMMH(
    5_000,
    θ -> MvNormal(θ, (0.1)*I(2)),
    θ -> local_level(θ; initial_observation = y[1]),
    prior
)

kf_pmmh = sample(rng, ll_pmmh, y, KF(); burn_in=1000)
pf_pmmh = sample(rng, ll_pmmh, y, PF(256, 1.0); burn_in=1000)

a = mean(getproperty.(kf_pmmh, :params))
b = mean(getproperty.(pf_pmmh, :params))

norm(a-b)

## SMC ########################################################################

# fix the seed so we can make comparison over small batches
ll_smc(smc,data) = begin
    rng = StableRNG(1234)
    return batch_tempered_smc(
        rng,
        smc,
        data,
        θ -> local_level(θ, initial_observation = data[1]),
        prior
    )
end

# generally the Kalman filter far out paces the particle filter
kf_smc = ll_smc(SMC(512, KF()), fred_data.pce)
pf_smc = ll_smc(SMC(64, PF(1024, 1.0)), fred_data.pce)

## SMOOTHER TESTS #############################################################

using StatsBase
using Plots

test_model = local_level(last(pf_pmmh).params; initial_observation = fred_data.pce[1])
ss = forward_trajectories(rng, test_model, fred_data.pce[1:50], PF(256, 1.0))

sm_particles,_ = smooth(
    rng, test_model, fred_data.pce[1:50], PF(256, 1.0)
)

plot(fred_data.pce)
plot(vcat(mean(sm_particles,dims=1)...))

mean(vcat(StateSpaceInference.get_parameters(kf_smc)'...), dims=1)