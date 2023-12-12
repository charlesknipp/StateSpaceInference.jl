using StateSpaceInference
using LinearAlgebra
using StableRNGs
using Distributions

# for demonstration I use (Harvey-Trimbur, 2003)
function HarveyTrimbur(
        n::Int64, m::Int64,
        params::NamedTuple{(:ση², :σε², :σκ², :ρ, :λ)};
        initial_observation::XT = 0.0,
        initial_covariance::ΣT = 1e4
    ) where {XT<:Number, ΣT<:Number}
    U_m = UpperTriangular(ones(m, m))
    i_n = diagm([zeros(n-1)..., 1])

    cosλ = cospi(params.λ)
    sinλ = sinpi(params.λ)

    Σψ = kron(i_n, params.σκ²*I(2))
    Σx = zeros(m, m)
    Σx[end, end] = params.σε²

    Tψ  = kron(I(n), params.ρ*[cosλ sinλ;-sinλ cosλ])
    Tψ += kron(diagm(1 => ones(n-1)), I(2))

    vecΓ = inv(I(4*n^2) - kron(Tψ, Tψ))*vec(kron(i_n, params.σκ²*I(2)))
    Σψ0 = reshape(vecΓ, 2*n, 2*n)
    Σx0 = Matrix(initial_covariance*I(m))

    z = zeros(1, 2*n+m)
    z[1, 1] = 1
    z[1, m+1] = 1

    init_state = zeros(2*n+m)
    init_state[1] = initial_observation

    return LinearGaussianStateSpaceModel(
        init_state,
        cat(Σx0, Σψ0, dims = (1, 2)),
        cat(U_m, Tψ, dims = (1, 2)),
        zeros(2*n+m),
        cat(Σx, Σψ, dims = (1, 2)),
        z,
        [params.ση²;;]
    )
end

function harvey_trimbur(
        n::Int64, m::Int64,
        params::AbstractVector;
        kwargs...
    )
    θ = NamedTuple{(:ση², :σε², :σκ², :ρ, :λ)}(params)
    return HarveyTrimbur(n, m, θ; kwargs...)
end

prior = product_distribution(
    LogNormal(),
    LogNormal(),
    LogNormal(),
    Uniform(0.0, 0.99),
    Beta(2.6377, 15.0577)
)

## READING DATA ###############################################################

using CSV, DataFrames

# for demonstration, I queried quarterly PCE (index) from FRED
fred_data = CSV.read("data/fred_data.csv",DataFrame)

## PMMH #######################################################################

rng = StableRNG(1234)
ht_pmmh = PMMH(
    100,
    θ -> MvNormal(θ, (0.005)*I(5)),
    θ -> harvey_trimbur(2, 1, θ; initial_observation = fred_data.pce[1]),
    prior
)

kf_pmmh = sample(rng, ht_pmmh, fred_data.pce[1:50], KF())
pf_pmmh = sample(rng, ht_pmmh, fred_data.pce[1:50], PF(256, 1.0))

## SMC ########################################################################

# fix the seed so we can make comparison over small batches
ht_smc(smc,data) = begin
    rng = StableRNG(1234)
    return batch_tempered_smc(
        rng,
        smc,
        data,
        θ -> harvey_trimbur(2, 2, θ, initial_observation = data[1]),
        prior
    )
end

# generally the Kalman filter far out paces the particle filter
kf_smc = ht_smc(SMC(512, KF()), fred_data.gdp)
pf_smc = ht_smc(SMC(64, PF(512, 1.0)), fred_data.gdp)

# more state particles would close this gap
norm(mean(kf_smc)-mean(pf_smc))


using Plots

# use the node structure to recover the "smoothed" states
plot(reverse(smoothed_states(pf_smc)[1,:]), label="smoothed trend")
plot!(2:256, fred_data.pce, label="data")

## SMOOTHER TESTS #############################################################

using StatsBase

test_model = harvey_trimbur(2, 1, last(kf_pmmh).params; initial_observation = fred_data.pce[1])
ss = forward_trajectories(rng,test_model, fred_data.pce[1:50], PF(256, 1.0))

sm_sts = smooth(rng,test_model,fred_data.pce[1:50],PF(256,1.0),FFBS(10))

getproperty.(ss[1][end],:state)