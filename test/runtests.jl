using StateSpaceInference
using Distributions
using Random
using LinearAlgebra

using Test
using Suppressor

# unweighted mean shouldn't be that different from the true average
parameter_mean(particles::StateSpaceInference.SMCState) = begin
    parameter_matrix = vcat(StateSpaceInference.get_parameters(particles)'...)
    return vec(mean(parameter_matrix', dims=2))
end

@testset "local level model" begin
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

    function local_level(params::AbstractVector)
        θ = NamedTuple{(:ση², :σε²)}(params)
        return LocalLevel(θ; initial_covariance = 1.0)
    end

    prior = product_distribution(
        LogNormal(),
        LogNormal()
    )

    rng = MersenneTwister(1234)
    true_params = rand(rng, prior)
    x, y = generate_data(rng, local_level(true_params), 100)

    @testset "filtering methods" begin
        kf_sample, kf_ll = sample(rng, local_level(true_params), y, KF())
        pf_sample, pf_ll = sample(rng, local_level(true_params), y, PF(1024, 1.0))

        @test abs(kf_ll-pf_ll) ≤ 5
        @test mean(kf_sample) ≈ mean(pf_sample) atol = 1e-1
    end

    @testset "particle marginal metropolis hastings" begin
        rw_kernel   = θ -> MvNormal(θ, (0.1)*I(2))
        pmmh_kernel = PMMH(1000, rw_kernel, local_level, prior)

        kf_sample = @suppress sample(rng, pmmh_kernel, y, KF(); burn_in = 200)
        pf_sample = @suppress_out sample(rng, pmmh_kernel, y, PF(256, 1.0); burn_in = 200)

        kf_mean = mean(getproperty.(kf_sample, :params))
        @test norm(kf_mean-true_params) ≤ 1

        pf_mean = mean(getproperty.(pf_sample, :params))
        @test norm(pf_mean-true_params) ≤ 1
    end

    @testset "sequential monte carlo algorithms" begin
        kf_particles = @suppress batch_tempered_smc(rng, SMC(128, KF()), y, local_level, prior)
        pf_particles = @suppress batch_tempered_smc(rng, SMC(128, PF(64, 1.0)), y, local_level, prior)

        kf_mean = parameter_mean(kf_particles)
        @test norm(kf_mean-true_params) ≤ 1

        pf_mean = parameter_mean(pf_particles)
        @test norm(pf_mean-true_params) ≤ 1
    end
end