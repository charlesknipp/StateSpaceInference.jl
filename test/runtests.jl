using StateSpaceInference
using Distributions
using Random
using LinearAlgebra
using StatsBase
using StatsFuns

using Test
using Suppressor

# unweighted mean shouldn't be that different from the true average
parameter_mean(particles::StateSpaceInference.SMCState) = begin
    parameter_matrix = vcat(StateSpaceInference.get_parameters(particles)'...)
    weights = StatsBase.weights(softmax(particles.log_weights))
    return vec(mean(parameter_matrix', weights; dims=2))
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
    true_params = [0.2, 1.7]
    x, y = generate_data(rng, local_level(true_params), 500)
    @show true_params

    @testset "filtering methods" begin
        kf_sample, kf_ll = sample(rng, local_level(true_params), y, KF())
        pf_sample, pf_ll = sample(rng, local_level(true_params), y, PF(1024, 1.0))

        @test abs(kf_ll-pf_ll) ≤ 20
        @test mean(kf_sample) ≈ mean(pf_sample) atol = 0.1
    end

    @testset "smoothing methods" begin
        ffbs_smooth, _ = smooth(rng, local_level(true_params), y, PF(256, 1.0), FFBS(64))
        ffbs_mean = vec(vcat(mean(ffbs_smooth; dims=1)...))

        path_smooth, _ = sample(rng, local_level(true_params), y, PF(256, 1.0); save_history=true)
        path_mean = mean(geneology(path_smooth), StatsBase.weights(path_smooth))
        path_mean = reverse(vec(path_mean)[2:end])

        # shows that the two methods produce similar results
        @test mean(abs2, ffbs_mean-path_mean) < 0.5
    end

    @testset "particle marginal metropolis hastings" begin
        rw_kernel   = θ -> MvNormal(θ, (0.05)*I(2))
        pmmh_kernel = PMMH(1000, rw_kernel, local_level, prior)

        kf_sample = sample(rng, pmmh_kernel, y, KF(); burn_in = 200)
        pf_sample = sample(rng, pmmh_kernel, y, PF(64, 1.0); burn_in = 200)

        kf_mean = mean(getproperty.(kf_sample, :params))
        kf_rmse = mean(
            x -> abs2.(x),
            [kf_sample[m].params - true_params for m in eachindex(kf_sample)]
        )

        pf_mean = mean(getproperty.(pf_sample, :params))
        pf_rmse = mean(
            x -> abs2.(x),
            [pf_sample[m].params - true_params for m in eachindex(pf_sample)]
        )

        @show pf_mean
        @show kf_mean

        @test norm(pf_rmse) < 1
        @test norm(kf_rmse) < 1
    end

    @testset "sequential monte carlo algorithms" begin
        kf_particles = batch_tempered_smc(rng, SMC(128, KF()), y, local_level, prior)
        pf_particles = batch_tempered_smc(rng, SMC(64, PF(128, 1.0)), y, local_level, prior)

        kf_mean = parameter_mean(kf_particles)
        kf_rmse = mean(
            x -> abs2.(x),
            [kf_particles.parameters[m] - true_params for m in eachindex(kf_particles.parameters)]
        )

        pf_mean = parameter_mean(pf_particles)
        pf_rmse = mean(
            x -> abs2.(x),
            [pf_particles.parameters[m] - true_params for m in eachindex(pf_particles.parameters)]
        )
        
        @show pf_mean
        @show kf_mean

        @test norm(pf_rmse) < 1
        @test norm(kf_rmse) < 1
    end
end