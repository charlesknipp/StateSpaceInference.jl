"""
    Abstract Filter
"""
abstract type AbstractFilter end

struct KalmanFilter <: AbstractFilter end

KF() = KalmanFilter()

function filter_step!(
        ::AbstractRNG,
        model::LinearGaussianStateSpaceModel,
        ::KalmanFilter
    )
    particle = Gaussian(model.X0, Matrix(model.Σ0))
    return particle, 0.0
end

# TODO: extend node structure to Gaussians, although I'm not sure if this makes sense
function filter_step!(
        ::AbstractRNG,
        model::LinearGaussianStateSpaceModel,
        particle::Gaussian,
        observation,
        ::KalmanFilter;
        save_history::Bool = false
    )
    particle = let A = model.A, Q = model.Q, μ = particle.μ, Σ = particle.Σ
        Gaussian(A*μ, A*Σ*A' + Q)
    end

    particle, residual, S = correct(particle, Gaussian(observation, model.R), model.B)
    log_marginal = logpdf(Gaussian(zero(residual), symmetrize!(S)), residual)
    return particle, log_marginal
end

function sample(
        rng::AbstractRNG,
        model::LinearGaussianStateSpaceModel,
        observations,
        filter::KalmanFilter;
        save_history::Bool = false
    )
    particle, log_evidence = filter_step!(rng, model, filter)

    for t in eachindex(observations)
        particle, logℓ = filter_step!(
            rng, model, particle, observations[t], filter;
            save_history = save_history
        )
        log_evidence  += logℓ
    end

    return particle, log_evidence
end

struct ParticleFilter <: AbstractFilter
    num_particles::Int64
    ess_threshold::Float64
    resampling_algorithm
end

PF(N::Int,ess_thrshld::Float64) = ParticleFilter(N, ess_thrshld, systematic_resampling)

# Particle Filter 
ess(weights) = inv(sum(abs2, weights))

function systematic_resampling(
        rng::AbstractRNG,
        weights::Vector{WT},
        n::Int64 = length(weights)
    ) where WT <: Real
    # pre-calculations
    @inbounds v = n*weights[1]
    u = oftype(v, rand(rng))

    # initialize sampling algorithm
    a = Array{Int64}(undef, n)
    idx = 1

    @inbounds for i in 1:n
        while v < u
            idx += 1
            v += n*weights[idx]
        end
        a[i] = idx
        u += one(u)
    end

    return a
end

function filter_step!(
        rng::AbstractRNG,
        model::SSMProblems.AbstractStateSpaceModel,
        filter::ParticleFilter
    )
    # fix this later...
    particles = map(1:filter.num_particles) do i
        state = SSMProblems.transition!!(rng, model)
        SSMProblems.Utils.Particle(state)
    end

    return ParticleContainer(particles), 0.0
end

function filter_step!(
        rng::AbstractRNG,
        model::SSMProblems.AbstractStateSpaceModel,
        particles::ParticleContainer,
        observation,
        filter::ParticleFilter;
        save_history::Bool = false
    )
    weights = softmax(particles.log_weights)
    log_marginals = zeros(length(particles))

    if ess(weights) <= (filter.ess_threshold*filter.num_particles)
        idx = filter.resampling_algorithm(rng, weights)
        particles = ParticleContainer(particles[idx])
    end

    # storing particle geneology is not ideal, but allows for smoothing
    for i in eachindex(particles)
        latent_state = SSMProblems.transition!!(rng, model, particles[i].state)
        particles[i] = if save_history
            SSMProblems.Utils.Particle(particles[i], latent_state)
        else
            SSMProblems.Utils.Particle(latent_state)
        end

        log_marginals[i] = SSMProblems.emission_logdensity(
            model,
            particles[i].state,
            observation
        )

        particles.log_weights[i] += log_marginals[i]
    end

    log_marginal = logsumexp(log_marginals)-log(length(particles))
    return particles, log_marginal
end

function sample(
        rng::AbstractRNG,
        model::SSMProblems.AbstractStateSpaceModel,
        observations::AbstractVector,
        filter::ParticleFilter;
        save_history::Bool = false
    )
    particles, log_evidence = filter_step!(rng, model, filter)
    for t in eachindex(observations)
        particles, log_marginal = filter_step!(
            rng, model, particles, observations[t], filter;
            save_history = save_history
        )
        log_evidence += log_marginal
    end

    return particles, log_evidence
end