abstract type AbstractSmoother end

struct ForwardFilterBackwardsSampler <: AbstractSmoother
    num_trajectories::Int
end

FFBS(M::Int) = ForwardFilterBackwardsSampler(M)

function randcat(rng::AbstractRNG, weights::AbstractVector{<:Real})
    T = eltype(weights)
    r = rand(rng, T)
    cp = weights[1]
    s = 1
    n = length(weights)
    while cp <= r && s < n
        @inbounds cp += weights[s += 1]
    end
    return s
end

function forward_trajectories(
        rng::AbstractRNG,
        model::SSMProblems.AbstractStateSpaceModel,
        observations::AbstractVector,
        filter::ParticleFilter
    )
    particles, log_evidence = filter_step!(rng, model, filter)
    trajectories = [particles]

    for t in eachindex(observations)
        particles, log_marginal = filter_step!(rng, model, particles, observations[t], filter)
        log_evidence += log_marginal
        push!(trajectories,particles)
    end

    return trajectories, log_evidence
end

function smooth(
        rng::AbstractRNG,
        model::SSMProblems.AbstractStateSpaceModel,
        observations::AbstractVector,
        filter::ParticleFilter,
        alg::ForwardFilterBackwardsSampler = FFBS(filter.num_particles)
    )
    T = lastindex(observations)
    M = alg.num_trajectories
    N = filter.num_particles

    filtered_particles, log_evidence = forward_trajectories(rng,model,observations,filter)
    filtered_states = hcat([getproperty.(filtered_particles[t],:state) for t in 1:T]...)
    log_weights = hcat(getproperty.(filtered_particles,:log_weights)...)

    backward_states = Array{Vector{Float64}}(undef,M,T)
    backward_weights = Vector{Float64}(undef,N)

    Bs = systematic_resampling(rng, weights(last(filtered_particles)), M)
    for m in 1:M
        backward_states[m,T] = filtered_states[Bs[m],T]
    end

    @inbounds for t in T-1:-1:1
        for m = 1:M
            for n = 1:N
                backwards_pass = transition_logdensity(
                    model,
                    filtered_states[n,t],
                    backward_states[m,t+1]
                )

                backward_weights[n] = log_weights[n,t] + backwards_pass
            end
            B = randcat(rng,StatsBase.weights(softmax(backward_weights)))
            backward_states[m,t] = filtered_states[B,t]
        end
    end

    return backward_states, log_evidence
end