struct SequentialMonteCarlo
    num_particles::Int
    chain_length::Int

    filter::AbstractFilter

    rs_threshold::Float64
    resampling_alg
end

function SMC(
        num_particles::Int,
        filter::AbstractFilter;
        chain_length::Int = 5,
        resampling_threshold::Float64 = 0.5
    )
    return SequentialMonteCarlo(
        num_particles,
        chain_length,
        filter,
        resampling_threshold,
        systematic_resampling
    )
end

# TODO: check type inference
function resample(
        rng::AbstractRNG,
        smc::SequentialMonteCarlo,
        weights::AbstractVector{<:Real}
    )
    return smc.resampling_alg(rng, weights)
end

minimum_ess(smc::SequentialMonteCarlo) = smc.num_particles*smc.rs_threshold


mutable struct SMCState{θT,XT}
    parameters::Vector{θT}
    states::Vector{XT}
    log_weights::Vector{Float64}
end

Base.show(io::IO, particles::SMCState{θT,XT}) where {θT, XT} = begin
    print(io, "Particles{$XT}: ($(mean(particles)))")
end

# TODO: check type inference
function initialize_particles(
        rng::AbstractRNG,
        model,
        prior::Distribution,
        filter::AbstractFilter,
        M::Int64
    )
    init_params = [rand(rng, prior) for _ in 1:M]
    return SMCState(
        init_params,
        [filter_step!(rng, model(init_params[m]), filter)[1] for m in 1:M],
        zeros(M)
    )
end

function initialize_particles(
        rng::AbstractRNG,
        model,
        prior::Distribution,
        smc::SequentialMonteCarlo
    )
    return initialize_particles(rng, model, prior, smc.filter, smc.num_particles)
end

rw_kernel(particles::SMCState) = begin
    parameters = vcat(get_parameters(particles)'...)
    weights = StatsBase.weights(softmax(particles.log_weights))
    Σθ = ((2.38^2)/size(parameters, 2))*cov(parameters, weights, 1)
    return θ -> MvNormal(θ, Σθ)
end

function batch_sample!(
        rng::AbstractRNG,
        particles::SMCState,
        model,
        smc::SequentialMonteCarlo,
        data::AbstractVector
    )
    M = smc.num_particles
    logposts = zeros(M)

    # for thread local reproducibility of parallel processing
    rngs = [deepcopy(rng) for _ in 1:M]
    seeds = rand(rng, UInt, M)

    Threads.@threads for m in 1:M
        Random.seed!(rngs[m], seeds[m])
        θm = model(particles.parameters[m])
        particles.states[m], logposts[m] = sample(rngs[m], θm, data, smc.filter)
    end

    return logposts
end

# TODO: there are far too many function arguments...
function move_particles!(
        rng::AbstractRNG,
        particles::SMCState,
        smc::SequentialMonteCarlo,
        data::AbstractVector,
        model,
        prior;
        temperature::Float64 = 1.0,
        online::Bool = false
    )
    weights = softmax(particles.log_weights)
    ess_tracker = if online
        @sprintf("t = %4d\tess = %7.2f", lastindex(data), ess(weights))
    else
        @sprintf("ξ = %1.4f\tess = %7.2f", temperature, ess(weights))
    end

    ## resampling
    idx = resample(rng, smc, weights)
    particles.parameters = particles.parameters[idx]
    particles.states = particles.states[idx]
    fill!(particles.log_weights, 0.0)

    ## rejuvenation
    kernel = rw_kernel(particles)
    chains = parallel_sample(
        rng,
        PMMH(smc.chain_length, kernel, model, prior),
        smc.filter,
        data,
        get_parameters(particles),
        temperature = temperature,
        msg = ess_tracker*"\t"
    )

    particles.parameters = getproperty.(chains, :params)
    particles.states = getproperty.(chains, :states)
    return getproperty.(chains, :log_evidence)
end

function batch_tempered_smc(
        rng::AbstractRNG,
        smc::SequentialMonteCarlo,
        data::AbstractVector,
        model,
        prior
    )
    particles = initialize_particles(rng, model, prior, smc)
    logposts  = batch_sample!(rng, particles, model, smc, data)
    logposts[isnan.(logposts)] .= -Inf 

    ξ = 0.0

    while ξ < 1.0
        lower_bound = oldξ = ξ
        upper_bound = 2.0
        local newξ

        ## temperature calculation
        while (upper_bound-lower_bound) > 1.e-10
            newξ = 0.5*(upper_bound+lower_bound)
            log_weights = (newξ-oldξ)*logposts
            weights = softmax(log_weights)

            if ess(weights) == minimum_ess(smc)
                break
            elseif ess(weights) < minimum_ess(smc)
                upper_bound = newξ
            else
                lower_bound = newξ
            end
        end

        ξ = min(newξ, 1.0)
        particles.log_weights = (ξ-oldξ)*logposts
        weights = softmax(particles.log_weights)
        ess_tracker = @sprintf("ξ = %1.4f\tess = %7.2f", ξ, ess(weights))

        if newξ ≥ 1.0
            println(ess_tracker)
            break
        end

        logposts = move_particles!(
            rng, particles, smc, data, model, prior;
            temperature = ξ
        )
    end

    return particles
end

function batch_tempered_smc(
        smc::SequentialMonteCarlo,
        data::AbstractVector,
        model,
        prior::Distribution
    )
    return batch_tempered_smc(Random.default_rng(), smc, data, model, prior)
end

function smc_iter(
        rng::AbstractRNG,
        smc::SequentialMonteCarlo,
        particles::SMCState,
        data::AbstractVector,
        model,
        prior::Distribution
    )
    M = smc.num_particles
    weights = StatsBase.weights(particles)
    ess_tracker = @sprintf("t = %4d\tess = %7.2f", lastindex(data), ess(weights))
    print("\r"*ess_tracker)

    if ess(weights) < minimum_ess(smc)
        move_particles!(
            rng, particles, smc, data, model, prior;
            online = true
        )
    end

    # for thread local reproducibility of parallel processing
    rngs = [deepcopy(rng) for _ in 1:M]
    seeds = rand(rng, UInt, M)

    log_marginals = zeros(M)
    Threads.@threads for m in 1:M
        Random.seed!(rngs[m], seeds[m])
        θm = particles.parameters[m]
        Xm = particles.states[m]

        log_marginals[m] = filter_step!(rngs[m], θm, Xm, data, smc.filter)
        particles.log_weights[m] += log_marginals[m]
    end

    return particles
end