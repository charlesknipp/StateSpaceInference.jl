struct ParticleMarginalMetropolisHastings
    chain_length::Int64
    proposal
    model
    prior
end

PMMH(iters::Int64, prop, model, prior) = ParticleMarginalMetropolisHastings(
    iters, prop, model, prior
)

struct MHState{θT, XT}
    params::θT
    states::XT
    log_evidence::Float64
end

logpost(mh::MHState) = mh.log_evidence
logprior(prior::Distribution,mh::MHState) = logpdf(prior, mh.params)

function mh_iter(
        rng::AbstractRNG,
        sampler::ParticleMarginalMetropolisHastings,
        prev_state::MHState,
        filter::AbstractFilter,
        data::AbstractVector,
        temperature::Float64
    )
    kernel = sampler.proposal
    prior  = sampler.prior

    prop_param = rand(rng, kernel(prev_state.params))

    if insupport(prior, prop_param)
        parameterized_model = sampler.model(prop_param)
        prop_states, prop_logpost = sample(rng, parameterized_model, data, filter)
    else
        prop_logpost = -Inf
    end

    # evaluate the log ratio
    log_ratio  = logpdf(prior, prop_param) - logprior(prior, prev_state)
    log_ratio += temperature*(prop_logpost - logpost(prev_state))

    # for asymmetric kernels, calculate the likelihoods
    log_ratio += logpdf(kernel(prop_param), prev_state.params)
    log_ratio -= logpdf(kernel(prev_state.params), prop_param)

    # replace the current value along the chain
    if -randexp(rng) ≤ log_ratio
        return true, MHState(prop_param, prop_states, prop_logpost)
    else
        return false, prev_state
    end
end

# running a single PMMH sampler
function mcmcsample(
        rng::AbstractRNG,
        sampler::ParticleMarginalMetropolisHastings,
        data::AbstractVector,
        filter::AbstractFilter;
        verbose::Bool = false,
        init_param = nothing,
        temperature::Float64 = 1.0
    )
    if isnothing(init_param) init_param = rand(rng, sampler.prior) end

    parameterized_model = sampler.model(init_param)
    init_states, init_likelihood = sample(rng, parameterized_model, data, filter)

    N = sampler.chain_length
    acc_flags = zeros(Int, N)
    mh_state  = MHState(init_param, init_states, init_likelihood)

    for n in 1:N
        acc_flag, mh_state = mh_iter(rng, sampler, mh_state, filter, data, temperature)
        acc_flags[n] = acc_flag

        verbose && @printf("\racceptance rate: %.4f", mean(acc_flags[1:n]))
    end

    verbose && print("\n")
    return acc_flags, mh_state
end

# for true PMCMC, this function preserves the entire chain
function sample(
        rng::AbstractRNG,
        sampler::ParticleMarginalMetropolisHastings,
        data::AbstractVector,
        filter::AbstractFilter;
        burn_in::Int64 = 1
    )
    init_param = rand(rng, sampler.prior)
    mh_states = []

    parameterized_model = sampler.model(init_param)
    init_states, init_likelihood = sample(rng, parameterized_model, data, filter)

    N = sampler.chain_length
    acc_flags = zeros(Int, N)
    mh_state  = MHState(init_param, init_states, init_likelihood)

    for n in 1:N
        acc_flag, mh_state = mh_iter(rng, sampler, mh_state, filter, data, 1.0)
        acc_flags[n] = acc_flag

        @printf("\racceptance rate: %.4f", mean(acc_flags[1:n]))
        push!(mh_states, mh_state)
    end

    print("\n")
    return mh_states[burn_in:end]
end

function serial_sample(
        rng::AbstractRNG,
        sampler::ParticleMarginalMetropolisHastings,
        filter::AbstractFilter,
        data::AbstractVector,
        init_params;
        temperature::Float64 = 1.0,
        msg::String = ""
    )
    M = length(init_params)
    acc_flags = zeros(Int64, M, sampler.chain_length)
    chains = Vector{Any}(undef, M)

    for m in 1:M
        acc_flags[m, :], chains[m] = mcmcsample(
            rng,
            sampler,
            data,
            filter;
            init_param = init_params[m],
            temperature = temperature
        )
    end

    acc_particles = mean(acc_flags, dims=2)
    acc_ratio = sum(acc_particles) / M
    @printf("\r%sacceptance rate: %.4f", msg, min(1.0,acc_ratio))

    print("\n")
    return chains
end

function parallel_sample(
        rng::AbstractRNG,
        sampler::ParticleMarginalMetropolisHastings,
        filter::AbstractFilter,
        data::AbstractVector,
        init_params;
        temperature::Float64 = 1.0,
        msg::String = ""
    )

    _init_param = init_params
    M = length(init_params)

    nchunks   = min(M, Threads.nthreads())
    chunksize = div(M, nchunks)
    interval  = 1:nchunks

    rngs = [deepcopy(rng) for _ in interval]
    datas = [deepcopy(data) for _ in interval]
    samplers = [deepcopy(sampler) for _ in interval]
    filters = [deepcopy(filter) for _ in interval]
    temps = [deepcopy(temperature) for _ in interval]

    # create a channel for progress tracking in parallel
    channel = Channel{Bool}(length(interval))
    task_list = zip(1:nchunks, rngs, datas, samplers, filters, temps)
    
    # preallocate the chains and define the child seeds
    chains = Vector{Any}(undef, M)
    seeds = rand(rng, UInt, M)
    acc_flags = zeros(Int, M, sampler.chain_length)

    @sync begin
        # track acceptance rates in parallel
        @async begin
            ctr = 0
            while take!(channel)
                ctr += 1
                acc_particles = mean(acc_flags, dims=2)
                acc_ratio = sum(acc_particles) / ctr
                @printf("\r%sacceptance rate: %.4f", msg, min(1.0, acc_ratio))
            end
        end

        @async begin
            @sync for (i, _rng, _data, _sampler, _filter, _temp) in task_list
                # determine the proper interval for parallelization
                chainidxs = if i == nchunks
                    ((i-1)*(chunksize)+1):M
                else
                    ((i-1)*(chunksize)+1):(i*(chunksize))
                end

                Threads.@spawn for chainidx in chainidxs
                    # seed the chunk defined random number generator
                    Random.seed!(_rng, seeds[chainidx])
                    acc_flags[chainidx, :], chains[chainidx] = mcmcsample(
                        _rng,
                        _sampler,
                        _data,
                        _filter,
                        init_param = _init_param[chainidx],
                        temperature = _temp
                    )

                    # update acceptance ratio calculation
                    put!(channel, true)
                end
            end
            # halt progress tracking
            put!(channel, false)
        end
    end

    print("\n")
    return chains
end

function sample(
        rng::AbstractRNG,
        sampler::ParticleMarginalMetropolisHastings,
        filter::AbstractFilter,
        data::AbstractVector,
        M::Int;
        parallel::Bool = false
    )
    init_params = [rand(rng, sampler.prior) for _ in 1:M]
    if parallel
        return parallel_sample(rng, sampler, filter, data, init_params)
    else
        return serial_sample(rng, sampler, filter, data, init_params)
    end
end