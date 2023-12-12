## FILTER STATISTICS ##########################################################

get_states(particles::ParticleContainer) = vcat(getproperty.(particles.vals,:state)'...)
StatsBase.weights(particles::ParticleContainer) = StatsBase.weights(softmax(particles.log_weights))

function StatsBase.mean(particles::ParticleContainer)
    return vec(StatsBase.mean(get_states(particles), StatsBase.weights(particles); dims=1))
end

function StatsBase.cov(particles::ParticleContainer)
    return StatsBase.cov(get_states(particles), StatsBase.weights(particles); dims=1)
end

function StatsBase.mean_and_cov(particles::ParticleContainer)
    μ, Σ = StatsBase.mean_and_cov(get_states(particles), StatsBase.weights(particles), 1)
    return vec(μ), Σ
end

StatsBase.mean_and_cov(particles::Gaussian) = (particles.μ, particles.Σ)

## GENERAL SMC STATISTICS #####################################################

get_parameters(particles::SMCState) = particles.parameters

function StatsBase.mean(particles::SMCState)
    return vec(StatsBase.mean(get_states(particles), StatsBase.weights(particles); dims=1))
end

function StatsBase.cov(particles::SMCState)
    return StatsBase.cov(get_states(particles), StatsBase.weights(particles); dims=1)
end

function StatsBase.mean_and_cov(particles::SMCState)
    μ, Σ = StatsBase.mean_and_cov(get_states(particles), StatsBase.weights(particles), 1)
    return vec(μ), Σ
end

## PARTICLE SMC STATISTICS ####################################################

function get_states(particles::SMCState{XT,<:ParticleContainer}) where XT
    return vcat([get_states.(particles.states)...]...)
end

function get_states(particles::SMCState{XT,<:Gaussian}) where XT
    return vcat(getproperty.(particles.states,:μ)'...)
end

function Base.pairs(particles::SMCState{XT,<:ParticleContainer}) where XT
    N = length(particles.states[1])
    return (
        eachrow(get_states(particles)),
        repeat(get_parameters(particles), inner=N)
    )
end

function StatsBase.weights(particles::SMCState{XT,<:ParticleContainer}) where XT
    param_weights = particles.log_weights
    state_weights = getproperty.(particles.states, :log_weights)
    log_weights = map(
        i -> param_weights[i] .+ state_weights[i],
        1:length(particles.states)
    )

    return StatsBase.weights(softmax(vcat(log_weights...)))
end

## GAUSSIAN SMC STATISTICS ####################################################

function Base.pairs(particles::SMCState{XT,<:Gaussian}) where XT
    return eachrow(get_states(particles)), get_parameters(particles)
end

# create a new function to handle jointly covariate Gaussians
function StatsBase.mean_and_cov(particles::SMCState{XT,<:Gaussian}) where XT
    weights = StatsBase.weights(particles)
    xdim = length(particles.states[1].μ)
    
    μ = μd = zeros(xdim,1)
    Σ = zeros(xdim,xdim)

    for i in eachindex(weights)
        wi = weights[i]
        if wi > 0.0
            state = particles.states[i]
            BLAS.axpy!(wi,state.μ,μ)
            BLAS.axpy!(wi,state.Σ,Σ)
        end
    end

    for i in eachindex(weights)
        wi = weights[i]
        if wi > 0.0
            state = particles.states[i]
            μd = state.μ - μ
            BLAS.axpy!(wi,μd*μd',Σ)
        end
    end

    return μ, Σ
end

function StatsBase.weights(particles::SMCState{XT,<:Gaussian}) where XT
    return StatsBase.weights(softmax(particles.log_weights))
end

function StatsBase.cov(particles::SMCState{XT,<:Gaussian}) where XT
    return StatsBase.mean_and_cov(particles)[2]
end

## APPLY FUNCTIONS ############################################################

# takes function f(states, parameters) and applies it to each particle
function StatsBase.mean(f::Function, particles::SMCState)
    mapped_items = map(f, pairs(particles)...)
    if eltype(mapped_items) <: NamedTuple
        names, items = reduce_named_tuples(mapped_items)
        means = map(x -> StatsBase.mean(x,StatsBase.weights(particles)), eachcol(items))
        return NamedTuple{names}(means)
    else
        return StatsBase.mean(mapped_items,weights(particles))
    end
end

function reduce_named_tuples(nts::Vector{<:NamedTuple{names}}) where names
    return names, vcat([[values(nt)...] for nt in nts]'...)
end

## LINEARIZING ################################################################

function geneology(particles::ParticleContainer)
    lineage = linearize.(particles)
    return map(x -> hcat(x...), lineage)
end

function smoothed_states(particles::ParticleContainer)
    paths = geneology(particles)
    return mean(paths, weights(particles))
end


function geneology(particles::SMCState{XT,<:ParticleContainer}) where XT
    return vcat(map(geneology,particles.states)...)
end

function smoothed_states(particles::SMCState{XT,<:ParticleContainer}) where XT
    paths = geneology(particles)
    return mean(paths, weights(particles))
end