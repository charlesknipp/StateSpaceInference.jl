module StateSpaceInference

using SSMProblems.Utils: Particle, linearize
using GaussianDistributions: correct, logpdf
using LinearAlgebra
using PDMats

using Random
using StatsFuns
using Distributions
using StatsBase
using SSMProblems
using GaussianDistributions

using Printf

import GaussianDistributions: Gaussian
import StatsBase: weights, mean, cov, mean_and_cov
import Statistics: mean, cov

# reexport transition!!, emission_logdensity, and transition_logdensity
using SSMProblems: AbstractStateSpaceModel, transition!!, emission_logdensity, transition_logdensity
export transition!!, emission_logdensity, transition_logdensity

# reexport sample
using StatsBase: sample
export sample

"""
    ParticleContainer{T}

ParticleContainer is a weighted collection of Particles
"""
mutable struct ParticleContainer{T<:Particle}
    vals::Vector{T}
    log_weights::Vector{Float64}
end

function ParticleContainer(particles::Vector{<:Particle})
    return ParticleContainer(particles, zeros(length(particles)))
end

Base.collect(pc::ParticleContainer) = pc.vals
Base.length(pc::ParticleContainer) = length(pc.vals)
Base.keys(pc::ParticleContainer) = LinearIndices(pc.vals)

Base.@propagate_inbounds Base.getindex(pc::ParticleContainer, i::Int) = pc.vals[i]
Base.@propagate_inbounds Base.getindex(pc::ParticleContainer, i::Vector{Int}) = pc.vals[i]
Base.setindex!(pc::ParticleContainer{T}, p::T, i::Int) where T = Base.setindex!(pc.vals, p, i)

function generate_data(rng, model, T::Int)
    y = []
    x = [rand(rng, f0(model))]

    for t in 1:T
        push!(y, rand(rng, g(x[t], model)))
        if t < T
            push!(x, rand(rng, f(x[end], model)))
        end
    end

    return x, y
end

include("linear_models.jl")
include("filters.jl")
include("pmmh.jl")
include("smc.jl")
include("utils.jl")
include("smoothers.jl")

export LinearGaussianStateSpaceModel, generate_data

export PF, ParticleFilter, KF, KalmanFilter
export sample, filter_step!

export PMMH, ParticleMarginalMetropolisHastings

export SMC, SequentialMonteCarlo
export batch_tempered_smc, initialize_particles, batch_sample!

export geneology, smoothed_states

export FFBS, ForwardFilterBackwardsSampler
export smooth, forward_trajectories

end
