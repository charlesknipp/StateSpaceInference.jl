module StateSpaceInference

using GaussianDistributions: correct, logpdf
using LinearAlgebra
using PDMats

using Random
using StatsFuns
using Distributions
using StatsBase
using GaussianDistributions

using Printf

import Distributions: sample
import GaussianDistributions: Gaussian
import StatsBase: weights, mean, cov, mean_and_cov
import Statistics: mean, cov

using SSMProblems

# import SSMProblems: Particle, ParticleContainer, linearize
# import SSMProblems: AbstractStateSpaceModel, transition!!, emission_logdensity

# export Particle, ParticleContainer, linearize
# export  AbstractStateSpaceModel, transition!!, emission_logdensity

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
