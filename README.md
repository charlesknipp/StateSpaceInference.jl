# StateSpaceInference

[![Build Status](https://github.com/charlesknipp/StateSpaceInference.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/charlesknipp/StateSpaceInference.jl/actions/workflows/CI.yml?query=branch%3Amain)

An intuitive approach to joint estimation of state space models. This package seeks to extend the framework defined by SSMProblems.jl specifically with parameter estimation in mind.

## Defining the State Space Model

Since this package extends SSMProblems.jl, state spaces are defined in a nearly identical fashion. The exception being linear Gaussian state spaces, which are defined by their linear system.

```
x[t+1] = A*x[t] + ε[t]      s.t.    ε[t] ~ N(0,Q)
  y[t] = B*x[t] + η[t]      s.t.    η[t] ~ N(0,R)
```

For example, one can define the local level model, with 2 parameters `(ση², σε²)` and a diffuse prior, like so:

```julia
function LocalLevel(params::NamedTuple{(:ση², :σε²)})
    return LinearGaussianStateSpaceModel(
        [0.0], [1.e6;;],
        [1.0;;], zeros(1), [params.σε²;;],
        [1.0;;], [params.ση²;;]
    )
end
```

And subsequently parameterize it with a named tuple for legibility.

```julia
θ = (ση²=0.7, σε²=0.2)
model = LocalLevel(θ)
```

## Filtering

The estimation of latent states, and unbiased log evidence, is calculated via filtering. For the linear Gaussian model, the Kalman filter yields an analytical solution to the problem. For non-linearities, the bootstrap particle filter is readily available.

Using the above model, one can extract a tuple of final state particles (or a `Gaussian` in the case of the Kalman filter) and the log evidence.

```julia
states, log_evidence = sample(rng, model, y, KF())
```

Instead of calling the wrapper, this can also be done iteratively using `filter_step!()` to update filtered states over time. For demonstration, one can program a bootstrap particle filter using the following:

```julia
# 1024 partilces, resampled at every time step
filter = PF(1024, 1.0)

# sample from the prior
particles, log_evidence = filter_step!(rng, model, filter)

# transition particles through observations
for t in eachindex(observations)
    particles, log_marginal = filter_step!(
        rng, model, particles, observations[t], filter;
        save_history = true
    )
    log_evidence += log_marginal
end
```

In addition to filtering methods, we can achieve particle smoothing using either particle geneology like so:

```julia
particles, _  = sample(rng, model, y, PF(1024, 1.0); save_history=true)
approx_smooth = geneology(particles)
```

Alternatively, forward-filtering backwards-sampling (FFBS) uses backwards simulation to achieve a smooth approximation.

```julia
# optionally pass FFBS(N) to set the number of recorded trajectories
ffbs_smooth = smooth(rng, model, y, PF(1024, 1.0))
```

## Parameter Estimation

To estimate this models parameters, we have to pass a function which takes a single vector as an input, and impose a prior to that function's inputs. This behavior mimics that of AdvancedMH.jl, but designed to only allocate what is necessary for parameter estimation.

```julia
function local_level(params::AbstractVector)
    θ = NamedTuple{(:ση², :σε²)}(params)
    return LocalLevel(θ; initial_covariance = 1.0)
end

prior = product_distribution(
    LogNormal(),
    LogNormal()
)
```

### Particle Markov Chain Monte Carlo

To estimate parameters in a stable, but time consuming manner, we can employ a Particle Marginal Metropolis Hastings (PMMH) sampler. For simplicity, we stick to a random walk proposal, which ensures parameter estimates fall within the support of the prior.

We perform parameter estimation via PMMH like so, throwing out the first 1000 draws:

```julia
rw_kernel = θ -> MvNormal(θ, (0.1)*I(2))
sampler = PMMH(5_000, rw_kernel, local_level, prior)

# we can also use a particle filter here too
post = sample(rng, sampler, y, KF(); burn_in = 1_000)
```

### Sequential Monte Carlo

For a more efficient sampler, we can employ the density tempered Sequential Monte Carlo (SMC) algorithm of (Duan-Fulop, 2015). Since SMC can also be done online, we must define it's sampler slightly differently:

```julia
# we use a PMMH step for particle rejuvenation with chain length set to 5 by default
sampler = SMC(256, PF(1024, 1.0); chain_length = 8)
post = batch_tempered_smc(rng, sampler, y, local_level, prior)
```

For online estimation via SMC, we look to SMC² by (Chopin, 2013). This algorithm propagates particles forward as new information is revealed; therefore we can design an iterative process using the `smc_iter()` function like so:

```julia
particles = initialize_particles(rng, model, prior, smc)
for t in eachindex(y)
    particles = smc_iter(rng, sampler, particles, y[1:t], local_level, prior)
    # [record and process particles here]
end
```

Where particles can be recorded or processed at each iteration.