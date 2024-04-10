# borrowed/transcribed from StateSpaceModels.jl
function cholesky_decomposition(x::AbstractMatrix{XT}) where XT
    if isposdef(x)
        return cholesky(x)
    elseif all(i -> i >= 0.0, eigvals(x))
        size_matrix = size(x, 1)
        chol_x = cholesky(x .+ I(size_matrix) .* floatmin(XT))

        chol_x.L[:, :]  = round.(chol_x.L; digits = 10)
        chol_x.U[:, :]  = round.(chol_x.U; digits = 10)
        chol_x.UL[:, :] = round.(chol_x.UL; digits = 10)

        return chol_x
    else
        @error("Matrix is not positive definite or semidefinite. Cholesky decomposition cannot be performed.", x)
    end
end

# for Kalman filter stability
symmetrize!(A::Symmetric) = A
symmetrize!(A) = Symmetric(A)
symmetrize!(A::Number) = A

# to unify filter_step processes, create a new constructor for Gaussian
function GaussianDistributions.Gaussian(x::Number, Σ::AbstractArray{<:Number})
    return Gaussian([x], Σ)
end

#=
TODO: the over reliance on PDMats here is pretty annoying, since it can be wildly
inefficient to cholesky especially for particle filters
=#
struct LinearGaussianStateSpaceModel{XT<:Number} <: AbstractStateSpaceModel
    """
        A state space model with linear dynamics and Gaussian noise.
        The model is defined by the following equations:
        x[0] = X0 + σ,                σ    ∼ N(0, Σ0)
        x[t] = Ax[t-1] + μ + ε[t],    ε[t] ∼ N(0, Q)
        y[t] = Bx[t] + η[t],          η[t] ∼ N(0, R)
    """
    X0::Vector{XT}
    Σ0::PDMat{XT}

    A::Matrix{XT}
    μ::Vector{XT}
    Q::PDMat{XT}

    B::Matrix{XT}
    R::PDMat{XT}
end

# TODO: Create a conversion from AbstractMatrix{XT} to PDMat{XT} instead of this
function LinearGaussianStateSpaceModel(
        X0::Vector{XT}, Σ0::AbstractMatrix{XT},
        A::Matrix{XT}, μ::Vector{XT}, Q::AbstractMatrix{XT},
        B::Matrix{XT}, R::AbstractMatrix{XT}
    ) where XT
    return LinearGaussianStateSpaceModel(
        X0, PDMat(Σ0, cholesky_decomposition(symmetrize!(Σ0))),
        A, μ, PDMat(Q, cholesky_decomposition(symmetrize!(Q))),
        B, PDMat(R, cholesky_decomposition(symmetrize!(R)))
    )
end

# Model densities
f0(model::LinearGaussianStateSpaceModel) = Gaussian(model.X0, model.Σ0)
f(x::Vector{<:Number}, model::LinearGaussianStateSpaceModel) = Gaussian(model.A*x + model.μ, model.Q)
g(x::Vector{<:Number}, model::LinearGaussianStateSpaceModel) = Gaussian(model.B*x, model.R)

# Sampling process
function SSMProblems.transition!!(
        rng::AbstractRNG,
        model::LinearGaussianStateSpaceModel
    )
    return rand(rng, f0(model))
end

function SSMProblems.transition!!(
        rng::AbstractRNG,
        model::LinearGaussianStateSpaceModel,
        state::Vector{Float64}
    )
    return rand(rng, f(state, model))
end

function SSMProblems.transition_logdensity(
        model::LinearGaussianStateSpaceModel,
        prev_state::Vector{XT},
        state::Vector{XT}
    ) where XT
    try
        return logpdf(f(prev_state, model), state)
    catch e
        # usually this is because of LAPACK errors, just watch out for those
        show(f(prev_state, model).Σ)
    end
end

function SSMProblems.emission_logdensity(
        model::LinearGaussianStateSpaceModel,
        state::Vector{XT},
        observation::Vector{<:Number}
    ) where XT
    return logpdf(g(state, model), observation)
end

function SSMProblems.emission_logdensity(
        model::LinearGaussianStateSpaceModel,
        state::Vector{XT},
        observation::Number
    ) where XT
    return logpdf(g(state, model), [observation])
end