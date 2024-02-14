using DrWatson 
@quickactivate "BNNGF"
using Random, Flux, Statistics, Distributions,LinearAlgebra,Infiltrator


global my_π = 0.5
global σ1 = 1.
global σ2 = 0.5

"""
Gaussian Sampling
"""
#a mutable gaussian data structreu 
mutable struct Gaussian
    μ::AbstractArray
    ρ::AbstractArray
    normal::Distribution
end



function gaussian(μ::AbstractArray,ρ::AbstractArray)
    return Gaussian(μ,ρ,Normal(0,1))
end

function sigma(my_gaussian::Gaussian)
    return log.(1 .+ exp.(my_gaussian.ρ))
end

function sample(my_gaussian::Gaussian)
    epsilon = rand(my_gaussian.normal, length(my_gaussian.ρ))
    return my_gaussian.μ .+ my_gaussian.ρ .* epsilon
end

#ln(P(w))
function  logpdf(my_gaussian::Gaussian, x::Vector{Float64})
  return  sum(-log(sqrt(2*pi)) - log.(sigma(my_gaussian)) - 0.5*sum((x-my_gaussian.μ).^2 ./ sigma(my_gaussian).^2))
end

my_ρ = rand(10)
my_μ = rand(10)
g = gaussian(my_μ,my_ρ)
my_sigma = sigma(g)

epsilon = rand(g.normal)
sample(g)


"""
Scale Mixture of Gaussians
"""
mutable struct ScaleMixtureGaussian
    π::Float64
    σ1::Float64
    σ2::Float64
    normal1::Distribution
    normal2::Distribution
end

#constructor
ScaleMixtureGaussian(π::Float64, σ1::Float64, σ2::Float64) = ScaleMixtureGaussian(π, σ1, σ2, Normal(0, σ1), Normal(0, σ2))

function log_prob(m::ScaleMixtureGaussian, x::Vector{Float64})
    return log.(m.π .* pdf.(m.normal1, x) .+ (1 .- m.π) .* pdf.(m.normal2, x)) |> sum
end



"""Baysian Layer"""

mutable struct BayesianLinear

    in_features::Int
    out_features::Int

    weight_mean::AbstractArray
    weight_rho::AbstractArray
    weight::Gaussian
    
    bias_mean::AbstractArray
    bias_rho::AbstractArray
    bias::Gaussian

    weight_prior::ScaleMixtureGaussian
    bias_prior::ScaleMixtureGaussian
    log_prior::Float64
    log_variational_posterior::Float64
end

function BayesianLinear(in_features::Int, out_features::Int)
    weight_mean = randn(out_features, in_features)
    weight_rho = randn(out_features, in_features) 
    weight = gaussian(weight_mean, weight_rho)

    bias_mean = randn(out_features, in_features)
    bias_rho = randn(out_features, in_features) 
    bias = gaussian(bias_mean, bias_rho)

    weight_prior = ScaleMixtureGaussian(my_π, σ1, σ2)
    bias_prior = ScaleMixtureGaussian(my_π, σ1, σ2)
    log_prior = 0 
    log_variational_posterior = 0

    return BayesianLinear(in_features, out_features, weight_mean, weight_rho, weight, bias_mean, bias_rho, bias, weight_prior, bias_prior, log_prior, log_variational_posterior)
end



function forward(BayesianLinear, x::AbstractArray,sample::Bool)
    if sample
        w = sample(BayesianLinear.weight)
        b = sample(BayesianLinear.bias)
    else
        w = BayesianLinear.weight.μ
        b = BayesianLinear.bias.μ
    end


    return w * x .+ b
end


BayesianLinear(in_features::Int, out_features::Int, weight_mean::AbstractArray, weight_rho::AbstractArray, bias_mean::AbstractArray, bias_rho::AbstractArray) = BayesianLinear(in_features, out_features, weight_mean, weight_rho, weight_mean, weight_rho, bias_mean, bias_rho, weight_mean, weight_rho, 0, 0)

    

in_features = 10
out_features = 10


BayesianLinear(in_features, out_features)


my_pi = 1.0
sigma1 = 1.
sigma2 = 1e-2








