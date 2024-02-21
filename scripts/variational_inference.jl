using DrWatson 
@quickactivate "BNNGF"
using Random, Flux, Statistics, Distributions,LinearAlgebra,Infiltrator,Functors


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
    #@infiltrate
    epsilon = rand(my_gaussian.normal, length(my_gaussian.ρ))#THIS SHOULD BE NORMAL
    epsilon =reshape(epsilon,size(my_gaussian.μ))
    return my_gaussian.μ .+ my_gaussian.ρ .* epsilon#THIS SHOULD BE SIGMA
end

#ln(P(w))
function  logpdf(my_gaussian::Gaussian, x::AbstractArray{Float64})
  #@infiltrate
  return  sum(-log(sqrt(2*pi)) .- log.(sigma(my_gaussian)) .- 0.5*sum((x-my_gaussian.μ).^2 ./ sigma(my_gaussian).^2))
end


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

function logpdf(m::ScaleMixtureGaussian, x::AbstractArray{Float64})
    #@infiltrate
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
  
    training::Bool
    w::AbstractArray
    b::AbstractArray
end

function BayesianLinear(in_features::Int, out_features::Int)
    weight_mean = randn(out_features, in_features)
    weight_rho = randn(out_features, in_features) 
    weight = gaussian(weight_mean, weight_rho)

    bias_mean = randn(out_features,1)
    bias_rho = randn(out_features, 1) 
    bias = gaussian(bias_mean, bias_rho)

    weight_prior = ScaleMixtureGaussian(my_π, σ1, σ2)
    bias_prior = ScaleMixtureGaussian(my_π, σ1, σ2)
    log_prior = 0 
    log_variational_posterior = 0
  

    training = false
    #create random initial sample of weights and biases
    w = sample(weight)
    b = sample(bias)


  return BayesianLinear(in_features, out_features, weight_mean, weight_rho, weight, bias_mean, bias_rho, bias, weight_prior, bias_prior, log_prior, log_variational_posterior,training,w,b)
end

@functor BayesianLinear

function(layer::BayesianLinear)(x::AbstractArray;sample_weights = true,calculate_logpdfs = false)

        if layer.training | sample_weights
          layer.w = sample(layer.weight)
          layer.b = sample(layer.bias)
        else
          layer.w = layer.weight.μ
          layer.b = layer.bias.μ
        end
      
        if layer.training | calculate_logpdfs
          layer.log_prior = logpdf(layer.weight_prior, layer.w) + logpdf(layer.bias_prior, layer.b)
          layer.log_variational_posterior = logpdf(layer.weight, layer.w) + logpdf(layer.bias, layer.b)
          
        else 
          layer.log_prior = 0
          layer.log_variational_posterior = 0
        end
    return layer.w * x .+ layer.b
end


function ReLU(x::AbstractArray)
    return max.(0,x)
end



"""Hidden Layers"""

mutable struct HiddenLayers{T}
    layers::AbstractArray{T}
end

function (h::HiddenLayers)(x::AbstractArray; sample_weights = false,calculate_logpdfs = true)
    for layer in h.layers
        x = layer(x;sample_weights = sample_weights,calculate_logpdfs = calculate_logpdfs) |> ReLU
    end
  return x
end


"""Bayesian Neural Network"""
mutable struct BayesianNeuralNetwork{T}
    input_layer::T
    hidden_layers::HiddenLayers{T}
    output_layer::T
end

@functor BayesianNeuralNetwork

function BayesianNeuralNetwork(input_size::Int,output_size::Int;hidden_size::Int = 10,hidden_layers::Int = 2)
  #construct the input layer
  input_layer = BayesianLinear(input_size,hidden_size)
  #constuc alternating chain of hidden layers and ReLU activation #compose functions
  hidden_layers = HiddenLayers([BayesianLinear(hidden_size,hidden_size) for i in 1:hidden_layers])
  #output_layer = BayesianLinear(hidden_size,output_size)
  output_layer = BayesianLinear(hidden_size,output_size)
  return BayesianNeuralNetwork(input_layer,hidden_layers,output_layer)
end

function (bnn::BayesianNeuralNetwork)(x::AbstractArray;sample_weights = false,calculate_logpdfs = true)
    x = bnn.input_layer(x,sample_weights = sample_weights,calculate_logpdfs = calculate_logpdfs) |> ReLU
    x = bnn.hidden_layers(x,sample_weights = sample_weights,calculate_logpdfs = calculate_logpdfs)
    return bnn.output_layer(x,sample_weights = sample_weights,calculate_logpdfs = calculate_logpdfs)
end



"""
Loss Functions
"""
#calculate total variationla posterior 
function total_variational_posterior(bnn::BayesianNeuralNetwork)
    return bnn.input_layer.log_variational_posterior + bnn.output_layer.log_variational_posterior +  mapreduce(hidden_layer -> hidden_layer.log_variational_posterior,+,bnn.hidden_layers.layers)
end
#calculate total log_prior
function total_log_prior(bnn::BayesianNeuralNetwork)
  return bnn.input_layer.log_prior + bnn.output_layer.log_prior +  sum(mapreduce(hidden_layer -> hidden_layer.log_prior,+,bnn.hidden_layers.layers))
end

#to calculate the total log likelihood - we need to calculate the log likelihood of the output layer
function  log_likelihood(y_est::Float64,y_samp::Float64;σ = 1)
  """
  log likelihood of the output layer,sample from gaussian PDF centered at y_samp with variance σ
  """
  return  sum(-log(sqrt(2*pi)) - log(σ) - 0.5*sum((y_est-y_samp).^2 ./ σ.^2))
end

function log_likelihood(bnn::BayesianNeuralNetwork,x_samp::Float64,y_samp::Float64)
 """
  log likelihood of the output layer,sample from gaussian PDF centered at y_samp with variance σ for a single data point in the training set
 """ 
  y_est = bnn.(x_samp)

  return log_likelihood(y_est,y_samp)
end

function log_likelihood(bnn::BayesianNeuralNetwork,x_samp::AbstractArray{Float64},y_samp::AbstractArray{Float64})
  """
  log likelihood of the output layer,sample from gaussian PDF centered at y_samp with variance σ for a batch of data
  """
  y_est = bnn.(x_samp)
  batch_ll = map(x -> log_likelihood(x[1],x[2]),zip(y_est,y_samp)) | sum
  return batch_ll
end

function elbo(bnn::BayesianNeuralNetwork,x_samp::AbstractArray{Float64},y_samp::Float64;n_samples = 10)
  """
  calculate the evidence lower bound
  """
  #monte carlo estimate of elbo
  outputs  = []
  log_priors = []
  log_variational_posteriors = []
  total_log_likelihoods =[]

  #monte carlo sampling of weights
  for i in 1:n_samples
    #generate the fist sample of the weights
    y_est_list = []#store the output of the network for each data point for a random weight sample
    push!(y_est_list,bnn(x_samp[1];sample_weights = true))
    for i in 2:length(x_samp) #use the same weights for the entire batch
      push!(y_est_list,bnn(x_samp[i];sample_weights = false))
    end
    #compute the log likelihood of the output given weight sample i
    log_likelihood_terms = map(x -> log_likelihood(x[1],x[2]),zip(y_est_list,y_samp)) 
    total_log_likelihood_i = sum(log_likelihood_terms)

    push!(log_priors,total_log_prior(bnn))
    push!(log_variational_posteriors,total_variational_posterior(bnn))
    push!(total_log_likelihoods,total_log_likelihood_i)
  end
    F_i =  log_variational_posteriors -. log_priors .- total_log_likelihoods #Q: is this the likelihood or what? Should this be the mean or sum? 

    return mean(F_i) #Q: Should this be the mean or the sum
 
  return 
end


"""
Testing
"""
#prior hyperparameters
my_pi = 1.0
sigma1 = 1.
sigma2 = 1e-2

bnn = BayesianNeuralNetwork(10,1)

input = rand(10)
l = BayesianLinear(10,10)





N = 10
input = rand(10)
output = sum(2 .* input.^2 .+ 3 .* input .+ 1 .+ randn(1))


bnn(input)

elbo(bnn,input,output,n_samples = 10)




