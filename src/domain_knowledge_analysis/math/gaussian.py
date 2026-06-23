import math
import torch

# Assumes p(z) = N(0, I) and q(z|x) = N(mean, exp(log_variance)) is a diagonal Gaussian distribution

def log_prob(z, mean, log_variance):
# log q(z) = -0.5 * sum_j [log(2π) + log_variance_j + (z_j - mean_j)^2 / variance_j ]

    log_p_z_j = -0.5 * (math.log(2*math.pi) + log_variance + torch.pow(z-mean, 2) / torch.exp(log_variance))
 
    log_p_z =  torch.sum(log_p_z_j, dim=1)

    return log_p_z

def sample(mean, log_variance):
# z = mean + exp(0.5 * log_variance) * epsilon, where epsilon ~ N(0, I)

    epsilon = torch.randn_like(mean)

    z = mean + torch.exp(0.5 * log_variance) * epsilon

    return z 

def kl_divergence(mean, log_variance):
# KL = 0.5 * sum_j [exp(log_variance_j) + mean_j^2 - 1 - log_variance_j]

    kl_j = 0.5 * (torch.exp(log_variance) + torch.pow(mean, 2) - 1 - log_variance)

    kl = torch.sum(kl_j, dim=1)

    return kl

