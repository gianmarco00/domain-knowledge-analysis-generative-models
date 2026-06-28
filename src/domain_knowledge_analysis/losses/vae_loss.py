from domain_knowledge_analysis.math import elbo, bernoulli_log_prob_from_logits, kl_divergence
import torch

def vae_loss(x, logits, mean, log_variance):

    reconstruction_loss = bernoulli_log_prob_from_logits(x, logits)

    kl_loss = kl_divergence(mean, log_variance)

    elbo_loss_per_image = elbo(reconstruction_loss, kl_loss)

    elbo_loss = torch.sum(elbo_loss_per_image)

    return -elbo_loss