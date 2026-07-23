from domain_knowledge_analysis.math import elbo_per_image, bernoulli_log_prob_from_logits, kl_divergence, continuous_bernoulli_log_prob_from_logits
import torch

def negative_vae_elbo_per_image(x, logits, mean, log_variance, log_prob_function=continuous_bernoulli_log_prob_from_logits):

    reconstruction_loss = log_prob_function(x, logits)

    kl_loss = kl_divergence(mean, log_variance)

    elbo_loss_per_image = elbo_per_image(reconstruction_loss, kl_loss)

    return -elbo_loss_per_image


def vae_loss(x, logits, mean, log_variance, log_prob_function=continuous_bernoulli_log_prob_from_logits):

    elbo_loss = negative_vae_elbo_per_image(
        x, 
        logits, 
        mean, 
        log_variance,
        log_prob_function=log_prob_function
    )

    return torch.mean(elbo_loss)