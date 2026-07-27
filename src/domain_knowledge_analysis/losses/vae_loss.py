import torch

from domain_knowledge_analysis.math import elbo_per_image, kl_divergence


class VAELoss(torch.nn.Module):
    def __init__(self, log_prob_function, beta=1.0):
        super().__init__()
        self.beta = beta
        self.log_prob_function = log_prob_function
        self._components = {}

    def negative_elbo_per_image(self, x, logits, mean, log_variance):
        reconstruction_log_prob = self.log_prob_function(x, logits)
        kl_loss = self.beta * kl_divergence(mean, log_variance)
        negative_elbo = -elbo_per_image(reconstruction_log_prob, kl_loss)

        self._components = {"Reconstruction Loss": -reconstruction_log_prob.detach(), "KL Loss": kl_loss.detach()}

        return negative_elbo

    def forward(self, x, logits, mean, log_variance):
        negative_elbo = self.negative_elbo_per_image(x, logits, mean, log_variance)
        return torch.mean(negative_elbo)

    def components(self):
        return self._components