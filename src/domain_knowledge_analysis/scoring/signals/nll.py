import torch

from domain_knowledge_analysis.math.bernoulli import bernoulli_log_prob_from_logits
from domain_knowledge_analysis.math.gaussian import kl_divergence
from domain_knowledge_analysis.math.elbo_function import elbo_per_image


class NLLEstimator:
    def __init__(self, model, model_architecture):
        self.model = model
        self.model_architecture = model_architecture

    def estimate(self, x):

        if self.model_architecture == "vae":
            return self.estimate_vae_nll(x)

        raise ValueError(f"Unsupported model for likelihood estimation: {self.model_architecture}")

    def estimate_vae_nll(self, x):
        self.model.eval()

        with torch.no_grad():
            logits, mean, log_variance = self.model(x)

            reconstruction_log_prob = bernoulli_log_prob_from_logits(x, logits)
            kl_divergence_term = kl_divergence(mean, log_variance)

            elbo_value = elbo_per_image(reconstruction_log_prob, kl_divergence_term)

            nll_per_image = -elbo_value

            return nll_per_image