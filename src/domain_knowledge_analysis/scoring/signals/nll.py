import torch

from domain_knowledge_analysis.losses import VAELoss
from domain_knowledge_analysis.math.bernoulli import bernoulli_log_prob_from_logits, continuous_bernoulli_log_prob_from_logits


class NLLEstimator:
    def __init__(self, model, model_architecture):
        self.model = model
        self.model_architecture = model_architecture.lower()

        if self.model.decoder_distribution_name == "bernoulli":
            self.log_prob_function = bernoulli_log_prob_from_logits
        elif self.model.decoder_distribution_name == "continuous_bernoulli":
            self.log_prob_function = continuous_bernoulli_log_prob_from_logits
        else:
            raise ValueError(f"Unsupported decoder distribution: {self.model.decoder_distribution_name}")

        self.vae_loss = VAELoss(log_prob_function=self.log_prob_function, beta=1.0)

    def estimate(self, x):
        if self.model_architecture == "vae":
            return self.estimate_vae_nll(x)

        raise ValueError(f"Unsupported model for likelihood estimation: {self.model_architecture}")

    def estimate_vae_nll(self, x):
        self.model.eval()
        device = next(self.model.parameters()).device
        x = x.to(device)

        with torch.no_grad():
            logits, mean, log_variance = self.model(x)
            nll_per_image = self.vae_loss.negative_elbo_per_image(x, logits, mean, log_variance)

        return nll_per_image