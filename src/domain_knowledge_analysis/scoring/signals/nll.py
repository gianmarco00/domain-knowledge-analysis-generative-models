import torch

from domain_knowledge_analysis.math.bernoulli import bernoulli_log_prob_from_logits
from domain_knowledge_analysis.math.gaussian import kl_divergence
from domain_knowledge_analysis.losses import negative_vae_elbo_per_image

from domain_knowledge_analysis.math.bernoulli import bernoulli_log_prob_from_logits, continuous_bernoulli_log_prob_from_logits


class NLLEstimator:
    def __init__(self, model, model_architecture):
        self.model = model
        self.model_architecture = model_architecture

        if self.model.decoder_distribution_name == "bernoulli":
            self.log_prob_function = bernoulli_log_prob_from_logits
        elif self.model.decoder_distribution_name == "continuous_bernoulli":
            self.log_prob_function = continuous_bernoulli_log_prob_from_logits
        else:
            raise ValueError("Unsupported decoder distribution: "f"{self.model.decoder_distribution_name}")

    def estimate(self, x):

        if self.model_architecture == "vae":
            return self.estimate_vae_nll(x)

        raise ValueError(f"Unsupported model for likelihood estimation: {self.model_architecture}")

    def estimate_vae_nll(self, x):
        self.model.eval()

        with torch.no_grad():
            logits, mean, log_variance = self.model(x)

            nll_per_image = negative_vae_elbo_per_image(
                x, 
                logits, 
                mean, 
                log_variance,
                log_prob_function=self.log_prob_function
            )

            return nll_per_image