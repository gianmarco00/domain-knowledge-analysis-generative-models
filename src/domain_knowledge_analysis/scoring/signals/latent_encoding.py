import torch

class LatentEncodingEstimator:
    def __init__(
        self,
        model,
        model_architecture,
    ):
        self.model = model
        self.model_architecture = model_architecture
        self.device = next(self.model.parameters()).device

    def estimate(self, x):
        if self.model_architecture == "vae":
            return self.estimate_vae_latent_encoding(x)

        raise ValueError(
            f"Unsupported model for Latent Encoding estimation: "
            f"{self.model_architecture}"
        )

    def estimate_vae_latent_encoding(self, x):
        with torch.no_grad():
            mu, logvar = self.model.encode(x)
            z = self.model.reparameterize(mu, logvar)
            return z