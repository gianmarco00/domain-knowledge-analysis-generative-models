import torch
from tqdm import tqdm

from domain_knowledge_analysis.losses import negative_vae_elbo_per_image
from domain_knowledge_analysis.math import fit_diagonal_gaussian, gaussian_log_prob


class GradNormEstimator:
    def __init__(
        self,
        calibration_dataloader,
        model,
        model_architecture,
    ):
        self.model = model
        self.model_architecture = model_architecture
        self.calibration_dataloader = calibration_dataloader
        self.device = next(self.model.parameters()).device

        self.model_layers_names = [
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]

        self.model_layers_parameters = [
            parameter
            for _, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]

        self.gradnorm_mean = None
        self.gradnorm_variance = None

        self.minimum_value = 1e-12

    def estimate(self, x):

        if (self.gradnorm_mean is None or self.gradnorm_variance is None):
            raise ValueError("GradNorm estimator was not calibrated")

        gradnorm = self.estimate_gradnorm_per_layer(x)

        log_gradnorm = torch.log(torch.clamp(gradnorm, min=self.minimum_value))

        score = -gaussian_log_prob(
            log_gradnorm,
            self.gradnorm_mean,
            torch.log(self.gradnorm_variance),
        )

        return score

    def estimate_gradnorm_per_layer(self, x):

        if self.model_architecture == "vae":
            return self.estimate_vae_gradnorm_per_layer(x)

        raise ValueError(
            f"Unsupported model for GradNorm estimation: "
            f"{self.model_architecture}"
        )

    def estimate_vae_gradnorm_per_layer(self, x):

        self.model.eval()

        gradnorms_per_image = []

        for image in x:
            image = image.unsqueeze(0)

            logits, mean, log_variance = self.model(image)

            loss = negative_vae_elbo_per_image(
                image,
                logits,
                mean,
                log_variance,
            )

            gradients = torch.autograd.grad(
                outputs=loss,
                inputs=self.model_layers_parameters,
            )

            gradnorms_per_layer = torch.stack([torch.sum(gradient ** 2) for gradient in gradients])

            gradnorms_per_image.append(gradnorms_per_layer)

        return torch.stack(gradnorms_per_image, dim=0)

    def calibrate(self):

        samples = []

        progress_bar = tqdm(self.calibration_dataloader, desc="Calibrating GradNorm Estimator")

        for batch in progress_bar:
            x = batch[0].to(self.device)

            gradnorms = self.estimate_gradnorm_per_layer(x)

            log_gradnorms = torch.log(torch.clamp(gradnorms,min=self.minimum_value))

            samples.append(log_gradnorms.detach())

        samples = torch.cat(samples, dim=0)

        self.gradnorm_mean, self.gradnorm_variance = fit_diagonal_gaussian(samples)

        self.gradnorm_variance = torch.clamp(self.gradnorm_variance, min=self.minimum_value)
