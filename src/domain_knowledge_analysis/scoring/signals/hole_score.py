import math

import torch
from tqdm import tqdm

from domain_knowledge_analysis.math.gaussian import gaussian_log_prob


class HoleScoreEstimator:
    def __init__(
        self,
        calibration_dataloader,
        model,
        model_architecture,
        n_images=8,
    ):
        self.model = model
        self.model_architecture = model_architecture
        self.calibration_dataloader = calibration_dataloader
        self.device = next(self.model.parameters()).device
        self.n_images = n_images

        self.q_in_mean = None
        self.q_in_log_variance = None
        self.hole_threshold = None
        self.summary = None

    def calibrate(self):
        if self.model_architecture != "vae":
            raise ValueError(
                f"Unsupported model for Hole Score estimation: "
                f"{self.model_architecture}"
            )

        self.model.eval()

        means = []
        log_variances = []
        validation_latents = []
        actual_images = []

        progress_bar = tqdm(
            self.calibration_dataloader,
            desc="Calibrating Hole Score Estimator",
        )

        with torch.no_grad():
            for batch in progress_bar:
                x = batch[0].to(self.device)
                mean, log_variance = self.model.encoder(x)
                z = self.model.reparametrize(mean, log_variance)

                means.append(mean)
                log_variances.append(log_variance)
                validation_latents.append(z)

                if len(actual_images) < self.n_images:
                    actual_images.extend(batch[0][:self.n_images])

        self.q_in_mean = torch.cat(means, dim=0)
        self.q_in_log_variance = torch.cat(log_variances, dim=0)

        validation_latents = torch.cat(validation_latents, dim=0)
        validation_h = self.estimate_h(validation_latents)
        self.hole_threshold = torch.quantile(validation_h, 0.99)

        actual_images = torch.stack(
            actual_images[:self.n_images],
            dim=0,
        )

        self.summary = self.compute_prior_summary(actual_images)

        return self.hole_threshold

    def estimate(self, x):
        if self.q_in_mean is None or self.q_in_log_variance is None:
            raise ValueError("Hole Score estimator was not calibrated")

        if self.model_architecture != "vae":
            raise ValueError(
                f"Unsupported model for Hole Score estimation: "
                f"{self.model_architecture}"
            )

        self.model.eval()

        with torch.no_grad():
            mean, log_variance = self.model.encoder(x)
            z = self.model.reparametrize(mean, log_variance)

        return self.estimate_h(z)

    def estimate_h(self, z):
        log_prior = gaussian_log_prob(
            z,
            torch.zeros_like(z),
            torch.zeros_like(z),
        )

        return log_prior - self.log_q_in(z)

    def log_q_in(self, z):
        log_q_in_batches = []

        for z_batch in z.split(64):
            z_batch = z_batch.unsqueeze(1)
            mean = self.q_in_mean.unsqueeze(0)
            log_variance = self.q_in_log_variance.unsqueeze(0)

            component_log_q = -0.5 * torch.sum(
                math.log(2 * math.pi)
                + log_variance
                + ((z_batch - mean) ** 2) / torch.exp(log_variance),
                dim=2,
            )

            log_q_in_batch = (
                torch.logsumexp(component_log_q, dim=1)
                - math.log(len(self.q_in_mean))
            )

            log_q_in_batches.append(log_q_in_batch)

        return torch.cat(log_q_in_batches, dim=0)

    def compute_prior_summary(self, actual_images):
        z = torch.randn(
            len(self.q_in_mean),
            self.q_in_mean.shape[1],
            device=self.device,
        )
        h = self.estimate_h(z)
        holemass = torch.mean((h > self.hole_threshold).float())

        sorted_indices = torch.argsort(h)
        middle_start = (len(sorted_indices) - self.n_images) // 2

        selected_indices = {
            "lowest": sorted_indices[:self.n_images],
            "middle": sorted_indices[
                middle_start:middle_start + self.n_images
            ],
            "highest": torch.flip(
                sorted_indices[-self.n_images:],
                dims=[0],
            ),
        }

        images = {"actual": actual_images.cpu()}
        image_scores = {}

        self.model.eval()

        with torch.no_grad():
            for group_name, indices in selected_indices.items():
                logits = self.model.decoder(z[indices])
                images[group_name] = torch.sigmoid(logits).cpu()
                image_scores[group_name] = h[indices].cpu()

        return {
            "holemass": holemass.cpu(),
            "hole_threshold": self.hole_threshold.cpu(),
            "images": images,
            "image_scores": image_scores,
        }
