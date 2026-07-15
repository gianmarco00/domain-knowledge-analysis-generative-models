import math

import torch
from tqdm import tqdm


class HoleScoreEstimator:
    def __init__(
        self,
        calibration_dataloader,
        model,
        model_architecture,
        num_prior_samples=4096,
        num_images_per_group=8,
        density_batch_size=256,
        density_component_batch_size=1024,
    ):
        self.model = model
        self.model_architecture = model_architecture
        self.calibration_dataloader = calibration_dataloader
        self.device = next(self.model.parameters()).device

        self.num_prior_samples = num_prior_samples
        self.num_images_per_group = num_images_per_group
        self.density_batch_size = density_batch_size
        self.density_component_batch_size = (
            density_component_batch_size
        )

        self.minimum_variance = 1e-12

        self.calibration_means = None
        self.calibration_log_variances = None
        self.hole_threshold = None
        self.holemass = None
        self.hole_images = None
        self.hole_image_scores = None

    def calibrate(self):
        if self.model_architecture != "vae":
            raise ValueError(
                f"Unsupported model for Hole Score estimation: "
                f"{self.model_architecture}"
            )

        self.model.eval()

        means = []
        log_variances = []
        calibration_latents = []

        progress_bar = tqdm(
            self.calibration_dataloader,
            desc="Calibrating Hole Score Estimator",
        )

        with torch.no_grad():
            for batch in progress_bar:
                x = batch[0].to(self.device)

                mean, log_variance = self.model.encoder(x)
                z = self.model.reparametrize(
                    mean,
                    log_variance,
                )

                means.append(mean.detach())
                log_variances.append(log_variance.detach())
                calibration_latents.append(z.detach())

        self.calibration_means = torch.cat(means, dim=0)
        self.calibration_log_variances = torch.cat(
            log_variances,
            dim=0,
        )
        self.calibration_log_variances = torch.clamp(
            self.calibration_log_variances,
            min=math.log(self.minimum_variance),
        )

        calibration_latents = torch.cat(
            calibration_latents,
            dim=0,
        )

        calibration_scores = self.compute_hole_score(
            calibration_latents
        )

        self.hole_threshold = torch.quantile(
            calibration_scores.detach().cpu(),
            0.99,
        ).to(self.device)

        self._compute_prior_hole_summary()

        return self.hole_threshold

    def estimate(self, x):
        if (
            self.calibration_means is None
            or self.calibration_log_variances is None
        ):
            raise ValueError("Hole Score estimator was not calibrated")

        if self.model_architecture == "vae":
            return self.estimate_vae_hole_score(x)

        raise ValueError(
            f"Unsupported model for Hole Score estimation: "
            f"{self.model_architecture}"
        )

    def estimate_vae_hole_score(self, x):
        self.model.eval()

        with torch.no_grad():
            mean, log_variance = self.model.encoder(x)
            z = self.model.reparametrize(
                mean,
                log_variance,
            )

            return self.compute_hole_score(z)

    def compute_hole_score(self, z):
        log_prior = self._standard_normal_log_prob(z)
        log_aggregate_posterior = (
            self._aggregate_posterior_log_prob(z)
        )

        return log_prior - log_aggregate_posterior

    def get_prior_hole_summary(self):
        if (
            self.holemass is None
            or self.hole_images is None
            or self.hole_image_scores is None
        ):
            raise ValueError(
                "Hole Score prior summary was not computed"
            )

        return {
            "holemass": self.holemass.detach().cpu(),
            "hole_threshold": self.hole_threshold.detach().cpu(),
            "images": {
                group_name: images.detach().cpu()
                for group_name, images in self.hole_images.items()
            },
            "image_scores": {
                group_name: scores.detach().cpu()
                for group_name, scores in self.hole_image_scores.items()
            },
        }

    def _compute_prior_hole_summary(self):
        z = torch.randn(
            self.num_prior_samples,
            self.latent_dim,
            device=self.device,
        )

        normal_radius_mask = self._normal_prior_radius_mask(z)
        z = z[normal_radius_mask]

        min_required_prior_samples = (
            3 * self.num_images_per_group
        )

        if len(z) < min_required_prior_samples:
            raise ValueError(
                "Not enough normal-radius prior samples to select "
                "hole-score images"
            )

        scores = self.compute_hole_score(z)

        self.holemass = torch.mean(
            (scores > self.hole_threshold).float()
        )

        selected_latents, selected_scores = (
            self._select_representative_latents(
                z=z,
                scores=scores,
            )
        )

        self.hole_images = self._decode_latent_groups(
            selected_latents
        )
        self.hole_image_scores = selected_scores

    def _select_representative_latents(self, z, scores):
        sorted_indices = torch.argsort(scores)
        median_score = torch.median(scores)
        middle_indices = torch.argsort(
            torch.abs(scores - median_score)
        )

        group_indices = {
            "lowest": sorted_indices[:self.num_images_per_group],
            "middle": middle_indices[:self.num_images_per_group],
            "highest": torch.flip(
                sorted_indices[-self.num_images_per_group:],
                dims=[0],
            ),
        }

        selected_latents = {}
        selected_scores = {}

        for group_name, indices in group_indices.items():
            selected_latents[group_name] = z[indices]
            selected_scores[group_name] = scores[indices]

        return selected_latents, selected_scores

    def _decode_latent_groups(self, selected_latents):
        images = {}

        self.model.eval()

        with torch.no_grad():
            for group_name, z in selected_latents.items():
                logits = self.model.decoder(z)
                images[group_name] = torch.sigmoid(logits)

        return images

    def _normal_prior_radius_mask(self, z):
        squared_norm = torch.sum(z ** 2, dim=1)

        lower_quantile = self._chi_square_quantile(
            probability=0.05,
            degrees_of_freedom=self.latent_dim,
        )
        upper_quantile = self._chi_square_quantile(
            probability=0.95,
            degrees_of_freedom=self.latent_dim,
        )

        return (
            (squared_norm >= lower_quantile)
            & (squared_norm <= upper_quantile)
        )

    def _standard_normal_log_prob(self, z):
        return -0.5 * (
            self.latent_dim * math.log(2 * math.pi)
            + torch.sum(z ** 2, dim=1)
        )

    def _aggregate_posterior_log_prob(self, z):
        log_probs = []

        for start in range(0, len(z), self.density_batch_size):
            z_batch = z[start:start + self.density_batch_size]
            log_probs.append(
                self._aggregate_posterior_log_prob_batch(
                    z_batch
                )
            )

        return torch.cat(log_probs, dim=0)

    def _aggregate_posterior_log_prob_batch(self, z):
        log_sum = None

        for start in range(
            0,
            len(self.calibration_means),
            self.density_component_batch_size,
        ):
            end = start + self.density_component_batch_size

            mean = self.calibration_means[start:end]
            log_variance = self.calibration_log_variances[start:end]

            component_log_probs = self._diagonal_gaussian_log_prob(
                z=z,
                mean=mean,
                log_variance=log_variance,
            )

            component_log_sum = torch.logsumexp(
                component_log_probs,
                dim=1,
            )

            if log_sum is None:
                log_sum = component_log_sum
            else:
                log_sum = torch.logaddexp(
                    log_sum,
                    component_log_sum,
                )

        return log_sum - math.log(len(self.calibration_means))

    def _diagonal_gaussian_log_prob(
        self,
        z,
        mean,
        log_variance,
    ):
        z = z.unsqueeze(1)
        mean = mean.unsqueeze(0)
        log_variance = log_variance.unsqueeze(0)

        log_prob_per_dimension = -0.5 * (
            math.log(2 * math.pi)
            + log_variance
            + ((z - mean) ** 2) / torch.exp(log_variance)
        )

        return torch.sum(log_prob_per_dimension, dim=2)

    def _chi_square_quantile(
        self,
        probability,
        degrees_of_freedom,
    ):
        try:
            return self._chi_square_quantile_by_bisection(
                probability=probability,
                degrees_of_freedom=degrees_of_freedom,
            )
        except NotImplementedError:
            return self._wilson_hilferty_chi_square_quantile(
                probability=probability,
                degrees_of_freedom=degrees_of_freedom,
            )

    def _chi_square_quantile_by_bisection(
        self,
        probability,
        degrees_of_freedom,
        iterations=80,
    ):
        distribution = torch.distributions.Chi2(
            torch.tensor(float(degrees_of_freedom))
        )

        low = torch.tensor(0.0)
        high = torch.tensor(max(float(degrees_of_freedom), 1.0))
        target = torch.tensor(float(probability))

        while distribution.cdf(high) < target:
            high = high * 2

        for _ in range(iterations):
            midpoint = (low + high) / 2

            if distribution.cdf(midpoint) < target:
                low = midpoint
            else:
                high = midpoint

        return high.to(self.device)

    def _wilson_hilferty_chi_square_quantile(
        self,
        probability,
        degrees_of_freedom,
    ):
        normal = torch.distributions.Normal(
            torch.tensor(0.0),
            torch.tensor(1.0),
        )
        z = normal.icdf(
            torch.tensor(float(probability))
        ).item()

        degrees_of_freedom = float(degrees_of_freedom)
        quantile = degrees_of_freedom * (
            1
            - 2 / (9 * degrees_of_freedom)
            + z * math.sqrt(2 / (9 * degrees_of_freedom))
        ) ** 3

        return torch.tensor(
            max(quantile, 0.0),
            device=self.device,
        )

    @property
    def latent_dim(self):
        return self.calibration_means.shape[1]
