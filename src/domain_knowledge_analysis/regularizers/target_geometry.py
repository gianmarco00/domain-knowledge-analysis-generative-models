"""Target geometry with source-calibrated tolerance (no synthetic transforms).

The soft objective has the exact variational representation

    eta * [E a - c_s]_+ = max_{0 <= lambda <= eta} lambda * (E a - c_s).

We optimize this saddle objective with stochastic primal/dual steps. In
particular, we never replace it with E [a - c_s]_+. Finite-step nonconvex
primal/dual optimization does not guarantee that the inner maximum is solved.
"""

from copy import deepcopy
import math
import warnings

import torch

from domain_knowledge_analysis.adapters.lora import LoRA


def positive_integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


class GeometryBatchSampler:
    """Independent uniform batches, distinct images within each batch.

    Calibration, target training and validation use this same sampling rule and
    batch size, with independent CPU generators. No class labels are consulted.
    """

    def __init__(self, dataset, batch_size, seed):
        self.dataset = dataset
        self.batch_size = positive_integer(batch_size, "geometry_batch_size", 2)
        if len(dataset) < batch_size:
            raise ValueError("Geometry dataset must contain at least geometry_batch_size images.")
        self.generator = torch.Generator().manual_seed(seed)

    def sample(self, device):
        indices = torch.randperm(len(self.dataset), generator=self.generator)[:self.batch_size]
        return torch.stack([self.dataset[int(index)][0] for index in indices]).to(device)


def centered_gram(images):
    if images.ndim < 2 or len(images) < 2:
        raise ValueError("Connected geometry requires at least two images.")
    vectors = images.flatten(start_dim=1)
    # Avoid half-precision Gram products/energies, preserving float64 for tests.
    if vectors.dtype in {torch.float16, torch.bfloat16}:
        vectors = vectors.float()
    centered = vectors - vectors.mean(dim=0, keepdim=True)
    return centered @ centered.T


def geometry_distortion(input_gram, output_images, energy_floor=1e-12):
    """||G(output)-G(input)||_F^2 / ||G(input)||_F^2.

    Reject degenerate input geometry rather than changing its denominator.
    There is no output trace, dimension, batch-size or response normalization.
    """
    energy = input_gram.square().sum()
    if not torch.isfinite(input_gram).all() or not torch.isfinite(energy) or energy <= energy_floor:
        raise ValueError("Input Gram energy is nonfinite or near zero; reject this geometry batch.")
    output_gram = centered_gram(output_images)
    if output_gram.shape != input_gram.shape:
        raise ValueError("Input and output geometry must use the same batch members.")
    distortion = (output_gram - input_gram).square().sum() / energy
    if not torch.isfinite(distortion):
        raise ValueError("Nonfinite target geometry distortion.")
    return distortion


def image_mean(decoder, codes, distribution):
    logits = decoder(codes)
    if distribution == "bernoulli":
        return logits.sigmoid()
    if distribution == "continuous_bernoulli":
        return torch.distributions.ContinuousBernoulli(logits=logits).mean
    raise ValueError(f"Unsupported deterministic decoder image mean: {distribution}.")


class TargetGeometryRegularizer:
    VALID_VIEWS = {"full", "target_only", "source_only"}

    def __init__(self, model, source_batches, calibration_batches=128, eta=100.0,
                 dual_learning_rate=0.01, view="full", energy_floor=1e-12):
        positive_integer(calibration_batches, "calibration_batches", 2)
        for value, name in ((eta, "eta"), (dual_learning_rate, "dual_learning_rate"),
                            (energy_floor, "energy_floor")):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if view not in self.VALID_VIEWS:
            raise ValueError(f"view must be one of {sorted(self.VALID_VIEWS)}.")
        self.model = model
        self.eta = float(eta)
        self.dual_learning_rate = float(dual_learning_rate)
        self.view = view
        self.energy_floor = float(energy_floor)
        self.dual_fraction = 0.0
        self.distribution = model.decoder_distribution_name
        self.device = next(model.parameters()).device

        # An independent source snapshot avoids temporarily toggling live LoRA
        # adapters during autograd, or allowing future base-weight updates into D0.
        source = deepcopy(model)
        for name, layer in list(source.named_modules()):
            if isinstance(layer, LoRA):
                if layer.merged:
                    raise ValueError("Calibrate target geometry before merging LoRA adapters.")
                source.set_submodule(name, layer.base_layer)
        source.requires_grad_(False)
        source.eval()
        self.source_decoder = source.decoder

        distortions = []
        with torch.no_grad():
            for _ in range(calibration_batches):
                images = source_batches.sample(self.device)
                codes, _ = source.encoder(images)
                outputs = image_mean(source.decoder, codes, self.distribution)
                distortions.append(geometry_distortion(centered_gram(images), outputs, self.energy_floor).item())
        values = torch.tensor(distortions, dtype=torch.float64)
        self.source_tolerance = values.mean().item()
        self.calibration = {
            "source_tolerance": self.source_tolerance,
            "standard_error": (values.std(unbiased=True) / math.sqrt(calibration_batches)).item(),
            "batch_distortion_std": values.std(unbiased=True).item(),
            "min": values.min().item(),
            "max": values.max().item(),
            "calibration_batches": calibration_batches,
            "geometry_batch_size": source_batches.batch_size,
            "sampler": "independent uniform batches without replacement within each batch",
        }
        if self.source_tolerance >= 1:
            raise ValueError(f"Uninformative source tolerance c_s={self.source_tolerance:.6g} >= 1; collapse would satisfy it.")
        if self.source_tolerance + 2 * self.calibration["standard_error"] >= 1 or self.source_tolerance >= 0.9:
            warnings.warn("Source tolerance is close to collapse distortion 1; inspect calibration before sweeping.", stacklevel=2)

    def distortions(self, images):
        # Deterministic posterior mean, with gradients to the adapted encoder.
        # Restore every module's mode even if the geometry check raises.
        modes = [(module, module.training) for module in self.model.modules()]
        self.model.eval()
        try:
            codes, _ = self.model.encoder(images)
            gram = centered_gram(images.detach())
            result = {}
            if self.view != "source_only":
                outputs = image_mean(self.model.decoder, codes, self.distribution)
                result["target_distortion"] = geometry_distortion(gram, outputs, self.energy_floor)
            if self.view != "target_only":
                # D0 parameters are frozen; deliberately DO NOT use no_grad here.
                outputs = image_mean(self.source_decoder, codes, self.distribution)
                result["source_chart_distortion"] = geometry_distortion(gram, outputs, self.energy_floor)
            result["worst_distortion"] = torch.stack(list(result.values())).max()
            return result
        finally:
            for module, training in modes:
                module.training = training

    @property
    def multiplier(self):
        return self.eta * self.dual_fraction

    def primal_loss(self, worst_distortion):
        return self.multiplier * (worst_distortion - self.source_tolerance)

    def update_dual(self, worst_distortion):
        # Normalized dual coordinate q=lambda/eta: lambda's ascent step size
        # is eta*dual_learning_rate. The sample violation is unbiased for E a-c_s.
        violation = float(worst_distortion.detach()) - self.source_tolerance
        self.dual_fraction = min(1.0, max(0.0, self.dual_fraction + self.dual_learning_rate * violation))

    def penalty(self, mean_worst_distortion):
        return max(0.0, float(mean_worst_distortion) - self.source_tolerance)

    def state_dict(self):
        return {"dual_fraction": self.dual_fraction, "eta": self.eta,
                "dual_learning_rate": self.dual_learning_rate, "view": self.view,
                "energy_floor": self.energy_floor, "calibration": self.calibration}
