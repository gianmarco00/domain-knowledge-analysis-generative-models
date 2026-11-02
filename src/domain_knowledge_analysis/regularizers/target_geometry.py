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
from torch.utils.data import Subset, TensorDataset

from domain_knowledge_analysis.adapters.lora import LoRA


def positive_integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


class GeometryBatchSampler:
    """Independent uniform batches, distinct images within each batch.

    Calibration, target training and validation use this same sampling rule and
    batch size, with independent generators on the image-storage device. No
    class labels are consulted. CPU datasets retain their original RNG stream.
    """

    def __init__(self, dataset, batch_size, seed):
        self.dataset = dataset
        self.batch_size = positive_integer(batch_size, "geometry_batch_size", 2)
        if len(dataset) < batch_size:
            raise ValueError("Geometry dataset must contain at least geometry_batch_size images.")
        self._images, self._indices = self._tensor_storage(dataset)
        self.sampling_device = self._images.device if self._images is not None else torch.device("cpu")
        self.generator = torch.Generator(device=self.sampling_device).manual_seed(seed)
        if self._indices is not None:
            self._indices = self._indices.to(self.sampling_device)

    @staticmethod
    def _tensor_storage(dataset):
        """Use existing tensor storage; don't cache or bypass arbitrary transforms."""
        indices = None
        while type(dataset) is Subset:
            subset_indices = torch.as_tensor(dataset.indices, dtype=torch.long)
            indices = subset_indices if indices is None else subset_indices[indices]
            dataset = dataset.dataset
        if type(dataset) is TensorDataset:
            if indices is not None:
                # Subset permits ordinary negative tensor indices; index_select
                # requires their nonnegative equivalents. Invalid indices still fail.
                indices = torch.where(indices < 0, indices + len(dataset), indices)
            return dataset.tensors[0], indices
        return None, None

    def sample(self, device):
        indices = torch.randperm(len(self.dataset), generator=self.generator,
                                 device=self.sampling_device)[:self.batch_size]
        if self._images is not None:
            if self._indices is not None:
                indices = self._indices[indices]
            return self._images.index_select(0, indices.to(self._images.device)).to(device)
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


def input_gram_energy(input_gram, energy_floor=1e-12):
    energy = input_gram.square().sum()
    # Finite positive squared energy also implies every Gram entry is finite.
    # Combine predicates before the single host read (important on accelerators).
    if not (torch.isfinite(energy) & (energy > energy_floor)):
        raise ValueError("Input Gram energy is nonfinite or near zero; reject this geometry batch.")
    return energy


def _distortion(input_gram, energy, output_images):
    output_gram = centered_gram(output_images)
    if output_gram.shape != input_gram.shape:
        raise ValueError("Input and output geometry must use the same batch members.")
    return (output_gram - input_gram).square().sum() / energy


def geometry_distortion(input_gram, output_images, energy_floor=1e-12):
    """||G(output)-G(input)||_F^2 / ||G(input)||_F^2.

    Reject degenerate input geometry rather than changing its denominator.
    There is no output trace, dimension, batch-size or response normalization.
    """
    energy = input_gram_energy(input_gram, energy_floor)
    distortion = _distortion(input_gram, energy, output_images)
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
                 dual_learning_rate=0.01, view="full", energy_floor=1e-12,
                 calibration=None):
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

        if calibration is None:
            distortions = []
            with torch.no_grad():
                for _ in range(calibration_batches):
                    images = source_batches.sample(self.device)
                    codes, _ = source.encoder(images)
                    outputs = image_mean(source.decoder, codes, self.distribution)
                    distortions.append(geometry_distortion(centered_gram(images), outputs, self.energy_floor))
            # Keep results on their device during calibration, then transfer once.
            # Float64 CPU reduction preserves the original tolerance arithmetic.
            values = torch.stack(distortions).cpu().double()
            calibration = {
                "source_tolerance": values.mean().item(),
                "standard_error": (values.std(unbiased=True) / math.sqrt(calibration_batches)).item(),
                "batch_distortion_std": values.std(unbiased=True).item(),
                "min": values.min().item(),
                "max": values.max().item(),
                "calibration_batches": calibration_batches,
                "geometry_batch_size": source_batches.batch_size,
                "sampler": "independent uniform batches without replacement within each batch",
            }
        if (calibration["calibration_batches"] != calibration_batches
                or calibration["geometry_batch_size"] != source_batches.batch_size
                or any(not math.isfinite(calibration[key]) or calibration[key] < 0
                       for key in ("source_tolerance", "standard_error", "batch_distortion_std", "min", "max"))):
            raise ValueError("Invalid or incompatible cached source calibration.")
        self.calibration = dict(calibration)
        self.source_tolerance = self.calibration["source_tolerance"]
        if self.source_tolerance >= 1:
            raise ValueError(f"Uninformative source tolerance c_s={self.source_tolerance:.6g} >= 1; collapse would satisfy it.")
        if self.source_tolerance + 2 * self.calibration["standard_error"] >= 1 or self.source_tolerance >= 0.9:
            warnings.warn("Source tolerance is close to collapse distortion 1; inspect calibration before sweeping.", stacklevel=2)

    def distortions(self, images, input_geometry=None, check_finite=True):
        # Deterministic posterior mean, with gradients to the adapted encoder.
        # Restore every module's mode even if the geometry check raises.
        modes = [(module, module.training) for module in self.model.modules()]
        self.model.eval()
        try:
            codes, _ = self.model.encoder(images)
            if input_geometry is None:
                gram = centered_gram(images.detach())
                energy = input_gram_energy(gram, self.energy_floor)
            else:
                # The trainer caches only validated, detached REAL input geometry.
                # Model outputs/codes must always be recomputed with current weights.
                gram, energy = input_geometry
            result = {}
            if self.view != "source_only":
                outputs = image_mean(self.model.decoder, codes, self.distribution)
                result["target_distortion"] = _distortion(gram, energy, outputs)
            if self.view != "target_only":
                # D0 parameters are frozen; deliberately DO NOT use no_grad here.
                outputs = image_mean(self.source_decoder, codes, self.distribution)
                result["source_chart_distortion"] = _distortion(gram, energy, outputs)
            stacked = torch.stack(list(result.values()))
            if check_finite and not torch.isfinite(stacked).all():
                raise ValueError("Nonfinite target geometry distortion.")
            result["worst_distortion"] = stacked.max()
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
        value = worst_distortion.detach() if isinstance(worst_distortion, torch.Tensor) else worst_distortion
        violation = float(value) - self.source_tolerance
        self.dual_fraction = min(1.0, max(0.0, self.dual_fraction + self.dual_learning_rate * violation))

    def penalty(self, mean_worst_distortion):
        return max(0.0, float(mean_worst_distortion) - self.source_tolerance)

    def state_dict(self):
        return {"dual_fraction": self.dual_fraction, "eta": self.eta,
                "dual_learning_rate": self.dual_learning_rate, "view": self.view,
                "energy_floor": self.energy_floor, "calibration": self.calibration}
