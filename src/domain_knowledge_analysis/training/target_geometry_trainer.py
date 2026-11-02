"""Opt-in target-geometry training; the existing Trainer is unchanged."""

from collections import defaultdict
import math

import torch

from .trainer import Trainer
from domain_knowledge_analysis.regularizers.target_geometry import (
    GeometryBatchSampler, centered_gram, input_gram_energy, positive_integer,
)


class GeometryCheckpointManager:
    """Keep optimizer-only dual/calibration state beside each model checkpoint."""

    def __init__(self, manager, regularizer, sampler):
        self.manager = manager
        self.regularizer = regularizer
        self.sampler = sampler

    def __getattr__(self, name):
        return getattr(self.manager, name)

    def _save(self, method, filename, *args):
        getattr(self.manager, method)(*args)
        state = {**self.regularizer.state_dict(), "sampler_state": self.sampler.generator.get_state()}
        torch.save(state, self.manager.checkpoint_path(filename).with_suffix(".regularizer.pt"))

    def save_best(self, *args):
        self._save("save_best", "best.pt", *args)

    def save_last(self, *args):
        self._save("save_last", "last.pt", *args)

    def save_with_name(self, *args):
        self._save("save_with_name", args[-1], *args)


class TargetGeometryTrainer(Trainer):
    def __init__(self, *args, geometry_batch_size=64, validation_batches=64,
                 geometry_seed=0, **kwargs):
        super().__init__(*args, **kwargs)
        positive_integer(validation_batches, "validation_batches")
        self.geometry_batch_size = geometry_batch_size
        self.validation_batches = validation_batches
        self.geometry_seed = geometry_seed
        self._cuda = torch.device(self.device).type == "cuda"
        self._validation_geometry = None
        self.geometry_sampler = GeometryBatchSampler(self.train_dataloader.dataset, geometry_batch_size, geometry_seed)
        # Validate availability now; use the identical batch size in all splits.
        GeometryBatchSampler(self.validate_dataloader.dataset, geometry_batch_size, geometry_seed + 1)
        if self.checkpoint_manager is not None:
            self.checkpoint_manager = GeometryCheckpointManager(self.checkpoint_manager, self.regularizer, self.geometry_sampler)

    def _geometry_summary(self, totals, count):
        result = {name: value / count for name, value in totals.items()}
        result["source_tolerance"] = self.regularizer.source_tolerance
        result["constraint_violation"] = result["worst_distortion"] - self.regularizer.source_tolerance
        result["regularizer"] = self.regularizer.penalty(result["worst_distortion"])
        result["dual_multiplier"] = self.regularizer.multiplier
        return result

    @staticmethod
    def _host_values(values):
        """One device-to-host transfer, keeping Python-double epoch accumulation."""
        return dict(zip(values, torch.stack([value.detach() for value in values.values()]).cpu().tolist()))

    @staticmethod
    def _active_geometry(values, host_values):
        """Avoid backward through the strictly losing view of max(F,H).

        At exact ties keep Tensor.max's original equal-share subgradient.
        Detach only the diagnostic losing view, releasing its decoder graph.
        The VAE loss still reaches every adapted encoder/decoder parameter,
        including ordinary zero-gradient/momentum updates on those parameters.
        """
        if "target_distortion" in values and "source_chart_distortion" in values:
            target = host_values["target_distortion"]
            source = host_values["source_chart_distortion"]
            if target != source:
                winner, loser = (("target_distortion", "source_chart_distortion") if target > source
                                 else ("source_chart_distortion", "target_distortion"))
                values["worst_distortion"] = values[winner]
                values[loser] = values[loser].detach()
        return values["worst_distortion"]

    def train_epoch(self):
        self.model.train()
        base_total, samples, steps = 0.0, 0, 0
        components, geometry = defaultdict(float), defaultdict(float)
        for batch in self.train_dataloader:
            images = batch[0].to(self.device)
            self.optimizer.zero_grad()
            logits, mean, log_variance = self.model(images)
            base_loss = self.loss(images, logits, mean, log_variance)
            monitors = {"vae_loss": base_loss}
            if callable(getattr(self.loss, "components", None)):
                for name, value in self.loss.components().items():
                    monitors[name] = value.sum()

            # A separate, full-size random geometry batch leaves the ordinary
            # VAE minibatch size, remainder handling and KL objective untouched.
            # Still evaluate both views and update the dual when lambda=0, but
            # an identically zero coefficient needs no geometry autograd graph.
            with torch.set_grad_enabled(self.regularizer.multiplier != 0):
                values = self.regularizer.distortions(self.geometry_sampler.sample(self.device),
                                                      check_finite=not self._cuda)
            host = self._host_values({**monitors, **values})
            if self._cuda and any(not math.isfinite(host[name]) for name in values):
                # Use the already-required scalar read for dual ascent/active-view
                # selection; reject invalid outputs before backward or Adam updates.
                raise ValueError("Nonfinite target geometry distortion.")
            worst = self._active_geometry(values, host)
            loss = base_loss + self.regularizer.primal_loss(worst)
            loss.backward()
            self.optimizer.step()
            self.regularizer.update_dual(host["worst_distortion"])

            base_total += host["vae_loss"] * len(images)
            for name in monitors:
                if name != "vae_loss":
                    components[name] += host[name]
            samples += len(images)
            steps += 1
            for name in values:
                geometry[name] += host[name]
        summary = self._geometry_summary(geometry, steps)
        components = {name: value / samples for name, value in components.items()}
        # Reporting only: this is a trajectory average, not a fixed-theta
        # population objective. Validation below evaluates fixed parameters.
        return base_total / samples + self.regularizer.eta * summary["regularizer"], {**components, **summary}

    @torch.no_grad()
    def validate_epoch(self):
        if self._cuda:
            return self._validate_cuda()
        self.model.eval()
        total, samples = 0.0, 0
        components, geometry = defaultdict(float), defaultdict(float)
        for batch in self.validate_dataloader:
            images = batch[0].to(self.device)
            logits, mean, log_variance = self.model(images)
            monitors = {"vae_loss": self.loss(images, logits, mean, log_variance)}
            samples += len(images)
            if callable(getattr(self.loss, "components", None)):
                for name, value in self.loss.components().items():
                    monitors[name] = value.sum()
            host = self._host_values(monitors)
            total += host["vae_loss"] * len(images)
            for name in monitors:
                if name != "vae_loss":
                    components[name] += host[name]
        # Reset only this independent evaluation generator for a fixed bank
        # across epochs. Training/dual state and train sampling are untouched.
        sampler = GeometryBatchSampler(self.validate_dataloader.dataset, self.geometry_batch_size, self.geometry_seed + 1)
        for _ in range(self.validation_batches):
            values = self.regularizer.distortions(sampler.sample(self.device))
            for name, value in self._host_values(values).items():
                geometry[name] += value
        summary = self._geometry_summary(geometry, self.validation_batches)
        components = {name: value / samples for name, value in components.items()}
        # One hinge AFTER the average, independent of the current dual iterate.
        return total / samples + self.regularizer.eta * summary["regularizer"], {**components, **summary}

    @torch.no_grad()
    def _validate_cuda(self):
        """Accumulate validation metrics on device and reuse fixed input geometry."""
        self.model.eval()
        totals, geometry, samples = {}, {}, 0

        def accumulate(destination, name, value, weight=1):
            # Match Python-double accumulation; do not change model precision.
            value = value.detach().double() * weight
            if name in destination:
                destination[name].add_(value)
            else:
                destination[name] = value

        for batch in self.validate_dataloader:
            images = batch[0].to(self.device)
            logits, mean, log_variance = self.model(images)
            accumulate(totals, "vae_loss", self.loss(images, logits, mean, log_variance), len(images))
            samples += len(images)
            if callable(getattr(self.loss, "components", None)):
                for name, value in self.loss.components().items():
                    accumulate(totals, name, value.sum())

        if self._validation_geometry is None:
            sampler = GeometryBatchSampler(self.validate_dataloader.dataset, self.geometry_batch_size,
                                           self.geometry_seed + 1)
            # Do not cache arbitrary/stochastic dataset transforms.
            if sampler._images is not None:
                bank = []
                for _ in range(self.validation_batches):
                    images = sampler.sample(self.device)
                    gram = centered_gram(images)
                    energy = input_gram_energy(gram, self.regularizer.energy_floor)
                    bank.append((images, (gram, energy)))
                self._validation_geometry = bank
        if self._validation_geometry is None:
            sampler = GeometryBatchSampler(self.validate_dataloader.dataset, self.geometry_batch_size,
                                           self.geometry_seed + 1)
            batches = ((sampler.sample(self.device), None) for _ in range(self.validation_batches))
        else:
            batches = self._validation_geometry
        for images, input_geometry in batches:
            values = self.regularizer.distortions(images, input_geometry=input_geometry, check_finite=False)
            for name, value in values.items():
                accumulate(geometry, name, value)
        host = self._host_values({**totals, **geometry})
        if any(not math.isfinite(host[name]) for name in geometry):
            raise ValueError("Nonfinite target geometry distortion.")
        summary = self._geometry_summary({name: host[name] for name in geometry}, self.validation_batches)
        components = {name: host[name] / samples for name in totals if name != "vae_loss"}
        return host["vae_loss"] / samples + self.regularizer.eta * summary["regularizer"], {**components, **summary}
