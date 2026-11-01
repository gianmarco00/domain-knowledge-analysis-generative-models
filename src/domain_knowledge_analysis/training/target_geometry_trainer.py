"""Opt-in target-geometry training; the existing Trainer is unchanged."""

from collections import defaultdict

import torch

from .trainer import Trainer
from domain_knowledge_analysis.regularizers.target_geometry import GeometryBatchSampler, positive_integer


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

    def train_epoch(self):
        self.model.train()
        base_total, samples, steps = 0.0, 0, 0
        components, geometry = defaultdict(float), defaultdict(float)
        for batch in self.train_dataloader:
            images = batch[0].to(self.device)
            self.optimizer.zero_grad()
            logits, mean, log_variance = self.model(images)
            base_loss = self.loss(images, logits, mean, log_variance)
            if callable(getattr(self.loss, "components", None)):
                for name, value in self.loss.components().items():
                    components[name] += value.sum().item()

            # A separate, full-size random geometry batch leaves the ordinary
            # VAE minibatch size, remainder handling and KL objective untouched.
            values = self.regularizer.distortions(self.geometry_sampler.sample(self.device))
            loss = base_loss + self.regularizer.primal_loss(values["worst_distortion"])
            loss.backward()
            self.optimizer.step()
            self.regularizer.update_dual(values["worst_distortion"])

            base_total += base_loss.item() * len(images)
            samples += len(images)
            steps += 1
            for name, value in values.items():
                geometry[name] += value.detach().item()
        summary = self._geometry_summary(geometry, steps)
        components = {name: value / samples for name, value in components.items()}
        # Reporting only: this is a trajectory average, not a fixed-theta
        # population objective. Validation below evaluates fixed parameters.
        return base_total / samples + self.regularizer.eta * summary["regularizer"], {**components, **summary}

    @torch.no_grad()
    def validate_epoch(self):
        self.model.eval()
        total, samples = 0.0, 0
        components, geometry = defaultdict(float), defaultdict(float)
        for batch in self.validate_dataloader:
            images = batch[0].to(self.device)
            logits, mean, log_variance = self.model(images)
            total += self.loss(images, logits, mean, log_variance).item() * len(images)
            samples += len(images)
            if callable(getattr(self.loss, "components", None)):
                for name, value in self.loss.components().items():
                    components[name] += value.sum().item()
        # Reset only this independent evaluation generator for a fixed bank
        # across epochs. Training/dual state and train sampling are untouched.
        sampler = GeometryBatchSampler(self.validate_dataloader.dataset, self.geometry_batch_size, self.geometry_seed + 1)
        for _ in range(self.validation_batches):
            values = self.regularizer.distortions(sampler.sample(self.device))
            for name, value in values.items():
                geometry[name] += value.item()
        summary = self._geometry_summary(geometry, self.validation_batches)
        components = {name: value / samples for name, value in components.items()}
        # One hinge AFTER the average, independent of the current dual iterate.
        return total / samples + self.regularizer.eta * summary["regularizer"], {**components, **summary}
