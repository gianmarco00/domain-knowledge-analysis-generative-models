"""Ablation plumbing for geometry parameters, independent of transform sweeps.

Reuse the existing training workers and reconstruction/FID scorers. Existing
transformation ablations, their selection rules and their figures are unchanged.
"""

from copy import deepcopy
from contextlib import ExitStack
from hashlib import sha256
from itertools import product
from pathlib import Path
import math

import torch

from domain_knowledge_analysis import utils
from domain_knowledge_analysis.regularizers.target_geometry import TargetGeometryRegularizer, positive_integer
from .ablation import Ablation
from .ablation_utils import model_training_completed, save_rows, save_yaml, training_config_id


def generate_target_geometry_runs(base_config, ablation_config, output_dir):
    base_regularizer = base_config["lora"]["regularizer"]
    if not base_regularizer or base_regularizer.get("mode") != "target_geometry":
        raise ValueError("Target geometry ablations require a target_geometry base config.")
    forbidden = {"m_values", "s_values", "semantic_transformations", "local_sweep", "extra_runs"} & ablation_config.keys()
    if forbidden:
        raise ValueError(f"Use geometry settings rather than transformation sweep settings: {sorted(forbidden)}.")
    for name in ("seeds", "eta_values", "geometry_batch_sizes"):
        if not isinstance(ablation_config.get(name), list) or not ablation_config[name]:
            raise ValueError(f"{name} must be a nonempty list.")
    runs = {}

    def add(seed, eta, batch_size=None, view="full", dual_learning_rate=None):
        if not math.isfinite(eta) or eta < 0:
            raise ValueError("Ablation eta must be finite and nonnegative.")
        config = deepcopy(base_config)
        config["seed"] = positive_integer(seed, "seed", 0)
        config["lora"]["pretrained_model"] = None
        regularizer = config["lora"]["regularizer"]
        if eta == 0:
            config["lora"]["regularizer"] = None
            view, batch_size, dual_learning_rate = "ordinary_lora", 0, 0.0
        else:
            positive_integer(batch_size, "geometry_batch_size", 2)
            if view not in TargetGeometryRegularizer.VALID_VIEWS:
                raise ValueError(f"Unknown geometry view: {view}.")
            if not math.isfinite(dual_learning_rate) or dual_learning_rate <= 0:
                raise ValueError("dual_learning_rate must be finite and positive.")
            regularizer.update(eta=float(eta), geometry_batch_size=batch_size, view=view,
                               dual_learning_rate=float(dual_learning_rate))
        model_id = training_config_id(config)
        if model_id in runs:
            return
        name = f"target_geometry_{view}_seed_{seed}_eta_{eta:g}_B_{batch_size}_{model_id}"
        model_dir = Path(output_dir) / "models" / name
        config["paths"]["run_dir"] = str(model_dir)
        config_path = model_dir / "config.yaml"
        if not (model_training_completed(model_dir) and config_path.exists()):
            save_yaml(config, config_path)
        runs[model_id] = {
            "model_id": model_id, "model_dir": model_dir, "config_path": config_path,
            "seed": seed, "eta": float(eta), "mode": "target_geometry" if eta else "ordinary_lora",
            "view": view, "geometry_batch_size": batch_size,
            "dual_learning_rate": float(dual_learning_rate),
        }

    rates = ablation_config.get("dual_learning_rates", [base_regularizer.get("dual_learning_rate", 0.01)])
    if not isinstance(rates, list) or not rates:
        raise ValueError("dual_learning_rates must be a nonempty list.")
    for seed in ablation_config["seeds"]:
        add(seed, 0.0)
        for eta, batch_size, rate in product(ablation_config["eta_values"], ablation_config["geometry_batch_sizes"], rates):
            add(seed, float(eta), batch_size, "full", float(rate))
        # Matched-budget one-view controls, with the same source tolerance.
        for control in ablation_config.get("controls", []):
            add(seed, float(control["eta"]), control["geometry_batch_size"], control["view"],
                float(control.get("dual_learning_rate", base_regularizer.get("dual_learning_rate", 0.01))))
    return list(runs.values())


class TargetGeometryAblation(Ablation):
    def __init__(self, config_path):
        self.config = utils.load_config(config_path)
        self.repo_root = utils.get_repo_root()
        self.base_config = utils.load_config(self.repo_root / self.config["base_config"])
        self.ablation_config = self.config["ablation"]
        self.output_dir = self.repo_root / self.ablation_config["output_dir"]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.runs = generate_target_geometry_runs(self.base_config, self.ablation_config, self.output_dir)
        self.fid = None
        self.fid_reference_key = None

    def run(self):
        print(f"Target geometry ablation contains {len(self.runs)} unique models.")
        self.train_models()
        metrics_path = self.output_dir / "results" / "metrics.csv"
        saved = self.load_saved_results(metrics_path)
        results = []
        for index, run in enumerate(self.runs):
            print(f"Scoring {index + 1}/{len(self.runs)}: {run['model_dir'].name}")
            config = utils.load_config(run["config_path"])
            target_loader = utils.create_testing_dataloaders(config, config["lora"]["transformed_dataset"])
            reference_key, fid_key = self.prepare_fid(target_loader)
            checkpoint = run["model_dir"] / "checkpoints" / "best.pt"
            checkpoint_hash = sha256(checkpoint.read_bytes()).hexdigest()
            cached = saved.get(run["model_id"], {})
            if cached.get("checkpoint_hash") == checkpoint_hash and cached.get("fid_key") == fid_key:
                results.append(cached)
                continue
            experiment = self.load_or_train(run)
            try:
                metrics = self.evaluate(experiment, target_loader, reference_key)
            finally:
                experiment.logger.close()
            row = {**{key: str(value) if isinstance(value, Path) else value for key, value in run.items()},
                   **metrics, "checkpoint_hash": checkpoint_hash, "fid_key": fid_key}
            state_path = checkpoint.with_suffix(".regularizer.pt")
            if run["eta"] and state_path.exists():
                state = torch.load(state_path, map_location="cpu", weights_only=True)
                row["source_tolerance"] = state["calibration"]["source_tolerance"]
                row["calibration_standard_error"] = state["calibration"]["standard_error"]
                row["dual_multiplier"] = state["eta"] * state["dual_fraction"]
            results.append(row)
            saved[run["model_id"]] = row
            save_rows(list(saved.values()), metrics_path)

        regularized = [row for row in results if row["eta"] and row["view"] == "full"]
        selected = min(regularized, key=lambda row: row["validation_adapted_target_mse"])
        save_yaml({"selection": "minimum validation target reconstruction MSE among full-view runs",
                   "selected_model": selected}, self.output_dir / "results" / "selected_model.yaml")
        # Also report across-seed means; individual seed selection is only for
        # qualitative inspection, not evidence for a winning hyperparameter.
        groups = {}
        for row in results:
            key = (row["view"], row["eta"], row["geometry_batch_size"], row["dual_learning_rate"])
            groups.setdefault(key, []).append(row)
        aggregates = []
        for (view, eta, batch_size, rate), rows in groups.items():
            aggregate = {"view": view, "eta": eta, "geometry_batch_size": batch_size,
                         "dual_learning_rate": rate, "seeds": len(rows)}
            for metric in ("validation_adapted_target_mse", "test_adapted_target_mse", "test_target_gain", "test_source_degradation", "test_target_fid"):
                values = torch.tensor([row[metric] for row in rows], dtype=torch.float64)
                aggregate[metric + "_mean"] = values.mean().item()
                aggregate[metric + "_std"] = values.std(unbiased=True).item() if len(rows) > 1 else 0.0
            aggregates.append(aggregate)
        save_rows(aggregates, self.output_dir / "results" / "summary.csv")
        self.plot_geometry_results(aggregates)
        from domain_knowledge_analysis.plotting.target_geometry_plotter import TargetGeometryPlotter
        TargetGeometryPlotter(self.output_dir, self.ablation_config).plot_metric_pairs(results)
        self.plot_reconstructions(selected)
        print(f"Results saved in {self.output_dir}")
        return results

    @staticmethod
    def load_saved_results(metrics_path):
        # Geometry metadata includes a text view; the transformation loader
        # intentionally knows nothing about these new columns.
        import csv
        if not metrics_path.exists():
            return {}
        with open(metrics_path, newline="") as file:
            rows = list(csv.DictReader(file))
        strings = {"model_id", "model_dir", "config_path", "mode", "view", "checkpoint_hash", "fid_key"}
        for row in rows:
            for key, value in row.items():
                if key not in strings and value != "":
                    row[key] = int(value) if key in {"seed", "geometry_batch_size", "fid_seed"} else float(value)
        return {row["model_id"]: row for row in rows}

    def plot_geometry_results(self, rows):
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        metrics = [("test_adapted_target_mse", "Target reconstruction MSE"),
                   ("test_source_degradation", "Source MSE increase"), ("test_target_fid", "Target FID")]
        for axis, (metric, title) in zip(axes, metrics):
            baseline = next(row for row in rows if row["view"] == "ordinary_lora")
            axis.axhline(baseline[metric + "_mean"], color="black", linestyle="--", label="Ordinary LoRA")
            groups = sorted({(r["view"], r["geometry_batch_size"], r["dual_learning_rate"]) for r in rows if r["eta"]})
            for view, batch_size, rate in groups:
                series = sorted((r for r in rows if (r["view"], r["geometry_batch_size"], r["dual_learning_rate"]) == (view, batch_size, rate)), key=lambda r: r["eta"])
                axis.errorbar([r["eta"] for r in series], [r[metric + "_mean"] for r in series],
                              yerr=[r[metric + "_std"] for r in series], marker="o", capsize=3,
                              label=f"{view}, B={batch_size}, dual step={rate:g}")
            axis.set(xscale="log", xlabel="eta", title=title)
        axes[0].legend(fontsize=7)
        figure.tight_layout()
        path = self.output_dir / "figures" / "target_geometry.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=180)
        plt.close(figure)

    def create_qualitative_results(self, selected):
        """Use fixed held-out inputs and common prior draws across all models."""
        regularized_run = next(r for r in self.runs if r["model_id"] == selected["model_id"])
        ordinary_run = next(r for r in self.runs if r["seed"] == selected["seed"] and not r["eta"])
        with ExitStack() as cleanup:
            ordinary = self.load_or_train(ordinary_run)
            cleanup.callback(ordinary.logger.close)
            regularized = self.load_or_train(regularized_run)
            cleanup.callback(regularized.logger.close)
            source_loader = utils.create_testing_dataloaders(ordinary.config)
            target_loader = utils.create_testing_dataloaders(ordinary.config, ordinary.config["lora"]["transformed_dataset"])
            count = positive_integer(self.ablation_config.get("num_qualitative_images", 6), "num_qualitative_images")
            seed = int(self.ablation_config.get("visual_seed", 123))

            def sample_images(dataset):
                if len(dataset) < count:
                    raise ValueError("Not enough held-out images for num_qualitative_images.")
                # Matching source/transported datasets use matching indices.
                indices = torch.randperm(len(dataset), generator=utils.create_random_generator(seed))[:count]
                return torch.stack([dataset[int(index)][0] for index in indices])

            source_images = sample_images(source_loader.dataset)
            target_images = sample_images(target_loader.dataset)
            latent_count = positive_integer(self.ablation_config.get("num_generation_images", count), "num_generation_images")
            latents = utils.sample_random_latents(latent_count, ordinary.model.encoder_params["latent_dim"], utils.create_random_generator(seed))

            def model_outputs(experiment):
                return {
                    "source": experiment.model.reconstruct_images(source_images).detach().cpu(),
                    "target": experiment.model.reconstruct_images(target_images).detach().cpu(),
                    "generation": experiment.model.generate_images(latent_count, latents).detach().cpu(),
                }

            ordinary.lora_manager.disable_adapters()
            base = model_outputs(ordinary)
            ordinary.lora_manager.enable_adapters()
            adapted = model_outputs(ordinary)
            regularized.lora_manager.enable_adapters()
            proposed = model_outputs(regularized)
            return {
                "source": {"input": source_images.cpu(), "base": base["source"], "lora": adapted["source"], "regularized": proposed["source"]},
                "target": {"input": target_images.cpu(), "base": base["target"], "lora": adapted["target"], "regularized": proposed["target"]},
                "generation": {"base": base["generation"], "lora": adapted["generation"], "regularized": proposed["generation"]},
                "latents": latents.cpu(), "selected_model": dict(selected), "visual_seed": seed,
            }

    def plot_reconstructions(self, selected):
        from domain_knowledge_analysis.plotting.target_geometry_plotter import TargetGeometryPlotter
        results = self.create_qualitative_results(selected)
        # Preserve the actual displayed samples and prior vectors for review.
        path = self.output_dir / "results" / "qualitative_examples.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(results, path)
        return TargetGeometryPlotter(self.output_dir, self.ablation_config).plot_qualitative(results)
