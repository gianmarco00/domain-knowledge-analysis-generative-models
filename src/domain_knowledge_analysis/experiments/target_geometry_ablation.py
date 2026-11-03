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
            for view in ("full", "target_only"):
                add(seed, float(eta), batch_size, view, float(rate))
        # Pair explicit full/target-only controls too, including off-grid ones.
        # Existing matching controls deduplicate by their unchanged training ID.
        for control in ablation_config.get("controls", []):
            views = ("full", "target_only") if control["view"] in {"full", "target_only"} else (control["view"],)
            for view in views:
                add(seed, float(control["eta"]), control["geometry_batch_size"], view,
                    float(control.get("dual_learning_rate", base_regularizer.get("dual_learning_rate", 0.01))))
    return list(runs.values())


def select_target_geometry_models(results):
    """Separate reconstruction and generation selections; lower errors are better."""
    from domain_knowledge_analysis.plotting.target_geometry_plotter import TargetGeometryPlotter
    from domain_knowledge_analysis.plotting.ablation_plotter import AblationPlotter

    baselines = AblationPlotter.ordinary_by_seed([row for row in results if not float(row["eta"])])
    candidates = {view: [row for row in results if row["view"] == view and float(row["eta"]) > 0]
                  for view in ("full", "target_only")}
    if any(not rows for rows in candidates.values()):
        raise ValueError("Model selection requires evaluated full and target-only models.")
    for rows in candidates.values():
        for row in rows:
            if row["seed"] not in baselines:
                raise ValueError("Every selected model requires its own ordinary LoRA baseline.")
            for metric in ("validation_adapted_target_mse", "test_adapted_target_mse", "test_target_fid"):
                if not math.isfinite(float(row[metric])) or float(row[metric]) < 0:
                    raise ValueError(f"Model selection requires finite, nonnegative {metric}.")

    def fid_key(row):
        return (float(row["test_target_fid"]), float(row["test_adapted_target_mse"]), str(row["model_id"]))

    def balanced_score(row):
        baseline = baselines[row["seed"]]
        return sum(TargetGeometryPlotter.metric_value(row, baseline, metric)
                   for metric in ("test_adapted_target_mse", "test_target_fid")) / 2

    target = min(candidates["target_only"], key=lambda row: (-balanced_score(row), *fid_key(row)))
    selections = {
        "full_reconstruction": min(candidates["full"], key=lambda row: (float(row["validation_adapted_target_mse"]), *fid_key(row))),
        "target_only_reconstruction": target,
        "full_generation": min(candidates["full"], key=fid_key),
        "target_only_generation": min(candidates["target_only"], key=fid_key),
    }
    return selections, balanced_score(target)


class TargetGeometryAblation(Ablation):
    def __init__(self, config_path, plots_only=False):
        self.config = utils.load_config(config_path)
        self.repo_root = utils.get_repo_root()
        self.base_config = utils.load_config(self.repo_root / self.config["base_config"])
        self.ablation_config = self.config["ablation"]
        self.output_dir = self.repo_root / self.ablation_config["output_dir"]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Plot-only runs consume saved scores/banks, without generating configs,
        # touching model directories, constructing FID, or loading a model.
        self.runs = ([] if plots_only else
                     generate_target_geometry_runs(self.base_config, self.ablation_config, self.output_dir))
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

        return self.render_results(results)

    def replot_saved_results(self):
        results = list(self.load_saved_results(self.output_dir / "results" / "metrics.csv").values())
        if not results:
            raise ValueError("No saved target-geometry metrics to plot. Run the ablation first.")
        return self.render_results(results, cached_only=True)

    def render_results(self, results, cached_only=False):
        selections, balanced_score = select_target_geometry_models(results)
        save_yaml({"selection": "minimum validation target reconstruction MSE among full-view runs",
                   "selected_model": selections["full_reconstruction"]}, self.output_dir / "results" / "selected_model.yaml")
        save_yaml({"selection_rules": {
            "full_reconstruction": "minimum validation target MSE; ties prefer lower test FID",
            "target_only_reconstruction": "maximum equal-weight mean of same-seed LoRA-relative test target MSE and FID improvements; ties prefer lower FID",
            "full_generation": "minimum test target FID among full models; ties prefer lower test target MSE",
            "target_only_generation": "minimum test target FID among target-only models; ties prefer lower test target MSE",
        }, "target_only_balanced_improvement_percent": balanced_score, "models": selections},
            self.output_dir / "results" / "selected_models.yaml")
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
        from domain_knowledge_analysis.plotting.target_geometry_plotter import TargetGeometryPlotter
        plotter = TargetGeometryPlotter(self.output_dir, self.ablation_config)
        plotter.plot_metric_pairs(results, selected_target_only=selections["target_only_reconstruction"])
        plotter.plot_target_only_metric_pairs(results)
        self.plot_selected_models(selections, results, cached_only=cached_only)
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

    @torch.no_grad()
    def create_qualitative_results(self, selected, generation_selected=None, common_bank=None):
        """Separate model choices, shared real inputs/prior draws, no training."""
        generation_selected = generation_selected or selected
        with ExitStack() as cleanup:
            loaded = {}

            def load(run):
                if run["model_id"] not in loaded:
                    experiment = self.load_or_train(run)
                    loaded[run["model_id"]] = experiment
                    cleanup.callback(experiment.logger.close)
                return loaded[run["model_id"]]

            def run_for(selection):
                return next(run for run in self.runs if run["model_id"] == selection["model_id"])

            def baseline_for(selection):
                return next(run for run in self.runs if run["seed"] == selection["seed"] and not run["eta"])

            ordinary = load(baseline_for(selected))
            count = positive_integer(self.ablation_config.get("num_qualitative_images", 6), "num_qualitative_images")
            seed = int(self.ablation_config.get("visual_seed", 123))
            latent_count = positive_integer(self.ablation_config.get("num_generation_images", count), "num_generation_images")
            if common_bank is None:
                source_loader = utils.create_testing_dataloaders(ordinary.config)
                target_loader = utils.create_testing_dataloaders(ordinary.config, ordinary.config["lora"]["transformed_dataset"])

                def sample_images(dataset):
                    if len(dataset) < count:
                        raise ValueError("Not enough held-out images for num_qualitative_images.")
                    indices = torch.randperm(len(dataset), generator=utils.create_random_generator(seed))[:count]
                    return torch.stack([dataset[int(index)][0] for index in indices])

                source_images = sample_images(source_loader.dataset)
                target_images = sample_images(target_loader.dataset)
                latents = utils.sample_random_latents(latent_count, ordinary.model.encoder_params["latent_dim"], utils.create_random_generator(seed))
            else:
                # The two modes use exactly the same image and latent banks,
                # even when their winning models have different training seeds.
                source_images = common_bank["source"]["input"]
                target_images = common_bank["target"]["input"]
                latents = common_bank["latents"]

            result = {}
            for split, images in (("source", source_images), ("target", target_images), ("generation", latents)):
                selection = generation_selected if split == "generation" else selected
                baseline = load(baseline_for(selection))
                proposed = load(run_for(selection))

                def outputs(experiment):
                    if split == "generation":
                        return experiment.model.generate_images(latent_count, images).detach().cpu()
                    return experiment.model.reconstruct_images(images).detach().cpu()

                baseline.lora_manager.disable_adapters()
                try:
                    base = outputs(baseline)
                finally:
                    baseline.lora_manager.enable_adapters()
                adapted = outputs(baseline)
                proposed.lora_manager.enable_adapters()
                values = {"base": base, "lora": adapted, "regularized": outputs(proposed)}
                if split != "generation":
                    values["input"] = images.cpu()
                result[split] = values
            return {**result, "latents": latents.cpu(), "selected_model": dict(selected),
                    "generation_selected_model": dict(generation_selected), "visual_seed": seed,
                    "view": selected.get("view", "full")}

    def plot_selected_models(self, selections, results, cached_only=False):
        baselines = {row["seed"]: row for row in results if not float(row["eta"])}
        common_bank = None
        paths = []
        for view in ("full", "target_only"):
            reconstruction = selections[f"{view}_reconstruction"]
            generation = selections[f"{view}_generation"]
            baseline_models = {"reconstruction": baselines[reconstruction["seed"]],
                               "generation": baselines[generation["seed"]]}
            paths.extend(self.plot_reconstructions(reconstruction, generation, view=view, common_bank=common_bank,
                                                  cached_only=cached_only, baseline_models=baseline_models))
            if view == "full":
                common_bank = torch.load(self.output_dir / "results" / "qualitative_examples_full.pt",
                                         map_location="cpu", weights_only=True)
        return paths

    def plot_reconstructions(self, selected, generation_selected=None, view="full", common_bank=None,
                             cached_only=False, baseline_models=None):
        from domain_knowledge_analysis.plotting.target_geometry_plotter import TargetGeometryPlotter
        generation_selected = generation_selected or selected
        path = self.output_dir / "results" / f"qualitative_examples_{view}.pt"
        results = torch.load(path, map_location="cpu", weights_only=True) if path.exists() else None

        def same_model(stored, expected):
            return all(stored.get(key) == expected.get(key) for key in ("model_id", "checkpoint_hash"))

        def matches(bank):
            if bank is None or not same_model(bank["selected_model"], selected):
                return False
            if not same_model(bank.get("generation_selected_model", {}), generation_selected):
                return False
            count = self.ablation_config.get("num_qualitative_images", 6)
            if (bank.get("visual_seed") != self.ablation_config.get("visual_seed", 123)
                    or len(bank["source"]["input"]) != count or len(bank["target"]["input"]) != count
                    or len(bank["latents"]) != self.ablation_config.get("num_generation_images", count)):
                return False
            if baseline_models is not None and any(
                    not same_model(bank.get("baseline_models", {}).get(role, {}), model)
                    for role, model in baseline_models.items()):
                return False
            if common_bank is not None and (any(not torch.equal(bank[split]["input"], common_bank[split]["input"])
                                               for split in ("source", "target"))
                                            or not torch.equal(bank["latents"], common_bank["latents"])):
                return False
            return True

        if not matches(results):
            if cached_only:
                raise ValueError(f"Saved {view} qualitative bank is missing or stale for the selected reconstruction/FID models. "
                                 "Rerun the ablation normally once to refresh it; completed models are reused.")
            results = self.create_qualitative_results(selected, generation_selected, common_bank=common_bank)
            results["baseline_models"] = baseline_models or {}
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(results, path)
        return TargetGeometryPlotter(self.output_dir, self.ablation_config).plot_qualitative(results, view=view)
