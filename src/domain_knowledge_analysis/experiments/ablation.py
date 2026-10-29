from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import csv
import torch

from domain_knowledge_analysis import utils
from domain_knowledge_analysis.experiments.experiment import Experiment
from domain_knowledge_analysis.experiments.ablation_utils import generate_model_runs, save_rows, save_yaml, select_best_models
from domain_knowledge_analysis.plotting.ablation_plotter import AblationPlotter
from domain_knowledge_analysis.scoring import AdaptationScorer


def train_ablation_model(config_path):
    config_path = Path(config_path)
    checkpoint_path = config_path.parent / "checkpoints" / "best.pt"

    if checkpoint_path.exists():
        return str(checkpoint_path)

    print(f"Training: {config_path.parent.name}")
    experiment = Experiment(config_path)
    experiment.adapt()
    experiment.logger.close()

    return str(checkpoint_path)


class Ablation:
    def __init__(self, config_path):
        self.config = utils.load_config(config_path)
        self.repo_root = utils.get_repo_root()
        self.base_config_path = self.repo_root / Path(self.config["base_config"])
        self.base_config = utils.load_config(self.base_config_path)
        self.ablation_config = self.config["ablation"]
        self.output_dir = self.repo_root / Path(self.ablation_config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.runs = generate_model_runs(self.base_config, self.ablation_config, self.output_dir)

    def run(self):
        print(f"Ablation contains {len(self.runs)} unique models.")
        self.train_models()
        results = []
        metrics_path = self.output_dir / "results" / "metrics.csv"
        saved_results = self.load_saved_results(metrics_path)

        for row in saved_results.values():
            row.setdefault("test_target_fid", "")
            row.setdefault("fid_seed", "")

        for index, run in enumerate(self.runs):
            print(f"\nModel {index + 1}/{len(self.runs)}: {run['model_dir'].name}")

            if run["model_id"] in saved_results and saved_results[run["model_id"]]["test_target_fid"] != "" and saved_results[run["model_id"]]["fid_seed"] == int(self.ablation_config["fid_seed"]):
                print("Using saved metrics.")
                results.append(saved_results[run["model_id"]])
                continue

            experiment = self.load_or_train(run)
            if run["model_id"] in saved_results:
                print("Computing missing FID.")
                transformation_config = experiment.config["lora"]["transformed_dataset"]
                target_test_dataloader = utils.create_testing_dataloaders(experiment.config, transformation_config)
                row = saved_results[run["model_id"]]
                row["test_target_fid"] = self.score_fid(experiment, target_test_dataloader)
                row["fid_seed"] = int(self.ablation_config["fid_seed"])
            else:
                metrics = self.evaluate(experiment)

                row = {
                    "model_id": run["model_id"],
                    "model_dir": str(run["model_dir"]),
                    "config_path": str(run["config_path"]),
                    "seed": run["seed"],
                    "eta": run["eta"],
                    "mode": run["mode"],
                    "m": run["m"],
                    "s": run["s"],
                    "subset": run["subset"],
                    "extra_run": run["extra_run"],
                    **metrics,
                }

            results.append(row)
            saved_results[run["model_id"]] = row
            save_rows(list(saved_results.values()), metrics_path)
            experiment.logger.close()
            del experiment

        best_models, selection_summary = select_best_models(results, self.ablation_config["target_gain_fraction"])
        best_model_rows = [{"selection": selection, "available": model is not None, "required_target_gain": selection_summary["required_target_gain"], **(model or {})} for selection, model in best_models.items()]
        selected_model = best_models["best_overall"] or best_models["best_target_gain"]
        qualitative_selection = "best_overall" if best_models["best_overall"] is not None else "best_target_gain"
        save_rows(best_model_rows, self.output_dir / "results" / "best_models.csv")
        save_yaml({**selection_summary, "qualitative_selection": qualitative_selection, "models": {selection: dict(model) if model is not None else None for selection, model in best_models.items()}}, self.output_dir / "results" / "best_models.yaml")

        plotter = AblationPlotter(self.output_dir, self.ablation_config)
        plotter.plot_figure_1(results)
        plotter.plot_figure_2(results)
        plotter.plot_figure_3(results)
        plotter.plot_figure_4(results)
        plotter.plot_fid(results)
        plotter.plot_qualitative(self.create_qualitative_results(selected_model))

        print("\nBest regularized models:")

        for selection, model in best_models.items():
            if model is None:
                print(f"{selection}: unavailable because no model reaches {100 * selection_summary['target_gain_fraction']:g}% of ordinary LoRA target gain")
            else:
                print(f"{selection}: eta={float(model['eta']):g}, M={int(model['m'])}, s={float(model['s']):g}")

        if best_models["best_overall"] is None:
            print(f"Figures 5-8 use best_target_gain as a fallback ({100 * selection_summary['best_regularized_target_fraction']:.2f}% of ordinary LoRA target gain).")

        print(f"Results saved in: {self.output_dir}")

        return results

    @staticmethod
    def load_saved_results(metrics_path):
        if not metrics_path.exists():
            return {}

        with open(metrics_path, newline="") as file:
            rows = list(csv.DictReader(file))

        text_fields = {"model_id", "model_dir", "config_path", "mode", "subset"}

        for row in rows:
            for key, value in row.items():
                if key in {"seed", "m", "fid_seed"} and value != "":
                    row[key] = int(value)
                elif key == "extra_run":
                    row[key] = value == "True"
                elif key not in text_fields and value != "":
                    row[key] = float(value)

        return {row["model_id"]: row for row in rows}

    def train_models(self):
        runs_to_train = [run for run in self.runs if not (run["model_dir"] / "checkpoints" / "best.pt").exists()]

        if not runs_to_train:
            print("All models are already trained.")
            return

        self.prepare_dataset_caches(runs_to_train)
        config_paths = [str(run["config_path"]) for run in runs_to_train]
        parallel_models = min(int(self.ablation_config.get("parallel_models", 1)), len(config_paths))
        print(f"Training {len(config_paths)} models with {parallel_models} parallel process(es).")

        if parallel_models == 1:
            for config_path in config_paths:
                train_ablation_model(config_path)
            return

        with ProcessPoolExecutor(max_workers=parallel_models, mp_context=get_context("spawn")) as executor:
            list(executor.map(train_ablation_model, config_paths))

    def prepare_dataset_caches(self, runs):
        prepared_seeds = set()

        for run in runs:
            if run["seed"] in prepared_seeds:
                continue

            config = utils.load_config(run["config_path"])
            utils.create_training_dataloaders(config, config["lora"]["transformed_dataset"])
            prepared_seeds.add(run["seed"])

    def load_or_train(self, run):
        experiment = Experiment(run["config_path"])
        checkpoint_path = run["model_dir"] / "checkpoints" / "best.pt"

        if checkpoint_path.exists():
            print("Loading existing model.")
        else:
            print("Training model.")
            experiment.adapt()

        experiment.model = experiment.checkpoint_manager.load_model(experiment.model, checkpoint_path, experiment.device)
        experiment.lora_manager.enable_adapters()

        return experiment

    def evaluate(self, experiment):
        transformation_config = experiment.config["lora"]["transformed_dataset"]
        _, source_validation_dataloader = utils.create_training_dataloaders(experiment.config)
        _, target_validation_dataloader = utils.create_training_dataloaders(experiment.config, transformation_config)
        source_test_dataloader = utils.create_testing_dataloaders(experiment.config)
        target_test_dataloader = utils.create_testing_dataloaders(experiment.config, transformation_config)

        validation_metrics = self.score_split(experiment, target_validation_dataloader, source_validation_dataloader)
        test_metrics = self.score_split(experiment, target_test_dataloader, source_test_dataloader)
        return {
            **{f"validation_{name}": value for name, value in validation_metrics.items()},
            **{f"test_{name}": value for name, value in test_metrics.items()},
            "test_target_fid": self.score_fid(experiment, target_test_dataloader),
            "fid_seed": int(self.ablation_config["fid_seed"]),
        }

    def score_fid(self, experiment, target_dataloader):
        from torchmetrics.image.fid import FrechetInceptionDistance

        device = experiment.device if experiment.device.type == "cuda" else torch.device("cpu")
        fid = FrechetInceptionDistance(feature=2048, normalize=True).set_dtype(torch.float64).to(device)
        generator = utils.create_random_generator(self.ablation_config["fid_seed"])
        latent_dim = experiment.model.encoder_params["latent_dim"]
        experiment.lora_manager.enable_adapters()
        experiment.model.eval()

        with torch.no_grad():
            for images, _ in target_dataloader:
                latents = utils.sample_random_latents(len(images), latent_dim, generator)
                generated = experiment.model.generate_images(len(images), latents)
                fid.update(images.to(device).repeat(1, 3, 1, 1), real=True)
                fid.update(generated.to(device).clamp(0, 1).repeat(1, 3, 1, 1), real=False)

        return fid.compute().item()

    def score_split(self, experiment, target_dataloader, source_dataloader):
        scorer = AdaptationScorer(experiment.model, experiment.lora_manager, target_dataloader, source_dataloader, experiment.config, experiment.device)
        scores = scorer.score()

        base_target_mse = scores["pretrained_target_mse"]
        base_source_mse = scores["pretrained_source_mse"]
        adapted_target_mse = scores["adapted_target_mse"]
        adapted_source_mse = scores["adapted_source_mse"]

        return {
            "base_target_mse": base_target_mse,
            "base_source_mse": base_source_mse,
            "adapted_target_mse": adapted_target_mse,
            "adapted_source_mse": adapted_source_mse,
            "target_gain": base_target_mse - adapted_target_mse,
            "source_degradation": adapted_source_mse - base_source_mse,
        }

    def create_qualitative_results(self, selected_model):
        seed = int(selected_model["seed"])
        num_images = int(self.ablation_config["num_qualitative_images"])

        ordinary_run = next(run for run in self.runs if run["seed"] == seed and run["eta"] == 0)
        source_loss_run = next(run for run in self.runs if run["seed"] == seed and run["mode"] == "source_loss")
        regularized_run = next(run for run in self.runs if run["model_id"] == selected_model["model_id"])

        ordinary_experiment = self.load_or_train(ordinary_run)
        source_loss_experiment = self.load_or_train(source_loss_run)
        regularized_experiment = self.load_or_train(regularized_run)
        transformation_config = ordinary_experiment.config["lora"]["transformed_dataset"]
        source_dataloader = utils.create_testing_dataloaders(ordinary_experiment.config)
        target_dataloader = utils.create_testing_dataloaders(ordinary_experiment.config, transformation_config)
        source_images = next(iter(source_dataloader))[0][:num_images]
        target_images = next(iter(target_dataloader))[0][:num_images]

        latent_dim = ordinary_experiment.model.encoder_params["latent_dim"]
        latent_generator = utils.create_random_generator(self.ablation_config["visual_seed"])
        latents = utils.sample_random_latents(num_images, latent_dim, latent_generator)

        ordinary_experiment.lora_manager.disable_adapters()
        base_source = ordinary_experiment.model.reconstruct_images(source_images).cpu()
        base_target = ordinary_experiment.model.reconstruct_images(target_images).cpu()
        base_generated = ordinary_experiment.model.generate_images(num_images, latents).cpu()

        ordinary_experiment.lora_manager.enable_adapters()
        lora_source = ordinary_experiment.model.reconstruct_images(source_images).cpu()
        lora_target = ordinary_experiment.model.reconstruct_images(target_images).cpu()
        lora_generated = ordinary_experiment.model.generate_images(num_images, latents).cpu()

        source_loss_experiment.lora_manager.enable_adapters()
        source_loss_source = source_loss_experiment.model.reconstruct_images(source_images).cpu()
        source_loss_target = source_loss_experiment.model.reconstruct_images(target_images).cpu()
        source_loss_generated = source_loss_experiment.model.generate_images(num_images, latents).cpu()

        regularized_experiment.lora_manager.enable_adapters()
        regularized_source = regularized_experiment.model.reconstruct_images(source_images).cpu()
        regularized_target = regularized_experiment.model.reconstruct_images(target_images).cpu()
        regularized_generated = regularized_experiment.model.generate_images(num_images, latents).cpu()
        responses = self.create_model_responses(ordinary_experiment, regularized_experiment, source_images, selected_model)

        ordinary_experiment.logger.close()
        source_loss_experiment.logger.close()
        regularized_experiment.logger.close()

        return {
            "source": {"input": source_images, "base": base_source, "lora": lora_source, "source_loss": source_loss_source, "regularized": regularized_source},
            "target": {"input": target_images, "base": base_target, "lora": lora_target, "source_loss": source_loss_target, "regularized": regularized_target},
            "generation": {"base": base_generated, "lora": lora_generated, "source_loss": source_loss_generated, "regularized": regularized_generated},
            "responses": responses,
            "selected_model": selected_model,
        }

    def create_model_responses(self, ordinary_experiment, regularized_experiment, source_images, selected_model):
        num_images = int(self.ablation_config["num_response_images"])
        transformation_names = self.ablation_config["response_transformations"]
        transformation_configs = [transformation for transformation in regularized_experiment.config["lora"]["regularizer"]["transformations"] if transformation["name"] in transformation_names]
        response_config = deepcopy(regularized_experiment.config)

        if not transformation_configs:
            transformation_configs = regularized_experiment.config["lora"]["regularizer"]["transformations"][:4]

        response_config["lora"]["regularizer"]["transformations"] = transformation_configs

        if "target_input_rms" in response_config["lora"]["regularizer"] or "transformation_strength" in response_config["lora"]["regularizer"]:
            anchor_points = utils.select_anchor_points(response_config, response_config["dataset"]["name"], ordinary_experiment.device)
            transformation_functions, _ = utils.create_regularizer_transformations(response_config, ordinary_experiment.device, anchor_points)
        else:
            transformation_functions, _ = utils.create_regularizer_transformations(response_config, ordinary_experiment.device)

        resolved_transformations = response_config["lora"]["regularizer"].get("calibration", {}).get("transformations", [])
        resolved_intensities = {transformation["name"]: transformation["resolved_intensity"] for transformation in resolved_transformations}
        images = source_images[:num_images].to(ordinary_experiment.device)
        transformed_images = torch.stack([transformation(images) for transformation in transformation_functions])

        def responses(experiment):
            experiment.model.eval()
            experiment.lora_manager.enable_adapters()

            with torch.no_grad():
                original_output = experiment.model.deterministic_forward(images)
                transformed_output = experiment.model.deterministic_forward(transformed_images.flatten(0, 1)).reshape(len(transformation_functions), num_images, *original_output.shape[1:])

            return (transformed_output - original_output.unsqueeze(0)).cpu()

        labels = [f"{transformation['name'].replace('_', ' ')} ({float(resolved_intensities.get(transformation['name'], transformation['intensity'])):g})" for transformation in transformation_configs]

        return {
            "input": images.cpu(),
            "transformed": transformed_images.cpu(),
            "lora": responses(ordinary_experiment),
            "regularized": responses(regularized_experiment),
            "transformation_labels": labels,
            "selected_model": selected_model,
        }
