from copy import deepcopy
from pathlib import Path

import torch

from domain_knowledge_analysis import utils
from domain_knowledge_analysis.experiments.experiment import Experiment
from domain_knowledge_analysis.experiments.ablation_utils import create_transformation_families, generate_model_runs, save_rows, save_yaml, select_eta
from domain_knowledge_analysis.plotting.ablation_plotter import AblationPlotter
from domain_knowledge_analysis.regularizers import TFRegularizer
from domain_knowledge_analysis.scoring import AdaptationScorer


class Ablation:
    def __init__(self, config_path):
        self.config = utils.load_config(config_path)
        self.repo_root = utils.get_repo_root()
        self.base_config_path = self.repo_root / Path(self.config["base_config"])
        self.base_config = utils.load_config(self.base_config_path)
        self.ablation_config = self.config["ablation"]
        self.output_dir = self.repo_root / Path(self.ablation_config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.evaluation_families = create_transformation_families(self.ablation_config, self.ablation_config["evaluation_mode"])
        self.runs = generate_model_runs(self.base_config, self.ablation_config, self.output_dir)

    def run(self):
        print(f"Ablation contains {len(self.runs)} unique models.")
        results = []

        for index, run in enumerate(self.runs):
            print(f"\nModel {index + 1}/{len(self.runs)}: {run['model_dir'].name}")
            experiment = self.load_or_train(run)
            metrics = self.evaluate(experiment)

            results.append({
                "model_id": run["model_id"],
                "model_dir": str(run["model_dir"]),
                "config_path": str(run["config_path"]),
                "seed": run["seed"],
                "eta": run["eta"],
                "mode": run["mode"],
                "m": run["m"],
                "subset_index": run["subset_index"],
                "subset": run["subset"],
                **metrics,
            })

            save_rows(results, self.output_dir / "results" / "metrics.csv")
            experiment.logger.close()
            del experiment

        eta_star, eta_summary = select_eta(
            results,
            self.ablation_config["eta_values"],
            self.ablation_config["eta_selection_mode"],
            self.ablation_config["eta_selection_m"],
            self.ablation_config["target_gain_fraction"],
        )

        save_rows(eta_summary, self.output_dir / "results" / "eta_selection.csv")
        save_yaml({"eta_star": eta_star}, self.output_dir / "results" / "selected_model.yaml")

        plotter = AblationPlotter(self.output_dir, self.ablation_config)
        plotter.plot_figure_1(results)
        plotter.plot_figure_2(results, eta_star)
        plotter.plot_heatmaps(results)
        plotter.plot_qualitative(self.create_qualitative_results(eta_star))

        print(f"\nSelected eta: {eta_star:g}")
        print(f"Results saved in: {self.output_dir}")

        return results

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
        gram_error = self.score_gram_error(experiment, source_test_dataloader)

        return {
            **{f"validation_{name}": value for name, value in validation_metrics.items()},
            **{f"test_{name}": value for name, value in test_metrics.items()},
            "held_out_gram_error": gram_error,
        }

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

    def score_gram_error(self, experiment, source_dataloader):
        num_images = int(self.ablation_config["gram_num_images"])
        batch_size = int(self.ablation_config["gram_batch_size"])
        images = []
        num_selected_images = 0

        for batch in source_dataloader:
            images.append(batch[0])
            num_selected_images += len(batch[0])

            if num_selected_images >= num_images:
                break

        images = torch.cat(images)[:num_images].to(experiment.device)
        evaluation_config = deepcopy(experiment.config)
        evaluation_config["lora"]["regularizer"] = {
            "eta": 1.0,
            "num_anchor_points": num_images,
            "transformations": deepcopy(self.evaluation_families),
        }

        transformations = utils.create_regularizer_transformations(evaluation_config, experiment.device)
        regularizer = TFRegularizer(experiment.model, experiment.lora_manager, transformations, images, anchor_batch_size=batch_size, calibration_batch_size=batch_size)

        with torch.no_grad():
            adapted_grams = regularizer.gram_matrix_batched(batch_size)
            source_grams = regularizer.source_gram_matrices
            gram_error = torch.square(adapted_grams - source_grams).mean(dim=(-2, -1)).mean()

        return gram_error.item()

    def create_qualitative_results(self, eta_star):
        seed = int(self.ablation_config["qualitative_seed"])
        num_images = int(self.ablation_config["num_qualitative_images"])
        final_mode = self.ablation_config["final_mode"]
        final_m = int(self.ablation_config["final_m"])

        ordinary_run = next(run for run in self.runs if run["seed"] == seed and run["eta"] == 0)

        if eta_star == 0:
            regularized_run = ordinary_run
        else:
            regularized_run = next(run for run in self.runs if run["seed"] == seed and run["eta"] == eta_star and run["mode"] == final_mode and run["m"] == final_m)

        ordinary_experiment = self.load_or_train(ordinary_run)
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

        regularized_experiment.lora_manager.enable_adapters()
        regularized_source = regularized_experiment.model.reconstruct_images(source_images).cpu()
        regularized_target = regularized_experiment.model.reconstruct_images(target_images).cpu()
        regularized_generated = regularized_experiment.model.generate_images(num_images, latents).cpu()

        ordinary_experiment.logger.close()
        regularized_experiment.logger.close()

        return {
            "source": {"input": source_images, "base": base_source, "lora": lora_source, "regularized": regularized_source},
            "target": {"input": target_images, "base": base_target, "lora": lora_target, "regularized": regularized_target},
            "generation": {"base": base_generated, "lora": lora_generated, "regularized": regularized_generated},
        }
