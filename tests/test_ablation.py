from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import pytest
import torch

from domain_knowledge_analysis.experiments import Ablation
from domain_knowledge_analysis.experiments.ablation import train_ablation_model


class FakeCheckpointManager:
    def __init__(self):
        self.loaded_path = None

    def load_model(self, model, checkpoint_path, device):
        self.loaded_path = checkpoint_path
        return "loaded model"


class FakeLoRAManager:
    def __init__(self):
        self.enabled = False

    def enable_adapters(self):
        self.enabled = True


class FakeLogger:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeExperiment:
    instances = []

    def __init__(self, config_path):
        self.config_path = config_path
        self.model = "source model"
        self.device = torch.device("cpu")
        self.checkpoint_manager = FakeCheckpointManager()
        self.lora_manager = FakeLoRAManager()
        self.logger = FakeLogger()
        self.adapt_calls = 0
        self.instances.append(self)

    def adapt(self):
        self.adapt_calls += 1


def make_run(tmp_path):
    return {
        "config_path": tmp_path / "config.yaml",
        "model_dir": tmp_path / "model",
    }


def test_load_or_train_trains_when_checkpoint_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("domain_knowledge_analysis.experiments.ablation.Experiment", FakeExperiment)
    FakeExperiment.instances.clear()
    ablation = Ablation.__new__(Ablation)
    run = make_run(tmp_path)

    experiment = ablation.load_or_train(run)

    assert experiment.adapt_calls == 1
    assert experiment.model == "loaded model"
    assert experiment.checkpoint_manager.loaded_path == run["model_dir"] / "checkpoints" / "best.pt"
    assert experiment.lora_manager.enabled


def test_load_or_train_reuses_existing_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr("domain_knowledge_analysis.experiments.ablation.Experiment", FakeExperiment)
    FakeExperiment.instances.clear()
    ablation = Ablation.__new__(Ablation)
    run = make_run(tmp_path)
    checkpoint_path = run["model_dir"] / "checkpoints" / "best.pt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.touch()

    experiment = ablation.load_or_train(run)

    assert experiment.adapt_calls == 0
    assert experiment.checkpoint_manager.loaded_path == checkpoint_path
    assert experiment.lora_manager.enabled


def test_train_ablation_model_trains_and_closes_logger(tmp_path, monkeypatch):
    monkeypatch.setattr("domain_knowledge_analysis.experiments.ablation.Experiment", FakeExperiment)
    FakeExperiment.instances.clear()
    config_path = tmp_path / "model" / "config.yaml"

    checkpoint_path = train_ablation_model(config_path)
    experiment = FakeExperiment.instances[0]

    assert experiment.adapt_calls == 1
    assert experiment.logger.closed
    assert checkpoint_path == str(config_path.parent / "checkpoints" / "best.pt")


def test_train_ablation_model_runs_in_spawned_processes(tmp_path):
    config_paths = []

    for model_name in ["first", "second"]:
        config_path = tmp_path / model_name / "config.yaml"
        checkpoint_path = config_path.parent / "checkpoints" / "best.pt"
        checkpoint_path.parent.mkdir(parents=True)
        checkpoint_path.touch()
        config_paths.append(config_path)

    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as executor:
        checkpoint_paths = list(executor.map(train_ablation_model, config_paths))

    assert checkpoint_paths == [str(path.parent / "checkpoints" / "best.pt") for path in config_paths]


def test_train_models_uses_two_processes_and_skips_existing_models(tmp_path, monkeypatch):
    class FakeExecutor:
        instance = None

        def __init__(self, max_workers, mp_context):
            self.max_workers = max_workers
            self.config_paths = []
            self.__class__.instance = self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def map(self, function, config_paths):
            self.config_paths = list(config_paths)
            return [None] * len(self.config_paths)

    monkeypatch.setattr("domain_knowledge_analysis.experiments.ablation.ProcessPoolExecutor", FakeExecutor)
    ablation = Ablation.__new__(Ablation)
    ablation.ablation_config = {"parallel_models": 2}
    ablation.runs = [make_run(tmp_path / "first"), make_run(tmp_path / "second"), make_run(tmp_path / "trained")]
    existing_checkpoint = ablation.runs[-1]["model_dir"] / "checkpoints" / "best.pt"
    existing_checkpoint.parent.mkdir(parents=True)
    existing_checkpoint.touch()
    prepared_runs = []
    ablation.prepare_dataset_caches = lambda runs: prepared_runs.extend(runs)

    ablation.train_models()

    assert FakeExecutor.instance.max_workers == 2
    assert FakeExecutor.instance.config_paths == [str(run["config_path"]) for run in ablation.runs[:2]]
    assert prepared_runs == ablation.runs[:2]


def test_prepare_dataset_caches_runs_once_per_seed(tmp_path, monkeypatch):
    runs = [
        {**make_run(tmp_path / "first"), "seed": 42},
        {**make_run(tmp_path / "second"), "seed": 42},
        {**make_run(tmp_path / "third"), "seed": 43},
    ]
    configs = {
        run["config_path"]: {"seed": run["seed"], "lora": {"transformed_dataset": {"type": "rotation"}}}
        for run in runs
    }
    prepared_seeds = []
    monkeypatch.setattr("domain_knowledge_analysis.experiments.ablation.utils.load_config", lambda path: configs[path])
    monkeypatch.setattr("domain_knowledge_analysis.experiments.ablation.utils.create_training_dataloaders", lambda config, transformation: prepared_seeds.append(config["seed"]))
    ablation = Ablation.__new__(Ablation)

    ablation.prepare_dataset_caches(runs)

    assert prepared_seeds == [42, 43]


def test_score_split_computes_target_gain_and_source_degradation(monkeypatch):
    class FakeScorer:
        def __init__(self, *args):
            pass

        def score(self):
            return {
                "pretrained_target_mse": 0.30,
                "adapted_target_mse": 0.10,
                "pretrained_source_mse": 0.01,
                "adapted_source_mse": 0.04,
            }

    monkeypatch.setattr("domain_knowledge_analysis.experiments.ablation.AdaptationScorer", FakeScorer)
    ablation = Ablation.__new__(Ablation)
    experiment = type("Experiment", (), {"model": object(), "lora_manager": object(), "config": {}, "device": torch.device("cpu")})()

    metrics = ablation.score_split(experiment, object(), object())

    assert metrics["target_gain"] == pytest.approx(0.20)
    assert metrics["source_degradation"] == pytest.approx(0.03)


def test_score_gram_error_averages_all_gram_entries(monkeypatch):
    class FakeRegularizer:
        def __init__(self, *args, **kwargs):
            self.source_gram_matrices = torch.zeros(2, 2, 2)

        def gram_matrix_batched(self, batch_size):
            return torch.ones(2, 2, 2)

    monkeypatch.setattr("domain_knowledge_analysis.experiments.ablation.TFRegularizer", FakeRegularizer)
    monkeypatch.setattr("domain_knowledge_analysis.experiments.ablation.utils.create_regularizer_transformations", lambda config, device: None)

    ablation = Ablation.__new__(Ablation)
    ablation.ablation_config = {"gram_num_images": 2, "gram_batch_size": 2}
    ablation.evaluation_families = []
    experiment = type("Experiment", (), {
        "config": {"lora": {}},
        "device": torch.device("cpu"),
        "model": object(),
        "lora_manager": object(),
    })()
    source_dataloader = [(torch.zeros(2, 1, 2, 2), torch.zeros(2))]

    assert ablation.score_gram_error(experiment, source_dataloader) == pytest.approx(1.0)


def test_model_responses_are_output_differences_for_both_models(monkeypatch):
    class ResponseModel(torch.nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.scale = scale

        def deterministic_forward(self, images):
            return self.scale * images

    class ResponseExperiment:
        def __init__(self, scale):
            self.device = torch.device("cpu")
            self.model = ResponseModel(scale)
            self.lora_manager = FakeLoRAManager()
            self.config = {"lora": {"regularizer": {"transformations": [{"name": "first", "intensity": 1}, {"name": "second", "intensity": 2}]}}}

    transformations = [lambda images: images + 0.1, lambda images: images - 0.2]
    monkeypatch.setattr("domain_knowledge_analysis.experiments.ablation.utils.create_regularizer_transformations", lambda config, device: (transformations, None))
    ablation = Ablation.__new__(Ablation)
    ablation.ablation_config = {"num_response_images": 2, "response_transformations": ["first", "second"]}
    selected_model = {"mode": "semantic_perturbations", "m": 10, "eta": 1000.0}
    results = ablation.create_model_responses(ResponseExperiment(2), ResponseExperiment(3), torch.zeros(2, 1, 2, 2), selected_model)

    torch.testing.assert_close(results["lora"][0], torch.full((2, 1, 2, 2), 0.2))
    torch.testing.assert_close(results["lora"][1], torch.full((2, 1, 2, 2), -0.4))
    torch.testing.assert_close(results["regularized"][0], torch.full((2, 1, 2, 2), 0.3))
    torch.testing.assert_close(results["regularized"][1], torch.full((2, 1, 2, 2), -0.6))
