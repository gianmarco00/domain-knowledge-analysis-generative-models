from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from multiprocessing import get_context
from types import SimpleNamespace
from unittest.mock import Mock

import csv
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
    (checkpoint_path.parent / "last.pt").touch()

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
        (checkpoint_path.parent / "last.pt").touch()
        config_paths.append(config_path)

    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as executor:
        checkpoint_paths = list(executor.map(train_ablation_model, config_paths))

    assert checkpoint_paths == [str(path.parent / "checkpoints" / "best.pt") for path in config_paths]


def test_train_models_uses_two_processes_and_skips_existing_models(tmp_path, monkeypatch, capsys):
    class FakePool:
        instance = None

        def __init__(self, processes, initializer, initargs):
            self.processes = processes
            self.initializer = initializer
            self.initargs = initargs
            self.config_paths = []
            self.closed = False
            self.joined = False
            self.__class__.instance = self

        def imap_unordered(self, function, config_paths):
            self.config_paths = list(config_paths)
            return iter([None] * len(self.config_paths))

        def close(self):
            self.closed = True

        def join(self):
            self.joined = True

    context = get_context("spawn")
    fake_context = SimpleNamespace(Queue=context.Queue, RLock=context.RLock, Pool=FakePool)
    monkeypatch.setattr("domain_knowledge_analysis.experiments.ablation.get_context", lambda _: fake_context)
    ablation = Ablation.__new__(Ablation)
    ablation.ablation_config = {"parallel_models": 2}
    ablation.runs = [make_run(tmp_path / "first"), make_run(tmp_path / "second"), make_run(tmp_path / "trained")]
    existing_checkpoint = ablation.runs[-1]["model_dir"] / "checkpoints" / "best.pt"
    existing_checkpoint.parent.mkdir(parents=True)
    existing_checkpoint.touch()
    (existing_checkpoint.parent / "last.pt").touch()
    prepared_runs = []
    ablation.prepare_dataset_caches = lambda runs: prepared_runs.extend(runs)

    ablation.train_models()

    assert FakePool.instance.processes == 2
    assert FakePool.instance.config_paths == [str(run["config_path"]) for run in ablation.runs[:2]]
    assert FakePool.instance.closed and FakePool.instance.joined
    assert prepared_runs == ablation.runs[:2]
    progress_output = capsys.readouterr().err
    assert "1/3" in progress_output
    assert "3/3" in progress_output


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


def test_load_saved_results_restores_metric_types(tmp_path):
    metrics_path = tmp_path / "metrics.csv"
    row = {"model_id": "abc", "model_dir": "model", "config_path": "config.yaml", "seed": 42, "eta": 10.0, "mode": "encoder_response", "m": 5, "s": 0.1, "subset": "rotation", "extra_run": False, "validation_target_gain": 0.02}

    with open(metrics_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    loaded_row = Ablation.load_saved_results(metrics_path)["abc"]

    assert loaded_row["seed"] == 42
    assert loaded_row["m"] == 5
    assert loaded_row["eta"] == 10.0
    assert loaded_row["s"] == 0.1
    assert loaded_row["validation_target_gain"] == 0.02
    assert loaded_row["extra_run"] is False


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


def test_fid_uses_target_test_images_and_prior_samples_with_adapters(tmp_path, monkeypatch):
    from torchmetrics.image import fid as fid_module

    class FakeFID:
        instances = []

        def __init__(self, feature, normalize, reset_real_features, antialias):
            assert feature == 2048
            assert normalize is True
            assert reset_real_features is False
            assert antialias is True
            self.inception = torch.nn.Linear(1, 1)
            self.real_features_sum = torch.zeros(1, dtype=torch.float64)
            self.real_features_cov_sum = torch.zeros(1, 1, dtype=torch.float64)
            self.real_features_num_samples = torch.tensor(0)
            self.real = []
            self.fake = []
            self.__class__.instances.append(self)

        def to(self, device):
            return self

        def reset(self):
            self.fake = []

        def update(self, images, real):
            (self.real if real else self.fake).append(images.clone())
            if real:
                self.real_features_num_samples += len(images)

        def compute(self):
            return torch.tensor(12.5)

    class FakeModel:
        encoder_params = {"latent_dim": 2}

        def __init__(self, manager):
            self.manager = manager
            self.latents = []

        def eval(self):
            pass

        def generate_images(self, count, latents):
            assert self.manager.enabled
            assert not torch.is_grad_enabled()
            self.latents.append(latents.clone())
            return torch.full((count, 1, 2, 2), 0.75)

    monkeypatch.setattr(fid_module, "FrechetInceptionDistance", FakeFID)
    images = torch.arange(20, dtype=torch.float32).reshape(5, 1, 2, 2) / 20
    dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(images, torch.zeros(5)), batch_size=2)
    manager = FakeLoRAManager()
    model = FakeModel(manager)
    experiment = type("Experiment", (), {"model": model, "lora_manager": manager, "device": torch.device("cpu")})()
    ablation = Ablation.__new__(Ablation)
    ablation.ablation_config = {"fid_seed": 123, "fid_batch_size": 2}
    ablation.output_dir = tmp_path
    ablation.fid = None
    ablation.fid_reference_key = None

    reference_key, fid_key = ablation.prepare_fid(dataloader)
    legacy_fid_key = ablation.legacy_fid_key
    assert legacy_fid_key is not None
    assert ablation.score_fid(experiment, dataloader, reference_key) == 12.5
    first_latents = torch.cat(model.latents)
    assert ablation.score_fid(experiment, dataloader, reference_key) == 12.5
    torch.testing.assert_close(torch.cat(model.latents[3:]), first_latents)
    assert len(FakeFID.instances) == 1
    torch.testing.assert_close(torch.cat(FakeFID.instances[0].real), images.repeat(1, 3, 1, 1))
    torch.testing.assert_close(torch.cat(FakeFID.instances[0].fake), torch.full((5, 3, 2, 2), 0.75))
    assert FakeFID.instances[0].real_features_num_samples.item() == 5

    ablation.fid_reference_key = None
    ablation.fid.real.clear()
    assert ablation.score_fid(experiment, dataloader, reference_key) == 12.5
    assert ablation.fid.real == []
    assert ablation.fid.real_features_num_samples.item() == 5

    images[0] = 0
    changed_reference_key, changed_fid_key = ablation.prepare_fid(dataloader)
    assert changed_reference_key != reference_key
    assert changed_fid_key != fid_key
    assert ablation.score_fid(experiment, dataloader, changed_reference_key) == 12.5
    assert ablation.fid.real_features_num_samples.item() == 5
    ablation.ablation_config["fid_seed"] = 456
    same_reference_key, different_seed_key = ablation.prepare_fid(dataloader)
    assert same_reference_key == changed_reference_key
    assert different_seed_key != changed_fid_key
    assert ablation.legacy_fid_key != legacy_fid_key


def test_saved_results_are_invalidated_when_checkpoint_changes(tmp_path, monkeypatch):
    metrics_path = tmp_path / "results" / "metrics.csv"
    metrics_path.parent.mkdir()
    checkpoint_path = tmp_path / "model" / "checkpoints" / "best.pt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"first checkpoint")
    row = {"model_id": "abc", "seed": 42, "m": 5, "extra_run": False, "test_target_fid": 12.5, "fid_seed": 123, "checkpoint_hash": sha256(checkpoint_path.read_bytes()).hexdigest(), "fid_key": "fid-key"}
    with open(metrics_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    ablation = Ablation.__new__(Ablation)
    ablation.output_dir = tmp_path
    ablation.ablation_config = {"fid_seed": 123, "target_gain_fraction": 0.99}
    ablation.runs = [{"model_id": "abc", "model_dir": checkpoint_path.parent.parent, "config_path": tmp_path / "config.yaml"}]
    ablation.train_models = Mock()
    ablation.prepare_fid = Mock(return_value=("reference", "fid-key"))
    ablation.load_or_train = Mock(side_effect=RuntimeError("checkpoint needs evaluation"))
    monkeypatch.setattr("domain_knowledge_analysis.experiments.ablation.utils.load_config", Mock(return_value={"lora": {"transformed_dataset": {}}}))
    monkeypatch.setattr("domain_knowledge_analysis.experiments.ablation.utils.create_testing_dataloaders", Mock(return_value=object()))
    monkeypatch.setattr("domain_knowledge_analysis.experiments.ablation.select_best_models", Mock(side_effect=RuntimeError("saved result reused")))

    with pytest.raises(RuntimeError, match="saved result reused"):
        ablation.run()
    ablation.load_or_train.assert_not_called()

    checkpoint_path.write_bytes(b"different checkpoint")
    with pytest.raises(RuntimeError, match="checkpoint needs evaluation"):
        ablation.run()


def test_existing_fid_with_legacy_key_is_reused_without_rewriting_metrics(tmp_path, monkeypatch):
    metrics_path = tmp_path / "results" / "metrics.csv"
    metrics_path.parent.mkdir()
    checkpoint_path = tmp_path / "model" / "checkpoints" / "best.pt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"checkpoint")
    row = {"model_id": "abc", "test_target_fid": 12.5,
           "checkpoint_hash": sha256(checkpoint_path.read_bytes()).hexdigest(), "fid_key": "legacy-key"}
    with open(metrics_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    original_metrics = metrics_path.read_bytes()

    ablation = Ablation.__new__(Ablation)
    ablation.output_dir = tmp_path
    ablation.ablation_config = {"target_gain_fraction": 0.99}
    ablation.runs = [{"model_id": "abc", "model_dir": checkpoint_path.parent.parent, "config_path": tmp_path / "config.yaml"}]
    ablation.train_models = Mock()
    ablation.legacy_fid_key = "legacy-key"
    ablation.prepare_fid = Mock(return_value=("reference", "new-key"))
    ablation.load_or_train = Mock(side_effect=AssertionError("FID should be reused"))
    monkeypatch.setattr("domain_knowledge_analysis.experiments.ablation.utils.load_config", Mock(return_value={"lora": {"transformed_dataset": {}}}))
    monkeypatch.setattr("domain_knowledge_analysis.experiments.ablation.utils.create_testing_dataloaders", Mock(return_value=object()))
    monkeypatch.setattr("domain_knowledge_analysis.experiments.ablation.select_best_models", Mock(side_effect=RuntimeError("saved result reused")))

    with pytest.raises(RuntimeError, match="saved result reused"):
        ablation.run()
    ablation.load_or_train.assert_not_called()
    assert metrics_path.read_bytes() == original_metrics


def test_load_saved_results_accepts_missing_fid_for_interrupted_migration(tmp_path):
    metrics_path = tmp_path / "metrics.csv"
    row = {"model_id": "abc", "seed": 42, "m": 5, "extra_run": False, "test_target_fid": ""}

    with open(metrics_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    assert Ablation.load_saved_results(metrics_path)["abc"]["test_target_fid"] == ""


def test_fid_migration_preserves_unprocessed_saved_metrics(tmp_path, monkeypatch):
    metrics_path = tmp_path / "results" / "metrics.csv"
    metrics_path.parent.mkdir()
    rows = [{"model_id": name, "seed": 42, "m": 5, "extra_run": False, "validation_target_gain": 0.1} for name in ["first", "second"]]

    with open(metrics_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    experiment = SimpleNamespace(config={"lora": {"transformed_dataset": {}}}, logger=FakeLogger())
    ablation = Ablation.__new__(Ablation)
    ablation.output_dir = tmp_path
    ablation.ablation_config = {"fid_seed": 123}
    ablation.runs = [{"model_id": name, "model_dir": tmp_path / name, "config_path": tmp_path / name / "config.yaml", "seed": 42, "eta": 10.0, "mode": "output_gram", "m": 5, "s": 0.1, "subset": "rotation", "extra_run": False} for name in ["first", "second"]]
    for run in ablation.runs:
        checkpoint = run["model_dir"] / "checkpoints" / "best.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
    ablation.train_models = Mock()
    ablation.load_or_train = Mock(side_effect=[experiment, RuntimeError("interrupted")])
    ablation.prepare_fid = Mock(return_value=("reference", "fid-key"))
    ablation.evaluate = Mock(return_value={"validation_target_gain": 0.1, "test_target_fid": 12.5, "fid_seed": 123})
    monkeypatch.setattr("domain_knowledge_analysis.experiments.ablation.utils.load_config", Mock(return_value=experiment.config))
    monkeypatch.setattr("domain_knowledge_analysis.experiments.ablation.utils.create_testing_dataloaders", Mock(return_value=object()))

    with pytest.raises(RuntimeError, match="interrupted"):
        ablation.run()

    saved = ablation.load_saved_results(metrics_path)
    assert saved["first"]["test_target_fid"] == 12.5
    assert saved["first"]["fid_seed"] == 123
    assert saved["second"]["test_target_fid"] == ""
    assert saved["second"]["validation_target_gain"] == 0.1


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
            self.config = {"dataset": {"name": "mnist"}, "lora": {"regularizer": {"transformation_strength": 0.1, "transformations": [{"name": "first", "intensity": 1}, {"name": "second", "intensity": 2}]}}}

    transformations = [lambda images: images + 0.1, lambda images: images - 0.2]
    anchor_points = torch.zeros(3, 1, 2, 2)
    received_anchor_points = []
    monkeypatch.setattr("domain_knowledge_analysis.experiments.ablation.utils.select_anchor_points", lambda config, dataset_name, device: anchor_points)
    monkeypatch.setattr("domain_knowledge_analysis.experiments.ablation.utils.create_regularizer_transformations", lambda config, device, anchors: (received_anchor_points.append(anchors) or transformations, None))
    ablation = Ablation.__new__(Ablation)
    ablation.ablation_config = {"num_response_images": 2, "response_transformations": ["first", "second"]}
    selected_model = {"mode": "output_gram", "m": 10, "eta": 1000.0, "s": 0.1}
    results = ablation.create_model_responses(ResponseExperiment(2), ResponseExperiment(3), torch.zeros(2, 1, 2, 2), selected_model)

    torch.testing.assert_close(results["lora"][0], torch.full((2, 1, 2, 2), 0.2))
    torch.testing.assert_close(results["lora"][1], torch.full((2, 1, 2, 2), -0.4))
    torch.testing.assert_close(results["regularized"][0], torch.full((2, 1, 2, 2), 0.3))
    torch.testing.assert_close(results["regularized"][1], torch.full((2, 1, 2, 2), -0.6))
    assert received_anchor_points[0] is anchor_points


def test_training_progress_writes_worker_messages_to_run_log(tmp_path, monkeypatch):
    class ProgressExperiment(FakeExperiment):
        def __init__(self, config_path, progress_position, progress_description, progress_stream):
            super().__init__(config_path)
            self.progress_position = progress_position
            self.progress_description = progress_description
            self.progress_stream = progress_stream
            print("Preparing model")

        def adapt(self):
            print("Training model")
            super().adapt()

    monkeypatch.setattr("domain_knowledge_analysis.experiments.ablation.Experiment", ProgressExperiment)
    ProgressExperiment.instances.clear()
    config_path = tmp_path / "model_1234567890" / "config.yaml"
    config_path.parent.mkdir()

    train_ablation_model(config_path, progress_position=2)

    experiment = ProgressExperiment.instances[0]
    assert experiment.progress_position == 2
    assert experiment.progress_description == "Model 1234567890"
    assert experiment.progress_stream is not None
    assert experiment.adapt_calls == 1
    assert experiment.logger.closed
    assert "Preparing model" in (config_path.parent / "training.log").read_text()
    assert "Training model" in (config_path.parent / "training.log").read_text()


def test_partial_checkpoint_does_not_skip_training(tmp_path, monkeypatch):
    monkeypatch.setattr("domain_knowledge_analysis.experiments.ablation.Experiment", FakeExperiment)
    FakeExperiment.instances.clear()
    config_path = tmp_path / "model" / "config.yaml"
    checkpoint_dir = config_path.parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "best.pt").touch()

    train_ablation_model(config_path)

    assert FakeExperiment.instances[0].adapt_calls == 1


def test_ctrl_c_terminates_and_joins_the_worker_pool(tmp_path, monkeypatch):
    class InterruptedPool:
        instance = None

        def __init__(self, processes, initializer, initargs):
            self.terminated = False
            self.joined = False
            self.__class__.instance = self

        def imap_unordered(self, function, config_paths):
            raise KeyboardInterrupt

        def terminate(self):
            self.terminated = True

        def join(self):
            self.joined = True

    context = get_context("spawn")
    fake_context = SimpleNamespace(Queue=context.Queue, RLock=context.RLock, Pool=InterruptedPool)
    monkeypatch.setattr("domain_knowledge_analysis.experiments.ablation.get_context", lambda _: fake_context)
    ablation = Ablation.__new__(Ablation)
    ablation.ablation_config = {"parallel_models": 2}
    ablation.runs = [make_run(tmp_path / "first"), make_run(tmp_path / "second")]
    ablation.prepare_dataset_caches = lambda runs: None

    with pytest.raises(KeyboardInterrupt):
        ablation.train_models()

    assert InterruptedPool.instance.terminated
    assert InterruptedPool.instance.joined
