import pytest
import torch

from domain_knowledge_analysis.experiments import Ablation


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


class FakeExperiment:
    instances = []

    def __init__(self, config_path):
        self.config_path = config_path
        self.model = "source model"
        self.device = torch.device("cpu")
        self.checkpoint_manager = FakeCheckpointManager()
        self.lora_manager = FakeLoRAManager()
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
