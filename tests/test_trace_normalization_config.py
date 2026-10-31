from types import SimpleNamespace

import pytest
import torch

from domain_knowledge_analysis.experiments.experiment import Experiment
from domain_knowledge_analysis.utils import load_config
from test_transformation_fingerprint import DummyLoRAManager, DummyModel, make_inputs


@pytest.mark.parametrize("configured, expected", [(None, True), (True, True), (False, False)])
def test_lora_config_controls_regularizer_during_adaptation(monkeypatch, configured, expected):
    import domain_knowledge_analysis.experiments.experiment as experiment_module

    config = load_config("config/lora_vae_mnist.yaml")
    if configured is None:
        config["lora"]["regularizer"].pop("trace_normalization")
    else:
        config["lora"]["regularizer"]["trace_normalization"] = configured
    anchors, transformations = make_inputs()
    model = DummyModel()
    experiment = Experiment.__new__(Experiment)
    experiment.config = config
    experiment.checkpoint_manager = SimpleNamespace(model_config={"dataset": {"name": config["dataset"]["name"]}})
    experiment.model = model
    experiment.lora_manager = DummyLoRAManager(model)
    experiment.lora_manager.trainable_parameters = lambda: model.parameters()
    experiment.device = torch.device("cpu")
    experiment.log_dir = None
    experiment.logger = None
    experiment.progress_position = None
    experiment.progress_description = "test"
    experiment.progress_stream = None
    experiment.print_tensorboard_instructions = lambda _: None

    monkeypatch.setattr(experiment_module.utils, "create_training_dataloaders", lambda **kwargs: (None, None))
    monkeypatch.setattr(experiment_module.utils, "create_optimizer", lambda *args: None)
    monkeypatch.setattr(experiment_module.utils, "create_lr_scheduler", lambda *args: (None, None))
    monkeypatch.setattr(experiment_module.utils, "create_loss", lambda *args: None)
    monkeypatch.setattr(experiment_module.utils, "select_anchor_points", lambda *args: anchors)
    monkeypatch.setattr(experiment_module.utils, "create_regularizer_transformations", lambda *args: transformations)
    captured = {}

    class FakeTrainer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def fit(self):
            pass

    monkeypatch.setattr(experiment_module, "Trainer", FakeTrainer)
    experiment.adapt()
    regularizer = captured["regularizer"]
    assert regularizer.trace_normalization is expected
    source_traces = regularizer.source_gram_matrices.diagonal(dim1=-2, dim2=-1).sum(-1)
    if expected:
        torch.testing.assert_close(source_traces, torch.ones_like(source_traces))
    else:
        assert not torch.allclose(source_traces, torch.ones_like(source_traces))
