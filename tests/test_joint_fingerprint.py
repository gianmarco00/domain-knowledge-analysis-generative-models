from types import SimpleNamespace

import pytest
import torch

from domain_knowledge_analysis.experiments.experiment import Experiment
from domain_knowledge_analysis.experiments.ablation_utils import create_training_config
from domain_knowledge_analysis.regularizers import TFRegularizer
from domain_knowledge_analysis.utils import load_config
from test_transformation_fingerprint import DummyLoRAManager, DummyModel, DummyVAE, make_inputs


def make_regularizer(mode="joint_gram", fingerprint_map="reconstruction", batch_size=3, calibration_batch_size=2):
    anchors, transforms = make_inputs()
    model = DummyVAE()
    regularizer = TFRegularizer(model, DummyLoRAManager(model), transforms, anchors, mode=mode, fingerprint_map=fingerprint_map, anchor_batch_size=batch_size, calibration_batch_size=calibration_batch_size)
    return model, regularizer


def test_joint_gram_matches_formula_and_uses_cross_anchor_entries():
    responses = torch.tensor([[[1., 2.], [2., 0.]], [[0., 1.], [1., -1.]]], dtype=torch.float64)
    rows = responses.reshape(4, 2)
    expected = rows @ rows.T
    expected = expected / expected.trace()
    actual = TFRegularizer.normalized_joint_gram(responses)
    assert actual.shape == (4, 4)
    torch.testing.assert_close(actual, expected)
    assert actual[0, 2] != 0


def test_joint_gram_rejects_independent_frames_and_gains_but_allows_shared_ones():
    responses = torch.tensor([[[1., 0.], [0., 1.]], [[1., 1.], [-1., 2.]]])
    rotation = torch.tensor([[0., -1.], [1., 0.]])
    source = TFRegularizer.normalized_joint_gram(responses)
    torch.testing.assert_close(TFRegularizer.normalized_joint_gram(3 * (responses @ rotation)), source)
    independently_rotated = responses.clone()
    independently_rotated[1] = independently_rotated[1] @ rotation
    independently_scaled = responses.clone()
    independently_scaled[1] *= 2
    # Local raw Grams cannot see the independent rotation.
    torch.testing.assert_close(responses @ responses.transpose(-1, -2), independently_rotated @ independently_rotated.transpose(-1, -2))
    assert not torch.allclose(TFRegularizer.normalized_joint_gram(independently_rotated), source)
    assert not torch.allclose(TFRegularizer.normalized_joint_gram(independently_scaled), source)
    assert (TFRegularizer.normalized_joint_gram(torch.zeros_like(responses)) - source).square().sum() > 0


def test_joint_gram_gradient_matches_finite_differences():
    torch.manual_seed(21)
    responses = torch.randn(3, 2, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(TFRegularizer.normalized_joint_gram, (responses,))


@pytest.mark.parametrize("fingerprint_map", ["reconstruction", "fixed_decoder"])
def test_joint_source_cache_loss_and_backpropagation(fingerprint_map):
    model, regularizer = make_regularizer(fingerprint_map=fingerprint_map)
    indices = torch.tensor([0, 2, 5])
    regularizer.anchor_indices = indices
    regularizer.anchor_position = 0
    response = regularizer.model_response(indices) / regularizer.input_scales.view(1, -1, 1)
    current = response.flatten(0, 1)
    source = regularizer.source_joint_responses[indices].flatten(0, 1)
    current_gram = current @ current.T
    source_gram = source @ source.T
    expected = (current_gram / current_gram.trace() - source_gram / source_gram.trace()).square().sum()
    loss = regularizer()
    torch.testing.assert_close(loss, expected)
    assert not regularizer.source_joint_responses.requires_grad
    loss.backward()
    assert model.decoder_delta.grad is not None and torch.isfinite(model.decoder_delta.grad)
    if fingerprint_map == "fixed_decoder":
        assert model.encoder_delta.grad is None
    else:
        assert model.encoder_delta.grad is not None and torch.isfinite(model.encoder_delta.grad)


@pytest.mark.parametrize("mode", ["output_gram", "joint_gram"])
def test_fixed_decoder_probes_are_frozen_source_latents(mode):
    model, regularizer = make_regularizer(mode=mode, fingerprint_map="fixed_decoder")
    with torch.no_grad():
        model.adapters_enabled = False
        images = torch.cat((regularizer.x.unsqueeze(1), regularizer.transformed_x), dim=1)
        means, _ = model.encoder(images.flatten(0, 1))
        expected = means.reshape(len(images), regularizer.num_transformations + 1, -1)
        source_response = regularizer.model_response()
        torch.testing.assert_close(regularizer.fixed_decoder_probes, expected)
        if mode == "joint_gram":
            torch.testing.assert_close(regularizer.source_joint_responses, source_response / regularizer.input_scales.view(1, -1, 1))
        else:
            torch.testing.assert_close(regularizer.source_gram_matrices, regularizer.gram_matrix(response_vector=source_response))
        model.adapters_enabled = True
        before = regularizer.model_response()
        model.encoder_delta.add_(10)
        torch.testing.assert_close(regularizer.model_response(), before)


@pytest.mark.parametrize("fingerprint_map", ["reconstruction", "fixed_decoder"])
def test_joint_calibration_is_independent_of_batch_size(fingerprint_map):
    _, first = make_regularizer(fingerprint_map=fingerprint_map, calibration_batch_size=2)
    _, second = make_regularizer(fingerprint_map=fingerprint_map, calibration_batch_size=7)
    torch.testing.assert_close(first.input_scales, second.input_scales)
    torch.testing.assert_close(first.source_joint_responses, second.source_joint_responses)
    if fingerprint_map == "fixed_decoder":
        torch.testing.assert_close(first.fixed_decoder_probes, second.fixed_decoder_probes)


def test_joint_batches_have_fixed_size_and_distinct_anchors():
    _, regularizer = make_regularizer(batch_size=3)
    for _ in range(12):
        indices = regularizer.sample_anchor_indices()
        assert len(indices) == 3 and len(indices.unique()) == 3
    _, full = make_regularizer(batch_size=100)
    assert len(full.sample_anchor_indices()) == 7


def test_joint_rejects_degenerate_reference_and_too_few_anchors():
    anchors, transforms = make_inputs()
    model = DummyModel()
    model.weight.data.zero_()
    with pytest.raises(ValueError, match="source response energy"):
        TFRegularizer(model, DummyLoRAManager(model), transforms, anchors, mode="joint_gram")
    with pytest.raises(ValueError, match="at least two"):
        TFRegularizer(model, DummyLoRAManager(model), transforms, anchors[:1], mode="joint_gram")


def test_calibration_failure_restores_model_and_adapter_state(monkeypatch):
    anchors, transforms = make_inputs()
    model = DummyVAE().train()
    manager = DummyLoRAManager(model)

    def fail(_):
        raise RuntimeError("encoder failure")

    monkeypatch.setattr(model, "encoder", fail)
    with pytest.raises(RuntimeError, match="encoder failure"):
        TFRegularizer(model, manager, transforms, anchors, mode="joint_gram", fingerprint_map="fixed_decoder")
    assert model.training and manager.adapters_enabled and model.adapters_enabled


def test_ablation_keeps_joint_map_and_removes_it_for_other_modes():
    config = load_config("config/lora_vae_mnist.yaml")
    config["lora"]["regularizer"]["fingerprint_map"] = "fixed_decoder"
    transforms = config["lora"]["regularizer"]["transformations"][:2]
    joint = create_training_config(config, 42, 1, transforms, regularizer_mode="joint_gram")
    source = create_training_config(config, 42, 1, [], regularizer_mode="source_loss")
    encoder = create_training_config(config, 42, 1, transforms, regularizer_mode="encoder_gram")
    assert joint["lora"]["regularizer"]["fingerprint_map"] == "fixed_decoder"
    assert "fingerprint_map" not in source["lora"]["regularizer"]
    assert "fingerprint_map" not in encoder["lora"]["regularizer"]


def test_joint_config_is_passed_to_training(monkeypatch):
    import domain_knowledge_analysis.experiments.experiment as module

    config = load_config("config/lora_vae_mnist.yaml")
    config["lora"]["regularizer"].update(mode="joint_gram", fingerprint_map="fixed_decoder", anchor_batch_size=3, trace_normalization=False)
    anchors, transforms = make_inputs()
    model = DummyVAE()
    experiment = Experiment.__new__(Experiment)
    experiment.config = config
    experiment.checkpoint_manager = SimpleNamespace(model_config={"dataset": {"name": config["dataset"]["name"]}})
    experiment.model = model
    experiment.lora_manager = DummyLoRAManager(model)
    experiment.lora_manager.trainable_parameters = lambda: model.parameters()
    experiment.device = torch.device("cpu")
    experiment.log_dir = experiment.logger = experiment.progress_position = experiment.progress_stream = None
    experiment.progress_description = "test"
    experiment.print_tensorboard_instructions = lambda _: None
    monkeypatch.setattr(module.utils, "create_training_dataloaders", lambda **kwargs: (None, None))
    monkeypatch.setattr(module.utils, "create_optimizer", lambda *args: None)
    monkeypatch.setattr(module.utils, "create_lr_scheduler", lambda *args: (None, None))
    monkeypatch.setattr(module.utils, "create_loss", lambda *args: None)
    monkeypatch.setattr(module.utils, "select_anchor_points", lambda *args: anchors)
    monkeypatch.setattr(module.utils, "create_regularizer_transformations", lambda *args: transforms)
    captured = {}
    class FakeTrainer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def fit(self):
            pass

    monkeypatch.setattr(module, "Trainer", FakeTrainer)
    experiment.adapt()
    regularizer = captured["regularizer"]
    assert regularizer.mode == "joint_gram"
    assert regularizer.fingerprint_map == "fixed_decoder"
    assert regularizer.anchor_batch_size == 3
    # Joint normalization is unconditional, even if the local-mode switch is false.
    gram = regularizer.normalized_joint_gram(regularizer.source_joint_responses[:3])
    torch.testing.assert_close(gram.trace(), torch.tensor(1.))
