from copy import deepcopy
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from domain_knowledge_analysis.regularizers.target_geometry import (
    GeometryBatchSampler, TargetGeometryRegularizer, centered_gram, geometry_distortion,
)
from domain_knowledge_analysis.training.target_geometry_trainer import TargetGeometryTrainer


class ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.map = nn.Linear(2, 2, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.map.weight.copy_(torch.eye(2))

    def forward(self, images):
        codes = self.map(images.flatten(1))
        return codes, torch.zeros_like(codes)


class ToyDecoder(nn.Module):
    def __init__(self, scale=0.9):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float64))
        self.register_buffer("permutation", torch.eye(2, dtype=torch.float64))

    def forward(self, codes):
        mean = 0.5 + self.scale * ((codes - 0.5) @ self.permutation.T)
        return mean.logit().reshape(-1, 1, 1, 2)


class ToyModel(nn.Module):
    def __init__(self, scale=0.9):
        super().__init__()
        self.encoder = ToyEncoder()
        self.decoder = ToyDecoder(scale)
        self.decoder_distribution_name = "bernoulli"
        self.encoder_params = {"latent_dim": 2}

    def forward(self, images):
        mean, logvar = self.encoder(images)
        return self.decoder(mean), mean, logvar


def toy_data():
    return torch.tensor([[0.3, 0.4], [0.6, 0.3], [0.4, 0.7], [0.7, 0.6]], dtype=torch.float64).reshape(-1, 1, 1, 2)


def make_regularizer(model=None, **kwargs):
    model = ToyModel() if model is None else model
    sampler = GeometryBatchSampler(TensorDataset(toy_data()), 4, 7)
    return TargetGeometryRegularizer(model, sampler, calibration_batches=4, **kwargs)


def test_connected_gram_and_scale_distortion_match_direct_formula():
    inputs = toy_data()
    vectors = inputs.flatten(1).T
    centered = vectors - vectors.mean(1, keepdim=True)
    gram = centered_gram(inputs)
    torch.testing.assert_close(gram, centered.T @ centered)
    for scale in (0.0, 0.9, 1.0, 2.0):
        distorted = geometry_distortion(gram, scale * inputs + 0.123)
        assert distorted.item() == pytest.approx((scale**2 - 1)**2, abs=1e-14)


def test_orthogonal_transport_imperfect_teacher_and_better_target_decoder():
    model = ToyModel()
    regularizer = make_regularizer(model)
    assert regularizer.source_tolerance == pytest.approx(0.0361)
    permutation = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.float64)
    with torch.no_grad():
        model.encoder.map.weight.copy_(permutation)
        model.decoder.permutation.copy_(permutation)
    targets = toy_data().flip(-1)
    values = regularizer.distortions(targets)
    assert values["target_distortion"].item() == pytest.approx(0.0361)
    assert values["source_chart_distortion"].item() == pytest.approx(0.0361)
    assert regularizer.penalty(values["worst_distortion"].item()) == pytest.approx(0, abs=1e-14)
    with torch.no_grad():
        model.decoder.scale.fill_(1)
    values = regularizer.distortions(targets)
    assert values["target_distortion"].item() == pytest.approx(0, abs=1e-14)
    assert values["source_chart_distortion"].item() == pytest.approx(0.0361)
    assert regularizer.penalty(values["worst_distortion"].item()) == pytest.approx(0, abs=1e-14)


def test_collapse_has_distortion_one_and_positive_penalty():
    model = ToyModel()
    regularizer = make_regularizer(model)
    with torch.no_grad():
        model.encoder.map.weight.zero_()
    worst = regularizer.distortions(toy_data())["worst_distortion"]
    assert worst.item() == pytest.approx(1)
    assert regularizer.penalty(worst.item()) == pytest.approx(0.9639)


def test_perfect_reconstruction_with_scaled_codes_still_violates_source_chart():
    model = ToyModel()
    regularizer = make_regularizer(model)
    class ScaledEncoder(nn.Module):
        def forward(self, images):
            codes = .5 + 2 * (images.flatten(1) - .5)
            return codes, torch.zeros_like(codes)
    model.encoder = ScaledEncoder()
    with torch.no_grad():
        model.decoder.scale.fill_(.5)
    values = regularizer.distortions(toy_data())
    assert values["target_distortion"].item() == pytest.approx(0, abs=1e-14)
    assert values["source_chart_distortion"].item() == pytest.approx((1.8**2 - 1)**2)
    assert regularizer.penalty(values["worst_distortion"].item()) > 4


def test_nonuniform_teacher_errors_transport_batchwise_and_in_expectation():
    class NonuniformDecoder(ToyDecoder):
        def forward(self, codes):
            offsets = codes - .5
            values = .5 + self.scale * offsets + .1 * offsets.square()
            # Orthogonal image permutation is applied after the source function.
            return (values @ self.permutation.T).logit().reshape(-1, 1, 1, 2)
    model = ToyModel()
    model.decoder = NonuniformDecoder()
    dataset = TensorDataset(toy_data())
    sampler = GeometryBatchSampler(dataset, 3, 19)
    regularizer = TargetGeometryRegularizer(model, sampler, calibration_batches=20)
    source_sampler = GeometryBatchSampler(dataset, 3, 19)
    values = []
    permutation = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.float64)
    with torch.no_grad():
        model.encoder.map.weight.copy_(permutation)
        model.decoder.permutation.copy_(permutation)
    for _ in range(20):
        inputs = source_sampler.sample("cpu")
        target = inputs.flip(-1)
        distortions = regularizer.distortions(target)
        source_outputs = regularizer.source_decoder(inputs.flatten(1)).sigmoid()
        source_distortion = geometry_distortion(centered_gram(inputs), source_outputs)
        torch.testing.assert_close(distortions["target_distortion"], source_distortion)
        torch.testing.assert_close(distortions["source_chart_distortion"], source_distortion)
        values.append(distortions["worst_distortion"].item())
    assert max(values) - min(values) > .001
    assert regularizer.penalty(sum(values) / len(values)) == pytest.approx(0, abs=1e-14)
    # A per-batch hinge would spuriously penalize this exact witness.
    assert sum(max(value-regularizer.source_tolerance, 0) for value in values) > 0


def test_frozen_source_decoder_passes_input_gradient_but_has_no_weight_gradient():
    model = ToyModel()
    regularizer = make_regularizer(model, view="source_only")
    values = regularizer.distortions(toy_data())
    values["worst_distortion"].backward()
    assert model.encoder.map.weight.grad.abs().sum() > 0
    assert model.decoder.scale.grad is None
    assert all(not p.requires_grad and p.grad is None for p in regularizer.source_decoder.parameters())
    source_before = deepcopy(regularizer.source_decoder.state_dict())
    with torch.no_grad():
        model.decoder.scale.fill_(0.6)
    for name, value in source_before.items():
        torch.testing.assert_close(regularizer.source_decoder.state_dict()[name], value)


def test_full_view_worse_target_decoder_gets_direct_gradient():
    model = ToyModel()
    regularizer = make_regularizer(model)
    with torch.no_grad():
        model.decoder.scale.fill_(0.6)
    values = regularizer.distortions(toy_data())
    assert values["target_distortion"] > values["source_chart_distortion"]
    values["worst_distortion"].backward()
    assert model.decoder.scale.grad.abs() > 0
    assert model.encoder.map.weight.grad.abs().sum() > 0


def test_dual_represents_hinge_after_expectation_and_has_correct_gradient():
    regularizer = make_regularizer(eta=10, dual_learning_rate=0.5)
    # Average violation is zero, despite an individual batch exceeding c_s.
    tolerance = regularizer.source_tolerance
    a = torch.tensor([0., 2 * tolerance], dtype=torch.float64, requires_grad=True)
    assert regularizer.penalty(a.mean().item()) == 0
    assert (a.detach() - tolerance).clamp_min(0).mean().item() > 0
    # Maximize the LINEAR expectation over the two endpoints [0, eta].
    for samples in (a, a + .2, a * .1):
        violation = samples.mean().item() - tolerance
        assert max(0, regularizer.eta * violation) == pytest.approx(regularizer.eta * regularizer.penalty(samples.mean().item()))
    regularizer.dual_fraction = .4
    regularizer.primal_loss(a.mean()).backward()
    torch.testing.assert_close(a.grad, torch.full_like(a, 2.))
    regularizer.update_dual(torch.tensor(100.))
    assert regularizer.multiplier == 10
    regularizer.update_dual(torch.tensor(-100.))
    assert regularizer.multiplier == 0


@pytest.mark.parametrize("images", [torch.ones(3, 2), torch.ones(1, 2), torch.full((3, 2), float('nan'))])
def test_degenerate_geometry_rejected(images):
    with pytest.raises(ValueError):
        geometry_distortion(centered_gram(images), images)


def test_uninformative_source_tolerance_is_rejected():
    with pytest.raises(ValueError, match="Uninformative source tolerance"):
        make_regularizer(ToyModel(scale=0))


def test_sampler_is_reproducible_distinct_and_does_not_change_global_rng():
    dataset = TensorDataset(torch.arange(20).reshape(10, 2))
    one, two = GeometryBatchSampler(dataset, 5, 7), GeometryBatchSampler(dataset, 5, 7)
    state = torch.random.get_rng_state()
    for _ in range(3):
        batch = one.sample("cpu")
        torch.testing.assert_close(batch, two.sample("cpu"))
        assert len(batch.unique(dim=0)) == 5
    torch.testing.assert_close(torch.random.get_rng_state(), state)
    with pytest.raises(ValueError):
        GeometryBatchSampler(dataset, 11, 0)


def make_trainer(tmp_path=None):
    from domain_knowledge_analysis.training.checkpoints import CheckpointManager
    model = ToyModel()
    regularizer = make_regularizer(model, eta=10)
    with torch.no_grad():
        model.decoder.scale.fill_(0.6)
    dataset = TensorDataset(toy_data())
    loader = DataLoader(dataset, batch_size=3)  # Includes a singleton VAE remainder.
    trainer = TargetGeometryTrainer(
        model=model, train_dataloader=loader, validate_dataloader=loader,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0001),
        lr_scheduler=None, lr_scheduler_start_epoch=0,
        loss=lambda x, logits, mean, logvar: (logits.sigmoid()-x).square().flatten(1).sum(1).mean(),
        epochs=1, device="cpu", regularizer=regularizer, regularizer_eta=10,
        geometry_batch_size=4, validation_batches=2,
        checkpoint_manager=CheckpointManager(tmp_path) if tmp_path else None,
    )
    return trainer


def test_training_validation_and_regularizer_checkpoint(tmp_path):
    trainer = make_trainer(tmp_path)
    before = deepcopy(trainer.model.state_dict())
    history = trainer.fit()
    assert len(history["train_loss"]) == 1
    assert trainer.regularizer.dual_fraction > 0
    assert not torch.equal(before["decoder.scale"], trainer.model.decoder.scale)
    state = torch.load(tmp_path / "checkpoints" / "last.regularizer.pt", weights_only=True)
    assert state["dual_fraction"] == trainer.regularizer.dual_fraction
    assert state["calibration"]["source_tolerance"] == pytest.approx(.0361)
    before_validation = deepcopy(trainer.model.state_dict())
    dual_before = trainer.regularizer.dual_fraction
    sampler_before = trainer.geometry_sampler.generator.get_state().clone()
    first, components = trainer.validate_epoch()
    second, _ = trainer.validate_epoch()
    assert first == second  # Deterministic toy likelihood and geometry bank.
    assert trainer.regularizer.dual_fraction == dual_before
    torch.testing.assert_close(trainer.geometry_sampler.generator.get_state(), sampler_before)
    for name, value in before_validation.items():
        torch.testing.assert_close(trainer.model.state_dict()[name], value)
    assert components["regularizer"] == pytest.approx(max(components["worst_distortion"] - .0361, 0))


def test_validation_hinge_is_after_average(monkeypatch):
    trainer = make_trainer()
    tolerance = trainer.regularizer.source_tolerance
    values = iter((0., 2 * tolerance))
    monkeypatch.setattr(trainer.regularizer, "distortions", lambda _: {"worst_distortion": torch.tensor(next(values), dtype=torch.float64)})
    _, components = trainer.validate_epoch()
    assert components["regularizer"] == pytest.approx(0, abs=1e-15)


def test_real_lora_snapshot_and_encoder_decoder_adapter_gradients():
    from domain_knowledge_analysis.adapters import LoRAManager
    model = ToyModel()
    model.decoder = nn.Sequential(nn.Linear(2, 2, dtype=torch.float64), nn.Unflatten(1, (1, 1, 2)))
    with torch.no_grad():
        model.decoder[0].weight.copy_(3.6 * torch.eye(2))
        model.decoder[0].bias.fill_(-1.8)
    manager = LoRAManager(model, {"lora": {"rank": 2, "alpha": 2, "layers_to_inject": {"encoder": ["fc"], "decoder": ["fc"]}}})
    # The repo's layer selector recognizes fully connected layers by type.
    manager.inject_adapters()
    assert manager.lora_layers
    regularizer = make_regularizer(model)
    source_outputs = regularizer.source_decoder(torch.tensor([[.4, .6]], dtype=torch.float64)).detach().clone()
    for layer in manager.lora_layers.values():
        with torch.no_grad():
            layer.lora_B.fill_(0.1)
    for view in ("source_only", "target_only"):
        model.zero_grad(set_to_none=True)
        regularizer.view = view
        regularizer.distortions(toy_data())["worst_distortion"].backward()
        encoder_layers = [layer for name, layer in manager.lora_layers.items() if name.startswith("encoder")]
        decoder_layers = [layer for name, layer in manager.lora_layers.items() if name.startswith("decoder")]
        assert any(layer.lora_B.grad is not None and layer.lora_B.grad.abs().sum() > 0 for layer in encoder_layers)
        if view == "source_only":
            assert all(layer.lora_B.grad is None for layer in decoder_layers)
        else:
            assert any(layer.lora_B.grad is not None and layer.lora_B.grad.abs().sum() > 0 for layer in decoder_layers)
    torch.testing.assert_close(regularizer.source_decoder(torch.tensor([[.4, .6]], dtype=torch.float64)), source_outputs)
