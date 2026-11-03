"""Numerical/optimizer equivalence checks for the geometry fast paths."""

import pytest
import torch
from torch.utils.data import Subset, TensorDataset

from domain_knowledge_analysis.regularizers.target_geometry import GeometryBatchSampler
from domain_knowledge_analysis.training.target_geometry_trainer import TargetGeometryTrainer
from test_target_geometry import make_regularizer, make_trainer, ToyModel, toy_data


@pytest.mark.parametrize("nested", [False, True, "negative"])
def test_vectorized_sampler_matches_original_images_indices_and_rng(nested):
    dataset = TensorDataset(torch.arange(80).reshape(20, 4))
    if nested == "negative":
        dataset = Subset(Subset(dataset, [-1, 7, -2, 15, 18, 3, 5, 9]), [-1, 3, 0, 5, 1])
    elif nested:
        dataset = Subset(Subset(dataset, [12, 7, 1, 15, 18, 3, 5, 9]), [7, 3, 0, 5, 1])
    sampler = GeometryBatchSampler(dataset, 4, 31)
    reference_generator = torch.Generator().manual_seed(31)
    for _ in range(5):
        indices = torch.randperm(len(dataset), generator=reference_generator)[:4]
        expected = torch.stack([dataset[int(index)][0] for index in indices])
        torch.testing.assert_close(sampler.sample("cpu"), expected, rtol=0, atol=0)
        torch.testing.assert_close(sampler.generator.get_state(), reference_generator.get_state(), rtol=0, atol=0)


def test_arbitrary_or_overridden_dataset_getitem_is_not_bypassed_or_cached():
    class TransformDataset(TensorDataset):
        calls = 0
        def __getitem__(self, index):
            self.calls += 1
            return (super().__getitem__(index)[0] + self.calls,)
    dataset = TransformDataset(torch.zeros(8, 2))
    sampler = GeometryBatchSampler(Subset(dataset, [1, 2, 3, 4]), 2, 1)
    assert dataset.calls == 0
    assert not torch.equal(sampler.sample("cpu"), sampler.sample("cpu"))
    assert dataset.calls == 4


def reference_train_epoch(trainer):
    """Original two-view backward and scalar reads, before performance changes."""
    trainer.model.train()
    total, samples, worst_total, steps = 0., 0, 0., 0
    for batch in trainer.train_dataloader:
        images = batch[0].to(trainer.device)
        trainer.optimizer.zero_grad()
        base = trainer.loss(images, *trainer.model(images))
        values = trainer.regularizer.distortions(trainer.geometry_sampler.sample(trainer.device))
        # Always backpropagate Tensor.max, including its zero-gradient branch.
        loss = base + trainer.regularizer.primal_loss(values["worst_distortion"])
        loss.backward()
        trainer.optimizer.step()
        trainer.regularizer.update_dual(values["worst_distortion"])
        total += base.item() * len(images)
        samples += len(images)
        steps += 1
        worst_total += values["worst_distortion"].item()
    return total / samples + trainer.regularizer.eta * trainer.regularizer.penalty(worst_total / steps)


@pytest.mark.parametrize("scale", [.6, .9, 1.])  # target worse, exact tie, source worse
@pytest.mark.parametrize("initial_dual", [0., .4])
@pytest.mark.parametrize("view", ["full", "target_only", "source_only"])
@pytest.mark.parametrize("learning_rate", [1e-4, .01])
def test_training_loss_gradients_adam_momentum_dual_and_sampler_match_reference(scale, initial_dual, view, learning_rate):
    optimized, reference = make_trainer(), make_trainer()
    for trainer in (optimized, reference):
        with torch.no_grad():
            trainer.model.decoder.scale.fill_(scale)
        trainer.regularizer.dual_fraction = initial_dual
        trainer.regularizer.view = view
        trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=learning_rate)
    # Multiple epochs exercise Adam's state, changing active views and lambda=0.
    for _ in range(3):
        expected_loss = reference_train_epoch(reference)
        actual_loss, _ = optimized.train_epoch()
        assert actual_loss == pytest.approx(expected_loss, abs=1e-13, rel=1e-13)
        assert optimized.regularizer.dual_fraction == reference.regularizer.dual_fraction
        for (name, parameter), (ref_name, expected) in zip(optimized.model.named_parameters(), reference.model.named_parameters()):
            assert name == ref_name
            torch.testing.assert_close(parameter, expected, rtol=1e-13, atol=1e-13)
            torch.testing.assert_close(parameter.grad, expected.grad, rtol=1e-13, atol=1e-13)
            for key, value in optimized.optimizer.state[parameter].items():
                torch.testing.assert_close(value, reference.optimizer.state[expected][key], rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(optimized.geometry_sampler.generator.get_state(), reference.geometry_sampler.generator.get_state())


def test_active_view_releases_losing_graph_and_preserves_exact_tie_gradient():
    for first, second in ((.2, .1), (.1, .2), (.2, .2)):
        leaves = [torch.tensor(first, requires_grad=True), torch.tensor(second, requires_grad=True)]
        values = {"target_distortion": leaves[0] * 2, "source_chart_distortion": leaves[1] * 2}
        values["worst_distortion"] = torch.stack(list(values.values())).max()
        host = TargetGeometryTrainer._host_values(values)
        worst = TargetGeometryTrainer._active_geometry(values, host)
        worst.backward()
        actual = torch.tensor([0. if leaf.grad is None else leaf.grad.item() for leaf in leaves])
        expected = torch.tensor([2., 0.] if first > second else [0., 2.] if second > first else [1., 1.])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if first != second:
            loser = "source_chart_distortion" if first > second else "target_distortion"
            assert values[loser].grad_fn is None


def test_resident_image_cache_uses_contents_and_preserves_custom_transforms():
    from domain_knowledge_analysis.utils.target_geometry_runtime import resident_tensor_dataset
    dataset = TensorDataset(torch.arange(32).reshape(8, 4))
    first, digest = resident_tensor_dataset(Subset(dataset, [7, 2, 0]), "cpu")
    second, second_digest = resident_tensor_dataset(Subset(dataset, [7, 2, 0]), "cpu")
    assert first is second and digest == second_digest
    torch.testing.assert_close(first.tensors[0], dataset.tensors[0][[7, 2, 0]])
    dataset.tensors[0][7, 0] += 1
    changed, changed_digest = resident_tensor_dataset(Subset(dataset, [7, 2, 0]), "cpu")
    assert changed is not first and changed_digest != digest
    class CustomDataset(TensorDataset):
        def __getitem__(self, index):
            raise AssertionError("Must not materialize arbitrary transforms")
    custom = CustomDataset(torch.ones(8, 2))
    assert resident_tensor_dataset(custom, "cpu") == (custom, None)


def test_tensor_loader_preserves_members_remainder_and_private_rng():
    from domain_knowledge_analysis.utils.target_geometry_runtime import TensorBatchLoader
    dataset = TensorDataset(torch.arange(28).reshape(7, 4))
    first = TensorBatchLoader(dataset, 3, True, 47)
    second = TensorBatchLoader(dataset, 3, True, 47)
    rng = torch.random.get_rng_state().clone()
    for _ in range(2):
        batches = list(first)
        assert len(first) == 3 and [len(batch[0]) for batch in batches] == [3, 3, 1]
        actual = torch.cat([batch[0] for batch in batches])
        expected = torch.cat([batch[0] for batch in second])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual[:, 0].sort().values, dataset.tensors[0][:, 0])
    torch.testing.assert_close(torch.random.get_rng_state(), rng)


def test_cached_calibration_skips_source_forwards_and_keeps_frozen_decoder(monkeypatch):
    calibration = make_regularizer().calibration
    model = ToyModel()
    def no_calibration(*args):
        raise AssertionError("Cached source calibration must not run encoder forwards")
    monkeypatch.setattr(model.encoder, "forward", no_calibration)
    sampler = GeometryBatchSampler(TensorDataset(toy_data()), 4, 7)
    from domain_knowledge_analysis.regularizers.target_geometry import TargetGeometryRegularizer
    regularizer = TargetGeometryRegularizer(model, sampler, calibration_batches=4, calibration=calibration)
    assert regularizer.calibration == calibration
    assert regularizer.source_decoder is not model.decoder
    assert all(not parameter.requires_grad for parameter in regularizer.source_decoder.parameters())


def test_calibration_cache_publishes_once_and_does_not_publish_failures(tmp_path):
    from domain_knowledge_analysis.utils.target_geometry_runtime import calibration_cache
    with calibration_cache(tmp_path, "source") as cached:
        assert cached["calibration"] is None
        cached["calibration"] = {"source_tolerance": .125}
    with calibration_cache(tmp_path, "source") as cached:
        assert cached["calibration"] == {"source_tolerance": .125}
    with pytest.raises(RuntimeError):
        with calibration_cache(tmp_path, "failed") as cached:
            cached["calibration"] = {"source_tolerance": .5}
            raise RuntimeError("calibration interrupted")
    assert not (tmp_path / "failed.json").exists()


def test_calibration_identity_reuses_eta_and_dual_but_rejects_changed_inputs(tmp_path):
    from types import SimpleNamespace
    from domain_knowledge_analysis.utils.target_geometry_runtime import calibration_cache_key
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"original source checkpoint")
    experiment = SimpleNamespace(pretrained_model_path=checkpoint, device=torch.device("cpu"),
                                 source_config={"seed": 17}, config={"seed": 42}, model=ToyModel())
    config = {"eta": 100., "view": "full", "dual_learning_rate": .01}
    original = calibration_cache_key(experiment, "source_images", 64, config)
    config.update(eta=1000., view="source_only", dual_learning_rate=.1)
    assert calibration_cache_key(experiment, "source_images", 64, config) == original
    assert calibration_cache_key(experiment, "different_images", 64, config) != original
    assert calibration_cache_key(experiment, "source_images", 128, config) != original
    checkpoint.write_bytes(b"changed source checkpoint")
    assert calibration_cache_key(experiment, "source_images", 64, config) != original


def test_device_validation_reuses_only_inputs_and_matches_original_objective():
    trainer = make_trainer()
    original = trainer.validate_epoch()
    # Exercise the device path's algebra with CPU tensors; no CUDA needed here.
    trainer._cuda = True
    optimized = trainer.validate_epoch()
    assert optimized == original
    bank = trainer._validation_geometry
    with torch.no_grad():
        trainer.model.decoder.scale.fill_(.7)
    updated = trainer.validate_epoch()
    assert trainer._validation_geometry is bank
    trainer._cuda = False
    assert updated == trainer.validate_epoch()


def test_device_training_finite_check_path_matches_reference():
    optimized, reference = make_trainer(), make_trainer()
    optimized._cuda = True
    expected = reference_train_epoch(reference)
    actual, _ = optimized.train_epoch()
    assert actual == pytest.approx(expected, rel=1e-13, abs=1e-13)
    for parameter, other in zip(optimized.model.parameters(), reference.model.parameters()):
        torch.testing.assert_close(parameter, other, rtol=1e-13, atol=1e-13)
