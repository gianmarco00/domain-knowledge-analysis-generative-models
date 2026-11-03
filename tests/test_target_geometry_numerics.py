from copy import deepcopy

import pytest
import torch

from domain_knowledge_analysis.losses import VAELoss
from domain_knowledge_analysis.math.bernoulli import bernoulli_log_prob_from_logits
from domain_knowledge_analysis.regularizers.target_geometry import centered_gram, geometry_distortion
from domain_knowledge_analysis.training.target_geometry_numerics import (
    OptimizerStepSnapshot, capture_rng, diagnose_geometry, invalid_named_tensors,
    trace_failure, trace_primal_failure,
)
from test_target_geometry import make_trainer, toy_data
from test_target_geometry_performance import reference_train_epoch


def overflow_trainer(monkeypatch, tmp_path=None):
    trainer = make_trainer(tmp_path)
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=.01)
    trainer.loss = VAELoss(bernoulli_log_prob_from_logits, beta=1.7)
    original = trainer.model.encoder.forward

    def forward(images):
        mean, _ = original(images)
        # Real KL formula, finite posterior means/logits, but exp(log_variance)
        # overflows. Calibration and deterministic geometry never use variance.
        log_variance = torch.ones_like(mean) * (1000 * trainer.model.encoder.map.weight[0, 0])
        return mean, log_variance

    monkeypatch.setattr(trainer.model.encoder, "forward", forward)
    return trainer


def test_unchecked_kl_overflow_can_cause_the_reported_geometry_error_on_the_next_batch(monkeypatch):
    trainer = overflow_trainer(monkeypatch)
    assert torch.isfinite(trainer.regularizer.distortions(toy_data())["worst_distortion"])
    with pytest.raises(ValueError, match="Nonfinite target geometry distortion"):
        reference_train_epoch(trainer)
    assert not torch.isfinite(trainer.model.encoder.map.weight).all()


@pytest.mark.parametrize("clipping_enabled", [False, True])
def test_nonfinite_vae_loss_is_rejected_before_adam_corrupts_weights_or_dual(monkeypatch, tmp_path, clipping_enabled):
    trainer = overflow_trainer(monkeypatch, tmp_path)
    trainer.gradient_clipping_max_norm = 100. if clipping_enabled else None
    before = deepcopy(trainer.model.state_dict())
    dual = trainer.regularizer.dual_fraction
    with pytest.raises(ValueError, match="nonfinite_forward.*before optimizer.step"):
        trainer.train_epoch()
    for name, value in before.items():
        torch.testing.assert_close(trainer.model.state_dict()[name], value, rtol=0, atol=0)
    assert trainer.regularizer.dual_fraction == dual
    assert not trainer.optimizer.state
    snapshot = torch.load(tmp_path / "checkpoints" / "nonfinite.pt", map_location="cpu", weights_only=True)
    report = snapshot["report"]
    assert report["stage"] == "nonfinite_forward"
    assert report["scalars"]["KL Loss"] == float("inf")
    assert report["training_posterior_log_variance"]["inf_count"] == 0
    assert report["training_posterior_variance"]["inf_count"] > 0
    assert report["geometry"]["target_image_mean"]["nan_count"] == 0
    assert report["nonfinite_parameters"] == []
    assert report["operation_trace"]["operator"] == "aten.exp.default"
    assert report["operation_trace"]["phase"] == "vae_forward"
    assert snapshot["regularizer"]["dual_fraction"] == dual
    assert not (tmp_path / "checkpoints" / "best.pt").exists()
    assert not (tmp_path / "checkpoints" / "last.pt").exists()


@pytest.mark.parametrize("clipping_enabled", [False, True])
def test_finite_loss_with_nonfinite_gradient_is_rejected_without_clipping_or_update(tmp_path, clipping_enabled):
    trainer = make_trainer(tmp_path)
    trainer.gradient_clipping_max_norm = 100. if clipping_enabled else None
    before = deepcopy(trainer.model.state_dict())
    handle = trainer.model.encoder.map.weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))
    try:
        with pytest.raises(ValueError, match="nonfinite_backward.*before optimizer.step"):
            trainer.train_epoch()
    finally:
        handle.remove()
    for name, value in before.items():
        torch.testing.assert_close(trainer.model.state_dict()[name], value, rtol=0, atol=0)
    assert trainer.regularizer.dual_fraction == 0
    snapshot = torch.load(tmp_path / "checkpoints" / "nonfinite.pt", map_location="cpu", weights_only=True)
    assert snapshot["report"]["nonfinite_gradients"] == ["encoder.map.weight"]
    assert torch.isinf(snapshot["gradients"]["encoder.map.weight"]).all()


def test_finiteness_guard_does_not_reject_large_finite_gradients_or_change_values():
    gradient = torch.full((16,), 1e30)
    before = gradient.clone()
    assert torch.isinf(gradient.norm())  # A norm-based float32 guard would falsely reject it.
    assert invalid_named_tensors([("large", gradient), ("unused", None)]) == []
    torch.testing.assert_close(gradient, before, rtol=0, atol=0)


def test_reporting_only_overflow_does_not_reject_a_finite_objective():
    trainer = make_trainer()
    loss = trainer.loss

    class ReportingOverflow:
        def __call__(self, *args):
            return loss(*args)
        def components(self):
            return {"large_report_only": torch.full((128,), 1e308, dtype=torch.float64)}

    trainer.loss = ReportingOverflow()
    value, components = trainer.train_epoch()
    assert torch.isfinite(torch.tensor(value))
    assert components["large_report_only"] == float("inf")


def test_failure_diagnostics_identify_the_decoder_and_preserve_modes_rng_and_gradients():
    trainer = make_trainer()
    trainer.model.encoder.eval()
    with torch.no_grad():
        trainer.model.decoder.scale.fill_(float("nan"))
    modes = [module.training for module in trainer.model.modules()]
    state = torch.random.get_rng_state().clone()
    report = diagnose_geometry(trainer.regularizer, toy_data())
    assert report["posterior_mean"]["nan_count"] == 0
    assert report["target_decoder_logits"]["nan_count"] > 0
    assert report["target_first_nonfinite_activation"]["module"] == "ToyDecoder"
    assert report["source_chart_image_mean"]["nan_count"] == 0
    assert [module.training for module in trainer.model.modules()] == modes
    assert all(parameter.grad is None for parameter in trainer.model.parameters())
    torch.testing.assert_close(torch.random.get_rng_state(), state)


def test_ready_target_only_config_retains_the_regularizer_and_uses_requested_adaptation_lr():
    from domain_knowledge_analysis.utils import load_config
    source = load_config("config/lora_vae_mnist_emnist.yaml")
    target = load_config("config/lora_vae_mnist_emnist_target_only.yaml")
    assert target["training"] == {**source["training"], "learning_rate": .01}
    expected = {**source["lora"]["regularizer"], "view": "target_only"}
    assert target["lora"]["regularizer"] == expected
    for key in ("model", "loss", "optimizer", "lr_scheduler", "dataloader", "dataset"):
        assert target[key] == source[key]


@pytest.mark.parametrize("case", ["perfect", "collapsed", "saturated", "small_input_energy"])
def test_gram_gradient_has_no_singularity_at_zero_output_energy_or_zero_distortion(case):
    images = toy_data()
    if case == "small_input_energy":
        scale = (1.1e-12 / centered_gram(images).square().sum()).pow(.25)
        images = .5 + scale * (images - .5)
    if case == "saturated":
        logits = torch.tensor([-1000., 1000.], dtype=torch.float64).repeat(4).reshape_as(images).requires_grad_()
    else:
        outputs = torch.full_like(images, .5) if case == "collapsed" else toy_data()
        logits = outputs.logit().requires_grad_()
    gram = centered_gram(images)
    outputs = logits.sigmoid()
    value = geometry_distortion(gram, outputs)
    gradient, = torch.autograd.grad(value, logits)
    assert torch.isfinite(value)
    assert torch.isfinite(gradient).all()
    if case == "collapsed":
        assert value.item() == pytest.approx(1.)
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), rtol=0, atol=0)
    if case == "perfect":
        assert value.item() == pytest.approx(0., abs=1e-28)


def test_failure_trace_identifies_a_real_backward_division_by_zero_without_changing_live_state():
    trainer = make_trainer()
    # Synthetic singularity to verify localization; the real geometry loss has
    # no sqrt and its detached denominator is checked before division.
    loss = lambda x, logits, mean, logvar: (mean - mean.detach()).sum().sqrt()
    before = deepcopy(trainer.model.state_dict())
    rng = capture_rng("cpu")
    report = trace_primal_failure(trainer.regularizer, loss, toy_data(), toy_data(), rng)
    assert report["status"] == "first_nonfinite_operation"
    assert report["phase"] == "primal_backward"
    assert report["operator"] == "aten.div.Tensor"
    assert report["inputs"][-1]["finite_min"] == report["inputs"][-1]["finite_max"] == 0
    for name, value in before.items():
        torch.testing.assert_close(trainer.model.state_dict()[name], value, rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in trainer.model.parameters())
    torch.testing.assert_close(torch.random.get_rng_state(), rng["cpu"], rtol=0, atol=0)


def test_optimizer_trace_reconstructs_first_step_and_finds_zero_denominator():
    parameter = torch.nn.Parameter(torch.tensor([1.]))
    parameter.grad = torch.zeros_like(parameter)
    # Deliberate zero-epsilon Adam, unlike the actual run's positive 1e-8.
    optimizer = torch.optim.Adam([parameter], lr=.01, eps=0)
    snapshot = OptimizerStepSnapshot()
    snapshot.capture(optimizer)
    optimizer.step()
    assert torch.isnan(parameter).all()
    replay = snapshot.replay_optimizer(optimizer)
    report = trace_failure(lambda trace: replay.step(), capture_rng("cpu"))
    assert report["operator"] == "aten.addcdiv_.default"
    assert report["inputs"][0]["finite_min"] == report["inputs"][0]["finite_max"] == 1.
    assert report["inputs"][0]["nan_count"] == 0
    assert report["outputs"][0]["nan_count"] == 1
    assert report["inputs"][-1]["finite_min"] == report["inputs"][-1]["finite_max"] == 0
    assert snapshot.entries[0][1].item() == 1.
    assert not snapshot.entries[0][2]  # State before the original first step.


def test_failed_optimizer_is_detected_in_the_same_batch_and_preserves_pre_step_evidence(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=.01, eps=0)
    trainer.loss = lambda x, logits, mean, logvar: (logits * 0).sum()
    before = deepcopy(trainer.model.state_dict())
    with pytest.raises(ValueError, match="nonfinite_optimizer.*during optimizer.step"):
        trainer.train_epoch()
    assert trainer.regularizer.dual_fraction == 0
    snapshot = torch.load(tmp_path / "checkpoints" / "nonfinite.pt", weights_only=True)
    assert snapshot["report"]["operation_trace"]["phase"] == "optimizer_step"
    assert snapshot["report"]["operation_trace"]["operator"] == "aten.addcdiv_.default"
    assert snapshot["pre_forward_rng_state"] is not None
    assert snapshot["report"]["nonfinite_updated_tensors"]
    for name, item in snapshot["pre_optimizer_state"].items():
        torch.testing.assert_close(item["parameter"], before[name], rtol=0, atol=0)
        assert not item["optimizer_state"]


def test_optimizer_snapshot_reuses_buffers_and_restores_existing_adam_moments():
    parameter = torch.nn.Parameter(torch.tensor([1., 2.]))
    optimizer = torch.optim.Adam([parameter], lr=.01)
    snapshot = OptimizerStepSnapshot()
    for step in range(3):
        parameter.grad = torch.tensor([.1, -.3])
        snapshot.capture(optimizer)
        if step == 1:
            buffer_ids = [id(value) for _, values in snapshot.groups for value in values]
        if step == 2:
            assert buffer_ids == [id(value) for _, values in snapshot.groups for value in values]
            before = deepcopy(optimizer.state_dict())
            before_parameter = parameter.detach().clone()
        optimizer.step()
    replay = snapshot.replay_optimizer(optimizer)
    torch.testing.assert_close(replay.param_groups[0]["params"][0], before_parameter, rtol=0, atol=0)
    for key, value in before["state"][0].items():
        torch.testing.assert_close(replay.state_dict()["state"][0][key], value, rtol=0, atol=0)
    replay.step()
    torch.testing.assert_close(replay.param_groups[0]["params"][0], parameter, rtol=0, atol=0)


def test_failure_replay_uses_pre_forward_randomness_and_restores_post_forward_rng():
    rng = capture_rng("cpu")
    expected = torch.randn(8)
    torch.randn(17)
    current = torch.random.get_rng_state().clone()
    sampled = []

    def fail(trace):
        sampled.append(torch.randn(8))
        torch.ones(1) / torch.zeros(1)

    report = trace_failure(fail, rng)
    assert report["operator"] == "aten.div.Tensor"
    torch.testing.assert_close(sampled[0], expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), current, rtol=0, atol=0)


def test_healthy_stochastic_updates_and_global_rng_match_original_training(monkeypatch):
    from types import MethodType
    optimized, reference = make_trainer(), make_trainer()

    def stochastic_forward(model, images):
        mean, logvar = model.encoder(images)
        return model.decoder(mean + .01 * torch.randn_like(mean)), mean, logvar

    for trainer in (optimized, reference):
        monkeypatch.setattr(trainer.model, "forward", MethodType(stochastic_forward, trainer.model))
        trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=.01)
        trainer.regularizer.dual_fraction = .4
    for _ in range(3):
        initial_rng = torch.random.get_rng_state().clone()
        expected_loss = reference_train_epoch(reference)
        expected_rng = torch.random.get_rng_state().clone()
        torch.random.set_rng_state(initial_rng)
        actual_loss, _ = optimized.train_epoch()
        assert actual_loss == expected_loss
        torch.testing.assert_close(torch.random.get_rng_state(), expected_rng, rtol=0, atol=0)
        assert optimized.regularizer.dual_fraction == reference.regularizer.dual_fraction
        for actual, expected in zip(optimized.model.parameters(), reference.model.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)
            for key, value in optimized.optimizer.state[actual].items():
                torch.testing.assert_close(value, reference.optimizer.state[expected][key], rtol=0, atol=0)
