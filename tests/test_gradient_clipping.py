"""Optimizer-level clipping must preserve the objective and the disabled path."""

from copy import deepcopy

import pytest
import torch

from domain_knowledge_analysis.training import Trainer
from domain_knowledge_analysis.training.gradient_clipping import (
    clip_optimizer_gradients, gradient_clipping_max_norm,
)
from domain_knowledge_analysis.utils import load_config
from test_target_geometry import make_trainer
from test_target_geometry_performance import reference_train_epoch


@pytest.mark.parametrize("settings", [None, {}, {"enabled": False, "max_norm": 100.}])
def test_disabled_clipping_preserves_exact_adam_updates_and_rng(settings):
    trainer = make_trainer()
    trainer.gradient_clipping_max_norm = gradient_clipping_max_norm(settings)
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=.01)
    parameters = list(trainer.model.parameters())
    expected = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
    optimizer = torch.optim.Adam(expected, lr=.01)
    rng = torch.random.get_rng_state().clone()
    for magnitude in (1., 1e10, .3):
        for actual, reference in zip(parameters, expected):
            actual.grad = torch.full_like(actual, magnitude)
            reference.grad = actual.grad.clone()
        trainer._optimizer_step()
        optimizer.step()
        for actual, reference in zip(parameters, expected):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            torch.testing.assert_close(actual.grad, reference.grad, rtol=0, atol=0)
            for key, value in trainer.optimizer.state[actual].items():
                torch.testing.assert_close(value, optimizer.state[reference][key], rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)


@pytest.mark.parametrize("max_norm", [0., -1., float("nan"), float("inf"), True, "100"])
def test_invalid_threshold_is_rejected(max_norm):
    with pytest.raises(ValueError, match="max_norm"):
        gradient_clipping_max_norm({"enabled": True, "max_norm": max_norm})


@pytest.mark.parametrize("settings", [True, {"enabled": "false"}, {"unknown": 1}])
def test_invalid_switch_or_typo_is_rejected(settings):
    with pytest.raises(ValueError, match="gradient_clipping"):
        gradient_clipping_max_norm(settings)


def test_global_norm_clips_before_adam_and_handles_finite_float32_norm_overflow():
    first = torch.nn.Parameter(torch.zeros(1))
    second = torch.nn.Parameter(torch.zeros(1))
    unused = torch.nn.Parameter(torch.zeros(1))
    # Separate groups must share ONE clipping coefficient, rather than each
    # receiving an independent norm budget. Parameters without gradients stay idle.
    optimizer = torch.optim.Adam([{"params": [first]}, {"params": [second, unused]}], lr=.01)
    first.grad = torch.tensor([3e30])
    second.grad = torch.tensor([4e30])
    assert torch.isinf(torch.cat([first.grad, second.grad]).norm())
    clip_optimizer_gradients(optimizer, 100.)
    torch.testing.assert_close(first.grad, torch.tensor([60.]))
    torch.testing.assert_close(second.grad, torch.tensor([80.]))
    assert unused.grad is None
    assert not optimizer.state
    optimizer.step()
    for parameter in (first, second):
        assert torch.isfinite(parameter).all()
        assert all(torch.isfinite(value).all() for value in optimizer.state[parameter].values())
        torch.testing.assert_close(optimizer.state[parameter]["exp_avg"], .1 * parameter.grad)
        torch.testing.assert_close(optimizer.state[parameter]["exp_avg_sq"], .001 * parameter.grad.square())
    assert unused not in optimizer.state


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_gradients_are_rejected_before_scaling_or_moment_update(bad):
    parameters = [torch.nn.Parameter(torch.ones(1)) for _ in range(2)]
    optimizer = torch.optim.Adam(parameters, lr=.01)
    parameters[0].grad = torch.tensor([3.])
    parameters[1].grad = torch.tensor([bad])
    with pytest.raises(RuntimeError, match="No optimizer update"):
        clip_optimizer_gradients(optimizer, 100.)
    assert parameters[0].grad.item() == 3.
    assert not optimizer.state
    assert all(parameter.item() == 1. for parameter in parameters)


def test_generic_baseline_training_clips_and_leaves_smaller_gradients_unchanged():
    geometry_trainer = make_trainer()
    trainer = Trainer(
        model=geometry_trainer.model, train_dataloader=geometry_trainer.train_dataloader,
        validate_dataloader=geometry_trainer.validate_dataloader,
        optimizer=torch.optim.Adam(geometry_trainer.model.parameters(), lr=.01),
        lr_scheduler=None, lr_scheduler_start_epoch=0, loss=geometry_trainer.loss,
        epochs=1, device="cpu", gradient_clipping={"enabled": True, "max_norm": .001},
    )
    norms = []
    trainer.optimizer.register_step_pre_hook(
        lambda optimizer, *_: norms.append(torch.cat([
            p.grad.flatten() for group in optimizer.param_groups for p in group["params"]
            if p.grad is not None]).norm().item())
    )
    trainer.train_epoch()
    assert len(norms) == len(trainer.train_dataloader)
    assert all(0 < norm <= .001 for norm in norms)
    for parameter in trainer.model.parameters():
        parameter.grad.fill_(1e-7)
    gradients = [parameter.grad.clone() for parameter in trainer.model.parameters()]
    trainer._optimizer_step()
    for parameter, gradient in zip(trainer.model.parameters(), gradients):
        torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)


@pytest.mark.parametrize("view", ["full", "target_only", "source_only"])
def test_combined_primal_gradient_is_clipped_once_with_unchanged_loss_dual_and_sampler(view):
    actual, expected = make_trainer(), make_trainer()
    for trainer in (actual, expected):
        trainer.regularizer.view = view
        trainer.regularizer.dual_fraction = .4
        trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=.01)
    actual.gradient_clipping_max_norm = gradient_clipping_max_norm({"enabled": True, "max_norm": .01})
    # Independent reference: original full-primal backward, standard PyTorch
    # global clipping on these float64 toy parameters, then ordinary Adam.
    def reference_clip(optimizer, *_):
        torch.nn.utils.clip_grad_norm_(
            [p for group in optimizer.param_groups for p in group["params"]], .01)
    expected.optimizer.register_step_pre_hook(reference_clip)
    for _ in range(3):
        reference_loss = reference_train_epoch(expected)
        actual_loss, _ = actual.train_epoch()
        assert actual_loss == pytest.approx(reference_loss, abs=1e-13, rel=1e-13)
        assert actual.regularizer.dual_fraction == expected.regularizer.dual_fraction
        assert actual.regularizer.source_tolerance == expected.regularizer.source_tolerance
        for p, ref in zip(actual.model.parameters(), expected.model.parameters()):
            torch.testing.assert_close(p, ref, rtol=1e-13, atol=1e-13)
            torch.testing.assert_close(p.grad, ref.grad, rtol=1e-13, atol=1e-13)
            for key, value in actual.optimizer.state[p].items():
                torch.testing.assert_close(value, expected.optimizer.state[ref][key], rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(actual.geometry_sampler.generator.get_state(),
                                   expected.geometry_sampler.generator.get_state(), rtol=0, atol=0)


def test_upper_emnist_sweep_clips_every_model_and_cannot_reuse_unclipped_models(tmp_path):
    from domain_knowledge_analysis.experiments.target_geometry_ablation import generate_target_geometry_runs
    sweep = load_config("config/ablation_vae_mnist_upper_emnist_target_geometry.yaml")
    base = load_config(sweep["base_config"])
    assert base["training"]["learning_rate"] == .01
    assert base["training"]["gradient_clipping"] == {"enabled": True, "max_norm": 100.}
    runs = generate_target_geometry_runs(base, sweep["ablation"], tmp_path / "clipped")
    assert len(runs) == 26
    assert {run["view"] for run in runs} == {"full", "target_only", "source_only", "ordinary_lora"}
    for run in runs:
        assert load_config(run["config_path"])["training"] == base["training"]
    old = deepcopy(base)
    del old["training"]["gradient_clipping"]
    previous = generate_target_geometry_runs(old, sweep["ablation"], tmp_path / "unclipped")
    assert {run["model_id"] for run in runs}.isdisjoint(run["model_id"] for run in previous)
    changed = deepcopy(base)
    changed["training"]["gradient_clipping"]["max_norm"] = 10.
    changed_runs = generate_target_geometry_runs(changed, sweep["ablation"], tmp_path / "different_cap")
    assert {run["model_id"] for run in runs}.isdisjoint(run["model_id"] for run in changed_runs)
