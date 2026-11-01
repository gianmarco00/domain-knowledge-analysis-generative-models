from itertools import product
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from domain_knowledge_analysis.experiments.target_geometry_ablation import TargetGeometryAblation
from domain_knowledge_analysis.plotting.target_geometry_plotter import TargetGeometryPlotter


def metric_rows():
    rows = []
    for seed, reference in ((42, 1.), (43, 2.)):
        rows.append({"view": "ordinary_lora", "seed": seed, "eta": 0.,
                     "test_source_degradation": .04 * reference, "test_target_gain": .08 * reference})
        for eta, size, rate in product((10., 100., 1000.), (64, 128), (.001, .01, .1)):
            # Seed 42: retention=50%, gain=75%. Seed 43: 75%, 100%.
            rows.append({"view": "full", "seed": seed, "eta": eta,
                         "geometry_batch_size": size, "dual_learning_rate": rate,
                         "test_source_degradation": .02, "test_target_gain": .06 if seed == 42 else .16})
    # Deliberately extreme control: it must never enter the full-method curves.
    rows.append({"view": "target_only", "seed": 42, "eta": 100., "geometry_batch_size": 64,
                 "dual_learning_rate": .01, "test_source_degradation": 100., "test_target_gain": -100.})
    return rows


def test_metric_normalization_uses_the_same_seed_and_retention_direction():
    rows = metric_rows()
    baseline = rows[0]
    proposed = rows[1]
    assert TargetGeometryPlotter.metric_value(proposed, baseline, "source_retention") == pytest.approx(50.)
    assert TargetGeometryPlotter.metric_value(proposed, baseline, "target_gain") == pytest.approx(75.)
    assert TargetGeometryPlotter.metric_value(baseline, baseline, "source_retention") == 0.
    assert TargetGeometryPlotter.metric_value(baseline, baseline, "target_gain") == 100.
    retained = {**proposed, "test_source_degradation": 0.}
    assert TargetGeometryPlotter.metric_value(retained, baseline, "source_retention") == 100.


def test_two_eta_panels_per_dual_rate_with_batch_size_colors_and_seed_error_bars(tmp_path, monkeypatch):
    figures = []
    real_close = plt.close
    monkeypatch.setattr(plt, "close", figures.append)
    paths = TargetGeometryPlotter(tmp_path, {}).plot_metric_pairs(metric_rows())
    assert [path.name for path in paths] == [f"eta_retention_gain_dual_{rate:g}.png" for rate in (.001, .01, .1)]
    assert all(path.stat().st_size > 1000 for path in paths)
    assert len(figures) == 3
    for figure in figures:
        assert len(figure.axes) == 2
        retention, gain = figure.axes
        assert retention.get_ylabel() == "Source retention (%) ↑"
        assert gain.get_ylabel() == "Target gain (%) ↑"
        assert all(axis.get_xscale() == "log" for axis in figure.axes)
        colors = []
        for axis, expected_mean in ((retention, 62.5), (gain, 87.5)):
            assert len(axis.containers) == 2
            for series in axis.containers:
                line = series.lines[0]
                assert list(line.get_xdata()) == [10., 100., 1000.]
                assert list(line.get_ydata()) == pytest.approx([expected_mean] * 3)
                # The STD of 50/75 (or 75/100) is sqrt(312.5).
                lower, upper = series.lines[2][0].get_segments()[0][:, 1]
                assert upper - lower == pytest.approx(2 * (312.5 ** .5))
            colors.append([series.lines[0].get_color() for series in axis.containers])
        assert colors[0] == colors[1]
        assert colors[0][0] != colors[0][1]
        real_close(figure)


def test_nonpositive_ordinary_changes_fall_back_to_raw_mse_without_division(tmp_path, monkeypatch):
    rows = metric_rows()
    for row in rows:
        if row["view"] == "ordinary_lora":
            row["test_source_degradation"] = 0.
            row["test_target_gain"] = -.02
    figures = []
    real_close = plt.close
    monkeypatch.setattr(plt, "close", figures.append)
    TargetGeometryPlotter(tmp_path, {}).plot_metric_pairs(rows)
    for figure in figures:
        assert all("MSE change" in axis.get_ylabel() for axis in figure.axes)
        assert list(figure.axes[0].containers[0].lines[0].get_ydata()) == pytest.approx([-.02] * 3)
        assert list(figure.axes[1].containers[0].lines[0].get_ydata()) == pytest.approx([.11] * 3)
        real_close(figure)


def test_missing_or_duplicate_same_seed_baseline_is_rejected(tmp_path):
    rows = metric_rows()
    with pytest.raises(ValueError, match="Every plotted seed"):
        TargetGeometryPlotter(tmp_path, {}).plot_metric_pairs([row for row in rows if row is not rows[0]])
    with pytest.raises(ValueError, match="Multiple ordinary LoRA"):
        TargetGeometryPlotter(tmp_path, {}).plot_metric_pairs([*rows, rows[0]])


class FakeModel:
    encoder_params = {"latent_dim": 2}

    def __init__(self, offset):
        self.offset = offset
        self.enabled = True
        self.latent_calls = []
        self.reconstruction_calls = []

    def reconstruct_images(self, images):
        self.reconstruction_calls.append(images.clone())
        return .25 * images + (self.offset if self.enabled else .05)

    def generate_images(self, count, latents):
        self.latent_calls.append(latents.clone())
        images = latents[:, 0].sigmoid().reshape(count, 1, 1, 1).expand(-1, 1, 4, 4)
        return images * .25 + (self.offset if self.enabled else .05)


def qualitative_ablation(monkeypatch, tmp_path):
    ablation = TargetGeometryAblation.__new__(TargetGeometryAblation)
    ablation.output_dir = tmp_path
    ablation.ablation_config = {"num_qualitative_images": 2, "num_generation_images": 3, "visual_seed": 123}
    selected = {"model_id": "selected", "seed": 42, "eta": 100., "geometry_batch_size": 64, "dual_learning_rate": .01}
    ablation.runs = [{"model_id": "baseline", "seed": 42, "eta": 0.}, selected]
    models = {"baseline": FakeModel(.15), "selected": FakeModel(.25)}
    closed = []
    def load(run):
        model = models[run["model_id"]]
        manager = SimpleNamespace(disable_adapters=lambda: setattr(model, "enabled", False),
                                  enable_adapters=lambda: setattr(model, "enabled", True))
        return SimpleNamespace(model=model, lora_manager=manager,
                               config={"lora": {"transformed_dataset": {"name": "rotated"}}},
                               logger=SimpleNamespace(close=lambda: closed.append(run["model_id"])))
    ablation.load_or_train = load
    images = torch.arange(160).float().reshape(10, 1, 4, 4) / 160
    source = DataLoader(TensorDataset(images), batch_size=4)
    target = DataLoader(TensorDataset(images.flip(-1)), batch_size=4)
    import domain_knowledge_analysis.experiments.target_geometry_ablation as module
    monkeypatch.setattr(module.utils, "create_testing_dataloaders", lambda config, transform=None: target if transform else source)
    return ablation, selected, models, closed


def test_qualitative_outputs_use_identical_inputs_and_latents_and_are_reproducible(tmp_path, monkeypatch):
    ablation, selected, models, closed = qualitative_ablation(monkeypatch, tmp_path)
    rng_state = torch.random.get_rng_state()
    first = ablation.create_qualitative_results(selected)
    second = ablation.create_qualitative_results(selected)
    torch.testing.assert_close(first["source"]["input"], second["source"]["input"])
    torch.testing.assert_close(first["target"]["input"], first["source"]["input"].flip(-1))
    torch.testing.assert_close(first["latents"], second["latents"])
    torch.testing.assert_close(torch.random.get_rng_state(), rng_state)
    for model in models.values():
        for latents in model.latent_calls:
            torch.testing.assert_close(latents, first["latents"])
    for split in ("source", "target"):
        bank = first[split]
        torch.testing.assert_close(bank["base"], bank["input"] * .25 + .05)
        torch.testing.assert_close(bank["lora"], bank["input"] * .25 + .15)
        torch.testing.assert_close(bank["regularized"], bank["input"] * .25 + .25)
    assert len(first["generation"]["base"]) == 3
    assert len(closed) == 4


def test_source_target_reconstructions_and_generations_render_and_save_sample_banks(tmp_path, monkeypatch):
    ablation, selected, _, _ = qualitative_ablation(monkeypatch, tmp_path)
    paths = ablation.plot_reconstructions(selected)
    assert {path.name for path in paths} == {"figure_5_source_reconstructions.png", "figure_6_target_reconstructions.png", "figure_7_generations.png"}
    assert all(path.stat().st_size > 1000 for path in paths)
    saved = torch.load(tmp_path / "results" / "qualitative_examples.pt", weights_only=True)
    assert set(saved) >= {"source", "target", "generation", "latents", "selected_model"}
    assert saved["latents"].shape == (3, 2)
    assert saved["selected_model"] == selected
