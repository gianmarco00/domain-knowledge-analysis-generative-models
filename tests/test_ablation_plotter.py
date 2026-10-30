import math

import matplotlib.pyplot as plt
import pytest
import torch

from domain_knowledge_analysis.plotting import AblationPlotter


def make_results():
    results = [
        {"mode": "ordinary_lora", "eta": 0.0, "m": 0, "s": 0.0, "test_target_gain": 0.08, "test_source_degradation": 0.04, "test_target_fid": 100.0},
        {"mode": "source_loss", "eta": 1.0, "m": 0, "s": 0.0, "test_target_gain": 0.075, "test_source_degradation": 0.01, "test_target_fid": 95.0},
    ]

    for eta in [10.0, 100.0]:
        for m in [5, 10]:
            for strength in [0.01, 0.5]:
                results.append({
                    "mode": "output_gram",
                    "eta": eta,
                    "m": m,
                    "s": strength,
                    "test_target_gain": 0.08 - eta / 100000 - strength / 1000,
                    "test_source_degradation": 0.04 - m / 1000,
                    "test_target_fid": 90.0,
                })

    for row in results:
        row.update(test_base_target_mse=0.2, test_base_source_mse=0.1)

    return results


def test_ablation_plotter_saves_the_four_metric_figures(tmp_path):
    plotter = AblationPlotter(tmp_path, {})
    plotter.plot_figure_1(make_results())
    selected = {"eta": 100.0, "s": 0.5}
    plotter.plot_figure_2(make_results(), selected)
    plotter.plot_figure_3(make_results(), selected)
    plotter.plot_figure_4(make_results())
    plotter.plot_fid(make_results(), selected)

    assert (tmp_path / "figures" / "figure_1_eta.png").is_file()
    assert (tmp_path / "figures" / "figure_2_transformation_count.png").is_file()
    assert (tmp_path / "figures" / "figure_3_transformation_strength.png").is_file()
    assert (tmp_path / "figures" / "figure_4_target_source_tradeoff.png").is_file()
    assert (tmp_path / "figures" / "figure_9_target_fid.png").is_file()


def test_ablation_plotter_saves_the_four_qualitative_figures(tmp_path):
    selected_model = {"mode": "output_gram", "m": 10, "eta": 1000.0, "s": 0.1}
    images = torch.rand(2, 1, 8, 8)
    qualitative_results = {
        "source": {"input": images, "base": images, "lora": images, "source_loss": images, "regularized": images},
        "target": {"input": images, "base": images, "lora": images, "source_loss": images, "regularized": images},
        "generation": {"base": images, "lora": images, "source_loss": images, "regularized": images},
        "responses": {
            "input": images,
            "transformed": torch.rand(2, 2, 1, 8, 8),
            "lora": torch.rand(2, 2, 1, 8, 8) - 0.5,
            "regularized": torch.rand(2, 2, 1, 8, 8) - 0.5,
            "transformation_labels": ["horizontal translation (2)", "isotropic scale (1.1)"],
            "selected_model": selected_model,
        },
        "selected_model": selected_model,
    }

    AblationPlotter(tmp_path, {}).plot_qualitative(qualitative_results)

    assert (tmp_path / "figures" / "figure_5_source_reconstructions.png").is_file()
    assert (tmp_path / "figures" / "figure_6_target_reconstructions.png").is_file()
    assert (tmp_path / "figures" / "figure_7_generations.png").is_file()
    assert (tmp_path / "figures" / "figure_8_model_responses.png").is_file()


def test_selected_model_label_does_not_expose_transformation_type():
    label = AblationPlotter.selected_model_label({"mode": "output_gram", "m": 10, "eta": 1000.0, "s": 0.1})

    assert label == "η=1000, M=10, s=0.1"
    assert "semantic" not in label


def test_qualitative_plots_use_configured_dataset_labels(tmp_path, monkeypatch):
    plotter = AblationPlotter(tmp_path, {"source_label": "Horizontal lines", "target_label": "Rotated horizontal lines"})
    titles = []
    monkeypatch.setattr(plotter, "plot_reconstructions", lambda results, title, filename, label: titles.append(title))
    monkeypatch.setattr(plotter, "plot_generations", lambda results, label: None)
    monkeypatch.setattr(plotter, "plot_model_responses", lambda results: None)

    plotter.plot_qualitative({
        "source": {},
        "target": {},
        "generation": {},
        "responses": {},
        "selected_model": {"eta": 10, "m": 5, "s": 0.1},
    })

    assert titles == ["Horizontal lines reconstructions", "Rotated horizontal lines reconstructions"]


def test_percentages_use_ordinary_lora_for_the_same_seed():
    baselines = {1: {"test_target_gain": 0.08, "test_source_degradation": 0.04},
                 2: {"test_target_gain": 0.04, "test_source_degradation": 0.02}}
    row = {"seed": 2, "test_target_gain": 0.06, "test_source_degradation": -0.01}

    assert AblationPlotter.percentage(row, "test_target_gain", baselines) == pytest.approx(150)
    assert AblationPlotter.percentage(row, "test_source_degradation", baselines) == pytest.approx(-50)
    assert AblationPlotter.percentage({"seed": 1, **baselines[1]}, "test_target_gain", baselines) == 100
    baselines[2]["test_source_degradation"] = 0
    assert math.isnan(AblationPlotter.percentage(row, "test_source_degradation", baselines))
    with pytest.raises(ValueError, match="No ordinary LoRA baseline"):
        AblationPlotter.percentage({"seed": 3}, "test_target_gain", baselines)


def test_plots_use_selected_parameters_and_connect_comparable_runs(tmp_path, monkeypatch):
    results = make_results()
    selected = {"eta": 100.0, "s": 0.5}
    plotter = AblationPlotter(tmp_path, {})
    figures = []
    monkeypatch.setattr(plt.Figure, "savefig", lambda figure, *args, **kwargs: figures.append(figure))
    monkeypatch.setattr(plt, "close", lambda *args: None)

    plotter.plot_figure_1(results)
    eta = figures[-1]
    assert len(eta.axes) == 4
    assert [axis.get_title() for axis in eta.axes] == ["s = 0.01", "s = 0.01", "s = 0.5", "s = 0.5"]
    for axis in eta.axes:
        # Two M series, each with eta=10 and 100; horizontal guides follow.
        assert [list(line.get_xdata()) for line in axis.lines[:2]] == [[10.0, 100.0], [10.0, 100.0]]
        assert [len(collection.get_offsets()) for collection in axis.collections[:2]] == [2, 2]

    plotter.plot_figure_2(results, selected)
    for axis in figures[-1].axes:
        assert [list(line.get_xdata()) for line in axis.lines[:2]] == [[5.0, 10.0], [5.0, 10.0]]
        assert [len(collection.get_offsets()) for collection in axis.collections[:2]] == [2, 2]
        # M=5 at eta=10: its target gain includes only s=0.5.
    assert figures[-1].axes[0].collections[0].get_offsets()[0, 1] == pytest.approx(100 * (0.08 - 10 / 100000 - 0.5 / 1000) / 0.08)

    plotter.plot_figure_3(results, selected)
    for axis in figures[-1].axes:
        assert [list(line.get_xdata()) for line in axis.lines[:2]] == [[0.01, 0.5], [0.01, 0.5]]
        assert [len(collection.get_offsets()) for collection in axis.collections[:2]] == [2, 2]
    assert figures[-1].axes[0].collections[0].get_offsets()[0, 1] == pytest.approx(100 * (0.08 - 100 / 100000 - 0.01 / 1000) / 0.08)

    plotter.plot_fid(results, selected)
    fid = figures[-1]
    assert [len(collection.get_offsets()) for collection in fid.axes[0].collections[:2]] == [2, 2]
    assert not fid.axes[0].lines
    assert fid.legends[0].get_title().get_text() == "M"
    assert "s=0.5" in fid._suptitle.get_text()


def test_lines_do_not_connect_different_seeds(tmp_path, monkeypatch):
    results = make_results()
    for row in results:
        row["seed"] = 1
    extra = [dict(row, seed=2) for row in results]
    figures = []
    monkeypatch.setattr(plt.Figure, "savefig", lambda figure, *args, **kwargs: figures.append(figure))
    monkeypatch.setattr(plt, "close", lambda *args: None)
    AblationPlotter(tmp_path, {}).plot_figure_1(results + extra)
    # Each M has one line per seed rather than one line across seeds.
    assert [list(line.get_xdata()) for line in figures[0].axes[0].lines[:4]] == [[10.0, 100.0]] * 4
