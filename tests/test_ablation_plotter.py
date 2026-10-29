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

    return results


def test_ablation_plotter_saves_the_four_metric_figures(tmp_path):
    plotter = AblationPlotter(tmp_path, {})
    plotter.plot_figure_1(make_results())
    plotter.plot_figure_2(make_results())
    plotter.plot_figure_3(make_results())
    plotter.plot_figure_4(make_results())
    plotter.plot_fid(make_results())

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
