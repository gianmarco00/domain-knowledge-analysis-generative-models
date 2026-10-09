import pytest
import torch

from domain_knowledge_analysis.plotting import AblationPlotter


def make_results():
    results = []

    for seed in [0, 1]:
        results.append({
            "seed": seed,
            "eta": 0.0,
            "mode": "ordinary_lora",
            "m": 0,
            "subset_index": 0,
            "test_target_gain": 0.08,
            "test_source_degradation": 0.04,
            "held_out_gram_error": 0.03,
        })

        for mode, m_values in {
            "semantic_perturbations": [5, 10],
            "random_perturbations": [5, 10],
            "both": [20],
        }.items():
            for m in m_values:
                num_subsets = 2 if m == 5 else 1

                for subset_index in range(num_subsets):
                    results.append({
                        "seed": seed,
                        "eta": 1.0,
                        "mode": mode,
                        "m": m,
                        "subset_index": subset_index,
                        "test_target_gain": 0.07 + 0.001 * m,
                        "test_source_degradation": 0.03 - 0.001 * m,
                        "held_out_gram_error": 0.02,
                    })

    return results


def test_ablation_plotter_saves_mode_colored_metric_figures(tmp_path):
    config = {
        "transformation_modes": ["semantic_perturbations", "random_perturbations", "both"],
        "eta_values": [0.0, 1.0],
        "m_values": {
            "semantic_perturbations": [5, 10],
            "random_perturbations": [5, 10],
            "both": [20],
        },
    }
    selected_model = {"mode": "semantic_perturbations", "m": 10, "eta": 1.0}

    plotter = AblationPlotter(tmp_path, config)
    plotter.plot_figure_1(make_results(), selected_model)
    plotter.plot_figure_2(make_results(), selected_model)
    plotter.plot_heatmaps(make_results())

    assert (tmp_path / "figures" / "figure_1_eta_tradeoff.png").is_file()
    assert (tmp_path / "figures" / "figure_2_transformation_count.png").is_file()
    assert (tmp_path / "figures" / "supplementary_eta_M_heatmaps.png").is_file()


def test_ablation_plotter_saves_model_response_grid(tmp_path):
    config = {"transformation_modes": [], "eta_values": [], "m_values": {}}
    selected_model = {"mode": "semantic_perturbations", "m": 10, "eta": 1000.0}
    responses = {
        "input": torch.rand(2, 1, 8, 8),
        "transformed": torch.rand(2, 2, 1, 8, 8),
        "lora": torch.rand(2, 2, 1, 8, 8) - 0.5,
        "regularized": torch.rand(2, 2, 1, 8, 8) - 0.5,
        "transformation_labels": ["rotation (15)", "isotropic scale (1.1)"],
        "selected_model": selected_model,
    }

    AblationPlotter(tmp_path, config).plot_model_responses(responses)

    assert (tmp_path / "figures" / "figure_4_model_responses.png").is_file()


def test_ablation_plotter_includes_sparse_extra_runs(tmp_path):
    config = {
        "transformation_modes": ["semantic_perturbations", "random_perturbations", "both"],
        "eta_values": [0.0, 1.0],
        "m_values": {
            "semantic_perturbations": [5, 10],
            "random_perturbations": [5, 10],
            "both": [20],
        },
        "extra_runs": [
            {"seed": 0, "eta": 10.0, "mode": "semantic_no_rotation", "transformations": ["a", "b", "c", "d", "e"]},
            {"seed": 0, "eta": 10.0, "mode": "semantic_perturbations", "transformations": [str(index) for index in range(10)]},
        ],
    }
    results = make_results()
    results.extend([
        {"seed": 0, "eta": 10.0, "mode": "semantic_no_rotation", "m": 5, "subset_index": 0, "test_target_gain": 0.079, "test_source_degradation": 0.018, "held_out_gram_error": 0.015},
        {"seed": 0, "eta": 10.0, "mode": "semantic_perturbations", "m": 10, "subset_index": 0, "test_target_gain": 0.078, "test_source_degradation": 0.017, "held_out_gram_error": 0.014},
    ])
    selected_model = {"mode": "semantic_perturbations", "m": 10, "eta": 10.0}
    plotter = AblationPlotter(tmp_path, config)

    plotter.plot_figure_1(results, selected_model)
    plotter.plot_figure_2(results, selected_model)
    plotter.plot_heatmaps(results)

    assert plotter.eta_values == [0.0, 1.0, 10.0]
    assert "semantic_no_rotation" in plotter.modes
    assert (tmp_path / "figures" / "figure_1_eta_tradeoff.png").is_file()
    assert (tmp_path / "figures" / "figure_2_transformation_count.png").is_file()
    assert (tmp_path / "figures" / "supplementary_eta_M_heatmaps.png").is_file()


def test_mean_ci_uses_student_t_for_three_seeds():
    mean, confidence_interval = AblationPlotter.mean_ci([1.0, 2.0, 3.0])

    assert mean == pytest.approx(2.0)
    assert confidence_interval == pytest.approx(2.484, abs=0.001)
