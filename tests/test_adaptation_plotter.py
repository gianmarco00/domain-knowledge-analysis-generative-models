import matplotlib.pyplot as plt
import pytest
import torch

from domain_knowledge_analysis.plotting import AdaptationPlotter


def make_results(num_images=2):
    def images(value):
        return torch.full((num_images, 1, 5, 5), value)

    return {
        "original_source_images": images(0.1),
        "pretrained_reconstructed_source_images": [image.unsqueeze(0) for image in images(0.2)],
        "adapted_reconstructed_source_images": images(0.3),
        "original_target_images": images(0.4),
        "pretrained_reconstructed_target_images": images(0.5),
        "adapted_reconstructed_target_images": [image for image in images(0.6)],
        "pretrained_source_mse": 0.0123,
        "adapted_source_mse": 0.0234,
        "pretrained_target_mse": 0.0345,
        "adapted_target_mse": 0.0456,
    }


def test_adaptation_plotter_saves_labeled_columns_in_scorer_order(tmp_path, monkeypatch):
    plotter = AdaptationPlotter(tmp_path)
    figures = []
    monkeypatch.setattr(plt, "close", figures.append)

    saved_path = plotter.plot(make_results())

    assert saved_path == tmp_path / "results" / "adaptation_reconstructions.png"
    assert saved_path.is_file()
    assert saved_path.stat().st_size > 0

    figure = figures[0]
    assert len(figure.axes) == 12
    assert [axis.images[0].get_array()[0, 0] for axis in figure.axes[:6]] == pytest.approx(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    )
    assert [axis.get_title() for axis in figure.axes[:6]] == [
        "Original",
        "Pretrained\nreconstruction\nMSE 0.0123",
        "Adapted\nreconstruction\nMSE 0.0234",
        "Original",
        "Pretrained\nreconstruction\nMSE 0.0345",
        "Adapted\nreconstruction\nMSE 0.0456",
    ]
    assert [axis.get_ylabel() for axis in (figure.axes[0], figure.axes[6])] == [
        "Example 1",
        "Example 2",
    ]


def test_adaptation_plotter_rejects_mismatched_image_counts(tmp_path):
    results = make_results()
    results["adapted_reconstructed_target_images"] = results[
        "adapted_reconstructed_target_images"
    ][:1]

    with pytest.raises(ValueError, match="same nonzero number"):
        AdaptationPlotter(tmp_path).plot(results)


def test_adaptation_plotter_wraps_regularizer_configuration_in_footer(tmp_path, monkeypatch):
    lora_config = {
        "regularizer": {
            "transformations": [
                ["horizontal_translation", [-2, 2]],
                ["vertical_translation", [-2, 2]],
                ["rotation", [-15, 15]],
                ["isotropic_scale", [1 / 1.1, 1.1]],
                ["aspect_ratio_deformation", [1 / 1.1, 1.1]],
                ["diagonal_shear", [-0.12, 0.12]],
                ["stroke_thickness", [-1, 1]],
            ],
            "eta": 1.0,
            "num_anchor_points": 300,
        },
    }
    figures = []
    monkeypatch.setattr(plt, "close", figures.append)

    AdaptationPlotter(tmp_path, lora_config).plot(make_results())

    footer_text = figures[0].axes[-1].texts[0].get_text()
    assert "\n" in footer_text
    assert "horizontal_translation: [-2, 2]" in footer_text
    assert "stroke_thickness: [-1, 1]" in footer_text
    assert "eta: 1" in footer_text
    assert "anchor points: 300" in footer_text
