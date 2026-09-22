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
