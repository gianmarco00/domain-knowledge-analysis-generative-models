import pytest
import torch

from domain_knowledge_analysis.plotting import Plotter

def test_plotter_creates_result_directory(tmp_path):
    plotter = Plotter(tmp_path)

    assert plotter.output_dir.exists()
    assert plotter.output_dir == tmp_path / "results"

def test_plotter_plot_histogram_creates_png_file(tmp_path):
    results = {
        "likelihood": {
            "in_distribution": torch.tensor([1.0, 2.0, 3.0]),
            "out_distribution": torch.tensor([2.0, 3.0, 4.0]),
        }
    }

    plotter = Plotter(tmp_path)

    save_path = plotter.plot_histogram(
        results=results,
        signal_name="likelihood",
        in_distribution_name="mnist",
        out_distribution_name="fashion_mnist",
        title="Trained on MNIST",
        filename="likelihood_mnist_vs_fashion_mnist.png",
    )

    assert save_path.exists()
    assert save_path.suffix == ".png"

def test_plotter_plot_creates_one_file_per_signal(tmp_path):
    results = {
        "likelihood": {
            "in_distribution": torch.tensor([1.0, 2.0, 3.0]),
            "out_distribution": torch.tensor([2.0, 3.0, 4.0]),
        }
    }

    plotter = Plotter(tmp_path)

    saved_paths = plotter.plot(
        results=results,
        in_distribution_name="mnist",
        out_distribution_name="fashion_mnist",
        title="Trained on MNIST",
    )

    assert "likelihood" in saved_paths
    assert saved_paths["likelihood"].exists()
    assert saved_paths["likelihood"].suffix == ".png"

def test_plotter_rejects_missing_signal(tmp_path):
    results = {
        "likelihood": {
            "in_distribution": torch.tensor([1.0, 2.0, 3.0]),
            "out_distribution": torch.tensor([2.0, 3.0, 4.0]),
        }
    }

    plotter = Plotter(tmp_path)

    with pytest.raises(ValueError, match="Signal 'likelihood_regret' not found"):
        plotter.plot_histogram(
            results=results,
            signal_name="likelihood_regret",
            in_distribution_name="mnist",
            out_distribution_name="fashion_mnist",
            title="Trained on MNIST",
            filename="missing.png",
        )

def test_plotter_accepts_list_scores(tmp_path):
    results = {
        "likelihood": {
            "in_distribution": [1.0, 2.0, 3.0],
            "out_distribution": [2.0, 3.0, 4.0],
        }
    }

    plotter = Plotter(tmp_path)

    save_path = plotter.plot_histogram(
        results=results,
        signal_name="likelihood",
        in_distribution_name="mnist",
        out_distribution_name="fashion_mnist",
        title="Trained on MNIST",
        filename="list_scores.png",
    )

    assert save_path.exists()