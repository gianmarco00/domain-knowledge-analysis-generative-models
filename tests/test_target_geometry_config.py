from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from domain_knowledge_analysis.experiments.target_geometry_ablation import generate_target_geometry_runs, TargetGeometryAblation
from domain_knowledge_analysis.utils import build_run_name, load_config
from test_target_geometry import ToyModel, toy_data


@pytest.mark.parametrize("dataset", ["mnist", "ellipsis"])
def test_new_configs_preserve_core_settings_and_generate_expected_sweep(dataset, tmp_path):
    original = load_config(f"config/lora_vae_{dataset}.yaml")
    base = load_config(f"config/lora_vae_{dataset}_target_geometry.yaml")
    for key in original:
        if key not in {"lora", "experiment"}:
            assert base[key] == original[key]
    for key in original["lora"]:
        if key != "regularizer":
            assert base["lora"][key] == original["lora"][key]
    ablation = load_config(f"config/ablation_vae_{dataset}_target_geometry.yaml")["ablation"]
    runs = generate_target_geometry_runs(base, ablation, tmp_path)
    assert len(runs) == 63
    assert len({run["model_id"] for run in runs}) == 63
    assert len([run for run in runs if run["eta"] == 0]) == 3
    assert {run["view"] for run in runs} == {"full", "target_only", "source_only", "ordinary_lora"}
    full = [run for run in runs if run["view"] == "full"]
    assert len(full) == 54
    from itertools import product
    assert {(run["seed"], run["eta"], run["geometry_batch_size"], run["dual_learning_rate"]) for run in full} == set(product(
        ablation["seeds"], ablation["eta_values"], ablation["geometry_batch_sizes"], ablation["dual_learning_rates"]))
    for run in runs:
        config = load_config(run["config_path"])
        for key in ("training", "model", "loss", "dataloader", "optimizer", "lr_scheduler"):
            assert config[key] == base[key]
        if run["eta"]:
            assert config["lora"]["regularizer"]["eta"] == run["eta"]
            assert "transformations" not in config["lora"]["regularizer"]
        else:
            assert config["lora"]["regularizer"] is None
    assert "target_geometry_full" in build_run_name(base)
    # Regeneration produces stable IDs and no duplicate baseline per batch size.
    assert [run["model_id"] for run in generate_target_geometry_runs(base, ablation, tmp_path)] == [run["model_id"] for run in runs]


def test_geometry_result_metadata_roundtrip(tmp_path):
    from domain_knowledge_analysis.experiments.ablation_utils import save_rows
    rows = [{"model_id": "abc", "mode": "target_geometry", "view": "source_only", "seed": 42,
             "eta": 100., "geometry_batch_size": 64, "dual_learning_rate": .01,
             "source_tolerance": .0361, "test_target_fid": 123.}]
    path = tmp_path / "metrics.csv"
    save_rows(rows, path)
    assert TargetGeometryAblation.load_saved_results(path)["abc"] == rows[0]


def test_experiment_uses_isolated_trainer_and_source_checkpoint_split(monkeypatch, tmp_path):
    import domain_knowledge_analysis.experiments.experiment as module
    import domain_knowledge_analysis.experiments.target_geometry as setup
    from domain_knowledge_analysis.experiments import Experiment
    from torch.utils.data import DataLoader, TensorDataset
    base = load_config("config/lora_vae_mnist_target_geometry.yaml")
    base["seed"] = 99
    base["lora"]["regularizer"].update(geometry_batch_size=4, calibration_batches=4, validation_batches=2)
    source_config = deepcopy(base)
    source_config["seed"] = 17
    model = ToyModel()
    experiment = Experiment.__new__(Experiment)
    experiment.config = base
    experiment.checkpoint_manager = SimpleNamespace(model_config=source_config)
    experiment.model = model
    experiment.lora_manager = SimpleNamespace(trainable_parameters=model.parameters)
    experiment.device = torch.device("cpu")
    experiment.log_dir = tmp_path
    experiment.logger = None
    experiment.progress_position = None
    experiment.progress_description = "test"
    experiment.progress_stream = None
    experiment.print_tensorboard_instructions = lambda _: None
    dataset = TensorDataset(toy_data())
    loader = DataLoader(dataset, batch_size=4)
    monkeypatch.setattr(module.utils, "create_training_dataloaders", lambda **kwargs: (loader, loader))
    monkeypatch.setattr(module.utils, "create_optimizer", lambda *args: None)
    monkeypatch.setattr(module.utils, "create_lr_scheduler", lambda *args: (None, 0))
    monkeypatch.setattr(module.utils, "create_loss", lambda *args: None)
    def held_out(config, name):
        assert config is source_config
        assert config["seed"] == 17
        return dataset, dataset
    monkeypatch.setattr(setup, "create_train_validation_datasets", held_out)
    monkeypatch.setattr(module.utils, "select_anchor_points", lambda *args: pytest.fail("Old regularizer must not run"))
    captured = {}
    class FakeTrainer:
        def __init__(self, **kwargs):
            captured.update(kwargs)
        def fit(self):
            pass
    monkeypatch.setattr(setup, "TargetGeometryTrainer", FakeTrainer)
    experiment.adapt()
    assert captured["regularizer"].source_tolerance == pytest.approx(.0361)
    assert captured["geometry_batch_size"] == 4
    calibration = load_config(tmp_path / "target_geometry_calibration.yaml")
    assert calibration["source_split_seed"] == 17
    assert calibration["calibration_seed"] == 99 + 17001


def test_geometry_plots_use_geometry_metadata(tmp_path):
    ablation = TargetGeometryAblation.__new__(TargetGeometryAblation)
    ablation.output_dir = tmp_path
    rows = []
    for view, eta, batch_size in (("ordinary_lora", 0., 0), ("full", 10., 64), ("full", 100., 64), ("source_only", 100., 64)):
        row = {"view": view, "eta": eta, "geometry_batch_size": batch_size, "dual_learning_rate": .01}
        for metric in ("test_adapted_target_mse", "test_source_degradation", "test_target_fid"):
            row[metric + "_mean"] = .1
            row[metric + "_std"] = .01
        rows.append(row)
    ablation.plot_geometry_results(rows)
    assert (tmp_path / "figures" / "target_geometry.png").stat().st_size > 1000


def test_standard_adaptation_plot_supports_new_base_config(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from domain_knowledge_analysis.plotting import AdaptationPlotter
    from test_adaptation_plotter import make_results
    figures = []
    monkeypatch.setattr(plt, "close", figures.append)
    base = load_config("config/lora_vae_mnist_target_geometry.yaml")
    path = AdaptationPlotter(tmp_path, base["lora"]).plot(make_results())
    assert path.is_file()
    footer = figures[0].axes[-1].texts[0].get_text()
    assert "Target geometry" in footer and "geometry batch: 64" in footer


def test_ablation_scoring_selection_and_cached_metadata(monkeypatch, tmp_path):
    config = load_config("config/lora_vae_mnist_target_geometry.yaml")
    sweep = {"seeds": [42], "eta_values": [100.], "geometry_batch_sizes": [64],
             "controls": [{"view": "target_only", "eta": 100., "geometry_batch_size": 64}]}
    ablation = TargetGeometryAblation.__new__(TargetGeometryAblation)
    ablation.output_dir = tmp_path
    ablation.ablation_config = sweep
    ablation.runs = generate_target_geometry_runs(config, sweep, tmp_path)
    ablation.train_models = lambda: None
    ablation.prepare_fid = lambda _: ("reference", "fid-key")
    ablation.plot_geometry_results = lambda _: None
    selections = []
    ablation.plot_reconstructions = selections.append
    scored = []
    def evaluate(experiment, *_):
        scored.append(experiment)
        # The one-view control scores better, but the selected full-method
        # model is explicitly chosen among full-view runs for inspection.
        return {"validation_adapted_target_mse": .2 if experiment.view == "full" else .1,
                "test_adapted_target_mse": .2, "test_source_degradation": .1, "test_target_gain": .1, "test_target_fid": 10.}
    ablation.evaluate = evaluate
    ablation.load_or_train = lambda run: SimpleNamespace(view=run["view"], logger=SimpleNamespace(close=lambda: None))
    import domain_knowledge_analysis.experiments.target_geometry_ablation as module
    monkeypatch.setattr(module.utils, "create_testing_dataloaders", lambda *args: None)
    for run in ablation.runs:
        checkpoint = run["model_dir"] / "checkpoints" / "best.pt"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"model")
        if run["eta"]:
            torch.save({"eta": 100., "dual_fraction": .25,
                        "calibration": {"source_tolerance": .04, "standard_error": .001}},
                       checkpoint.with_suffix(".regularizer.pt"))
    first = ablation.run()
    assert len(scored) == 3
    assert selections[-1]["view"] == "full"
    assert (tmp_path / "results" / "summary.csv").exists()
    assert (tmp_path / "results" / "selected_model.yaml").exists()
    second = ablation.run()
    assert len(scored) == 3  # Reuses metrics, including their view strings.
    assert [row["model_id"] for row in first] == [row["model_id"] for row in second]
    assert next(row for row in second if row["view"] == "full")["dual_multiplier"] == 25.
