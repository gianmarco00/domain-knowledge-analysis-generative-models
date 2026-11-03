from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from domain_knowledge_analysis.experiments.target_geometry_ablation import generate_target_geometry_runs, TargetGeometryAblation
from domain_knowledge_analysis.utils import build_run_name, load_config
from test_target_geometry import ToyModel, toy_data


@pytest.mark.parametrize("dataset", ["mnist", "ellipsis", "mnist_emnist"])
def test_paired_sweep_preserves_base_settings_and_covers_every_combination(dataset, tmp_path):
    from itertools import product
    sweep_config = load_config(f"config/ablation_vae_{dataset}_target_geometry.yaml")
    base = load_config(sweep_config["base_config"])
    ablation = sweep_config["ablation"]
    runs = generate_target_geometry_runs(base, ablation, tmp_path)
    grid = set(product(ablation["seeds"], ablation["eta_values"],
                       ablation["geometry_batch_sizes"], ablation["dual_learning_rates"]))
    assert len(runs) == len(ablation["seeds"]) + 2 * len(grid) + len(ablation["seeds"]) * len(ablation.get("controls", []))
    assert len({run["model_id"] for run in runs}) == len(runs)
    assert len([run for run in runs if run["eta"] == 0]) == len(ablation["seeds"])
    assert {run["view"] for run in runs} == {"full", "target_only", "source_only", "ordinary_lora"}
    for view in ("full", "target_only"):
        assert {(run["seed"], run["eta"], run["geometry_batch_size"], run["dual_learning_rate"])
                for run in runs if run["view"] == view} == grid
    for run in runs:
        config = load_config(run["config_path"])
        for key in ("training", "model", "loss", "dataloader", "optimizer", "lr_scheduler"):
            assert config[key] == base[key]
        for key in base["lora"]:
            if key != "regularizer":
                assert config["lora"][key] == base["lora"][key]
        if run["eta"]:
            assert config["lora"]["regularizer"]["eta"] == run["eta"]
            assert config["lora"]["regularizer"]["view"] == run["view"]
            assert "transformations" not in config["lora"]["regularizer"]
        else:
            assert config["lora"]["regularizer"] is None
    assert "target_geometry_full" in build_run_name(base)
    assert [run["model_id"] for run in generate_target_geometry_runs(base, ablation, tmp_path)] == [run["model_id"] for run in runs]


def test_old_target_controls_deduplicate_and_off_grid_controls_remain_paired(tmp_path):
    base = load_config("config/lora_vae_mnist_target_geometry.yaml")
    sweep = {"seeds": [42], "eta_values": [10.], "geometry_batch_sizes": [64],
             "dual_learning_rates": [.01], "controls": [
                 {"view": "target_only", "eta": 10., "geometry_batch_size": 64},
                 {"view": "target_only", "eta": 100., "geometry_batch_size": 128}]}
    runs = generate_target_geometry_runs(base, sweep, tmp_path)
    assert len(runs) == 5  # One baseline and two paired settings, no duplicate.
    full = {(row["eta"], row["geometry_batch_size"]) for row in runs if row["view"] == "full"}
    target = {(row["eta"], row["geometry_batch_size"]) for row in runs if row["view"] == "target_only"}
    assert full == target == {(10., 64), (100., 128)}


def test_expanding_target_grid_reuses_completed_models_and_old_training_ids(tmp_path, monkeypatch):
    from domain_knowledge_analysis.experiments.ablation_utils import training_config_id
    import domain_knowledge_analysis.experiments.ablation as worker_module
    base = load_config("config/lora_vae_mnist_target_geometry.yaml")
    sweep = {"seeds": [42], "eta_values": [10., 100.], "geometry_batch_sizes": [64],
             "dual_learning_rates": [.01]}
    # Construct pre-change full runs and the old single target-only control
    # directly from the training config; plot/grid changes must keep their IDs.
    expected = set()
    for eta, view in ((0., "ordinary_lora"), (10., "full"), (100., "full"), (100., "target_only")):
        config = deepcopy(base)
        if eta == 0:
            config["lora"]["regularizer"] = None
        else:
            config["lora"]["regularizer"].update(eta=eta, view=view, geometry_batch_size=64, dual_learning_rate=.01)
        expected.add(training_config_id(config))
    runs = generate_target_geometry_runs(base, sweep, tmp_path)
    assert expected <= {row["model_id"] for row in runs}
    previous_configs = {}
    for row in runs:
        if row["model_id"] in expected:
            checkpoint_dir = row["model_dir"] / "checkpoints"
            checkpoint_dir.mkdir()
            for filename in ("best.pt", "last.pt"):
                (checkpoint_dir / filename).write_bytes(b"completed")
            previous_configs[row["model_id"]] = row["config_path"].read_bytes()
    runs = generate_target_geometry_runs(base, sweep, tmp_path)
    for row in runs:
        if row["model_id"] in previous_configs:
            assert row["config_path"].read_bytes() == previous_configs[row["model_id"]]
    trained = []
    def train(config_path, **kwargs):
        model_dir = Path(config_path).parent
        trained.append(model_dir)
        checkpoint_dir = model_dir / "checkpoints"
        checkpoint_dir.mkdir()
        for filename in ("best.pt", "last.pt"):
            (checkpoint_dir / filename).write_bytes(b"new target-only model")
    monkeypatch.setattr(worker_module, "train_ablation_model", train)
    ablation = TargetGeometryAblation.__new__(TargetGeometryAblation)
    ablation.runs = runs
    ablation.ablation_config = {"parallel_models": 1}
    ablation.prepare_dataset_caches = lambda *_: None
    ablation.train_models()
    assert trained == [row["model_dir"] for row in runs if row["model_id"] not in expected]
    assert len(trained) == 1
    ablation.prepare_dataset_caches = lambda *_: pytest.fail("All runs are now completed")
    ablation.train_models()
    assert len(trained) == 1


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
    from domain_knowledge_analysis.plotting.target_geometry_plotter import TargetGeometryPlotter
    rows = []
    for view, eta, batch_size in (("ordinary_lora", 0., 0), ("full", 10., 64), ("full", 100., 64), ("source_only", 100., 64)):
        row = {"view": view, "seed": 42, "eta": eta, "geometry_batch_size": batch_size, "dual_learning_rate": .01}
        for metric, _ in TargetGeometryPlotter.METRICS:
            row[metric] = .1
        rows.append(row)
    TargetGeometryPlotter(tmp_path, {}).plot_metric_pairs(rows)
    assert (tmp_path / "figures" / "figure_1_eta_all.png").stat().st_size > 1000
    assert (tmp_path / "figures" / "figure_2_eta_zoom.png").stat().st_size > 1000
    assert not (tmp_path / "figures" / "target_geometry.png").exists()


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
    # Exercise the real completed-run check: rendering/scoring must never
    # dispatch workers or prepare training data for existing best+last models.
    ablation.prepare_dataset_caches = lambda *_: pytest.fail("Completed models must not be retrained")
    ablation.prepare_fid = lambda _: ("reference", "fid-key")
    selections = []
    ablation.plot_selected_models = lambda models, *_args, **_kwargs: selections.append(models)
    scored = []
    def evaluate(experiment, *_):
        scored.append(experiment)
        # The one-view control scores better, but the selected full-method
        # model is explicitly chosen among full-view runs for inspection.
        return {"validation_adapted_target_mse": .2 if experiment.view == "full" else .1,
                "test_adapted_target_mse": .2, "test_adapted_source_mse": .3,
                "test_source_degradation": .1, "test_target_gain": .1, "test_target_fid": 10.}
    ablation.evaluate = evaluate
    ablation.load_or_train = lambda run: SimpleNamespace(view=run["view"], logger=SimpleNamespace(close=lambda: None))
    import domain_knowledge_analysis.experiments.target_geometry_ablation as module
    monkeypatch.setattr(module.utils, "create_testing_dataloaders", lambda *args: None)
    for run in ablation.runs:
        checkpoint = run["model_dir"] / "checkpoints" / "best.pt"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"model")
        (checkpoint.parent / "last.pt").write_bytes(b"model")
        if run["eta"]:
            torch.save({"eta": 100., "dual_fraction": .25,
                        "calibration": {"source_tolerance": .04, "standard_error": .001}},
                       checkpoint.with_suffix(".regularizer.pt"))
    first = ablation.run()
    assert len(scored) == 3
    assert selections[-1]["full_reconstruction"]["view"] == "full"
    assert selections[-1]["target_only_reconstruction"]["view"] == "target_only"
    assert (tmp_path / "results" / "summary.csv").exists()
    assert (tmp_path / "results" / "selected_model.yaml").exists()
    second = ablation.run()
    assert len(scored) == 3  # Reuses metrics, including their view strings.
    assert [row["model_id"] for row in first] == [row["model_id"] for row in second]
    assert next(row for row in second if row["view"] == "full")["dual_multiplier"] == 25.
