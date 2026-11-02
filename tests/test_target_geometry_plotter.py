from itertools import product
from types import SimpleNamespace

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from domain_knowledge_analysis.experiments.target_geometry_ablation import TargetGeometryAblation
from domain_knowledge_analysis.plotting.target_geometry_plotter import TargetGeometryPlotter


def metric_rows():
    rows = []
    for seed, reference in ((42, 1.), (43, 2.)):
        baseline = {"view": "ordinary_lora", "seed": seed, "eta": 0.,
                    "test_adapted_target_mse": .01 * reference,
                    "test_adapted_source_mse": .04 * reference, "test_target_fid": 20. * reference}
        rows.append(baseline)
        for eta, size, rate in product((10., 100., 1000.), (64, 128), (.001, .01, .1)):
            rows.append({**baseline, "view": "full", "eta": eta,
                         "geometry_batch_size": size, "dual_learning_rate": rate,
                         "test_adapted_target_mse": baseline["test_adapted_target_mse"] * (.9 if seed == 42 else .8),
                         "test_adapted_source_mse": baseline["test_adapted_source_mse"] * (.5 if seed == 42 else .4),
                         "test_target_fid": baseline["test_target_fid"] * (.95 if seed == 42 else .85)})
    rows.append({"view": "target_only", "seed": 42, "eta": 100., "geometry_batch_size": 64,
                 "dual_learning_rate": .01, "test_adapted_target_mse": 100.,
                 "test_adapted_source_mse": 100., "test_target_fid": 1000.})
    rows.append({"view": "source_only", "seed": 42, "eta": 100., "geometry_batch_size": 64,
                 "dual_learning_rate": .01, "test_adapted_target_mse": .02,
                 "test_adapted_source_mse": .01, "test_target_fid": 18.})
    return rows


def capture_figures(monkeypatch):
    figures, real_close = [], plt.close
    def close(figure):
        if isinstance(figure, Figure):
            figures.append(figure)
        else:
            real_close(figure)
    monkeypatch.setattr(plt, "close", close)
    return figures, real_close


def test_percent_error_reduction_uses_same_seed_and_zero_lora_baseline():
    rows = metric_rows()
    baseline, proposed = rows[:2]
    for metric, expected in (("test_adapted_target_mse", 10.), ("test_adapted_source_mse", 50.), ("test_target_fid", 5.)):
        assert TargetGeometryPlotter.metric_value(proposed, baseline, metric) == pytest.approx(expected)
        assert TargetGeometryPlotter.metric_value(baseline, baseline, metric) == 0.
    second_baseline = next(row for row in rows if row["seed"] == 43 and not row["eta"])
    second_model = next(row for row in rows if row["seed"] == 43 and row["eta"])
    assert TargetGeometryPlotter.metric_value(second_model, second_baseline, "test_target_fid") == pytest.approx(15.)


def test_stacked_lr_rows_three_metrics_controls_and_consistent_batch_colors(tmp_path, monkeypatch):
    figures, real_close = capture_figures(monkeypatch)
    paths = TargetGeometryPlotter(tmp_path, {}).plot_metric_pairs(metric_rows())
    assert [path.name for path in paths] == ["figure_1_eta_all.png", "figure_2_eta_zoom.png"]
    assert all(path.stat().st_size > 1000 for path in paths)
    assert len(figures) == 2
    for figure in figures:
        assert len(figure.axes) == 9  # Three LR rows, target/source/FID columns.
        assert {text.get_text() for text in figure.legends[0].texts} >= {"B = 64", "B = 128", "Full", "Target only", "Source only"}
        for row_index, rate in enumerate((.001, .01, .1)):
            axes = figure.axes[3 * row_index:3 * row_index + 3]
            for axis, (_, title) in zip(axes, TargetGeometryPlotter.METRICS):
                assert axis.get_xscale() == "log"
                assert axis.get_ylabel() == f"{title} improvement (%) ↑"
                assert f"= {rate:g}" in axis.get_title()
        # Both controls appear at their measured LR, rather than being copied
        # into unmatched learning rates. Their marker and line style distinguish them.
        all_markers = {line.get_marker() for axis in figure.axes for line in axis.lines}
        if figure is figures[0]:
            assert {"o", "s", "P"} <= all_markers
            colors = [[line.get_color() for line in axis.lines[:4]] for axis in figure.axes[:3]]
            assert colors[0] == colors[1] == colors[2]
            assert colors[0][0] != colors[0][1]
        real_close(figure)


def fid_zoom_rows():
    baseline = {"view": "ordinary_lora", "seed": 42, "eta": 0.,
                "test_adapted_target_mse": .01, "test_adapted_source_mse": .04, "test_target_fid": 20.}
    rows = [baseline]
    for eta, fids in ((1., (20., 20.)), (1000., (17., 18.)),
                      (10000., (40., 22.)), (100000., (60., 80.))):
        for size, fid in zip((64, 128), fids):
            rows.append({**baseline, "view": "full", "eta": eta, "test_target_fid": fid,
                         "geometry_batch_size": size, "dual_learning_rate": .01})
    return rows


def test_zoom_best_improvement_plus_five_selection_clipping_and_dotted_lines(tmp_path, monkeypatch):
    figures, real_close = capture_figures(monkeypatch)
    plotter = TargetGeometryPlotter(tmp_path, {})
    series = plotter.metric_series(fid_zoom_rows())["test_target_fid"]
    assert plotter.zoom_radius(series) == pytest.approx(20.)  # FID 20 -> 17: 15% + 5 pp.
    assert plotter.selected_etas(series, .01, 20.) == [1., 1000., 10000.]
    plotter.plot_metric_pairs(fid_zoom_rows())
    all_figure, zoom = figures
    assert list(all_figure.axes[2].get_xticks()) == [1., 1000., 10000., 100000.]
    fid = zoom.axes[2]
    assert fid.get_ylim() == pytest.approx((-20., 20.))
    assert list(fid.get_xticks()) == [1., 1000., 10000.]
    assert list(zoom.axes[0].get_xticks()) == [1., 1000., 10000., 100000.]  # Selection is per metric.
    dotted = [line for line in fid.lines if list(line.get_xdata()) == [1000., 10000.] and line.get_linestyle() == ":"]
    assert len(dotted) == 1
    assert list(dotted[0].get_ydata()) == pytest.approx([15., -20.])
    assert dotted[0].get_color() == plt.get_cmap("tab10")(0)
    assert any(list(point) == [10000., -20.] for collection in fid.collections for point in collection.get_offsets())
    assert not any(100000. in line.get_xdata() for line in fid.lines)
    for figure in figures:
        real_close(figure)


def test_zoom_uses_seed_means_and_individual_sample_inclusion():
    plotter = TargetGeometryPlotter
    series = {(.01, "full", 64, 42): [(1., 10.), (10., 100.), (100., -100.)],
              (.01, "full", 64, 43): [(1., 20.), (10., -80.), (100., -110.)]}
    assert plotter.zoom_radius(series) == 20.  # Best mean is 15, not the lucky 100% seed.
    assert plotter.selected_etas(series, .01, 20.) == [1.]
    assert plotter.zoom_radius({(.01, "full", 64, 42): [(1., -15.), (10., -100.)]}) == 20.
    assert plotter.zoom_radius({(.01, "full", 64, 42): [(1., 0.)]}) == 5.
    assert plotter.selected_etas({(.01, "full", 64, 42): [(1., -20.), (10., 20.)]}, .01, 20.) == [1., 10.]


def test_zoom_does_not_connect_across_removed_eta_with_a_solid_line():
    figure, axis = plt.subplots()
    TargetGeometryPlotter.draw_series(axis, [(1., 0.), (10., -100.), (100., 10.)], "blue", "full", {1., 100.}, 15.)
    assert axis.lines[0].get_linestyle() == ":"
    assert list(axis.lines[0].get_xdata()) == [1., 100.]
    plt.close(figure)


def test_invalid_reference_and_missing_or_duplicate_baselines_are_rejected(tmp_path):
    rows = metric_rows()
    plotter = TargetGeometryPlotter(tmp_path, {})
    with pytest.raises(ValueError, match="Every plotted seed"):
        plotter.plot_metric_pairs(rows[1:])
    with pytest.raises(ValueError, match="Multiple ordinary LoRA"):
        plotter.plot_metric_pairs([*rows, rows[0]])
    bad_reference = {**rows[0], "test_target_fid": 0.}
    with pytest.raises(ValueError, match="finite and positive"):
        plotter.plot_metric_pairs([bad_reference, *rows[1:]])
    with pytest.raises(ValueError, match="Duplicate seed"):
        plotter.plot_metric_pairs([*rows, rows[1]])


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
    assert {path.name for path in paths} == {"figure_3_source_reconstructions.png", "figure_4_target_reconstructions.png", "figure_5_generations.png"}
    assert all(path.stat().st_size > 1000 for path in paths)
    saved = torch.load(tmp_path / "results" / "qualitative_examples.pt", weights_only=True)
    assert set(saved) >= {"source", "target", "generation", "latents", "selected_model"}
    assert saved["latents"].shape == (3, 2)
    assert saved["selected_model"] == selected


def test_plots_only_reuses_scores_and_image_banks_without_models_fid_or_training(tmp_path, monkeypatch):
    from domain_knowledge_analysis.experiments.ablation_utils import save_rows, save_yaml
    import domain_knowledge_analysis.experiments.target_geometry_ablation as module
    rows = metric_rows()
    for index, row in enumerate(rows):
        row.update(model_id=str(index), checkpoint_hash="saved-checkpoint",
                   geometry_batch_size=row.get("geometry_batch_size", 0),
                   dual_learning_rate=row.get("dual_learning_rate", 0.),
                   validation_adapted_target_mse=row["test_adapted_target_mse"],
                   test_target_gain=0., test_source_degradation=0.)
    selected = min((row for row in rows if row["view"] == "full"), key=lambda row: row["validation_adapted_target_mse"])
    metrics = tmp_path / "results" / "metrics.csv"
    save_rows(rows, metrics)
    original_metrics = metrics.read_bytes()
    images = torch.zeros(1, 1, 4, 4)
    bank = {key: {name: images for name in ("input", "base", "lora", "regularized")} for key in ("source", "target")}
    bank["generation"] = {name: images for name in ("base", "lora", "regularized")}
    bank["selected_model"] = dict(selected)
    torch.save(bank, tmp_path / "results" / "qualitative_examples.pt")
    figures = tmp_path / "figures"
    figures.mkdir()
    for name in ("target_geometry.png", "target_reconstructions.png", "eta_retention_gain_dual_0.01.png",
                 "figure_5_source_reconstructions.png", "figure_6_target_reconstructions.png", "figure_7_generations.png"):
        (figures / name).write_bytes(b"obsolete")
    config = tmp_path / "ablation.yaml"
    save_yaml({"base_config": "config/lora_vae_mnist_target_geometry.yaml",
               "ablation": {"output_dir": str(tmp_path)}}, config)
    def forbidden(*args, **kwargs):
        pytest.fail("Plot-only execution must not generate training configs, train, load models, score, or construct FID")
    monkeypatch.setattr(module, "generate_target_geometry_runs", forbidden)
    for method in ("train_models", "load_or_train", "evaluate", "prepare_fid"):
        monkeypatch.setattr(TargetGeometryAblation, method, forbidden)
    ablation = TargetGeometryAblation(config, plots_only=True)
    returned = ablation.replot_saved_results()
    assert len(returned) == len(rows)
    assert metrics.read_bytes() == original_metrics
    assert {path.name for path in figures.iterdir()} == {
        "figure_1_eta_all.png", "figure_2_eta_zoom.png", "figure_3_source_reconstructions.png",
        "figure_4_target_reconstructions.png", "figure_5_generations.png"}
    assert not (tmp_path / "models").exists()


def test_plots_only_cli_dispatches_without_calling_run(monkeypatch):
    import scripts.run_ablation as command
    import domain_knowledge_analysis.experiments.target_geometry_ablation as module
    calls = []
    class FakeAblation:
        def __init__(self, config_path, plots_only=False):
            assert plots_only
        def replot_saved_results(self):
            calls.append("replot")
        def run(self):
            pytest.fail("Plot-only CLI must not enter training/scoring")
    monkeypatch.setattr(module, "TargetGeometryAblation", FakeAblation)
    monkeypatch.setattr("sys.argv", ["run_ablation.py", "config/ablation_vae_mnist_target_geometry.yaml", "--plots-only"])
    command.main()
    assert calls == ["replot"]
