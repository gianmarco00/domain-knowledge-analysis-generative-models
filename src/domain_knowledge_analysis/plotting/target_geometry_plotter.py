"""Result figures for target geometry, independent of transformation sweeps."""

from collections import defaultdict
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from .ablation_plotter import AblationPlotter


class TargetGeometryPlotter:
    def __init__(self, output_dir, config):
        self.output_dir = Path(output_dir) / "figures"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config

    @staticmethod
    def metric_value(row, baseline, metric, normalized=True):
        """Normalize per seed before averaging, never a ratio of seed means.

        Retention measures recovery of the source degradation incurred by
        ordinary LoRA: source VAE=100%, ordinary LoRA=0%. Target gain uses the
        existing ablation convention: source VAE=0%, ordinary LoRA=100%.
        If a LoRA reference change is nonpositive, plot the raw MSE change
        instead for that entire metric, so the 'higher is better' meaning holds.
        """
        if metric == "source_retention":
            degradation = float(row["test_source_degradation"])
            if not normalized:
                return -degradation
            reference = float(baseline["test_source_degradation"])
            return 100 * (1 - degradation / reference)
        if metric == "target_gain":
            gain = float(row["test_target_gain"])
            if not normalized:
                return gain
            return 100 * gain / float(baseline["test_target_gain"])
        raise ValueError(f"Unknown target geometry plot metric: {metric}.")

    def plot_metric_pairs(self, results):
        full = [row for row in results if row["view"] == "full" and float(row["eta"]) > 0]
        if not full:
            raise ValueError("No full-view target geometry models to plot.")
        baselines = AblationPlotter.ordinary_by_seed([row for row in results if not float(row["eta"])])
        seeds = {row["seed"] for row in full}
        if seeds - baselines.keys():
            raise ValueError("Every plotted seed requires its own ordinary LoRA baseline.")
        if any(not math.isfinite(float(baselines[seed][key])) for seed in seeds for key in ("test_source_degradation", "test_target_gain")):
            raise ValueError("Ordinary LoRA reference metrics must be finite.")
        normalized = {
            metric: all(math.isfinite(float(baselines[seed][key])) and float(baselines[seed][key]) > 0 for seed in seeds)
            for metric, key in (("source_retention", "test_source_degradation"), ("target_gain", "test_target_gain"))
        }
        batch_sizes = sorted({int(row["geometry_batch_size"]) for row in full})
        colors = {size: plt.get_cmap("tab10")(index % 10) for index, size in enumerate(batch_sizes)}
        rates = sorted({float(row["dual_learning_rate"]) for row in full})
        paths = []
        for rate in rates:
            rows = [row for row in full if float(row["dual_learning_rate"]) == rate]
            eta_values = sorted({float(row["eta"]) for row in rows})
            figure, axes = plt.subplots(1, 2, figsize=(13, 6.5))
            try:
                for axis, metric, title in zip(axes, ("source_retention", "target_gain"), ("Source retention", "Target gain")):
                    for size in batch_sizes:
                        groups = defaultdict(list)
                        for row in rows:
                            if int(row["geometry_batch_size"]) == size:
                                groups[float(row["eta"])].append(row)
                        xs, means, deviations = [], [], []
                        for eta, group in sorted(groups.items()):
                            if len({row["seed"] for row in group}) != len(group):
                                raise ValueError("Duplicate seed for a geometry batch size/dual rate/eta setting.")
                            values = torch.tensor([self.metric_value(row, baselines[row["seed"]], metric, normalized[metric]) for row in group], dtype=torch.float64)
                            if not torch.isfinite(values).all():
                                raise ValueError("Nonfinite retention/gain data; cannot plot results.")
                            xs.append(eta)
                            means.append(values.mean().item())
                            deviations.append(values.std(unbiased=True).item() if len(values) > 1 else 0.)
                        if xs:
                            axis.errorbar(xs, means, yerr=deviations, color=colors[size], marker="o",
                                          capsize=4, linewidth=1.7, label=f"B = {size}")
                    # Ordinary LoRA is a common reference, not another B curve.
                    ordinary_values = [self.metric_value(baselines[seed], baselines[seed], metric, normalized[metric]) for seed in seeds]
                    reference = sum(ordinary_values) / len(ordinary_values)
                    axis.axhline(reference, color="0.35", linestyle="--", linewidth=1.1, label="Ordinary LoRA")
                    axis.axhline(100. if metric == "source_retention" and normalized[metric] else 0.,
                                 color="0.65", linestyle=":", linewidth=1.1, label="Source VAE")
                    units = "%" if normalized[metric] else "MSE change"
                    axis.set(xscale="log", xlabel="Regularization weight η", ylabel=f"{title} ({units}) ↑", title=title)
                    axis.set_xticks(eta_values, [f"{eta:g}" for eta in eta_values])
                    axis.set_xlim(eta_values[0] / 1.4, eta_values[-1] * 1.4)
                    axis.grid(color="0.9", linestyle="--", linewidth=.7)
                    axis.legend(fontsize=9)
                figure.suptitle(f"Target geometry · dual learning rate = {rate:g}")
                retention_formula = (
                    "Source retention (%) = 100 × (1 − source degradation / ordinary-LoRA source degradation)."
                    if normalized["source_retention"] else
                    "Source retention (MSE change) = source-VAE source MSE − model source MSE; the LoRA reference change is nonpositive."
                )
                gain_formula = (
                    "Target gain (%) = 100 × target gain / ordinary-LoRA target gain."
                    if normalized["target_gain"] else
                    "Target gain (MSE change) = source-VAE target MSE − model target MSE; the LoRA reference change is nonpositive."
                )
                figure.text(.5, .125, retention_formula, ha="center", fontsize=8.5)
                figure.text(.5, .09, gain_formula, ha="center", fontsize=8.5)
                figure.text(.5, .055, "Held-out test images; normalize against the same seed, then average. Error bars: ±1 sample standard deviation across seeds.", ha="center", fontsize=8.5)
                figure.text(.5, .025, "Full two-view method. Values are not clipped; higher is better. One-view controls appear in the overview and CSV summaries.", ha="center", fontsize=8.5)
                figure.tight_layout(rect=(0, .17, 1, .94))
                path = self.output_dir / f"eta_retention_gain_dual_{rate:g}.png"
                figure.savefig(path, dpi=200)
                paths.append(path)
            finally:
                plt.close(figure)
        return paths

    @staticmethod
    def selected_model_label(selected):
        return (f"η={float(selected['eta']):g}, B={int(selected['geometry_batch_size'])}, "
                f"dual rate={float(selected['dual_learning_rate']):g}, seed={int(selected['seed'])}")

    def plot_qualitative(self, results):
        selected = self.selected_model_label(results["selected_model"])
        columns = [("base", "Source VAE"), ("lora", "Ordinary LoRA"), ("regularized", "Target geometry")]
        paths = []
        for split in ("source", "target", "generation"):
            data = results[split]
            is_generation = split == "generation"
            split_columns = columns if is_generation else [("input", "Input"), *columns]
            count = len(data["base"])
            if count == 0 or any(len(data[key]) != count for key, _ in split_columns):
                raise ValueError("Qualitative image banks must have equal, nonzero lengths.")
            figure, axes = plt.subplots(count, len(split_columns), figsize=(3 * len(split_columns), 2 * count + 1.5), squeeze=False)
            try:
                for column, (key, label) in enumerate(split_columns):
                    axes[0, column].set_title(label, fontsize=12, pad=10)
                    for row, image in enumerate(data[key]):
                        AblationPlotter.show_image(axes[row, column], image)
                if is_generation:
                    title = "Prior generation using the same latent vectors"
                    filename = "figure_7_generations.png"
                    caption = f"One shared set of N(0, I) latent vectors for every model; visual seed = {self.config.get('visual_seed', 123)}."
                else:
                    label = self.config.get(f"{split}_label", split.title())
                    title = f"{label} reconstructions"
                    filename = f"figure_{5 if split == 'source' else 6}_{split}_reconstructions.png"
                    caption = "Identical held-out inputs across columns; reconstruction uses each encoder's posterior mean."
                # Reserve physical header/footer space even for a one-row grid.
                height = figure.get_figheight()
                figure.suptitle(title, fontsize=15, y=1 - .10 / height, va="top")
                figure.text(.5, 1 - .45 / height, f"Selected by validation target MSE: {selected}", ha="center", va="top", fontsize=10)
                figure.text(.5, .18 / height, caption, ha="center", va="center", fontsize=9)
                figure.tight_layout(rect=(0, .40 / height, 1, 1 - .9 / height))
                path = self.output_dir / filename
                figure.savefig(path, dpi=180)
                paths.append(path)
                if split == "target":
                    # Preserve the first version's target-figure filename.
                    figure.savefig(self.output_dir / "target_reconstructions.png", dpi=180)
            finally:
                plt.close(figure)
        return paths
