"""Result figures for target geometry, independent of transformation sweeps."""

from collections import defaultdict
import math
from pathlib import Path
from statistics import fmean

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .ablation_plotter import AblationPlotter


class TargetGeometryPlotter:
    METRICS = (("test_adapted_target_mse", "Target MSE"),
               ("test_adapted_source_mse", "Source MSE"),
               ("test_target_fid", "Target FID"))
    VIEWS = {"full": ("Full", "o", "-"),
             "target_only": ("Target only", "s", "--"),
             "source_only": ("Source only", "P", "-.")}
    ZOOM_PADDING = 5.0  # Percentage points, not 5% of the best improvement.

    def __init__(self, output_dir, config):
        self.output_dir = Path(output_dir) / "figures"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config

    @staticmethod
    def metric_value(row, baseline, metric):
        """Percentage error reduction against ordinary LoRA for the SAME seed."""
        reference, value = float(baseline[metric]), float(row[metric])
        if not math.isfinite(reference) or reference <= 0:
            raise ValueError(f"Ordinary LoRA {metric} must be finite and positive for percent comparisons.")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Model {metric} must be finite and nonnegative.")
        return 100 * (reference - value) / reference

    def metric_series(self, results):
        rows = [row for row in results if float(row["eta"]) > 0]
        if not rows:
            raise ValueError("No target geometry models to plot.")
        baselines = AblationPlotter.ordinary_by_seed([row for row in results if not float(row["eta"])])
        if {row["seed"] for row in rows} - baselines.keys():
            raise ValueError("Every plotted seed requires its own ordinary LoRA baseline.")
        seen = set()
        for row in rows:
            if row["view"] not in self.VIEWS:
                raise ValueError(f"Unknown target geometry view: {row['view']}.")
            key = (float(row["dual_learning_rate"]), row["view"], int(row["geometry_batch_size"]),
                   row["seed"], float(row["eta"]))
            if key in seen:
                raise ValueError("Duplicate seed for a geometry view/batch size/dual rate/eta setting.")
            if not math.isfinite(key[0]) or key[0] <= 0 or not math.isfinite(key[-1]):
                raise ValueError("Plotted eta and dual learning rates must be finite and positive.")
            seen.add(key)
        metrics = {}
        for metric, _ in self.METRICS:
            series = defaultdict(list)
            for row in rows:
                key = (float(row["dual_learning_rate"]), row["view"], int(row["geometry_batch_size"]), row["seed"])
                series[key].append((float(row["eta"]), self.metric_value(row, baselines[row["seed"]], metric)))
            metrics[metric] = {key: sorted(points) for key, points in series.items()}
        return metrics

    @classmethod
    def zoom_radius(cls, series):
        """Best mean across seeds for a (rate, view, B, eta), plus 5 pp.

        Share this metric's range across LR rows. If every setting is worse
        than LoRA, include the least-bad setting and the same padding instead
        of creating an empty zoom. A lucky single seed cannot set the scale.
        """
        groups = defaultdict(list)
        for (rate, view, size, _), points in series.items():
            for eta, value in points:
                groups[rate, view, size, eta].append(value)
        best = max(fmean(values) for values in groups.values())
        return abs(best) + cls.ZOOM_PADDING

    @staticmethod
    def selected_etas(series, rate, radius):
        # Selection uses actual individual seeds, not a mean or error-bar
        # overlap: retain eta iff at least one plotted sample is in the window.
        return sorted({eta for key, points in series.items() if key[0] == rate
                       for eta, value in points if -radius <= value <= radius})

    @classmethod
    def draw_series(cls, axis, points, color, view, selected, radius=None):
        _, marker, style = cls.VIEWS[view]
        visible = [(index, eta, value) for index, (eta, value) in enumerate(points) if eta in selected]
        if radius is None:
            axis.plot([eta for _, eta, _ in visible], [value for _, _, value in visible],
                      color=color, marker=marker, linestyle=style, markersize=6,
                      markeredgecolor="white", markeredgewidth=.6, linewidth=1.4, alpha=.85)
            return
        for previous, current in zip(visible, visible[1:]):
            first, x0, y0 = previous
            last, x1, y1 = current
            excursion = any(abs(value) > radius for _, value in points[first:last + 1])
            axis.plot([x0, x1], [max(-radius, min(radius, y0)), max(-radius, min(radius, y1))],
                      color=color, linestyle=":" if excursion else style, linewidth=1.4, alpha=.85)
        inside = [(eta, value) for _, eta, value in visible if abs(value) <= radius]
        axis.scatter([eta for eta, _ in inside], [value for _, value in inside], color=color,
                     marker=marker, s=40, edgecolor="white", linewidth=.6, zorder=3)
        # Boundary triangles explicitly identify retained samples whose values
        # lie outside the window; never silently present them as in-range scores.
        for direction, boundary_marker in ((1, "^"), (-1, "v")):
            outside = [eta for _, eta, value in visible if direction * value > radius]
            axis.scatter(outside, [direction * radius] * len(outside), color=color,
                         marker=boundary_marker, s=45, edgecolor="white", linewidth=.6,
                         clip_on=False, zorder=4)

    def plot_metric_pairs(self, results):
        """One LR row and three score columns, both complete and zoomed views."""
        metrics = self.metric_series(results)
        keys = list(next(iter(metrics.values())))
        rates = sorted({key[0] for key in keys})
        batch_sizes = sorted({key[2] for key in keys})
        colors = {size: plt.get_cmap("tab10")(index % 10) for index, size in enumerate(batch_sizes)}
        views = [view for view in self.VIEWS if any(key[1] == view for key in keys)]
        radii = {metric: self.zoom_radius(series) for metric, series in metrics.items()}
        paths = []
        for zoom, filename in ((False, "figure_1_eta_all.png"), (True, "figure_2_eta_zoom.png")):
            height = 2.6 + 3.6 * len(rates)
            figure, axes = plt.subplots(len(rates), 3, figsize=(17, height), squeeze=False, sharey="col")
            try:
                for row_index, rate in enumerate(rates):
                    for axis, (metric, title) in zip(axes[row_index], self.METRICS):
                        series = metrics[metric]
                        radius = radii[metric] if zoom else None
                        selected = (self.selected_etas(series, rate, radius) if zoom else
                                    sorted({eta for key, points in series.items() if key[0] == rate for eta, _ in points}))
                        for (series_rate, view, size, _), points in series.items():
                            if series_rate == rate:
                                self.draw_series(axis, points, colors[size], view, set(selected), radius)
                        axis.axhline(0, color="0.4", linestyle=":", linewidth=1)
                        axis.set(xscale="log", xlabel="Regularization weight η",
                                 ylabel=f"{title} improvement (%) ↑", title=f"{title} · dual learning rate = {rate:g}")
                        axis.set_xticks(selected, [f"{eta:g}" for eta in selected])
                        axis.minorticks_off()
                        if selected:
                            axis.set_xlim(selected[0] / 1.4, selected[-1] * 1.4)
                        else:
                            axis.set_xlim(1, 10)
                            axis.text(.5, .5, "No eta has a sample in the zoom window", transform=axis.transAxes,
                                      ha="center", va="center", fontsize=9)
                        if zoom:
                            axis.set_ylim(-radius, radius)
                        axis.grid(color="0.9", linestyle="--", linewidth=.7)
                handles = [Line2D([0], [0], color=colors[size], marker="o", label=f"B = {size}") for size in batch_sizes]
                handles += [Line2D([0], [0], color="0.3", marker=self.VIEWS[view][1],
                                   linestyle=self.VIEWS[view][2], label=self.VIEWS[view][0]) for view in views]
                handles.append(Line2D([0], [0], color="0.4", linestyle=":", label="Ordinary LoRA (0%)"))
                figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1 - .4 / height),
                              ncol=min(len(handles), 8), frameon=False)
                figure.suptitle("Target geometry: improvement over ordinary LoRA" + (" · automatic zoom" if zoom else " · all models"))
                figure.text(.5, .85 / height, "Improvement (%) = 100 × (ordinary LoRA score − model score) / ordinary LoRA score.  Positive is better; 0% is unregularized LoRA.", ha="center", fontsize=9)
                figure.text(.5, .60 / height, "Scores: held-out target MSE, source MSE and target FID. Each point is one seed, normalized to its own LoRA baseline; lines connect the same seed/view/B.", ha="center", fontsize=9)
                caption = ("Per metric: symmetric range = |best mean improvement across seeds| + 5 percentage points, shared across learning rates."
                           if zoom else "Color: geometry batch size B. Marker/line style: full, target-only or source-only view. All evaluated eta values are shown.")
                figure.text(.5, .35 / height, caption, ha="center", fontsize=9)
                if zoom:
                    figure.text(.5, .10 / height, "Eta is omitted per panel only if every sample is outside its window. Boundary triangles mark off-scale values; dotted segments touch or skip off-scale samples.", ha="center", fontsize=8.5)
                figure.tight_layout(rect=(0, 1.15 / height, 1, 1 - 1.05 / height))
                path = self.output_dir / filename
                figure.savefig(path, dpi=200)
                paths.append(path)
            finally:
                plt.close(figure)
        # Delete only this plotter's obsolete generated outputs, after successful rendering.
        for path in self.output_dir.glob("eta_retention_gain_dual_*.png"):
            path.unlink()
        (self.output_dir / "target_geometry.png").unlink(missing_ok=True)
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
                    filename = "figure_5_generations.png"
                    caption = f"One shared set of N(0, I) latent vectors for every model; visual seed = {self.config.get('visual_seed', 123)}."
                else:
                    label = self.config.get(f"{split}_label", split.title())
                    title = f"{label} reconstructions"
                    filename = f"figure_{3 if split == 'source' else 4}_{split}_reconstructions.png"
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
            finally:
                plt.close(figure)
        for filename in ("target_reconstructions.png", "figure_5_source_reconstructions.png",
                         "figure_6_target_reconstructions.png", "figure_7_generations.png"):
            (self.output_dir / filename).unlink(missing_ok=True)
        return paths
