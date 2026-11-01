import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class AblationPlotter:
    REGULARIZED_COLOR = "#0072B2"
    ORDINARY_COLOR = "#7A7A7A"
    SOURCE_LOSS_COLOR = "#CC79A7"

    def __init__(self, output_dir, config):
        self.output_dir = Path(output_dir) / "figures"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config

    @staticmethod
    def split_rows(results):
        ordinary = [row for row in results if float(row["eta"]) == 0]
        source_loss = [row for row in results if row.get("mode") == "source_loss" and float(row["eta"]) != 0]
        regularized = [row for row in results if float(row["eta"]) != 0 and row.get("mode") != "source_loss"]
        return regularized, ordinary, source_loss

    @staticmethod
    def ordinary_by_seed(ordinary_rows):
        baselines = {}
        for row in ordinary_rows:
            seed = row.get("seed")
            if seed in baselines:
                raise ValueError(f"Multiple ordinary LoRA baselines for seed {seed}.")
            baselines[seed] = row
        return baselines

    @staticmethod
    def percentage(row, metric, baselines):
        """Compare each signed MSE change with ordinary LoRA for the same seed."""
        seed = row.get("seed")
        if seed not in baselines:
            raise ValueError(f"No ordinary LoRA baseline for seed {seed}.")
        reference = float(baselines[seed][metric])
        return 100 * float(row[metric]) / reference if reference != 0 else math.nan

    @staticmethod
    def add_metric_formulas(figure, baselines):
        lines = [
            "Target gain (%) = 100 × [MSE_target(S) − MSE_target(P)] / [MSE_target(S) − MSE_target(L)]",
            "Source degradation (%) = 100 × [MSE_source(P) − MSE_source(S)] / [MSE_source(L) − MSE_source(S)]",
            "MSE = mean squared reconstruction error, measured on the target or source images.",
            "S = Source VAE, trained on source images.  L = LoRA VAE, adapted to target images without a regularizer.",
            "P = plotted model: L, source-loss LoRA, or regularized LoRA (S adapted to target images with a regularizer).",
        ]
        if any(float(row[metric]) <= 0 for row in baselines.values() for metric in ("test_target_gain", "test_source_degradation")):
            lines.append("Zero LoRA change is undefined; a negative LoRA change reverses the percentage direction.")
        height = figure.get_figheight()
        for index, line in enumerate(reversed(lines)):
            figure.text(0.5, (0.18 + index * 0.20) / height, line, ha="center", va="center", fontsize=9)

    @staticmethod
    def metric_label(metric, baselines):
        values = [float(row[metric]) for row in baselines.values() if float(row[metric]) != 0]
        label = "Target gain" if metric == "test_target_gain" else "Source degradation"
        preferred = "↑" if metric == "test_target_gain" else "↓"
        opposite = "↓" if preferred == "↑" else "↑"
        direction = preferred if values and all(value > 0 for value in values) else opposite if values and all(value < 0 for value in values) else ""
        return f"{label} (%) {direction}".rstrip()

    @staticmethod
    def set_log_ticks(axis, positive_values):
        if not positive_values:
            return
        linthresh = positive_values[0] / 2
        axis.set_xscale("symlog", linthresh=linthresh)
        def label(value):
            exponent = round(math.log10(value))
            if value >= 1000 and math.isclose(value, 10 ** exponent, rel_tol=1e-9):
                return rf"$10^{{{exponent}}}$"
            return f"{value:g}"
        axis.set_xticks([0.0, *positive_values], ["0", *[label(value) for value in positive_values]])
        axis.set_xlim(-linthresh * 0.5, positive_values[-1] * 1.5)

    @staticmethod
    def legend_handles(color_values, colors):
        handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[value], markeredgecolor="white", markersize=8, label=f"{value:g}") for value in color_values]
        handles.append(Line2D([0], [0], marker="D", color="none", markerfacecolor=AblationPlotter.ORDINARY_COLOR, markersize=8, label="Ordinary LoRA"))
        handles.append(Line2D([0], [0], marker="*", color="none", markerfacecolor=AblationPlotter.SOURCE_LOSS_COLOR, markersize=12, label="Source-loss LoRA"))
        return handles

    def plot_figure_1(self, results):
        regularized, _, _ = self.split_rows(results)
        strengths = sorted({float(row["s"]) for row in regularized})
        self.plot_metric_pair(results, "eta", "m", "η", "M", "Effect of regularization weight", "figure_1_eta.png", facets=strengths)

    def plot_figure_2(self, results, selected_model, selection="best_overall"):
        strength = float(selected_model["s"])
        self.plot_metric_pair(results, "m", "eta", "Number of transformations M", "η", f"Effect of transformation count · s={strength:g} ({self.selection_label(selection)})", "figure_2_transformation_count.png", filters={"s": strength})

    def plot_figure_3(self, results, selected_model, selection="best_overall"):
        eta = float(selected_model["eta"])
        self.plot_metric_pair(results, "s", "m", "Transformation strength s", "M", f"Effect of transformation strength · η={eta:g} ({self.selection_label(selection)})", "figure_3_transformation_strength.png", filters={"eta": eta})

    @staticmethod
    def selection_label(selection):
        return {"best_overall": "best overall", "best_target_gain": "best target gain fallback", "manual": "manually selected"}[selection]

    def plot_metric_pair(self, results, x_name, color_name, x_label, color_label, title, filename, facets=None, filters=None):
        regularized, ordinary, source_loss = self.split_rows(results)
        for key, value in (filters or {}).items():
            regularized = [row for row in regularized if float(row[key]) == value]
        baselines = self.ordinary_by_seed(ordinary)
        color_values = sorted({float(row[color_name]) for row in regularized})
        colors = {value: plt.get_cmap("tab10")(index % 10) for index, value in enumerate(color_values)}
        metrics = [(metric, self.metric_label(metric, baselines)) for metric in ("test_target_gain", "test_source_degradation")]
        facet_values = facets if facets is not None else [None]
        if not facet_values:
            raise ValueError("No regularized models to plot.")
        height = 2.9 + 3.7 * len(facet_values)
        figure, axes = plt.subplots(len(facet_values), 2, figsize=(12, height), squeeze=False)

        for facet_index, strength in enumerate(facet_values):
            facet_rows = regularized if strength is None else [row for row in regularized if float(row["s"]) == strength]
            for axis, (metric, y_label) in zip(axes[facet_index], metrics):
                for color_value in color_values:
                    rows = [row for row in facet_rows if float(row[color_name]) == color_value]
                    # Keep independent seeds and run types separate; repeated x values
                    # are ambiguous, so leave those groups as points only.
                    groups = {}
                    for row in rows:
                        key = (row.get("seed"), row.get("mode"), row.get("extra_run", False))
                        groups.setdefault(key, []).append(row)
                    for group in groups.values():
                        ordered = sorted(group, key=lambda row: float(row[x_name]))
                        xs = [float(row[x_name]) for row in ordered]
                        if len(xs) > 1 and len(set(xs)) == len(xs):
                            axis.plot(xs, [self.percentage(row, metric, baselines) for row in ordered], color=colors[color_value], linewidth=1.2, alpha=0.8, zorder=1)
                    axis.scatter([float(row[x_name]) for row in rows], [self.percentage(row, metric, baselines) for row in rows], s=55, color=colors[color_value], edgecolor="white", linewidth=0.6, alpha=0.8, zorder=2)

                axis.scatter([0.0] * len(ordinary), [self.percentage(row, metric, baselines) for row in ordinary], s=72, marker="D", color=self.ORDINARY_COLOR, edgecolor="white", linewidth=0.7, zorder=3)
                axis.scatter([float(row[x_name]) for row in source_loss], [self.percentage(row, metric, baselines) for row in source_loss], s=150, marker="*", color=self.SOURCE_LOSS_COLOR, edgecolor="white", linewidth=0.7, zorder=4)
                axis.axhline(0, color="0.75", linewidth=0.8)
                axis.axhline(100, color="0.7", linewidth=0.8, linestyle=":")
                axis.set_xlabel(x_label)
                axis.set_ylabel(y_label)
                if strength is not None:
                    axis.set_title(f"s = {strength:g}")
                axis.grid(color="0.9", linestyle="--", linewidth=0.7)
                if x_name in {"eta", "s"}:
                    positive_values = sorted({float(row[x_name]) for row in facet_rows + source_loss if float(row[x_name]) > 0})
                    self.set_log_ticks(axis, positive_values)
                else:
                    axis.set_xticks(sorted({0, *[int(row[x_name]) for row in facet_rows + source_loss]}))

        handles = self.legend_handles(color_values, colors)
        figure.legend(handles=handles, title=color_label, loc="upper center", bbox_to_anchor=(0.5, 1 - 0.38 / height), ncol=min(len(handles), 6), frameon=False)
        figure.suptitle(title)
        self.add_metric_formulas(figure, baselines)
        # Reserve a fixed amount of space for the legend and formula block.
        figure.tight_layout(rect=(0, 1.35 / height, 1, 1 - 1.12 / height))
        figure.savefig(self.output_dir / filename, dpi=200)
        plt.close(figure)

    def plot_figure_4(self, results):
        regularized, ordinary, source_loss = self.split_rows(results)
        baselines = self.ordinary_by_seed(ordinary)
        figure, axis = plt.subplots(figsize=(12, 7.5))
        for rows, color, marker, size, label in [
            (regularized, self.REGULARIZED_COLOR, "o", 58, "Regularized LoRA"),
            (ordinary, self.ORDINARY_COLOR, "D", 82, "Ordinary LoRA"),
            (source_loss, self.SOURCE_LOSS_COLOR, "*", 160, "Source-loss LoRA"),
        ]:
            axis.scatter([self.percentage(row, "test_source_degradation", baselines) for row in rows], [self.percentage(row, "test_target_gain", baselines) for row in rows], s=size, marker=marker, color=color, edgecolor="white", linewidth=0.6, alpha=0.8, label=label)
        axis.set_xlabel(self.metric_label("test_source_degradation", baselines))
        axis.set_ylabel(self.metric_label("test_target_gain", baselines))
        axis.set_title("Target learning versus source preservation")
        axis.grid(color="0.9", linestyle="--", linewidth=0.7)
        axis.legend()
        self.add_metric_formulas(figure, baselines)
        figure.tight_layout(rect=(0, 1.35 / figure.get_figheight(), 1, 1))
        figure.savefig(self.output_dir / "figure_4_target_source_tradeoff.png", dpi=200)
        plt.close(figure)

    def plot_fid(self, results, selected_model, selection="best_overall"):
        regularized, ordinary, source_loss = self.split_rows(results)
        strength = float(selected_model["s"])
        regularized = [row for row in regularized if float(row["s"]) == strength]
        m_values = sorted({int(row["m"]) for row in regularized})
        colors = {value: plt.get_cmap("tab10")(index % 10) for index, value in enumerate(m_values)}
        figure, axis = plt.subplots(figsize=(10, 6))
        for m in m_values:
            rows = [row for row in regularized if int(row["m"]) == m]
            axis.scatter([float(row["eta"]) for row in rows], [float(row["test_target_fid"]) for row in rows], s=55, color=colors[m], edgecolor="white", linewidth=0.6, alpha=0.8)
        axis.scatter([0.0] * len(ordinary), [float(row["test_target_fid"]) for row in ordinary], s=72, marker="D", color=self.ORDINARY_COLOR, edgecolor="white", linewidth=0.7, zorder=3)
        axis.scatter([float(row["eta"]) for row in source_loss], [float(row["test_target_fid"]) for row in source_loss], s=150, marker="*", color=self.SOURCE_LOSS_COLOR, edgecolor="white", linewidth=0.7, zorder=4)
        positive_values = sorted({float(row["eta"]) for row in regularized + source_loss if float(row["eta"]) > 0})
        self.set_log_ticks(axis, positive_values)
        handles = self.legend_handles(m_values, colors)
        figure.legend(handles=handles, title="M", loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=min(len(handles), 6), frameon=False)
        axis.set_xlabel("η")
        axis.set_ylabel("Target FID ↓")
        target_label = self.config.get("target_label", "Rotated MNIST")
        figure.suptitle(f"Generated images versus held-out {target_label} · s={strength:g} ({self.selection_label(selection)})")
        axis.grid(color="0.9", linestyle="--", linewidth=0.7)
        figure.tight_layout(rect=(0, 0, 1, 0.82))
        figure.savefig(self.output_dir / "figure_9_target_fid.png", dpi=200)
        plt.close(figure)

    @staticmethod
    def relative_improvement(row, metric, baselines):
        """Percentage reduction in a test error compared with ordinary LoRA."""
        seed = row.get("seed")
        if seed not in baselines:
            raise ValueError(f"No ordinary LoRA baseline for seed {seed}.")
        baseline = float(baselines[seed][metric])
        value = float(row[metric])
        if not math.isfinite(baseline) or baseline <= 0 or not math.isfinite(value):
            return math.nan
        return 100 * (baseline - value) / baseline

    def plot_local_sweep(self, results, model_ids):
        regularized, ordinary, _ = self.split_rows(results)
        rows = [row for row in regularized if row["model_id"] in model_ids]
        if not rows:
            raise ValueError("No evaluated models match the local sweep.")
        baselines = self.ordinary_by_seed(ordinary)
        strengths = sorted({float(row["s"]) for row in rows})
        m_values = sorted({int(row["m"]) for row in rows})
        seeds = sorted({row.get("seed") for row in rows}, key=str)
        colors = {m: plt.get_cmap("tab10")(index % 10) for index, m in enumerate(m_values)}
        line_styles = ["-", "--", ":", "-."]
        metrics = [
            ("test_adapted_target_mse", "Target MSE improvement (%) ↑"),
            ("test_adapted_source_mse", "Source MSE improvement (%) ↑"),
            ("test_target_fid", "Target FID improvement (%) ↑"),
        ]
        height = 2.3 + 3.7 * len(strengths)
        figure, axes = plt.subplots(len(strengths), 3, figsize=(16, height), squeeze=False)

        for row_index, strength in enumerate(strengths):
            strength_rows = [row for row in rows if float(row["s"]) == strength]
            eta_values = sorted({float(row["eta"]) for row in strength_rows})
            for axis, (metric, label) in zip(axes[row_index], metrics):
                for m in m_values:
                    for seed_index, seed in enumerate(seeds):
                        series = sorted((row for row in strength_rows if int(row["m"]) == m and row.get("seed") == seed), key=lambda row: float(row["eta"]))
                        if not series:
                            continue
                        axis.plot([float(row["eta"]) for row in series],
                                  [self.relative_improvement(row, metric, baselines) for row in series],
                                  color=colors[m], linestyle=line_styles[seed_index % len(line_styles)],
                                  marker="o", markersize=6, markeredgecolor="white", markeredgewidth=0.6,
                                  linewidth=1.4, alpha=0.9)
                axis.axhline(0, color="0.45", linewidth=0.9, linestyle=":")
                axis.set_xscale("log")
                axis.set_xticks(eta_values, [f"{value:g}" for value in eta_values])
                axis.set_xlim(eta_values[0] / 1.3, eta_values[-1] * 1.3)
                axis.set_xlabel("Regularization weight η")
                axis.set_ylabel(label)
                axis.set_title(f"s = {strength:g}")
                axis.grid(color="0.9", linestyle="--", linewidth=0.7)

        handles = [Line2D([0], [0], color=colors[m], marker="o", markeredgecolor="white", label=f"{m:g}") for m in m_values]
        if len(seeds) > 1:
            handles += [Line2D([0], [0], color="0.35", linestyle=line_styles[index % len(line_styles)], label=f"Seed {seed}") for index, seed in enumerate(seeds)]
        figure.legend(handles=handles, title="M", loc="upper center", bbox_to_anchor=(0.5, 1 - 0.38 / height), ncol=min(len(handles), 8), frameon=False)
        figure.suptitle("Local parameter sweep: improvement over ordinary LoRA")
        figure.text(0.5, 0.54 / height, "Improvement (%) = 100 × (ordinary LoRA score − regularized LoRA score) / ordinary LoRA score", ha="center", fontsize=10)
        figure.text(0.5, 0.27 / height, "Score = test target MSE, test source MSE, or target FID; each model uses the ordinary LoRA baseline for the same seed.  Above 0% is better.", ha="center", fontsize=9)
        figure.tight_layout(rect=(0, 0.9 / height, 1, 1 - 1.05 / height))
        figure.savefig(self.output_dir / "figure_10_local_parameter_sweep.png", dpi=200)
        plt.close(figure)

    def plot_qualitative(self, results):
        selected_label = self.selected_model_label(results["selected_model"])
        source_label = self.config.get("source_label", "Clean MNIST")
        target_label = self.config.get("target_label", "Rotated MNIST")
        self.plot_reconstructions(results["source"], f"{source_label} reconstructions", "figure_5_source_reconstructions.png", selected_label)
        self.plot_reconstructions(results["target"], f"{target_label} reconstructions", "figure_6_target_reconstructions.png", selected_label)
        self.plot_generations(results["generation"], selected_label)
        self.plot_model_responses(results["responses"])

    def plot_reconstructions(self, results, title, filename, selected_label):
        columns = [("input", "Input"), ("base", "Original VAE"), ("lora", "LoRA"), ("source_loss", "Source-loss LoRA"), ("regularized", f"Regularized LoRA\n{selected_label}")]
        num_images = len(results["input"])
        figure, axes = plt.subplots(num_images, len(columns), figsize=(10, 2 * num_images), squeeze=False)

        for row in range(num_images):
            for column, (key, label) in enumerate(columns):
                self.show_image(axes[row, column], results[key][row])

                if row == 0:
                    axes[row, column].set_title(label)

        figure.suptitle(title)
        figure.tight_layout()
        figure.savefig(self.output_dir / filename, dpi=200)
        plt.close(figure)

    def plot_generations(self, results, selected_label):
        columns = [("base", "Original VAE"), ("lora", "LoRA"), ("source_loss", "Source-loss LoRA"), ("regularized", f"Regularized LoRA\n{selected_label}")]
        num_images = len(results["base"])
        figure, axes = plt.subplots(num_images, len(columns), figsize=(8, 2 * num_images), squeeze=False)

        for row in range(num_images):
            for column, (key, label) in enumerate(columns):
                self.show_image(axes[row, column], results[key][row])

                if row == 0:
                    axes[row, column].set_title(label)

        figure.suptitle("Generation using the same latent vectors")
        figure.tight_layout()
        figure.savefig(self.output_dir / "figure_7_generations.png", dpi=200)
        plt.close(figure)

    def plot_model_responses(self, results):
        num_transformations = len(results["transformation_labels"])
        num_images = len(results["input"])
        num_rows = num_transformations * num_images
        selected_label = self.selected_model_label(results["selected_model"])
        response_limit = max(results["lora"].abs().max().item(), results["regularized"].abs().max().item(), 1e-6)
        figure, axes = plt.subplots(num_rows, 4, figsize=(11, 1.9 * num_rows), squeeze=False)
        response_image = None

        for transformation_index, transformation_label in enumerate(results["transformation_labels"]):
            for image_index in range(num_images):
                row = transformation_index * num_images + image_index
                self.show_image(axes[row, 0], results["input"][image_index])
                self.show_image(axes[row, 1], results["transformed"][transformation_index, image_index])
                response_image = self.show_response(axes[row, 2], results["lora"][transformation_index, image_index], response_limit)
                self.show_response(axes[row, 3], results["regularized"][transformation_index, image_index], response_limit)
                axes[row, 0].set_ylabel(f"{transformation_label}\nexample {image_index + 1}", fontsize=8)

                if row == 0:
                    axes[row, 0].set_title("Input", fontsize=10, pad=10)
                    axes[row, 1].set_title("Transformed\ninput", fontsize=10, pad=10)
                    axes[row, 2].set_title("Ordinary LoRA\nresponse", fontsize=10, pad=10)
                    axes[row, 3].set_title("Regularized LoRA\nresponse", fontsize=10, pad=10)

        colorbar = figure.colorbar(response_image, ax=axes[:, 2:].ravel().tolist(), fraction=0.02, pad=0.02)
        colorbar.set_label("h(T(x)) − h(x)")
        source_label = self.config.get("source_label", "Clean MNIST")
        figure.suptitle(f"Model responses on {source_label}", y=0.99)
        figure.text(0.5, 0.955, f"Selected regularized LoRA: {selected_label}", ha="center", fontsize=10)
        figure.text(0.5, 0.93, "Red increases output intensity; blue decreases it", ha="center", fontsize=9)
        figure.subplots_adjust(left=0.15, right=0.9, top=0.84, bottom=0.03, wspace=0.16, hspace=0.18)
        figure.savefig(self.output_dir / "figure_8_model_responses.png", dpi=200)
        plt.close(figure)

    @staticmethod
    def selected_model_label(selected_model):
        return f"η={float(selected_model['eta']):g}, M={int(selected_model['m'])}, s={float(selected_model['s']):g}"

    @staticmethod
    def show_response(axis, response, limit):
        response = response.detach().cpu()
        image = axis.imshow(response[0], cmap="coolwarm", vmin=-limit, vmax=limit)
        axis.set_xticks([])
        axis.set_yticks([])
        return image

    @staticmethod
    def show_image(axis, image):
        image = image.detach().cpu()

        if image.shape[0] == 1:
            axis.imshow(image[0], cmap="gray", vmin=0, vmax=1)
        else:
            axis.imshow(image.permute(1, 2, 0), vmin=0, vmax=1)

        axis.set_xticks([])
        axis.set_yticks([])
