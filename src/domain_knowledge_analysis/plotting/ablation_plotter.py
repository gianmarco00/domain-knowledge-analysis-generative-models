from pathlib import Path
from statistics import fmean
import math

import matplotlib.pyplot as plt
import torch
from scipy.stats import t


class AblationPlotter:
    MODE_COLORS = {
        "semantic_perturbations": "tab:blue",
        "random_perturbations": "tab:orange",
        "both": "tab:green",
    }

    MODE_LABELS = {
        "semantic_perturbations": "Semantic",
        "random_perturbations": "Random",
        "both": "Semantic + random",
    }

    def __init__(self, output_dir, config):
        self.output_dir = Path(output_dir) / "figures"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.modes = config["transformation_modes"]
        self.eta_values = sorted(set([0.0, *[float(eta) for eta in config["eta_values"]]]))

    def rows_for(self, results, eta, mode, m):
        if eta == 0:
            return [row for row in results if row["eta"] == 0]

        return [row for row in results if row["eta"] == eta and row["mode"] == mode and row["m"] == m]

    @staticmethod
    def mean_ci(values):
        values = torch.tensor(values, dtype=torch.float32)
        mean = values.mean().item()

        if len(values) == 1:
            return mean, 0.0

        critical_value = t.ppf(0.975, df=len(values) - 1)
        ci = critical_value * values.std(unbiased=True).item() / math.sqrt(len(values))
        return mean, ci

    @staticmethod
    def grouped_values(rows, group_name, metric):
        groups = {}

        for row in rows:
            groups.setdefault(row[group_name], []).append(float(row[metric]))

        return [fmean(values) for values in groups.values()]

    def plot_figure_1(self, results):
        figure, axis = plt.subplots(figsize=(9, 6.5))
        ordinary_rows = [row for row in results if row["eta"] == 0]
        ordinary_source = [float(row["test_source_degradation"]) for row in ordinary_rows]
        ordinary_target = [float(row["test_target_gain"]) for row in ordinary_rows]
        ordinary_source_mean, ordinary_source_ci = self.mean_ci(ordinary_source)
        ordinary_target_mean, ordinary_target_ci = self.mean_ci(ordinary_target)

        for mode in self.modes:
            color = self.MODE_COLORS[mode]
            full_m = max(self.config["m_values"][mode])
            mean_points = []

            for eta in self.eta_values:
                if eta == 0:
                    mean_points.append((ordinary_source_mean, ordinary_target_mean))
                    continue

                rows = self.rows_for(results, eta, mode, full_m)
                source_values = [float(row["test_source_degradation"]) for row in rows]
                target_values = [float(row["test_target_gain"]) for row in rows]
                source_mean, source_ci = self.mean_ci(source_values)
                target_mean, target_ci = self.mean_ci(target_values)

                axis.scatter(source_values, target_values, s=18, color=color, alpha=0.18)
                axis.errorbar(source_mean, target_mean, xerr=source_ci, yerr=target_ci, fmt="o", color=color, markersize=7, capsize=3)
                axis.annotate(f"η={eta:g}", (source_mean, target_mean), xytext=(5, 5), textcoords="offset points", color=color, fontsize=8)
                mean_points.append((source_mean, target_mean))

            axis.plot([point[0] for point in mean_points], [point[1] for point in mean_points], color=color, linewidth=1.5, label=self.MODE_LABELS[mode])

        axis.errorbar(ordinary_source_mean, ordinary_target_mean, xerr=ordinary_source_ci, yerr=ordinary_target_ci, fmt="D", color="0.25", markersize=7, capsize=3, label="Ordinary LoRA")
        axis.annotate("η=0", (ordinary_source_mean, ordinary_target_mean), xytext=(5, 5), textcoords="offset points", color="0.25", fontsize=8)
        axis.scatter([0], [0], marker="*", s=180, color="black", label="Original VAE")
        axis.axhline(0, color="0.85", linewidth=1)
        axis.axvline(0, color="0.85", linewidth=1)
        axis.set_xlabel("Source degradation ↓")
        axis.set_ylabel("Target gain ↑")
        axis.set_title("Effect of regularization strength")
        axis.legend()
        axis.text(0.02, 0.98, "Ideal: top-left", transform=axis.transAxes, va="top", fontsize=9)

        figure.text(0.5, 0.01, "Faint points are seeds; error bars are 95% t confidence intervals over seeds.", ha="center", fontsize=9)
        figure.tight_layout(rect=[0, 0.04, 1, 1])
        figure.savefig(self.output_dir / "figure_1_eta_tradeoff.png", dpi=200)
        plt.close(figure)

    def plot_figure_2(self, results, eta_star):
        metrics = [
            ("test_target_gain", "Target gain ↑"),
            ("test_source_degradation", "Source degradation ↓"),
            ("held_out_gram_error", "Held-out Gram error ↓"),
        ]

        figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
        ordinary_rows = [row for row in results if row["eta"] == 0]

        for axis_index, (axis, (metric, label)) in enumerate(zip(axes, metrics)):
            ordinary_values = [float(row[metric]) for row in ordinary_rows]
            ordinary_mean, ordinary_ci = self.mean_ci(ordinary_values)
            ordinary_label = "Ordinary LoRA" if axis_index == 0 else None
            axis.scatter([0] * len(ordinary_values), ordinary_values, s=18, color="0.3", alpha=0.18)
            axis.errorbar(0, ordinary_mean, yerr=ordinary_ci, fmt="D", color="0.3", markersize=7, capsize=4, label=ordinary_label)

            if eta_star != 0:
                for mode in self.modes:
                    color = self.MODE_COLORS[mode]
                    m_values = self.config["m_values"][mode]
                    means = []

                    for m in m_values:
                        rows = self.rows_for(results, eta_star, mode, int(m))
                        seed_values = self.grouped_values(rows, "seed", metric)
                        subset_values = self.grouped_values(rows, "subset_index", metric)
                        mean, ci = self.mean_ci(seed_values)
                        means.append(mean)

                        offsets = torch.linspace(-0.12, 0.12, len(subset_values)) if len(subset_values) > 1 else torch.zeros(1)
                        axis.scatter([m + offset.item() for offset in offsets], subset_values, s=20, color=color, alpha=0.25)
                        axis.errorbar(m, mean, yerr=ci, fmt="o", color=color, markersize=8, capsize=4)

                    mode_label = self.MODE_LABELS[mode] if axis_index == 0 else None
                    axis.plot([0, *m_values], [ordinary_mean, *means], color=color, linewidth=1.5, label=mode_label)

            axis.set_xticks(sorted(set([0, *(m for mode in self.modes for m in self.config["m_values"][mode])])))
            axis.set_xlabel("Number of transformation families M")
            axis.set_ylabel(label)

        axes[0].legend()
        figure.suptitle(f"Transformation-family ablation at η*={eta_star:g}")
        figure.text(0.5, 0.01, "Faint points are subset means; error bars are 95% t confidence intervals over seed means.", ha="center", fontsize=9)
        figure.tight_layout(rect=[0, 0.04, 1, 1])
        figure.savefig(self.output_dir / "figure_2_transformation_count.png", dpi=200)
        plt.close(figure)

    def plot_heatmaps(self, results):
        metrics = [
            ("test_target_gain", "Target gain ↑", "viridis"),
            ("test_source_degradation", "Source degradation ↓", "viridis_r"),
        ]

        figure, axes = plt.subplots(len(self.modes), len(metrics), figsize=(13, 4 * len(self.modes)), squeeze=False)
        matrices = {}

        for mode in self.modes:
            for metric, title, color_map in metrics:
                matrices[mode, metric] = torch.tensor([
                    [fmean(float(row[metric]) for row in self.rows_for(results, eta, mode, int(m))) for eta in self.eta_values]
                    for m in self.config["m_values"][mode]
                ])

        metric_limits = {
            metric: (
                min(matrix.min().item() for (mode, matrix_metric), matrix in matrices.items() if matrix_metric == metric),
                max(matrix.max().item() for (mode, matrix_metric), matrix in matrices.items() if matrix_metric == metric),
            )
            for metric, title, color_map in metrics
        }

        for row_index, mode in enumerate(self.modes):
            m_values = self.config["m_values"][mode]

            for column_index, (metric, title, color_map) in enumerate(metrics):
                axis = axes[row_index, column_index]
                matrix = matrices[mode, metric]
                value_min, value_max = metric_limits[metric]
                image = axis.imshow(matrix, aspect="auto", origin="lower", cmap=color_map, vmin=value_min, vmax=value_max)
                axis.set_xticks(range(len(self.eta_values)), [f"{eta:g}" for eta in self.eta_values])
                axis.set_yticks(range(len(m_values)), [str(m) for m in m_values])
                axis.set_xlabel("η")
                axis.set_ylabel("M")
                axis.set_title(f"{self.MODE_LABELS[mode]} — {title}")

                for matrix_row in range(len(m_values)):
                    for matrix_column in range(len(self.eta_values)):
                        value = matrix[matrix_row, matrix_column].item()
                        red, green, blue, _ = image.cmap(image.norm(value))
                        text_color = "black" if 0.299 * red + 0.587 * green + 0.114 * blue > 0.5 else "white"
                        axis.text(matrix_column, matrix_row, f"{value:.3g}", ha="center", va="center", color=text_color, fontsize=8)

                figure.colorbar(image, ax=axis)

        figure.tight_layout()
        figure.savefig(self.output_dir / "supplementary_eta_M_heatmaps.png", dpi=200)
        plt.close(figure)

    def plot_qualitative(self, results):
        self.plot_reconstructions(results["source"], "Clean MNIST reconstructions", "figure_3a_source_reconstructions.png")
        self.plot_reconstructions(results["target"], "Rotated MNIST reconstructions", "figure_3b_target_reconstructions.png")
        self.plot_generations(results["generation"])

    def plot_reconstructions(self, results, title, filename):
        columns = [
            ("input", "Input"),
            ("base", "Original VAE"),
            ("lora", "LoRA"),
            ("regularized", "Regularized LoRA"),
        ]

        num_images = len(results["input"])
        figure, axes = plt.subplots(num_images, len(columns), figsize=(8, 2 * num_images), squeeze=False)

        for row in range(num_images):
            for column, (key, label) in enumerate(columns):
                self.show_image(axes[row, column], results[key][row])

                if row == 0:
                    axes[row, column].set_title(label)

        figure.suptitle(title)
        figure.tight_layout()
        figure.savefig(self.output_dir / filename, dpi=200)
        plt.close(figure)

    def plot_generations(self, results):
        columns = [
            ("base", "Original VAE"),
            ("lora", "LoRA"),
            ("regularized", "Regularized LoRA"),
        ]

        num_images = len(results["base"])
        figure, axes = plt.subplots(num_images, len(columns), figsize=(6, 2 * num_images), squeeze=False)

        for row in range(num_images):
            for column, (key, label) in enumerate(columns):
                self.show_image(axes[row, column], results[key][row])

                if row == 0:
                    axes[row, column].set_title(label)

        figure.suptitle("Generation using the same latent vectors")
        figure.tight_layout()
        figure.savefig(self.output_dir / "figure_3c_generations.png", dpi=200)
        plt.close(figure)

    @staticmethod
    def show_image(axis, image):
        image = image.detach().cpu()

        if image.shape[0] == 1:
            axis.imshow(image[0], cmap="gray", vmin=0, vmax=1)
        else:
            axis.imshow(image.permute(1, 2, 0), vmin=0, vmax=1)

        axis.set_xticks([])
        axis.set_yticks([])
