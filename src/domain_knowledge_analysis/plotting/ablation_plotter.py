from pathlib import Path
from statistics import fmean
import math

import matplotlib.pyplot as plt
import torch
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy.stats import t


class AblationPlotter:
    MODE_COLORS = {
        "semantic_perturbations": "#0072B2",
        "random_perturbations": "#D55E00",
        "both": "#009E73",
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

    def plot_figure_1(self, results, selected_model):
        figure, axis = plt.subplots(figsize=(8.5, 6.5))
        ordinary_rows = [row for row in results if row["eta"] == 0]
        ordinary_source = [float(row["test_source_degradation"]) for row in ordinary_rows]
        ordinary_target = [float(row["test_target_gain"]) for row in ordinary_rows]
        ordinary_source_mean = fmean(ordinary_source)
        ordinary_target_mean = fmean(ordinary_target)
        positive_eta_values = [eta for eta in self.eta_values if eta != 0]
        markers = ["o", "s", "^", "P", "X", "v"]
        eta_markers = {eta: markers[index % len(markers)] for index, eta in enumerate(positive_eta_values)}

        for mode in self.modes:
            color = self.MODE_COLORS[mode]
            full_m = max(self.config["m_values"][mode])

            for eta in positive_eta_values:
                rows = self.rows_for(results, eta, mode, full_m)
                source_values = [1000 * float(row["test_source_degradation"]) for row in rows]
                target_values = [1000 * float(row["test_target_gain"]) for row in rows]
                source_mean = fmean(source_values)
                target_mean = fmean(target_values)

                axis.scatter(source_values, target_values, s=28, marker=eta_markers[eta], color=color, alpha=0.25)
                axis.scatter(source_mean, target_mean, s=95, marker=eta_markers[eta], color=color, edgecolor="white", linewidth=0.9, zorder=3)

                if mode == selected_model["mode"] and full_m == selected_model["m"] and eta == selected_model["eta"]:
                    axis.scatter(source_mean, target_mean, s=190, marker="o", facecolor="none", edgecolor="black", linewidth=1.8, zorder=4)

        axis.scatter([1000 * value for value in ordinary_source], [1000 * value for value in ordinary_target], s=28, marker="D", color="0.25", alpha=0.25)
        axis.scatter(1000 * ordinary_source_mean, 1000 * ordinary_target_mean, s=105, marker="D", color="0.25", edgecolor="white", linewidth=0.9, zorder=3)
        axis.set_xlabel("Source degradation (×10⁻³) ↓")
        axis.set_ylabel("Target gain (×10⁻³) ↑")
        axis.set_title("Target learning versus source preservation")
        axis.grid(color="0.9", linestyle="--", linewidth=0.7)
        axis.margins(x=0.12, y=0.18)
        axis.text(0.02, 0.98, "Better", transform=axis.transAxes, va="top", fontweight="bold")
        axis.annotate("", xy=(0.02, 0.90), xytext=(0.13, 0.79), xycoords="axes fraction", arrowprops={"arrowstyle": "->", "color": "0.25"})

        mode_handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=self.MODE_COLORS[mode], markeredgecolor="white", markersize=9, label=self.MODE_LABELS[mode]) for mode in self.modes]
        eta_handles = [Line2D([0], [0], marker=eta_markers[eta], color="0.3", linestyle="none", markersize=8, label=f"η={eta:g}") for eta in positive_eta_values]
        other_handles = [Line2D([0], [0], marker="D", color="none", markerfacecolor="0.25", markersize=8, label="Ordinary LoRA"), Line2D([0], [0], marker="o", color="black", markerfacecolor="none", markersize=10, label="Selected model")]
        mode_legend = axis.legend(handles=mode_handles, title="Transformation type", loc="lower left")
        axis.add_artist(mode_legend)
        axis.legend(handles=[*eta_handles, *other_handles], title="Regularization", loc="lower right")

        figure.text(0.5, 0.01, "Axes are zoomed to adapted models. Original VAE is at (0, 0), outside this view. Small points are individual seeds.", ha="center", fontsize=9)
        figure.tight_layout(rect=[0, 0.04, 1, 1])
        figure.savefig(self.output_dir / "figure_1_eta_tradeoff.png", dpi=200)
        plt.close(figure)

    def plot_figure_2(self, results, selected_model):
        metrics = [
            ("test_target_gain", "Target gain retained (%) ↑", 100.0),
            ("test_source_degradation", "Source damage reduced (%) ↑", 0.0),
            ("held_out_gram_error", "Gram error reduced (%) ↑", 0.0),
        ]

        figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
        ordinary_rows = [row for row in results if row["eta"] == 0]
        eta_star = float(selected_model["eta"])
        mode_offsets = {"semantic_perturbations": -0.18, "random_perturbations": 0.18, "both": 0.0}

        for axis_index, (axis, (metric, label, reference)) in enumerate(zip(axes, metrics)):
            ordinary_mean = fmean(float(row[metric]) for row in ordinary_rows)

            def relative(value):
                if metric == "test_target_gain":
                    return 100 * value / ordinary_mean

                return 100 * (ordinary_mean - value) / ordinary_mean

            axis.scatter(0, reference, s=105, marker="D", color="0.25", edgecolor="white", linewidth=0.9, zorder=3)

            for mode in self.modes:
                color = self.MODE_COLORS[mode]

                for m in self.config["m_values"][mode]:
                    rows = self.rows_for(results, eta_star, mode, int(m))
                    seed_values = [relative(value) for value in self.grouped_values(rows, "seed", metric)]
                    subset_values = [relative(value) for value in self.grouped_values(rows, "subset_index", metric)]
                    position = m + mode_offsets[mode]
                    offsets = torch.linspace(-0.10, 0.10, len(subset_values)) if len(subset_values) > 1 else torch.zeros(1)

                    axis.scatter([position + offset.item() for offset in offsets], subset_values, s=26, color=color, alpha=0.22)
                    axis.scatter([position] * len(seed_values), seed_values, s=34, marker="x", color=color, alpha=0.65)
                    axis.scatter(position, fmean(seed_values), s=95, color=color, edgecolor="white", linewidth=0.9, zorder=3)

                    if mode == selected_model["mode"] and m == selected_model["m"]:
                        axis.scatter(position, fmean(seed_values), s=190, facecolor="none", edgecolor="black", linewidth=1.8, zorder=4)

            axis.set_xticks(sorted(set([0, *(m for mode in self.modes for m in self.config["m_values"][mode])])))
            axis.set_xlabel("Number of transformation families M")
            axis.set_ylabel(label)
            axis.axhline(reference, color="0.75", linestyle="--", linewidth=1)
            axis.grid(axis="y", color="0.9", linestyle="--", linewidth=0.7)

        handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=self.MODE_COLORS[mode], markeredgecolor="white", markersize=9, label=self.MODE_LABELS[mode]) for mode in self.modes]
        handles.extend([Line2D([0], [0], marker="D", color="none", markerfacecolor="0.25", markersize=8, label="Ordinary LoRA"), Line2D([0], [0], marker="o", color="black", markerfacecolor="none", markersize=10, label="Selected model")])
        axes[0].legend(handles=handles)
        figure.suptitle(f"Transformation-family ablation at η*={eta_star:g}")
        figure.text(0.5, 0.01, "Values are relative to ordinary LoRA. Small circles are subset means; × markers are seed means.", ha="center", fontsize=9)
        figure.tight_layout(rect=[0, 0.04, 1, 1])
        figure.savefig(self.output_dir / "figure_2_transformation_count.png", dpi=200)
        plt.close(figure)

    def plot_heatmaps(self, results):
        ordinary_rows = [row for row in results if row["eta"] == 0]
        ordinary_target = fmean(float(row["test_target_gain"]) for row in ordinary_rows)
        ordinary_source = fmean(float(row["test_source_degradation"]) for row in ordinary_rows)
        eta_values = [eta for eta in self.eta_values if eta != 0]
        configurations = [(mode, int(m)) for mode in self.modes for m in self.config["m_values"][mode]]
        labels = [f"{self.MODE_LABELS[mode]} (M={m})" for mode, m in configurations]
        target_matrix = torch.tensor([[100 * fmean(float(row["test_target_gain"]) for row in self.rows_for(results, eta, mode, m)) / ordinary_target for eta in eta_values] for mode, m in configurations])
        source_matrix = torch.tensor([[100 * (ordinary_source - fmean(float(row["test_source_degradation"]) for row in self.rows_for(results, eta, mode, m))) / ordinary_source for eta in eta_values] for mode, m in configurations])
        figure, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        target_image = axes[0].imshow(target_matrix, aspect="auto", cmap="YlGn", vmin=95, vmax=max(100, target_matrix.max().item()))
        source_limit = max(abs(source_matrix.min().item()), abs(source_matrix.max().item()))
        source_image = axes[1].imshow(source_matrix, aspect="auto", cmap="RdYlGn", norm=TwoSlopeNorm(vmin=-source_limit, vcenter=0, vmax=source_limit))

        for axis, image, matrix, title, decimals in [(axes[0], target_image, target_matrix, "Target gain retained versus LoRA (%)", 2), (axes[1], source_image, source_matrix, "Source damage reduced versus LoRA (%)", 1)]:
            axis.set_xticks(range(len(eta_values)), [f"{eta:g}" for eta in eta_values])
            axis.set_yticks(range(len(labels)), labels)
            axis.set_xlabel("η")
            axis.set_title(title)

            for row in range(len(labels)):
                for column in range(len(eta_values)):
                    value = matrix[row, column].item()
                    red, green, blue, _ = image.cmap(image.norm(value))
                    text_color = "black" if 0.299 * red + 0.587 * green + 0.114 * blue > 0.52 else "white"
                    axis.text(column, row, f"{value:.{decimals}f}", ha="center", va="center", color=text_color, fontsize=9)

            figure.colorbar(image, ax=axis, shrink=0.82)

        figure.suptitle("Regularized models relative to ordinary LoRA")
        figure.tight_layout()
        figure.savefig(self.output_dir / "supplementary_eta_M_heatmaps.png", dpi=200)
        plt.close(figure)

    def plot_qualitative(self, results):
        selected_label = self.selected_model_label(results["selected_model"])
        self.plot_reconstructions(results["source"], "Clean MNIST reconstructions", "figure_3a_source_reconstructions.png", selected_label)
        self.plot_reconstructions(results["target"], "Rotated MNIST reconstructions", "figure_3b_target_reconstructions.png", selected_label)
        self.plot_generations(results["generation"], selected_label)
        self.plot_model_responses(results["responses"])

    def plot_reconstructions(self, results, title, filename, selected_label):
        columns = [
            ("input", "Input"),
            ("base", "Original VAE"),
            ("lora", "LoRA"),
            ("regularized", f"Regularized LoRA\n{selected_label}"),
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

    def plot_generations(self, results, selected_label):
        columns = [
            ("base", "Original VAE"),
            ("lora", "LoRA"),
            ("regularized", f"Regularized LoRA\n{selected_label}"),
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

    def plot_model_responses(self, results):
        num_transformations = len(results["transformation_labels"])
        num_images = len(results["input"])
        num_rows = num_transformations * num_images
        selected_label = self.selected_model_label(results["selected_model"])
        response_limit = max(results["lora"].abs().max().item(), results["regularized"].abs().max().item(), 1e-6)
        figure, axes = plt.subplots(num_rows, 4, figsize=(8.5, 1.9 * num_rows), squeeze=False)
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
                    axes[row, 0].set_title("Input")
                    axes[row, 1].set_title("Transformed input")
                    axes[row, 2].set_title("Ordinary LoRA response")
                    axes[row, 3].set_title(f"Regularized response\n{selected_label}")

        colorbar = figure.colorbar(response_image, ax=axes[:, 2:].ravel().tolist(), fraction=0.02, pad=0.02)
        colorbar.set_label("h(T(x)) − h(x)")
        figure.suptitle("Model responses on clean MNIST\nRed increases output intensity; blue decreases it")
        figure.subplots_adjust(left=0.17, right=0.89, top=0.93, bottom=0.03, wspace=0.08, hspace=0.12)
        figure.savefig(self.output_dir / "figure_4_model_responses.png", dpi=200)
        plt.close(figure)

    @classmethod
    def selected_model_label(cls, selected_model):
        return f"{cls.MODE_LABELS[selected_model['mode']]}, η={selected_model['eta']:g}, M={selected_model['m']}"

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
