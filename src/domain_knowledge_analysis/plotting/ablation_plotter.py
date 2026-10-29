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

    def plot_figure_1(self, results):
        self.plot_metric_pair(results, "eta", "m", "η", "M", "Effect of regularization weight", "figure_1_eta.png")

    def plot_figure_2(self, results):
        self.plot_metric_pair(results, "m", "eta", "Number of transformations M", "η", "Effect of transformation count", "figure_2_transformation_count.png")

    def plot_figure_3(self, results):
        self.plot_metric_pair(results, "s", "m", "Transformation strength s", "M", "Effect of transformation strength", "figure_3_transformation_strength.png")

    def plot_metric_pair(self, results, x_name, color_name, x_label, color_label, title, filename):
        regularized_rows = [row for row in results if float(row["eta"]) != 0 and row.get("mode") != "source_loss"]
        source_loss_rows = [row for row in results if row.get("mode") == "source_loss"]
        ordinary_rows = [row for row in results if float(row["eta"]) == 0]
        color_values = sorted({float(row[color_name]) for row in regularized_rows})
        colors = {value: plt.get_cmap("tab10")(index) for index, value in enumerate(color_values)}
        metrics = [("test_target_gain", "Target gain (×10⁻³) ↑"), ("test_source_degradation", "Source degradation (×10⁻³) ↓")]
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)

        for axis, (metric, y_label) in zip(axes, metrics):
            for color_value in color_values:
                rows = [row for row in regularized_rows if float(row[color_name]) == color_value]
                axis.scatter([float(row[x_name]) for row in rows], [1000 * float(row[metric]) for row in rows], s=55, color=colors[color_value], edgecolor="white", linewidth=0.6, alpha=0.8)

            axis.scatter([0.0] * len(ordinary_rows), [1000 * float(row[metric]) for row in ordinary_rows], s=72, marker="D", color=self.ORDINARY_COLOR, edgecolor="white", linewidth=0.7, zorder=3)
            axis.scatter([float(row[x_name]) for row in source_loss_rows], [1000 * float(row[metric]) for row in source_loss_rows], s=150, marker="*", color=self.SOURCE_LOSS_COLOR, edgecolor="white", linewidth=0.7, zorder=4)
            axis.axhline(0, color="0.75", linewidth=0.8)
            axis.set_xlabel(x_label)
            axis.set_ylabel(y_label)
            axis.grid(color="0.9", linestyle="--", linewidth=0.7)

            if x_name in {"eta", "s"}:
                positive_values = sorted({float(row[x_name]) for row in regularized_rows + source_loss_rows if float(row[x_name]) > 0})
                axis.set_xscale("symlog", linthresh=positive_values[0] / 2)
                axis.set_xticks([0.0, *positive_values], ["0", *[f"{value:g}" for value in positive_values]])
            else:
                axis.set_xticks(sorted({0, *[int(row[x_name]) for row in regularized_rows + source_loss_rows]}))

        handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[value], markeredgecolor="white", markersize=8, label=f"{value:g}") for value in color_values]
        handles.append(Line2D([0], [0], marker="D", color="none", markerfacecolor=self.ORDINARY_COLOR, markersize=8, label="Ordinary LoRA"))
        handles.append(Line2D([0], [0], marker="*", color="none", markerfacecolor=self.SOURCE_LOSS_COLOR, markersize=12, label="Source-loss LoRA"))
        axes[1].legend(handles=handles, title=color_label, loc="best")
        figure.suptitle(title)
        figure.tight_layout()
        figure.savefig(self.output_dir / filename, dpi=200)
        plt.close(figure)

    def plot_figure_4(self, results):
        regularized_rows = [row for row in results if float(row["eta"]) != 0 and row.get("mode") != "source_loss"]
        source_loss_rows = [row for row in results if row.get("mode") == "source_loss"]
        ordinary_rows = [row for row in results if float(row["eta"]) == 0]
        figure, axis = plt.subplots(figsize=(7.5, 6))
        axis.scatter([1000 * float(row["test_source_degradation"]) for row in regularized_rows], [1000 * float(row["test_target_gain"]) for row in regularized_rows], s=58, color=self.REGULARIZED_COLOR, edgecolor="white", linewidth=0.6, alpha=0.8, label="Regularized LoRA")
        axis.scatter([1000 * float(row["test_source_degradation"]) for row in ordinary_rows], [1000 * float(row["test_target_gain"]) for row in ordinary_rows], s=82, marker="D", color=self.ORDINARY_COLOR, edgecolor="white", linewidth=0.7, label="Ordinary LoRA", zorder=3)
        axis.scatter([1000 * float(row["test_source_degradation"]) for row in source_loss_rows], [1000 * float(row["test_target_gain"]) for row in source_loss_rows], s=160, marker="*", color=self.SOURCE_LOSS_COLOR, edgecolor="white", linewidth=0.7, label="Source-loss LoRA", zorder=4)
        axis.set_xlabel("Source degradation (×10⁻³) ↓")
        axis.set_ylabel("Target gain (×10⁻³) ↑")
        axis.set_title("Target learning versus source preservation")
        axis.grid(color="0.9", linestyle="--", linewidth=0.7)
        axis.legend()
        figure.tight_layout()
        figure.savefig(self.output_dir / "figure_4_target_source_tradeoff.png", dpi=200)
        plt.close(figure)

    def plot_fid(self, results):
        regularized_rows = [row for row in results if float(row["eta"]) != 0 and row.get("mode") != "source_loss"]
        source_loss_rows = [row for row in results if row.get("mode") == "source_loss"]
        ordinary_rows = [row for row in results if float(row["eta"]) == 0]
        m_values = sorted({int(row["m"]) for row in regularized_rows})
        strengths = sorted({float(row["s"]) for row in regularized_rows})
        colors = {value: plt.get_cmap("tab10")(index % 10) for index, value in enumerate(m_values)}
        markers = ["o", "s", "^", "v", "P", "X", "<", ">", "h", "p"]
        figure, axis = plt.subplots(figsize=(8, 5))

        for m in m_values:
            for index, strength in enumerate(strengths):
                rows = [row for row in regularized_rows if int(row["m"]) == m and float(row["s"]) == strength]
                axis.scatter([float(row["eta"]) for row in rows], [float(row["test_target_fid"]) for row in rows], s=55, color=colors[m], marker=markers[index % len(markers)], edgecolor="white", linewidth=0.6, alpha=0.8)

        axis.scatter([0.0] * len(ordinary_rows), [float(row["test_target_fid"]) for row in ordinary_rows], s=72, marker="D", color=self.ORDINARY_COLOR, edgecolor="white", linewidth=0.7, label="Ordinary LoRA")
        axis.scatter([float(row["eta"]) for row in source_loss_rows], [float(row["test_target_fid"]) for row in source_loss_rows], s=150, marker="*", color=self.SOURCE_LOSS_COLOR, edgecolor="white", linewidth=0.7, label="Source-loss LoRA")
        positive_values = sorted({float(row["eta"]) for row in regularized_rows + source_loss_rows if float(row["eta"]) > 0})

        if positive_values:
            axis.set_xscale("symlog", linthresh=positive_values[0] / 2)
            axis.set_xticks([0.0, *positive_values], ["0", *[f"{value:g}" for value in positive_values]])

        handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[m], markeredgecolor="white", markersize=8, label=f"M={m}") for m in m_values]
        handles += [Line2D([0], [0], marker=markers[index % len(markers)], color="none", markerfacecolor="0.4", markeredgecolor="white", markersize=8, label=f"s={strength:g}") for index, strength in enumerate(strengths)]
        baseline_handles, _ = axis.get_legend_handles_labels()
        axis.legend(handles=handles + baseline_handles, loc="best")
        axis.set_xlabel("η")
        axis.set_ylabel("Target FID ↓")
        axis.set_title("Generated images versus held-out rotated MNIST")
        axis.grid(color="0.9", linestyle="--", linewidth=0.7)
        figure.tight_layout()
        figure.savefig(self.output_dir / "figure_9_target_fid.png", dpi=200)
        plt.close(figure)

    def plot_qualitative(self, results):
        selected_label = self.selected_model_label(results["selected_model"])
        self.plot_reconstructions(results["source"], "Clean MNIST reconstructions", "figure_5_source_reconstructions.png", selected_label)
        self.plot_reconstructions(results["target"], "Rotated MNIST reconstructions", "figure_6_target_reconstructions.png", selected_label)
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
        figure.suptitle("Model responses on clean MNIST", y=0.99)
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
