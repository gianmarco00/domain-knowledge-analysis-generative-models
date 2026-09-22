from pathlib import Path

import matplotlib.pyplot as plt
import torch


class AdaptationPlotter:
    def __init__(self, log_dir):
        self.output_dir = Path(log_dir) / "results"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot(self, results, filename="adaptation_reconstructions.png"):
        columns = [
            ("original_source_images", "Original", None),
            ("pretrained_reconstructed_source_images", "Pretrained\nreconstruction", "pretrained_source_mse"),
            ("adapted_reconstructed_source_images", "Adapted\nreconstruction", "adapted_source_mse"),
            ("original_target_images", "Original", None),
            ("pretrained_reconstructed_target_images", "Pretrained\nreconstruction", "pretrained_target_mse"),
            ("adapted_reconstructed_target_images", "Adapted\nreconstruction", "adapted_target_mse"),
        ]
        image_batches = [self._as_image_batch(results[key]) for key, _, _ in columns]
        num_images = len(image_batches[0])
        if num_images == 0 or any(len(batch) != num_images for batch in image_batches):
            raise ValueError("All six image groups must contain the same nonzero number of images.")

        figure = plt.figure(figsize=(14, 2.2 * num_images + 2.5), facecolor="white")
        grid = figure.add_gridspec(
            num_images,
            7,
            width_ratios=[1, 1, 1, 0.25, 1, 1, 1],
            left=0.08,
            right=0.98,
            bottom=0.09,
            top=0.82,
            wspace=0.12,
            hspace=0.16,
        )

        top_axes = []
        for row in range(num_images):
            for column, (_, heading, mse_key) in enumerate(columns):
                grid_column = column if column < 3 else column + 1
                axis = figure.add_subplot(grid[row, grid_column])
                self._show_image(axis, image_batches[column][row])
                axis.set_xticks([])
                axis.set_yticks([])

                if column == 0:
                    axis.set_ylabel(f"Example {row + 1}", rotation=0, labelpad=36, va="center")

                if row == 0:
                    if mse_key is not None:
                        heading += f"\nMSE {float(results[mse_key]):.4g}"
                    axis.set_title(heading, fontsize=10, pad=12)
                    top_axes.append(axis)

        source_center = (top_axes[0].get_position().x0 + top_axes[2].get_position().x1) / 2
        target_center = (top_axes[3].get_position().x0 + top_axes[5].get_position().x1) / 2

        figure.suptitle("Reconstruction before and after adaptation", fontsize=16, y=0.98)
        figure.text(source_center, 0.91, "SOURCE", ha="center", fontsize=12, weight="bold")
        figure.text(target_center, 0.91, "TARGET", ha="center", fontsize=12, weight="bold")
        figure.text(
            0.5,
            0.035,
            "Each row shows the same position in the source and target test batches. "
            "MSE is the mean per-pixel squared error over the full test split (lower is better).",
            ha="center",
            fontsize=9,
            color="0.35",
        )

        save_path = self.output_dir / filename
        figure.savefig(save_path, dpi=180, facecolor="white")
        plt.close(figure)
        return save_path

    @staticmethod
    def _as_image_batch(images):
        if torch.is_tensor(images):
            batch = images.detach().cpu()
        else:
            samples = []
            for image in images:
                image = torch.as_tensor(image).detach().cpu()
                if image.ndim == 4 and image.shape[0] == 1:
                    image = image[0]
                samples.append(image)
            if not samples:
                raise ValueError("Image groups must not be empty.")
            batch = torch.stack(samples)

        if batch.ndim != 4 or batch.shape[1] not in (1, 3):
            raise ValueError("Images must have shape [N, 1 or 3, H, W].")
        return batch

    @staticmethod
    def _show_image(axis, image):
        if image.shape[0] == 1:
            axis.imshow(image[0].numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        else:
            axis.imshow(image.permute(1, 2, 0).numpy(), vmin=0, vmax=1, interpolation="nearest")
