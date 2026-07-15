from pathlib import Path

import matplotlib.pyplot as plt
import torch

from domain_knowledge_analysis.scoring import compute_auroc

from .pca import compute_pca
from .tsne import compute_tsne


class Plotter:
    def __init__(
        self,
        log_dir,
        seed,
        max_tsne_samples_per_dataset=2000,
    ):
        self.output_dir = Path(log_dir) / "results"
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.latent_visualization_dir = (
            self.output_dir
            / "latent_visualization"
        )

        self.latent_visualization_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.seed = seed

        self.max_tsne_samples_per_dataset = (
            max_tsne_samples_per_dataset
        )

    def plot(
        self,
        results,
        in_distribution_name,
        out_distribution_names,
        title,
    ):
        saved_paths = {}

        for signal_name in results:

            if signal_name == "latent_encoding":
                latent_embeddings = results[signal_name]

                saved_paths[
                    "latent_embeddings_tsne"
                ] = self.plot_tsne(
                    embeddings=latent_embeddings,
                    in_distribution_name=in_distribution_name,
                    out_distribution_names=out_distribution_names,
                    title=title,
                    filename="latent_embeddings_tsne.png",
                )

                saved_paths[
                    "latent_embeddings_pca"
                ] = self.plot_pca(
                    embeddings=latent_embeddings,
                    in_distribution_name=in_distribution_name,
                    out_distribution_names=out_distribution_names,
                    title=title,
                    filename="latent_embeddings_pca.png",
                )

                continue

            if signal_name == "hole_score":
                saved_paths["hole_score_grid"] = (
                    self.plot_hole_score_grid(
                        hole_score_results=results[signal_name],
                        title=title,
                        filename="hole_score_grid.png",
                    )
                )

                continue

            in_distribution_scores = (
                results[signal_name]["in_distribution"]
            )

            for out_distribution_name in out_distribution_names:
                out_distribution_scores = results[signal_name][
                    out_distribution_name
                ]

                auroc = compute_auroc(
                    in_distribution_scores,
                    out_distribution_scores,
                )

                histogram_filename = (
                    f"{signal_name}_"
                    f"{in_distribution_name}_vs_"
                    f"{out_distribution_name}_histogram.png"
                )

                ecdf_filename = (
                    f"{signal_name}_"
                    f"{in_distribution_name}_vs_"
                    f"{out_distribution_name}_ecdf.png"
                )

                saved_paths[
                    f"{signal_name}_"
                    f"{out_distribution_name}_histogram"
                ] = self.plot_histogram(
                    in_distribution_scores=in_distribution_scores,
                    out_distribution_scores=out_distribution_scores,
                    signal_name=signal_name,
                    in_distribution_name=in_distribution_name,
                    out_distribution_name=out_distribution_name,
                    title=title,
                    filename=histogram_filename,
                    auroc=auroc,
                )

                saved_paths[
                    f"{signal_name}_"
                    f"{out_distribution_name}_ecdf"
                ] = self.plot_ecdf(
                    in_distribution_scores=in_distribution_scores,
                    out_distribution_scores=out_distribution_scores,
                    signal_name=signal_name,
                    in_distribution_name=in_distribution_name,
                    out_distribution_name=out_distribution_name,
                    title=title,
                    filename=ecdf_filename,
                    auroc=auroc,
                )

        return saved_paths

    def plot_histogram(
        self,
        in_distribution_scores,
        out_distribution_scores,
        signal_name,
        in_distribution_name,
        out_distribution_name,
        title,
        filename,
        auroc,
        bins=80,
        alpha=0.75,
    ):
        in_distribution_scores = self._to_cpu_1d_tensor(
            in_distribution_scores
        )

        out_distribution_scores = self._to_cpu_1d_tensor(
            out_distribution_scores
        )

        in_distribution_scores = torch.log1p(
            in_distribution_scores
        )

        out_distribution_scores = torch.log1p(
            out_distribution_scores
        )

        all_scores = torch.cat([
            in_distribution_scores,
            out_distribution_scores,
        ])

        bin_edges = torch.linspace(
            all_scores.min(),
            all_scores.max(),
            bins + 1,
        ).numpy()

        in_distribution_weights = (
            torch.ones_like(in_distribution_scores)
            / len(in_distribution_scores)
        )

        out_distribution_weights = (
            torch.ones_like(out_distribution_scores)
            / len(out_distribution_scores)
        )

        save_path = self.output_dir / filename

        plt.figure(figsize=(7, 6))

        plt.hist(
            in_distribution_scores.numpy(),
            bins=bin_edges,
            weights=in_distribution_weights.numpy(),
            alpha=alpha,
            label=in_distribution_name.upper(),
        )

        plt.hist(
            out_distribution_scores.numpy(),
            bins=bin_edges,
            weights=out_distribution_weights.numpy(),
            alpha=alpha,
            label=out_distribution_name.upper(),
        )

        plt.title(
            f"{title}\nAUROC = {auroc:.4f}",
            fontsize=18,
        )

        plt.xlabel(
            f"log(1 + "
            f"{self._format_signal_name(signal_name)})"
        )

        plt.ylabel("Proportion of samples")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        return save_path

    def plot_ecdf(
        self,
        in_distribution_scores,
        out_distribution_scores,
        signal_name,
        in_distribution_name,
        out_distribution_name,
        title,
        filename,
        auroc,
    ):
        in_distribution_scores = self._to_cpu_1d_tensor(
            in_distribution_scores
        )

        out_distribution_scores = self._to_cpu_1d_tensor(
            out_distribution_scores
        )

        in_distribution_scores = torch.sort(
            in_distribution_scores
        ).values

        out_distribution_scores = torch.sort(
            out_distribution_scores
        ).values

        in_distribution_ecdf = torch.arange(
            1,
            len(in_distribution_scores) + 1,
            dtype=torch.float32,
        ) / len(in_distribution_scores)

        out_distribution_ecdf = torch.arange(
            1,
            len(out_distribution_scores) + 1,
            dtype=torch.float32,
        ) / len(out_distribution_scores)

        save_path = self.output_dir / filename

        plt.figure(figsize=(7, 6))

        plt.step(
            in_distribution_scores.numpy(),
            in_distribution_ecdf.numpy(),
            where="post",
            label=in_distribution_name.upper(),
        )

        plt.step(
            out_distribution_scores.numpy(),
            out_distribution_ecdf.numpy(),
            where="post",
            label=out_distribution_name.upper(),
        )

        plt.title(
            f"{title}\nAUROC = {auroc:.4f}",
            fontsize=18,
        )

        plt.xlabel(
            self._format_signal_name(signal_name)
        )

        plt.ylabel("Cumulative proportion")
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        return save_path

    def plot_hole_score_grid(
        self,
        hole_score_results,
        title,
        filename,
    ):
        images_by_group = hole_score_results["images"]
        scores_by_group = hole_score_results.get(
            "image_scores",
            {},
        )
        holemass = float(
            torch.as_tensor(hole_score_results["holemass"])
        )

        group_labels = [
            ("lowest", "Lowest H"),
            ("middle", "Middle H"),
            ("highest", "Highest H"),
        ]

        n_rows = len(group_labels)
        n_columns = max(
            len(images_by_group[group_name])
            for group_name, _ in group_labels
        )

        save_path = self.output_dir / filename

        figure, axes = plt.subplots(
            n_rows,
            n_columns,
            figsize=(2 * n_columns, 2.3 * n_rows),
            squeeze=False,
        )

        for row_index, (group_name, group_label) in enumerate(
            group_labels
        ):
            images = self._to_cpu_image_tensor(
                images_by_group[group_name]
            )
            scores = scores_by_group.get(group_name)

            if scores is not None:
                scores = self._to_cpu_1d_tensor(scores)

            for column_index in range(n_columns):
                axis = axes[row_index][column_index]
                axis.set_xticks([])
                axis.set_yticks([])

                if column_index >= len(images):
                    axis.axis("off")
                    continue

                image = self._prepare_image_for_plot(
                    images[column_index]
                )

                if image.ndim == 2:
                    axis.imshow(
                        image,
                        cmap="gray",
                        vmin=0,
                        vmax=1,
                    )
                else:
                    axis.imshow(image)

                if scores is not None:
                    axis.set_title(
                        f"H={float(scores[column_index]):.2f}",
                        fontsize=8,
                    )

            axes[row_index][0].set_ylabel(
                group_label,
                rotation=0,
                labelpad=42,
                va="center",
                fontsize=12,
            )

        figure.suptitle(
            f"{title}\nHoleMass = {holemass:.4f}",
            fontsize=18,
        )

        figure.tight_layout(rect=[0, 0, 1, 0.92])
        figure.savefig(save_path)
        plt.close(figure)

        return save_path

    def plot_tsne(
        self,
        embeddings,
        in_distribution_name,
        out_distribution_names,
        title,
        filename,
        alpha=0.6,
        point_size=8,
    ):
        dataset_embeddings = self._collect_dataset_embeddings(
            embeddings=embeddings,
            in_distribution_name=in_distribution_name,
            out_distribution_names=out_distribution_names,
        )

        dataset_embeddings = (
            self._sample_equal_size_embeddings(
                dataset_embeddings
            )
        )

        all_embeddings = torch.cat(
            list(dataset_embeddings.values()),
            dim=0,
        )

        tsne_embeddings = compute_tsne(
            embeddings=all_embeddings.numpy(),
            seed=self.seed,
        )

        save_path = (
            self.latent_visualization_dir
            / filename
        )

        plt.figure(figsize=(8, 7))

        start_index = 0

        for (
            dataset_name,
            embeddings_tensor,
        ) in dataset_embeddings.items():

            end_index = (
                start_index
                + len(embeddings_tensor)
            )

            dataset_tsne = tsne_embeddings[
                start_index:end_index
            ]

            plt.scatter(
                dataset_tsne[:, 0],
                dataset_tsne[:, 1],
                s=point_size,
                alpha=alpha,
                label=dataset_name.upper(),
            )

            start_index = end_index

        plt.title(
            f"{title}\nLatent embeddings t-SNE",
            fontsize=18,
        )

        plt.xlabel("t-SNE dimension 1")
        plt.ylabel("t-SNE dimension 2")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        return save_path

    def plot_pca(
        self,
        embeddings,
        in_distribution_name,
        out_distribution_names,
        title,
        filename,
        alpha=0.6,
        point_size=8,
    ):
        dataset_embeddings = self._collect_dataset_embeddings(
            embeddings=embeddings,
            in_distribution_name=in_distribution_name,
            out_distribution_names=out_distribution_names,
        )

        all_embeddings = torch.cat(
            list(dataset_embeddings.values()),
            dim=0,
        )

        (
            pca_embeddings,
            explained_variance_ratio,
        ) = compute_pca(
            embeddings=all_embeddings.numpy()
        )

        save_path = (
            self.latent_visualization_dir
            / filename
        )

        plt.figure(figsize=(8, 7))

        start_index = 0

        for (
            dataset_name,
            embeddings_tensor,
        ) in dataset_embeddings.items():

            end_index = (
                start_index
                + len(embeddings_tensor)
            )

            dataset_pca = pca_embeddings[
                start_index:end_index
            ]

            plt.scatter(
                dataset_pca[:, 0],
                dataset_pca[:, 1],
                s=point_size,
                alpha=alpha,
                label=dataset_name.upper(),
            )

            start_index = end_index

        pc1_variance = (
            float(explained_variance_ratio[0])
            * 100
        )

        pc2_variance = (
            float(explained_variance_ratio[1])
            * 100
        )

        plt.title(
            f"{title}\nLatent embeddings PCA",
            fontsize=18,
        )

        plt.xlabel(
            f"PC1 ({pc1_variance:.1f}% variance)"
        )

        plt.ylabel(
            f"PC2 ({pc2_variance:.1f}% variance)"
        )

        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        return save_path

    def _collect_dataset_embeddings(
        self,
        embeddings,
        in_distribution_name,
        out_distribution_names,
    ):
        dataset_embeddings = {
            in_distribution_name:
                self._to_cpu_2d_tensor(
                    embeddings["in_distribution"]
                )
        }

        for dataset_name in out_distribution_names:
            dataset_embeddings[dataset_name] = (
                self._to_cpu_2d_tensor(
                    embeddings[dataset_name]
                )
            )

        return dataset_embeddings

    def _sample_equal_size_embeddings(
        self,
        dataset_embeddings,
    ):
        smallest_dataset_size = min(
            len(embeddings)
            for embeddings
            in dataset_embeddings.values()
        )

        sample_size = min(
            smallest_dataset_size,
            self.max_tsne_samples_per_dataset,
        )

        generator = torch.Generator().manual_seed(
            self.seed
        )

        sampled_embeddings = {}

        for (
            dataset_name,
            embeddings,
        ) in dataset_embeddings.items():

            indices = torch.randperm(
                len(embeddings),
                generator=generator,
            )[:sample_size]

            sampled_embeddings[dataset_name] = (
                embeddings[indices]
            )

        return sampled_embeddings

    def _to_cpu_1d_tensor(self, scores):
        if not isinstance(scores, torch.Tensor):
            scores = torch.tensor(scores)

        return scores.detach().cpu().flatten()

    def _to_cpu_2d_tensor(self, embeddings):
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings)

        return embeddings.detach().cpu()

    def _to_cpu_image_tensor(self, images):
        if not isinstance(images, torch.Tensor):
            images = torch.tensor(images)

        return images.detach().cpu()

    def _prepare_image_for_plot(self, image):
        image = torch.clamp(
            image.detach().cpu(),
            min=0,
            max=1,
        )

        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = image.permute(1, 2, 0)

        if image.ndim == 3 and image.shape[-1] == 1:
            image = image.squeeze(-1)

        return image.numpy()

    def _format_signal_name(self, signal_name):
        signal_names = {
            "likelihood": "NLL via negative ELBO",
            "typicality": "Typicality score",
            "gradnorm": "GradNorm score",
            "hole_score": "Hole score",
        }

        return signal_names.get(
            signal_name,
            signal_name,
        )
