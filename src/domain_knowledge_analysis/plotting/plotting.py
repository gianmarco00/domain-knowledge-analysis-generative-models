from pathlib import Path

import matplotlib.pyplot as plt
import torch

from domain_knowledge_analysis.evaluation import compute_auroc


class Plotter:
    def __init__(self, log_dir):
        self.output_dir = Path(log_dir) / "results"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot(
        self,
        results,
        in_distribution_name,
        out_distribution_names,
        title,
    ):
        saved_paths = {}

        for signal_name in results:
            in_distribution_scores = results[signal_name]["in_distribution"]

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
                    f"{signal_name}_{out_distribution_name}_histogram"
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
                    f"{signal_name}_{out_distribution_name}_ecdf"
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
            f"log(1 + {self._format_signal_name(signal_name)})"
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

    def _to_cpu_1d_tensor(self, scores):
        if not isinstance(scores, torch.Tensor):
            scores = torch.tensor(scores)

        return scores.detach().cpu().flatten()

    def _format_signal_name(self, signal_name):
        signal_names = {
            "likelihood": "NLL via negative ELBO",
            "typicality": "Typicality score",
            "gradnorm": "GradNorm score",
        }

        return signal_names.get(
            signal_name,
            signal_name,
        )