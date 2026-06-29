from pathlib import Path

import matplotlib.pyplot as plt
import torch


class Plotter:
    def __init__(self, log_dir):
        self.output_dir = Path(log_dir) / "results"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot(
        self,
        results,
        in_distribution_name,
        out_distribution_name,
        title,
    ):
        saved_paths = {}

        for signal_name in results:
            filename = (
                f"{signal_name}_"
                f"{in_distribution_name}_vs_{out_distribution_name}.png"
            )

            saved_paths[signal_name] = self.plot_histogram(
                results=results,
                signal_name=signal_name,
                in_distribution_name=in_distribution_name,
                out_distribution_name=out_distribution_name,
                title=title,
                filename=filename,
            )

        return saved_paths

    def plot_histogram(
        self,
        results,
        signal_name,
        in_distribution_name,
        out_distribution_name,
        title,
        filename,
        bins=50,
        alpha=0.75,
    ):
        if signal_name not in results:
            raise ValueError(f"Signal '{signal_name}' not found in results.")

        signal_results = results[signal_name]

        in_distribution_scores = self._to_cpu_1d_tensor(
            signal_results["in_distribution"]
        )

        out_distribution_scores = self._to_cpu_1d_tensor(
            signal_results["out_distribution"]
        )

        save_path = self.output_dir / filename

        plt.figure(figsize=(7, 6))

        plt.hist(
            in_distribution_scores.numpy(),
            bins=bins,
            alpha=alpha,
            label=in_distribution_name.upper(),
        )

        plt.hist(
            out_distribution_scores.numpy(),
            bins=bins,
            alpha=alpha,
            label=out_distribution_name.upper(),
        )

        plt.title(title, fontsize=18)
        plt.xlabel(signal_name)
        plt.ylabel("# of samples")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        return save_path

    def _to_cpu_1d_tensor(self, scores):
        if not isinstance(scores, torch.Tensor):
            scores = torch.tensor(scores)

        scores = scores.detach().cpu().flatten()

        if scores.dim() != 1:
            raise ValueError(f"Expected 1D scores, got shape {scores.shape}.")

        return scores