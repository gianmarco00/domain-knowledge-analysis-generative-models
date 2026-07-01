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
            full_filename = (
                f"{signal_name}_"
                f"{in_distribution_name}_vs_{out_distribution_name}_full.png"
            )

            zoom_filename = (
                f"{signal_name}_"
                f"{in_distribution_name}_vs_{out_distribution_name}_zoom.png"
            )

            saved_paths[f"{signal_name}_full"] = self.plot_histogram(
                results=results,
                signal_name=signal_name,
                in_distribution_name=in_distribution_name,
                out_distribution_name=out_distribution_name,
                title=title,
                filename=full_filename,
                zoom=False,
                bins=80,
            )

            saved_paths[f"{signal_name}_zoom"] = self.plot_histogram(
                results=results,
                signal_name=signal_name,
                in_distribution_name=in_distribution_name,
                out_distribution_name=out_distribution_name,
                title=f"{title} — zoom",
                filename=zoom_filename,
                zoom=True,
                bins=120,
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
        zoom=False,
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

        x_min, x_max = self._get_plot_range(
            in_distribution_scores=in_distribution_scores,
            out_distribution_scores=out_distribution_scores,
            zoom=zoom,
        )

        bin_edges = torch.linspace(x_min, x_max, bins + 1)

        save_path = self.output_dir / filename

        plt.figure(figsize=(7, 6))

        plt.hist(
            in_distribution_scores.numpy(),
            bins=bin_edges.numpy(),
            alpha=alpha,
            label=in_distribution_name.upper(),
        )

        plt.hist(
            out_distribution_scores.numpy(),
            bins=bin_edges.numpy(),
            alpha=alpha,
            label=out_distribution_name.upper(),
        )

        plt.title(title, fontsize=18)
        plt.xlabel(self._format_signal_name(signal_name))
        plt.ylabel("# of samples")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        return save_path

    def _get_plot_range(
        self,
        in_distribution_scores,
        out_distribution_scores,
        zoom,
    ):
        if not zoom:
            all_scores = torch.cat(
                [in_distribution_scores, out_distribution_scores],
                dim=0,
            )

            return all_scores.min().item(), all_scores.max().item()

        in_min = in_distribution_scores.min().item()
        in_max = in_distribution_scores.max().item()
        in_range = in_max - in_min

        out_min = out_distribution_scores.min().item()
        out_max = out_distribution_scores.max().item()
        out_range = out_max - out_min

        if in_range <= out_range:
            compact_min = in_min
            compact_max = in_max
            compact_range = in_range
        else:
            compact_min = out_min
            compact_max = out_max
            compact_range = out_range

        if compact_range <= 0:
            compact_range = 1.0

        #half_range = compact_range / 2.0

        x_min = compact_max - compact_range #half_range
        x_max = compact_max + compact_range #half_range

        return x_min, x_max

    def _to_cpu_1d_tensor(self, scores):
        if not isinstance(scores, torch.Tensor):
            scores = torch.tensor(scores)

        scores = scores.detach().cpu().flatten()

        if scores.dim() != 1:
            raise ValueError(f"Expected 1D scores, got shape {scores.shape}.")

        return scores

    def _format_signal_name(self, signal_name):
        signal_names = {
            "likelihood": "NLL via negative ELBO",
            "typicality": "Typicality score",
        }

        return signal_names.get(signal_name, signal_name)