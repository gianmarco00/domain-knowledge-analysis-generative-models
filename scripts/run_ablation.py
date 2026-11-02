from pathlib import Path
import argparse

from domain_knowledge_analysis.experiments import Ablation


def main():
    default_config = Path(__file__).resolve().parents[1] / "config" / "ablation_vae_mnist.yaml"
    parser = argparse.ArgumentParser(description="Run an ablation study.")
    parser.add_argument("config", nargs="?", type=Path, default=default_config, help="Path to the ablation YAML file.")
    parser.add_argument("--plots-only", action="store_true",
                        help="Regenerate target-geometry figures from saved metrics and image banks; never train or score models.")
    args = parser.parse_args()

    try:
        from domain_knowledge_analysis.utils import load_config

        if load_config(args.config)["ablation"].get("regularizer_mode") == "target_geometry":
            from domain_knowledge_analysis.experiments.target_geometry_ablation import TargetGeometryAblation
            ablation = TargetGeometryAblation(args.config, plots_only=args.plots_only)
            if args.plots_only:
                ablation.replot_saved_results()
                return
        else:
            if args.plots_only:
                parser.error("--plots-only currently supports target_geometry ablations.")
            ablation = Ablation(args.config)
        ablation.run()
    except KeyboardInterrupt:
        print("\nAblation stopped. Completed model checkpoints are preserved.")
        raise SystemExit(130)


if __name__ == "__main__":
    main()
