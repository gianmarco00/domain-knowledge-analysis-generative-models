from pathlib import Path
import argparse

from domain_knowledge_analysis.experiments import Ablation


def main():
    default_config = Path(__file__).resolve().parents[1] / "config" / "ablation_vae_mnist.yaml"
    parser = argparse.ArgumentParser(description="Run an ablation study.")
    parser.add_argument("config", nargs="?", type=Path, default=default_config, help="Path to the ablation YAML file.")
    args = parser.parse_args()

    try:
        ablation = Ablation(args.config)
        ablation.run()
    except KeyboardInterrupt:
        print("\nAblation stopped. Completed model checkpoints are preserved.")
        raise SystemExit(130)


if __name__ == "__main__":
    main()
