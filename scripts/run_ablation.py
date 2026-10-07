from pathlib import Path

from domain_knowledge_analysis.experiments import Ablation


def main():
    config_path = Path(__file__).resolve().parents[1] / "config" / "ablation_vae_mnist.yaml"
    ablation = Ablation(config_path)
    ablation.run()


if __name__ == "__main__":
    main()
