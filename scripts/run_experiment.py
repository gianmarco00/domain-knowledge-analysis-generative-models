from pathlib import Path

from domain_knowledge_analysis.experiments import Experiment


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "vae_mnist.yaml"


def main():
    experiment = Experiment(DEFAULT_CONFIG_PATH)
    experiment.score()


if __name__ == "__main__":
    main()
