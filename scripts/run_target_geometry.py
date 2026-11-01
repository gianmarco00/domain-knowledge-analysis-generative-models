"""Train one target-geometry base LoRA config without launching an ablation."""

import argparse
from pathlib import Path

from domain_knowledge_analysis.experiments import Experiment
from domain_knowledge_analysis.utils import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Target geometry base LoRA YAML.")
    args = parser.parse_args()
    regularizer = load_config(args.config)["lora"]["regularizer"]
    if not regularizer or regularizer.get("mode") != "target_geometry":
        parser.error("Config must select lora.regularizer.mode: target_geometry.")
    experiment = Experiment(args.config)
    try:
        experiment.adapt()
    finally:
        experiment.logger.close()


if __name__ == "__main__":
    main()
