from pathlib import Path

from domain_knowledge_analysis.experiments import Experiment

experiments_yamls = {
    "vae_mnist.yaml", 
    #"vae_fashionmnist.yaml", 
    #"vae_kmnist.yaml", 
    #"vae_organamnist.yaml"
}

DEFAULT_CONFIG_PATHS = {Path(__file__).resolve().parents[1] / "config" / experiment_yaml for experiment_yaml in experiments_yamls}


def main():
    
    for yaml, path in zip(experiments_yamls, DEFAULT_CONFIG_PATHS):

        experiment = Experiment(path)
        print(f"Running {yaml} experiment")
        experiment.score()


if __name__ == "__main__":
    main()
