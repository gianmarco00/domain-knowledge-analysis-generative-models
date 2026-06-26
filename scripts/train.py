from domain_knowledge_analysis.experiments import Experiment


def main():

    experiment_name = "vae_mnist"
    config_path = f"/Users/gianmarcoalbano/domain-knowledge-analysis-generative-models/config/{experiment_name}.yaml"

    experiment = Experiment(config_path)

    experiment.train()