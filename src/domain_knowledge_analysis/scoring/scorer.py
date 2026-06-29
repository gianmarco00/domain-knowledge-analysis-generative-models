import torch

from domain_knowledge_analysis.scoring.signals import NLLEstimator


class Scorer:
    def __init__(
        self,
        model,
        in_distribution_dataloader,
        out_distribution_dataloader,
        config,
        device,
    ):
        self.model = model
        self.in_distribution_dataloader = in_distribution_dataloader
        self.out_distribution_dataloader = out_distribution_dataloader
        self.config = config
        self.device = device

        self.model.to(self.device)

        self.estimators = self.create_estimators()

    def create_estimators(self):
        estimators = {}
        signal_config = self.config["scoring"]["signals"]

        if signal_config["likelihood"]["enabled"]:
            estimators["likelihood"] = NLLEstimator(
                model=self.model,
                model_architecture=self.config["model"]["name"].lower(),
            )

        return estimators

    def score(self):
        results = {}

        for signal_name, estimator in self.estimators.items():
            in_distribution_scores = self.score_dataloader(
                estimator=estimator,
                dataloader=self.in_distribution_dataloader,
            )

            out_distribution_scores = self.score_dataloader(
                estimator=estimator,
                dataloader=self.out_distribution_dataloader,
            )

            results[signal_name] = {
                "in_distribution": in_distribution_scores,
                "out_distribution": out_distribution_scores,
            }

        return results

    def score_dataloader(self, estimator, dataloader):
        all_scores = []

        for batch in dataloader:
            x = batch[0].to(self.device)

            scores = estimator.estimate(x)

            all_scores.append(scores.detach().cpu())

        return torch.cat(all_scores, dim=0)
