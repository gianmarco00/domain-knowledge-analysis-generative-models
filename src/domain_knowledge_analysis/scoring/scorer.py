import torch

from domain_knowledge_analysis.scoring.signals import NLLEstimator, TypicalityEstimator, GradNormEstimator


class Scorer:
    def __init__(
        self,
        model,
        in_distribution_dataloader,
        out_distribution_dataloader,
        calibration_dataloader,
        config,
        device,
    ):
        self.model = model
        self.in_distribution_dataloader = in_distribution_dataloader
        self.out_distribution_dataloader = out_distribution_dataloader
        self.calibration_dataloader = calibration_dataloader
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

        if signal_config["typicality"]["enabled"]:
            estimators["typicality"] = TypicalityEstimator(
                model=self.model,
                model_architecture=self.config["model"]["name"].lower(),
                calibration_dataloader=self.calibration_dataloader
            )

        if signal_config["gradnorm"]["enabled"]:
            estimators["gradnorm"] = GradNormEstimator(
                model=self.model,
                model_architecture=self.config["model"]["name"].lower(),
                calibration_dataloader=self.calibration_dataloader
            )

        return estimators
    
    def calibrate_estimators(self):

        for estimator in self.estimators.values():
            if hasattr(estimator, "calibrate"):
                estimator.calibrate()

    def score(self):
        results = {}

        self.calibrate_estimators()

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
