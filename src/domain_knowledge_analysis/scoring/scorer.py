import torch
from tqdm import tqdm

from domain_knowledge_analysis.scoring.signals import (
    GradNormEstimator,
    HoleScoreEstimator,
    LatentEncodingEstimator,
    NLLEstimator,
    TypicalityEstimator,
)


class Scorer:
    def __init__(
        self,
        model,
        in_distribution_dataloader,
        out_distribution_dataloaders,
        calibration_dataloader,
        config,
        device,
    ):
        self.model = model
        self.in_distribution_dataloader = in_distribution_dataloader
        self.out_distribution_dataloaders = out_distribution_dataloaders
        self.calibration_dataloader = calibration_dataloader
        self.config = config
        self.device = device

        self.model.to(self.device)

        self.estimators = self.create_estimators()

    def create_estimators(self):
        estimators = {}
        signal_config = self.config["scoring"]["signals"]

        if signal_config.get("likelihood", {}).get("enabled", False):
            estimators["likelihood"] = NLLEstimator(
                model=self.model,
                model_architecture=self.config["model"]["name"].lower(),
            )

        if signal_config.get("typicality", {}).get("enabled", False):
            estimators["typicality"] = TypicalityEstimator(
                model=self.model,
                model_architecture=self.config["model"]["name"].lower(),
                calibration_dataloader=self.calibration_dataloader
            )

        if signal_config.get("gradnorm", {}).get("enabled", False):
            estimators["gradnorm"] = GradNormEstimator(
                model=self.model,
                model_architecture=self.config["model"]["name"].lower(),
                calibration_dataloader=self.calibration_dataloader
            )

        if signal_config.get("latent_encoding", {}).get("enabled", False):
            estimators["latent_encoding"] = LatentEncodingEstimator(
                model=self.model,
                model_architecture=self.config["model"]["name"].lower(),
            )

        if signal_config.get("hole_score", {}).get("enabled", False):
            estimators["hole_score"] = HoleScoreEstimator(
                model=self.model,
                model_architecture=self.config["model"]["name"].lower(),
                calibration_dataloader=self.calibration_dataloader,
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

            results[signal_name] = {
                "in_distribution": in_distribution_scores,
            }

            for out_distribution_name, out_distribution_dataloader in self.out_distribution_dataloaders.items():

                out_distribution_scores = self.score_dataloader(
                    estimator=estimator,
                    dataloader=out_distribution_dataloader,
                )

                results[signal_name].update({
                    out_distribution_name: out_distribution_scores,
                })

            if signal_name == "hole_score":
                results[signal_name].update(estimator.summary)

        return results

    def score_dataloader(self, estimator, dataloader):
        all_scores = []

        progress_bar = tqdm(dataloader, desc=f"Scoring {estimator.__class__.__name__}")

        for batch in progress_bar:
            x = batch[0].to(self.device)

            scores = estimator.estimate(x)

            all_scores.append(scores.detach().cpu())

        return torch.cat(all_scores, dim=0)
