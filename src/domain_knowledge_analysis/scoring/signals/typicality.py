import torch
from .nll import NLLEstimator


class TypicalityEstimator():
    def __init__(self, calibration_dataloader,  model, model_architecture):

        self.model = model
        self.model_architecture = model_architecture
        self.calibration_dataloader = calibration_dataloader
        self.device = next(self.model.parameters()).device

        self.nll_estimator = NLLEstimator(self.model, self.model_architecture)

        self.entropy_estimate = None

    def calibrate(self):
        nll_total = torch.tensor(0.0, device=self.device)
        nll_count = 0

        for batch in self.calibration_dataloader:
            x = batch[0].to(self.device)
            nll_batch = self.nll_estimator.estimate(x)

            nll_total += torch.sum(nll_batch)
            nll_count += nll_batch.numel()

        self.entropy_estimate = nll_total / nll_count

        return self.entropy_estimate
        

    def estimate(self, x):

        if self.entropy_estimate is None:
            raise ValueError("Typicality estimator was not calibrated")

        nll_per_image = self.nll_estimator.estimate(x)

        typicality_per_image = torch.abs(nll_per_image - self.entropy_estimate)

        return typicality_per_image