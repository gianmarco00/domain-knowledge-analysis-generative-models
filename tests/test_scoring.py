import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from domain_knowledge_analysis.scoring.signals import NLLEstimator, TypicalityEstimator
from domain_knowledge_analysis.scoring import Scorer


class DummyVae(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy_parameter = torch.nn.Parameter(torch.zeros(1))
        self.eval_was_called = False

    def eval(self):
        self.eval_was_called = True
        return super().eval()

    def forward(self, x):
        batch_size = x.shape[0]

        logits = torch.zeros_like(x)
        mean = torch.zeros(batch_size, 2, device=x.device)
        log_variance = torch.zeros(batch_size, 2, device=x.device)

        return logits, mean, log_variance


def expected_dummy_nll(batch_size):
    return torch.full(
        size=(batch_size,),
        fill_value=4 * torch.log(torch.tensor(2.0)),
    )


def make_dataloader(num_samples, batch_size):
    x = torch.zeros(num_samples, 1, 2, 2)

    return DataLoader(
        TensorDataset(x),
        batch_size=batch_size,
        shuffle=False,
    )


def test_nll_estimator_returns_negative_elbo_per_image_for_vae():
    model = DummyVae()
    estimator = NLLEstimator(model=model, model_architecture="vae")

    x = torch.zeros(3, 1, 2, 2)

    scores = estimator.estimate(x)

    assert scores.shape == (3,)
    assert torch.allclose(scores.cpu(), expected_dummy_nll(batch_size=3))
    assert model.eval_was_called


def test_nll_estimator_does_not_return_gradients():
    model = DummyVae()
    estimator = NLLEstimator(model=model, model_architecture="vae")

    x = torch.zeros(3, 1, 2, 2, requires_grad=True)

    scores = estimator.estimate(x)

    assert not scores.requires_grad


def test_nll_estimator_rejects_unsupported_model_architecture():
    model = DummyVae()
    estimator = NLLEstimator(model=model, model_architecture="diffusion")

    x = torch.zeros(3, 1, 2, 2)

    with pytest.raises(ValueError, match="Unsupported model"):
        estimator.estimate(x)


def test_typicality_estimator_calibrates_entropy_estimate():
    model = DummyVae()
    calibration_dataloader = make_dataloader(num_samples=5, batch_size=2)

    estimator = TypicalityEstimator(
        model=model,
        model_architecture="vae",
        calibration_dataloader=calibration_dataloader,
    )

    entropy_estimate = estimator.calibrate()

    expected_entropy = 4 * torch.log(torch.tensor(2.0))

    assert torch.allclose(entropy_estimate.cpu(), expected_entropy)


def test_typicality_estimator_requires_calibration_before_estimate():
    model = DummyVae()
    calibration_dataloader = make_dataloader(num_samples=5, batch_size=2)

    estimator = TypicalityEstimator(
        model=model,
        model_architecture="vae",
        calibration_dataloader=calibration_dataloader,
    )

    x = torch.zeros(3, 1, 2, 2)

    with pytest.raises(ValueError, match="not calibrated"):
        estimator.estimate(x)


def test_typicality_estimator_returns_zero_when_nll_equals_entropy():
    model = DummyVae()
    calibration_dataloader = make_dataloader(num_samples=5, batch_size=2)

    estimator = TypicalityEstimator(
        model=model,
        model_architecture="vae",
        calibration_dataloader=calibration_dataloader,
    )

    estimator.calibrate()

    x = torch.zeros(3, 1, 2, 2)

    scores = estimator.estimate(x)

    assert scores.shape == (3,)
    assert torch.allclose(scores.cpu(), torch.zeros(3))


def test_typicality_estimator_uses_absolute_difference():
    model = DummyVae()
    calibration_dataloader = make_dataloader(num_samples=5, batch_size=2)

    estimator = TypicalityEstimator(
        model=model,
        model_architecture="vae",
        calibration_dataloader=calibration_dataloader,
    )

    nll_value = 4 * torch.log(torch.tensor(2.0))
    estimator.entropy_estimate = nll_value + 1.0

    x = torch.zeros(3, 1, 2, 2)

    scores = estimator.estimate(x)

    assert scores.shape == (3,)
    assert torch.allclose(scores.cpu(), torch.ones(3))


def test_scorer_returns_likelihood_scores_for_both_datasets():
    model = DummyVae()

    in_distribution_dataloader = make_dataloader(num_samples=5, batch_size=2)
    out_distribution_dataloader = make_dataloader(num_samples=7, batch_size=3)
    calibration_dataloader = make_dataloader(num_samples=6, batch_size=2)

    config = {
        "model": {
            "name": "vae",
        },
        "scoring": {
            "signals": {
                "likelihood": {
                    "enabled": True,
                },
                "typicality": {
                    "enabled": False,
                },
            }
        },
    }

    scorer = Scorer(
        model=model,
        in_distribution_dataloader=in_distribution_dataloader,
        out_distribution_dataloader=out_distribution_dataloader,
        calibration_dataloader=calibration_dataloader,
        config=config,
        device=torch.device("cpu"),
    )

    results = scorer.score()

    assert "likelihood" in results
    assert "typicality" not in results
    assert results["likelihood"]["in_distribution"].shape == (5,)
    assert results["likelihood"]["out_distribution"].shape == (7,)


def test_scorer_returns_likelihood_and_typicality_when_both_enabled():
    model = DummyVae()

    in_distribution_dataloader = make_dataloader(num_samples=5, batch_size=2)
    out_distribution_dataloader = make_dataloader(num_samples=7, batch_size=3)
    calibration_dataloader = make_dataloader(num_samples=6, batch_size=2)

    config = {
        "model": {
            "name": "vae",
        },
        "scoring": {
            "signals": {
                "likelihood": {
                    "enabled": True,
                },
                "typicality": {
                    "enabled": True,
                },
            }
        },
    }

    scorer = Scorer(
        model=model,
        in_distribution_dataloader=in_distribution_dataloader,
        out_distribution_dataloader=out_distribution_dataloader,
        calibration_dataloader=calibration_dataloader,
        config=config,
        device=torch.device("cpu"),
    )

    results = scorer.score()

    assert "likelihood" in results
    assert "typicality" in results

    assert results["likelihood"]["in_distribution"].shape == (5,)
    assert results["likelihood"]["out_distribution"].shape == (7,)

    assert results["typicality"]["in_distribution"].shape == (5,)
    assert results["typicality"]["out_distribution"].shape == (7,)

    assert torch.allclose(results["typicality"]["in_distribution"], torch.zeros(5))
    assert torch.allclose(results["typicality"]["out_distribution"], torch.zeros(7))


def test_scorer_preserves_score_values_across_batches():
    model = DummyVae()

    dataloader = make_dataloader(num_samples=5, batch_size=2)
    calibration_dataloader = make_dataloader(num_samples=6, batch_size=2)

    config = {
        "model": {
            "name": "vae",
        },
        "scoring": {
            "signals": {
                "likelihood": {
                    "enabled": True,
                },
                "typicality": {
                    "enabled": False,
                },
            }
        },
    }

    scorer = Scorer(
        model=model,
        in_distribution_dataloader=dataloader,
        out_distribution_dataloader=dataloader,
        calibration_dataloader=calibration_dataloader,
        config=config,
        device=torch.device("cpu"),
    )

    estimator = NLLEstimator(model=model, model_architecture="vae")
    scores = scorer.score_dataloader(estimator=estimator, dataloader=dataloader)

    assert scores.shape == (5,)
    assert torch.allclose(scores.cpu(), expected_dummy_nll(batch_size=5))