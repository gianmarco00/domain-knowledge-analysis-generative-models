import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from domain_knowledge_analysis.scoring.signals import NLLEstimator
from domain_knowledge_analysis.scoring import Scorer


class DummyVae(torch.nn.Module):
    def __init__(self):
        super().__init__()
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


def test_nll_estimator_returns_negative_elbo_per_image_for_vae():
    model = DummyVae()
    estimator = NLLEstimator(model=model, model_architecture="vae")

    x = torch.zeros(3, 1, 2, 2)

    scores = estimator.estimate(x)

    expected_score_per_image = torch.full(
        size=(3,),
        fill_value=4 * torch.log(torch.tensor(2.0)),
    )

    assert scores.shape == (3,)
    assert torch.allclose(scores, expected_score_per_image)
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


def test_scorer_returns_scores_for_both_datasets():
    model = DummyVae()

    in_distribution_x = torch.zeros(5, 1, 2, 2)
    out_distribution_x = torch.zeros(7, 1, 2, 2)

    in_distribution_dataloader = DataLoader(
        TensorDataset(in_distribution_x),
        batch_size=2,
        shuffle=False,
    )

    out_distribution_dataloader = DataLoader(
        TensorDataset(out_distribution_x),
        batch_size=3,
        shuffle=False,
    )

    config = {
        "model": {
            "name": "vae",
        },
        "scoring": {
            "signals": {
                "likelihood": {
                    "enabled": True,
                }
            }
        },
    }

    scorer = Scorer(
        model=model,
        in_distribution_dataloader=in_distribution_dataloader,
        out_distribution_dataloader=out_distribution_dataloader,
        config=config,
        device=torch.device("cpu"),
    )

    results = scorer.score()

    assert "likelihood" in results
    assert results["likelihood"]["in_distribution"].shape == (5,)
    assert results["likelihood"]["out_distribution"].shape == (7,)


def test_scorer_preserves_score_values_across_batches():
    model = DummyVae()

    x = torch.zeros(5, 1, 2, 2)

    dataloader = DataLoader(
        TensorDataset(x),
        batch_size=2,
        shuffle=False,
    )

    config = {
        "model": {
            "name": "vae",
        },
        "scoring": {
            "signals": {
                "likelihood": {
                    "enabled": True,
                }
            }
        },
    }

    scorer = Scorer(
        model=model,
        in_distribution_dataloader=dataloader,
        out_distribution_dataloader=dataloader,
        config=config,
        device=torch.device("cpu"),
    )

    estimator = NLLEstimator(model=model, model_architecture="vae")
    scores = scorer.score_dataloader(estimator=estimator, dataloader=dataloader)

    expected_scores = torch.full(
        size=(5,),
        fill_value=4 * torch.log(torch.tensor(2.0)),
    )

    assert scores.shape == (5,)
    assert torch.allclose(scores, expected_scores)