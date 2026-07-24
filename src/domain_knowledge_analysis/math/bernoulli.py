import torch
from torch.distributions import ContinuousBernoulli


def bernoulli_log_prob_from_logits(x, logits):

    if not (x.shape == logits.shape):
        raise ValueError(f"Shape mismatch: x has shape {x.shape}, but logits has shape {logits.shape}. They must be the same.")

    if x.dim() < 2:
        raise ValueError(f"Input x must have at least 2 dimensions (batch_size, ...), but got {x.dim()} dimensions.")
    
    
    log_prob_i_j = torch.add(torch.multiply(x, torch.nn.functional.logsigmoid(logits)), torch.multiply((1 - x), torch.nn.functional.logsigmoid(-logits)))

    log_prob = torch.sum(log_prob_i_j, dim=[1, 2, 3])  # Sum over all pixels for each image in the batch

    return log_prob


CONTINUOUS_BERNOULLI_EPS = 1e-4


def continuous_bernoulli_probs_from_logits(logits):
    """Convert decoder logits to the bounded CB parameter lambda."""
    return torch.sigmoid(logits).clamp(
        min=CONTINUOUS_BERNOULLI_EPS,
        max=1.0 - CONTINUOUS_BERNOULLI_EPS,
    )

def continuous_bernoulli_log_prob_from_logits(x, logits):
    if x.shape != logits.shape:
        raise ValueError(
            f"Shape mismatch: x has shape {x.shape}, "
            f"but logits has shape {logits.shape}."
        )

    if x.dim() < 2:
        raise ValueError(
            "Input must contain a batch dimension and "
            "at least one feature dimension."
        )

    lambda_ = continuous_bernoulli_probs_from_logits(logits)

    distribution = ContinuousBernoulli(probs=lambda_)
    log_prob_per_pixel = distribution.log_prob(x)

    return log_prob_per_pixel.flatten(start_dim=1).sum(dim=1)

