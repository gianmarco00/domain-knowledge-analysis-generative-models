import torch

def bernoulli_log_prob_from_logits(x, logits):
    """
        This function computes the log probability of a binary image x given the decoder's output logits.
            - x is a binary image (0s and 1s). 
            - logits are the raw outputs from the decoder, which can be converted to probabilities using a sigmoid function.
    
        The Bernoulli distribution is used for modeling binary data. The log probability is computed using the formula:
            log_prob = x * log(sigmoid(logits)) + (1 - x) * log(1 - sigmoid(logits))

        Assumes:
            - x is [batch_size, channels, height, width]
            - logits is [batch_size, channels, height, width]

    """
    if not (x.shape == logits.shape):
        raise ValueError(f"Shape mismatch: x has shape {x.shape}, but logits has shape {logits.shape}. They must be the same.")

    if x.dim() < 2:
        raise ValueError(f"Input x must have at least 2 dimensions (batch_size, ...), but got {x.dim()} dimensions.")
    
    
    log_prob_i_j = torch.add(torch.multiply(x, torch.nn.functional.logsigmoid(logits)), torch.multiply((1 - x), torch.nn.functional.logsigmoid(-logits)))

    log_prob = torch.sum(log_prob_i_j, dim=[1, 2, 3])  # Sum over all pixels for each image in the batch

    return log_prob

