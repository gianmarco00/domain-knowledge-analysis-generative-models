import torch

def elbo_per_image(reconstruction_term, kl_divergence_term):
    """
    Computes the Evidence Lower Bound (ELBO) given the reconstruction term and KL divergence term.

    """
    if not (reconstruction_term.shape == kl_divergence_term.shape):
        raise ValueError(f"Shape mismatch: reconstruction_term has shape {reconstruction_term.shape}, but kl_divergence_term has shape {kl_divergence_term.shape}. They must be the same.")
    
    if not (reconstruction_term.dim() == 1 and kl_divergence_term.dim() == 1):
        raise ValueError(f"Both reconstruction_term and kl_divergence_term must be 1D tensors, but got {reconstruction_term.dim()}D and {kl_divergence_term.dim()}D respectively.")
    
    elbo_value = torch.sub(reconstruction_term, kl_divergence_term)

    return elbo_value