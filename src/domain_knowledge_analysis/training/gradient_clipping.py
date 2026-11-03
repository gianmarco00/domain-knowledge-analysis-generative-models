"""Optional global gradient clipping, independent of the loss/regularizer."""

from collections.abc import Mapping
import math

import torch


def gradient_clipping_max_norm(settings):
    """Missing settings leave the original optimizer path unchanged."""
    if settings is None:
        return None
    if not isinstance(settings, Mapping):
        raise ValueError("training.gradient_clipping must contain enabled and max_norm.")
    if settings.keys() - {"enabled", "max_norm"}:
        raise ValueError("Unknown training.gradient_clipping setting.")
    enabled = settings.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("training.gradient_clipping.enabled must be true or false.")
    max_norm = settings.get("max_norm", 100.0)
    if (isinstance(max_norm, bool) or not isinstance(max_norm, (int, float))
            or not math.isfinite(max_norm) or max_norm <= 0):
        raise ValueError("training.gradient_clipping.max_norm must be finite and positive.")
    return float(max_norm) if enabled else None


@torch.no_grad()
def clip_optimizer_gradients(optimizer, max_norm):
    """Clip the complete primal gradient once, before updating Adam's moments.

    Norm reductions and scaling stay on the gradient device. CUDA/CPU use
    float64 for the reduction only: squaring large, finite float32 gradients
    must not overflow the norm calculation itself. Model/gradient/Adam dtypes
    are unchanged. Nonfinite gradients fail before any scaling or update.
    """
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]
                  if parameter.grad is not None]
    if not parameters:
        return
    device = parameters[0].grad.device
    # MPS does not support float64. Retain PyTorch's ordinary float32 norm there.
    norm_dtype = torch.float32 if device.type == "mps" else torch.float64
    norms = [torch.linalg.vector_norm(parameter.grad, dtype=norm_dtype).to(device)
             for parameter in parameters]
    total_norm = torch.linalg.vector_norm(torch.stack(norms))
    if not torch.isfinite(total_norm):
        raise RuntimeError("Nonfinite total gradient norm; gradient clipping cannot repair "
                           "nonfinite gradients. No optimizer update was applied.")
    # The public PyTorch operation applies one common factor to every gradient,
    # using foreach kernels on CUDA/CPU and leaving sub-threshold gradients intact.
    torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, total_norm)

