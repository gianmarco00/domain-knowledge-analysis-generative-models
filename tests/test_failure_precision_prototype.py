"""Validate the proposed numerical representation, without enabling it in training."""

import math

import torch

from scripts.analyze_target_geometry_failure import adam_step_with_rms_state


def test_rms_representation_matches_standard_adam_with_existing_moments():
    generator = torch.Generator().manual_seed(123)
    reference = torch.nn.Parameter(torch.randn(20, dtype=torch.float64, generator=generator))
    candidate = torch.nn.Parameter(reference.detach().clone())
    expected = torch.optim.Adam([reference], lr=.01)
    actual = torch.optim.Adam([candidate], lr=.01)
    # Begin with ordinary moments, as in the captured checkpoint.
    for _ in range(3):
        gradient = torch.randn(20, dtype=torch.float64, generator=generator)
        reference.grad, candidate.grad = gradient.clone(), gradient.clone()
        expected.step()
        actual.step()
    for step in range(1, 1001):
        gradient = torch.randn(20, dtype=torch.float64, generator=generator) * 10**(3 * math.sin(step))
        reference.grad, candidate.grad = gradient.clone(), gradient.clone()
        expected.step()
        adam_step_with_rms_state(actual)
    torch.testing.assert_close(candidate, reference, rtol=1e-12, atol=1e-12)
    left, right = actual.state[candidate], expected.state[reference]
    torch.testing.assert_close(left["step"], right["step"], rtol=0, atol=0)
    torch.testing.assert_close(left["exp_avg"], right["exp_avg"], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(left["exp_avg_sq_sqrt"], right["exp_avg_sq"].sqrt(), rtol=1e-12, atol=1e-12)


def test_rms_representation_avoids_squared_gradient_overflow_without_clipping():
    expected = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    actual = torch.nn.Parameter(expected.detach().clone())
    ordinary = torch.optim.Adam([expected], lr=.01)
    stable = torch.optim.Adam([actual], lr=.01)
    gradient = torch.tensor([1e195, -1e195], dtype=torch.float64)
    expected.grad, actual.grad = gradient.clone(), gradient.clone()
    ordinary.step()
    adam_step_with_rms_state(stable)
    assert torch.isinf(ordinary.state[expected]["exp_avg_sq"]).all()
    assert torch.isfinite(stable.state[actual]["exp_avg_sq_sqrt"]).all()
    torch.testing.assert_close(actual, torch.tensor([-.01, .01], dtype=torch.float64))
    torch.testing.assert_close(actual.grad, gradient, rtol=0, atol=0)
