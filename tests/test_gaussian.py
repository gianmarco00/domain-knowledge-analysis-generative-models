import torch
import math.gaussian as gaussian



def test_log_prob_returns_one_value_per_batch_element():

    batch_size = 4
    latent_dim = 3

    z = torch.randn(batch_size, latent_dim)
    mean = torch.randn(batch_size, latent_dim)
    log_variance = torch.randn(batch_size, latent_dim)

    result = gaussian.log_prob(z, mean, log_variance)

    assert result.shape == (batch_size,)

def test_log_prob_returns_correct_value():

    batch_size = 4
    latent_dim = 3

    z = torch.zeros(batch_size, latent_dim)
    mean = torch.zeros(batch_size, latent_dim)
    log_variance = torch.zeros(batch_size, latent_dim)

    result = gaussian.log_prob(z, mean, log_variance)
    expected_value = -0.5 * latent_dim * math.log(2 * math.pi)

    assert torch.allclose(result, torch.full((batch_size,), expected_value))

def test_sample_returns_correct_shape():

    batch_size = 4
    latent_dim = 3

    mean = torch.randn(batch_size, latent_dim)
    log_variance = torch.randn(batch_size, latent_dim)

    result = gaussian.sample(mean, log_variance)

    assert result.shape == (batch_size, latent_dim)

def test_kl_divergence_returns_one_value_per_batch_element():

    batch_size = 4
    latent_dim = 3

    mean = torch.randn(batch_size, latent_dim)
    log_variance = torch.randn(batch_size, latent_dim)

    result = gaussian.kl_divergence(mean, log_variance)

    assert result.shape == (batch_size,)

def test_kl_divergence_equals_zero_for_same_distributions():

    batch_size = 4
    latent_dim = 3

    mean = torch.zeros(batch_size, latent_dim)
    log_variance = torch.zeros(batch_size, latent_dim)

    result = gaussian.kl_divergence(mean, log_variance)

    assert torch.allclose(result, 0.0)
