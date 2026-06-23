import math
import torch
import domain_knowledge_analysis.math.bernoulli as bernoulli

def test_bernoulli_log_prob_returns_one_value_per_batch_element():

    batch_size = 4
    img_size = 28

    x = torch.randn(batch_size, 1, img_size, img_size)
    logits = torch.randn(batch_size, 1, img_size, img_size)

    result = bernoulli.log_prob_from_logits(x, logits)

    assert result.shape == (batch_size,)

def test_bernoulli_log_prob_returns_correct_value():

    batch_size = 4
    img_size = 28

    x = torch.zeros(batch_size, 1, img_size, img_size)
    logits = torch.zeros(batch_size, 1, img_size, img_size)

    result = bernoulli.log_prob_from_logits(x, logits)
    expected_value = img_size * img_size * math.log(0.5)

    assert torch.allclose(result, torch.full((batch_size,), expected_value))

def test_bernoulli_log_prob_returns_correct_value():

    batch_size = 4
    img_size = 28

    large_value = 10000.0
    x_zero = torch.zeros(batch_size, 1, img_size, img_size)
    x_one = torch.ones(batch_size, 1, img_size, img_size)
    logits = torch.full((batch_size, 1, img_size, img_size), large_value)

    result_zero = bernoulli.log_prob_from_logits(x_zero, logits)
    result_one = bernoulli.log_prob_from_logits(x_one, logits)

    assert torch.all(result_one > result_zero), "Log probability for x=1 should be greater than for x=0 when logits are large positive values."
