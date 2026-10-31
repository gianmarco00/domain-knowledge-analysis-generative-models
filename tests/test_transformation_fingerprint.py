import pytest
import torch

from domain_knowledge_analysis.losses import VAELoss
from domain_knowledge_analysis.math import bernoulli_log_prob_from_logits
from domain_knowledge_analysis.models.vae import Vae
from domain_knowledge_analysis.regularizers import TFRegularizer


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.adapters_enabled = True
        self.batch_sizes = []

    def deterministic_forward(self, x):
        self.batch_sizes.append(len(x))
        output = self.weight * x.flatten(start_dim=1)

        if self.adapters_enabled:
            output = output + 0.2 * output.square()

        return output


class DummyLoRAManager:
    def __init__(self, model):
        self.model = model
        self.adapters_enabled = True

    def disable_adapters(self):
        self.model.adapters_enabled = False
        self.adapters_enabled = False

    def enable_adapters(self):
        self.model.adapters_enabled = True
        self.adapters_enabled = True


class DummyVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_delta = torch.nn.Parameter(torch.tensor(0.1))
        self.decoder_delta = torch.nn.Parameter(torch.tensor(0.1))
        self.encoder_params = {"latent_dim": 2}
        self.decoder_distribution_name = "bernoulli"
        self.adapters_enabled = True

    def encoder(self, x):
        x = x.flatten(start_dim=1)
        mean = x[:, :2]
        log_variance = 0.2 * x[:, 2:]

        if self.adapters_enabled:
            mean = mean + self.encoder_delta * x[:, 2:]
            log_variance = log_variance + self.encoder_delta * x[:, :2]

        return mean, log_variance

    def decoder(self, z):
        logits = torch.stack([z[:, 0], z[:, 1], z[:, 0] - z[:, 1], z[:, 0] + z[:, 1]], dim=1)

        if self.adapters_enabled:
            logits = logits + self.decoder_delta * logits.square()

        return logits

    def decoder_distribution(self, logits, decoder_distribution_name):
        return torch.sigmoid(logits)

    def deterministic_forward(self, x):
        mean, _ = self.encoder(x)
        return self.decoder_distribution(self.decoder(mean), self.decoder_distribution_name)


class DummySourceVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        logits = self.weight * x
        mean = self.weight * x.flatten(start_dim=1)[:, :2]
        log_variance = torch.zeros_like(mean)
        return logits, mean, log_variance


class AddPattern:
    def __init__(self, pattern):
        self.pattern = torch.tensor(pattern, dtype=torch.float32).reshape(1, 2, 2)
        self.num_calls = 0

    def __call__(self, x):
        self.num_calls += 1
        return x + self.pattern.to(device=x.device, dtype=x.dtype)


def make_inputs():
    torch.manual_seed(7)
    anchor_points = torch.rand(7, 1, 2, 2)
    transformations = [
        AddPattern([0.1, 0.0, 0.0, 0.0]),
        AddPattern([0.0, 0.2, 0.0, 0.0]),
        AddPattern([0.0, 0.0, 0.3, 0.1]),
    ]
    return anchor_points, (transformations, torch.ones(len(transformations)))


def full_calibration(model, lora_manager, transformations, anchor_points):
    transformation_functions, _ = transformations
    transformed_x = torch.stack([transformation(anchor_points) for transformation in transformation_functions], dim=1)
    input_delta = (transformed_x - anchor_points.unsqueeze(1)).flatten(start_dim=2)
    input_scales = input_delta.square().mean(dim=(0, 2)).clamp_min(1e-12).sqrt()
    input_scale_matrix = torch.outer(input_scales, input_scales)

    was_training = model.training
    model.eval()
    lora_manager.disable_adapters()

    with torch.no_grad():
        num_anchor_points = len(anchor_points)
        num_transformations = len(transformation_functions)
        transformed_output = model.deterministic_forward(transformed_x.flatten(0, 1)).reshape(num_anchor_points, num_transformations, -1)
        original_output = model.deterministic_forward(anchor_points).flatten(start_dim=1).unsqueeze(1)
        response = transformed_output - original_output
        output_dimension = response.shape[-1]
        gram_matrices = (response @ response.transpose(-1, -2)) / output_dimension
        gram_matrices = gram_matrices / input_scale_matrix
        trace = torch.diagonal(gram_matrices, dim1=-2, dim2=-1).sum(dim=-1)
        gram_matrices = gram_matrices / trace.clamp_min(1e-12).view(-1, 1, 1)

    lora_manager.enable_adapters()

    if was_training:
        model.train()

    return input_scale_matrix, gram_matrices


def make_encoder_regularizer(mode, calibration_batch_size=2, decoder_weight=0.0, decoder_seed=0):
    anchor_points, transformations = make_inputs()
    model = DummyVAE()
    regularizer = TFRegularizer(model, DummyLoRAManager(model), transformations, anchor_points, anchor_batch_size=len(anchor_points), calibration_batch_size=calibration_batch_size, mode=mode, response_weight=2.0, decoder_weight=decoder_weight, decoder_seed=decoder_seed)
    return model, regularizer


def test_batched_calibration_matches_original_full_calibration():
    anchor_points, transformations = make_inputs()
    model = DummyModel().train()
    lora_manager = DummyLoRAManager(model)
    expected_input_scale, expected_source_grams = full_calibration(model, lora_manager, transformations, anchor_points)
    model.batch_sizes.clear()

    regularizer = TFRegularizer(model, lora_manager, transformations, anchor_points, calibration_batch_size=2)

    torch.testing.assert_close(regularizer.input_scale_matrix, expected_input_scale, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(regularizer.source_gram_matrices, expected_source_grams, rtol=1e-6, atol=1e-7)
    assert model.training
    assert model.adapters_enabled
    assert not regularizer.source_gram_matrices.requires_grad


def test_different_calibration_batch_sizes_produce_the_same_result():
    anchor_points, transformations = make_inputs()
    first_model = DummyModel()
    second_model = DummyModel()
    first_regularizer = TFRegularizer(first_model, DummyLoRAManager(first_model), transformations, anchor_points, calibration_batch_size=len(anchor_points))
    second_regularizer = TFRegularizer(second_model, DummyLoRAManager(second_model), transformations, anchor_points, calibration_batch_size=2)

    torch.testing.assert_close(first_regularizer.input_scale_matrix, second_regularizer.input_scale_matrix, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(first_regularizer.source_gram_matrices, second_regularizer.source_gram_matrices, rtol=1e-6, atol=1e-7)


def test_batched_gram_matrix_matches_full_gram_matrix():
    anchor_points, transformations = make_inputs()
    model = DummyModel()
    regularizer = TFRegularizer(model, DummyLoRAManager(model), transformations, anchor_points, calibration_batch_size=2)

    full_grams = regularizer.gram_matrix()
    batched_grams = regularizer.gram_matrix_batched(batch_size=2)

    torch.testing.assert_close(batched_grams, full_grams, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("trace_normalization", [True, False])
def test_output_gram_normalization_calibration_loss_and_gradients(trace_normalization):
    anchor_points, transformations = make_inputs()
    model = DummyModel()
    manager = DummyLoRAManager(model)
    regularizer = TFRegularizer(model, manager, transformations, anchor_points, anchor_batch_size=len(anchor_points), calibration_batch_size=2, trace_normalization=trace_normalization)

    def expected_grams(response):
        scaled = response / regularizer.input_scales.view(1, -1, 1)
        grams = scaled @ scaled.transpose(-1, -2) / scaled.shape[-1]
        if trace_normalization:
            traces = grams.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
            grams = grams / traces.clamp_min(1e-12).view(-1, 1, 1)
        return grams

    manager.disable_adapters()
    with torch.no_grad():
        expected_source = expected_grams(regularizer.model_response())
    manager.enable_adapters()
    response = regularizer.model_response()
    expected_current = expected_grams(response)
    torch.testing.assert_close(regularizer.source_gram_matrices, expected_source)
    torch.testing.assert_close(regularizer.gram_matrix(response_vector=response), expected_current)
    torch.testing.assert_close(regularizer.gram_matrix_batched(batch_size=2), expected_current)

    scaled_grams = regularizer.gram_matrix(response_vector=2 * response)
    torch.testing.assert_close(scaled_grams, expected_current * (1 if trace_normalization else 4))
    regularizer.anchor_indices = torch.arange(len(anchor_points))
    regularizer.anchor_position = 0
    loss = regularizer()
    torch.testing.assert_close(loss, (expected_current - expected_source).square().mean())
    loss.backward()
    assert model.weight.grad is not None and torch.isfinite(model.weight.grad)


def test_output_gram_trace_floor_handles_zero_responses():
    anchor_points, transformations = make_inputs()
    model = DummyModel()
    regularizer = TFRegularizer(model, DummyLoRAManager(model), transformations, anchor_points)
    response = torch.zeros(len(anchor_points), len(transformations[0]), 4)
    torch.testing.assert_close(regularizer.gram_matrix(response_vector=response), torch.zeros(len(anchor_points), 3, 3))


def test_calibration_never_sends_more_than_one_chunk_to_the_model():
    anchor_points, transformations = make_inputs()
    model = DummyModel()
    TFRegularizer(model, DummyLoRAManager(model), transformations, anchor_points, calibration_batch_size=2)

    assert max(model.batch_sizes) <= 2 * len(transformations[0])


def test_transformed_anchor_images_are_cached():
    anchor_points, transformations = make_inputs()
    model = DummyModel()
    regularizer = TFRegularizer(model, DummyLoRAManager(model), transformations, anchor_points, calibration_batch_size=2)
    calls_after_calibration = [transformation.num_calls for transformation in transformations[0]]

    regularizer()
    regularizer()

    assert regularizer.transformed_x.shape == (len(anchor_points), len(transformations[0]), 1, 2, 2)
    assert calls_after_calibration == [1, 1, 1]
    assert [transformation.num_calls for transformation in transformations[0]] == calls_after_calibration


def test_regularizer_loss_still_backpropagates_after_batched_calibration():
    anchor_points, transformations = make_inputs()
    model = DummyModel()
    regularizer = TFRegularizer(model, DummyLoRAManager(model), transformations, anchor_points, anchor_batch_size=3, calibration_batch_size=2)

    loss = regularizer()
    loss.backward()

    assert loss.requires_grad
    assert model.weight.grad is not None
    assert torch.isfinite(model.weight.grad)


def test_source_loss_uses_the_training_loss_on_one_anchor_minibatch():
    anchor_points = torch.rand(12, 1, 2, 2)
    model = DummySourceVAE()
    loss = VAELoss(bernoulli_log_prob_from_logits, beta=1.7)
    regularizer = TFRegularizer(model, DummyLoRAManager(model), None, anchor_points, mode="source_loss", loss=loss)
    regularizer.anchor_indices = torch.arange(len(anchor_points))
    regularizer.anchor_position = 0
    source_x = anchor_points[:10]
    expected = loss(source_x, *model(source_x))
    actual = regularizer()

    torch.testing.assert_close(actual, expected)
    assert regularizer.anchor_position == 10
    assert not hasattr(regularizer, "transformed_x")

    actual.backward()
    assert torch.isfinite(model.weight.grad)


def test_source_loss_requires_the_training_loss():
    anchor_points = torch.rand(7, 1, 2, 2)
    model = DummySourceVAE()

    with pytest.raises(ValueError, match="training loss"):
        TFRegularizer(model, DummyLoRAManager(model), None, anchor_points, mode="source_loss")


def test_regularizer_loss_averages_all_gram_entries():
    anchor_points, transformations = make_inputs()
    model = DummyModel()
    regularizer = TFRegularizer(model, DummyLoRAManager(model), transformations, anchor_points, anchor_batch_size=3, calibration_batch_size=2)
    anchor_indices = torch.arange(3)
    regularizer.anchor_indices = anchor_indices
    regularizer.anchor_position = 0

    current_grams = regularizer.gram_matrix(anchor_indices)
    source_grams = regularizer.source_gram_matrices[anchor_indices]
    expected_loss = torch.square(current_grams - source_grams).mean()

    torch.testing.assert_close(regularizer(), expected_loss)


def test_regularizer_verifies_target_input_rms():
    anchor_points = torch.rand(4, 1, 2, 2)
    transformations = ([AddPattern([0.1, 0.0, 0.0, 0.0])], torch.tensor([0.1]))
    model = DummyModel()

    regularizer = TFRegularizer(model, DummyLoRAManager(model), transformations, anchor_points, target_input_rms=0.05)

    torch.testing.assert_close(regularizer.input_scales, torch.tensor([0.05]))


def test_regularizer_verifies_per_transformation_input_scales():
    anchor_points = torch.rand(4, 1, 2, 2)
    transformations = ([AddPattern([0.1, 0.0, 0.0, 0.0]), AddPattern([0.0, 0.2, 0.0, 0.0])], torch.tensor([0.1, 0.2]))
    model = DummyModel()

    regularizer = TFRegularizer(model, DummyLoRAManager(model), transformations, anchor_points, target_input_scales=[0.05, 0.1])

    torch.testing.assert_close(regularizer.input_scales, torch.tensor([0.05, 0.1]))


def test_regularizer_rejects_unresolved_input_scale():
    anchor_points = torch.rand(4, 1, 2, 2)
    transformations = ([AddPattern([0.0, 0.0, 0.0, 0.0])], torch.tensor([0.0]))
    model = DummyModel()

    with pytest.raises(ValueError, match="greater than 1e-6"):
        TFRegularizer(model, DummyLoRAManager(model), transformations, anchor_points)


def test_encoder_representation_contains_mean_and_standard_deviation():
    model, regularizer = make_encoder_regularizer("encoder_response")
    mean, log_variance = model.encoder(regularizer.x)
    expected = torch.cat([mean, torch.exp(0.5 * log_variance)], dim=-1)

    torch.testing.assert_close(regularizer.model_representation(regularizer.x), expected)


def test_encoder_response_loss_matches_its_three_terms():
    model, regularizer = make_encoder_regularizer("encoder_response", decoder_weight=3.0)
    regularizer.anchor_indices = torch.arange(regularizer.num_anchor_points)
    regularizer.anchor_position = 0
    encoder_output, response = regularizer.model_response(return_original=True)
    geometry = regularizer.transformation_geometry(response)
    latents = regularizer.source_latents
    decoder_output = regularizer.decoder_output(latents.flatten(0, 1)).reshape(regularizer.num_anchor_points, 3, -1)
    encoder_loss = torch.square(encoder_output - regularizer.source_encoder_outputs).mean()
    response_loss = torch.square(geometry - regularizer.source_transformation_geometries).mean()
    decoder_loss = torch.square(decoder_output - regularizer.source_decoder_outputs).mean()
    expected = encoder_loss + 2.0 * response_loss + 3.0 * decoder_loss

    torch.testing.assert_close(regularizer(), expected)


def test_encoder_response_gives_the_exact_transformed_posterior_error_identity():
    _, regularizer = make_encoder_regularizer("encoder_response")
    encoder_output, response = regularizer.model_response(return_original=True)
    current_geometry = regularizer.transformation_geometry(response)
    source_geometry = regularizer.source_transformation_geometries
    source_encoder_output = regularizer.source_encoder_outputs
    scales = regularizer.input_scales.view(1, -1, 1)
    transformed_error = encoder_output.unsqueeze(1) + response - source_encoder_output.unsqueeze(1) - scales * source_geometry
    expected = (encoder_output - source_encoder_output).unsqueeze(1) + scales * (current_geometry - source_geometry)

    torch.testing.assert_close(transformed_error, expected)


def test_encoder_gram_uses_normalized_responses_without_trace_normalization():
    _, regularizer = make_encoder_regularizer("encoder_gram")
    response = regularizer.model_response()
    normalized_response = response / regularizer.input_scales.view(1, -1, 1)
    expected = (normalized_response @ normalized_response.transpose(-1, -2)) / normalized_response.shape[-1]

    torch.testing.assert_close(regularizer.transformation_geometry(response), expected)
    torch.testing.assert_close(regularizer.transformation_geometry(2 * response), 4 * expected)


def test_decoder_source_codes_do_not_depend_on_calibration_batch_size():
    _, full_batch = make_encoder_regularizer("encoder_response", calibration_batch_size=7, decoder_weight=1.0, decoder_seed=13)
    _, small_batches = make_encoder_regularizer("encoder_response", calibration_batch_size=2, decoder_weight=1.0, decoder_seed=13)

    torch.testing.assert_close(full_batch.source_encoder_outputs, small_batches.source_encoder_outputs)
    torch.testing.assert_close(full_batch.source_transformation_geometries, small_batches.source_transformation_geometries)
    torch.testing.assert_close(full_batch.source_latents, small_batches.source_latents)
    torch.testing.assert_close(full_batch.source_decoder_outputs, small_batches.source_decoder_outputs)


def test_encoder_response_loss_reaches_encoder_and_decoder_adapters():
    model, regularizer = make_encoder_regularizer("encoder_response", decoder_weight=1.0)

    regularizer().backward()

    assert torch.isfinite(model.encoder_delta.grad)
    assert torch.isfinite(model.decoder_delta.grad)
    assert model.training
    assert model.adapters_enabled


@pytest.mark.parametrize("argument, value, message", [("mode", "unknown", "mode"), ("response_weight", -1.0, "response_weight"), ("decoder_weight", float("nan"), "decoder_weight"), ("trace_normalization", "false", "trace_normalization")])
def test_regularizer_rejects_invalid_modes_and_weights(argument, value, message):
    anchor_points, transformations = make_inputs()
    model = DummyVAE()
    arguments = {argument: value}

    with pytest.raises(ValueError, match=message):
        TFRegularizer(model, DummyLoRAManager(model), transformations, anchor_points, **arguments)


def test_vae_deterministic_forward_returns_differentiable_pixel_probabilities():
    encoder_params = {
        "latent_dim": 4,
        "out_channels": [8, 8],
        "kernels": [3, 3],
        "strides": [2, 1],
        "paddings": [1, 1],
    }
    model = Vae([1, 8, 8], encoder_params, "bernoulli")
    images = torch.rand(2, 1, 8, 8)

    reconstructions = model.deterministic_forward(images)
    reconstructions.mean().backward()

    assert reconstructions.min() >= 0
    assert reconstructions.max() <= 1
    assert any(parameter.grad is not None for parameter in model.parameters())
