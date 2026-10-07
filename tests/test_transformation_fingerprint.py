import torch

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

    def disable_adapters(self):
        self.model.adapters_enabled = False

    def enable_adapters(self):
        self.model.adapters_enabled = True


class AddPattern:
    def __init__(self, pattern):
        self.pattern = torch.tensor(pattern, dtype=torch.float32).reshape(1, 2, 2)

    def __call__(self, x):
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


def test_calibration_never_sends_more_than_one_chunk_to_the_model():
    anchor_points, transformations = make_inputs()
    model = DummyModel()
    TFRegularizer(model, DummyLoRAManager(model), transformations, anchor_points, calibration_batch_size=2)

    assert max(model.batch_sizes) <= 2 * len(transformations[0])


def test_transformed_anchor_images_are_not_cached():
    anchor_points, transformations = make_inputs()
    model = DummyModel()
    regularizer = TFRegularizer(model, DummyLoRAManager(model), transformations, anchor_points, calibration_batch_size=2)

    assert not hasattr(regularizer, "transformed_x")


def test_regularizer_loss_still_backpropagates_after_batched_calibration():
    anchor_points, transformations = make_inputs()
    model = DummyModel()
    regularizer = TFRegularizer(model, DummyLoRAManager(model), transformations, anchor_points, anchor_batch_size=3, calibration_batch_size=2)

    loss = regularizer()
    loss.backward()

    assert loss.requires_grad
    assert model.weight.grad is not None
    assert torch.isfinite(model.weight.grad)


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
