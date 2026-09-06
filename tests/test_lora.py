
import pytest
import copy
import torch
import torch.nn as nn

from domain_knowledge_analysis.adapters import LoRAConv2d, LoRAConvTranspose2d, LoRALinear, LoRAManager
from domain_knowledge_analysis.utils import select_layers
from domain_knowledge_analysis.models import Vae


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 2, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(2, 4, kernel_size=3, padding=1))
        self.fc_mean = nn.Linear(4, 2)
        self.fc_log_variance = nn.Linear(4, 2)
        self.conv_transpose = nn.ConvTranspose2d(4, 2, kernel_size=3)

    def forward(self, x):
        x = self.conv(x)
        x = x.mean(dim=(2, 3))
        return self.fc_mean(x), self.fc_log_variance(x)


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 4)
        self.conv = nn.Sequential(nn.Conv2d(4, 4, kernel_size=3, padding=1))
        self.conv_transpose = nn.Sequential(nn.ConvTranspose2d(4, 2, kernel_size=3), nn.ReLU(), nn.ConvTranspose2d(2, 1, kernel_size=3))

    def forward(self, z):
        x = self.fc(z)
        x = x.reshape(z.shape[0], 4, 1, 1)
        x = self.conv(x)
        return self.conv_transpose(x)


class TinyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TinyEncoder()
        self.decoder = TinyDecoder()

    def forward(self, x):
        mean, _ = self.encoder(x)
        return self.decoder(mean)


def test_select_layers_returns_expected_layers():
    vae = TinyVAE()
    layers_to_select = {"encoder": ["fc", "conv"], "decoder": ["fc", "conv", "convtranspose"]}

    selected_layers = select_layers(vae, layers_to_select)

    expected_names = {"encoder.conv.0", "encoder.conv.2", "encoder.fc_mean", "encoder.fc_log_variance", "decoder.fc", "decoder.conv.0", "decoder.conv_transpose.0", "decoder.conv_transpose.2"}

    assert set(selected_layers) == expected_names
    assert selected_layers["encoder.conv.0"] is vae.encoder.conv[0]
    assert selected_layers["encoder.fc_mean"] is vae.encoder.fc_mean
    assert selected_layers["decoder.fc"] is vae.decoder.fc
    assert selected_layers["decoder.conv_transpose.0"] is vae.decoder.conv_transpose[0]
    assert "encoder.conv_transpose" not in selected_layers


def set_nonzero_lora_parameters(layer):
    with torch.no_grad():
        layer.lora_A.fill_(0.25)
        layer.lora_B.fill_(0.50)


def assert_merge_preserves_output(layer, x):
    set_nonzero_lora_parameters(layer)

    original_base_output = layer.base_layer(x)
    output_before_merge = layer(x)
    original_weight = layer.base_layer.weight.detach().clone()
    expected_delta_weight = torch.full_like(original_weight, 0.25)
    expected_weight_after_merge = original_weight + expected_delta_weight

    assert not torch.allclose(original_base_output, output_before_merge)
    torch.testing.assert_close(layer.delta_weight(), expected_delta_weight)

    layer.merge()

    output_after_merge = layer.base_layer(x)
    output_from_merged_wrapper = layer(x)

    torch.testing.assert_close(layer.base_layer.weight, expected_weight_after_merge)
    torch.testing.assert_close(output_before_merge, output_after_merge, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(output_after_merge, output_from_merged_wrapper, rtol=1e-5, atol=1e-6)


def test_lora_linear_merge_preserves_output():
    base_layer = nn.Linear(in_features=3, out_features=2, bias=True)
    lora_layer = LoRALinear(base_layer=base_layer, rank=2, alpha=2)
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    assert_merge_preserves_output(lora_layer, x)


def test_lora_conv2d_merge_preserves_output():
    base_layer = nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=2, padding=1, bias=True)
    lora_layer = LoRAConv2d(base_layer=base_layer, rank=2, alpha=2)
    x = torch.ones(2, 2, 8, 8)

    assert_merge_preserves_output(lora_layer, x)


def test_lora_convtranspose2d_merge_preserves_output():
    base_layer = nn.ConvTranspose2d(in_channels=3, out_channels=2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
    lora_layer = LoRAConvTranspose2d(base_layer=base_layer, rank=2, alpha=2)
    x = torch.ones(2, 3, 4, 4)

    assert_merge_preserves_output(lora_layer, x)


@pytest.fixture
def config():
    return {"lora": {"layers_to_inject": {"encoder": ["fc", "conv"], "decoder": ["fc", "convtranspose"]}, "rank": 2, "alpha": 4}}


def test_inject_replaces_only_selected_layers(config):
    model = TinyVAE()
    original_encoder_conv_0 = model.encoder.conv[0]
    original_encoder_conv_2 = model.encoder.conv[2]
    original_encoder_fc_mean = model.encoder.fc_mean
    original_encoder_fc_log_variance = model.encoder.fc_log_variance
    original_encoder_conv_transpose = model.encoder.conv_transpose
    original_decoder_fc = model.decoder.fc
    original_decoder_conv = model.decoder.conv
    original_decoder_conv_0 = model.decoder.conv[0]
    original_decoder_conv_transpose_0 = model.decoder.conv_transpose[0]
    original_decoder_conv_transpose_2 = model.decoder.conv_transpose[2]

    injector = LoRAManager(model=model, config=config)
    injector.inject_adapters()

    assert isinstance(model.encoder.conv[0], LoRAConv2d)
    assert isinstance(model.encoder.conv[2], LoRAConv2d)
    assert isinstance(model.encoder.fc_mean, LoRALinear)
    assert isinstance(model.encoder.fc_log_variance, LoRALinear)
    assert isinstance(model.decoder.fc, LoRALinear)
    assert isinstance(model.decoder.conv_transpose[0], LoRAConvTranspose2d)
    assert isinstance(model.decoder.conv_transpose[2], LoRAConvTranspose2d)

    assert model.encoder.conv[0].base_layer is original_encoder_conv_0
    assert model.encoder.conv[2].base_layer is original_encoder_conv_2
    assert model.encoder.fc_mean.base_layer is original_encoder_fc_mean
    assert model.encoder.fc_log_variance.base_layer is original_encoder_fc_log_variance
    assert model.decoder.fc.base_layer is original_decoder_fc
    assert model.decoder.conv_transpose[0].base_layer is original_decoder_conv_transpose_0
    assert model.decoder.conv_transpose[2].base_layer is original_decoder_conv_transpose_2

    assert model.encoder.conv_transpose is original_encoder_conv_transpose
    assert model.decoder.conv is original_decoder_conv
    assert model.decoder.conv[0] is original_decoder_conv_0
    assert injector.injected is True


def test_injection_preserves_initial_model_output(config):
    torch.manual_seed(0)
    model = TinyVAE().eval()
    x = torch.randn(2, 1, 4, 4)

    with torch.no_grad():
        output_before_injection = model(x)

    injector = LoRAManager(model=model, config=config)
    injector.inject_adapters()
    model.eval()

    with torch.no_grad():
        output_after_injection = model(x)

    torch.testing.assert_close(output_before_injection, output_after_injection)


def test_lora_parameters_are_registered_in_model(config):
    model = TinyVAE()
    injector = LoRAManager(model=model,   config=config)
    injector.inject_adapters()

    parameter_names = set(dict(model.named_parameters()))

    assert "encoder.conv.0.lora_A" in parameter_names
    assert "encoder.conv.0.lora_B" in parameter_names
    assert "encoder.conv.2.lora_A" in parameter_names
    assert "encoder.conv.2.lora_B" in parameter_names
    assert "encoder.fc_mean.lora_A" in parameter_names
    assert "encoder.fc_mean.lora_B" in parameter_names
    assert "encoder.fc_log_variance.lora_A" in parameter_names
    assert "encoder.fc_log_variance.lora_B" in parameter_names
    assert "decoder.fc.lora_A" in parameter_names
    assert "decoder.fc.lora_B" in parameter_names
    assert "decoder.conv_transpose.0.lora_A" in parameter_names
    assert "decoder.conv_transpose.0.lora_B" in parameter_names
    assert "decoder.conv_transpose.2.lora_A" in parameter_names
    assert "decoder.conv_transpose.2.lora_B" in parameter_names


def test_injecting_twice_raises_error(config):
    injector = LoRAManager(model=TinyVAE(),   config=config)
    injector.inject_adapters()

    with pytest.raises(RuntimeError, match="LoRA has already been injected"):
        injector.inject_adapters()


@pytest.mark.parametrize("layer, expected_type", [(nn.Linear(4, 2), LoRALinear), (nn.Conv2d(1, 2, kernel_size=3), LoRAConv2d), (nn.ConvTranspose2d(2, 1, kernel_size=3), LoRAConvTranspose2d)])
def test_create_lora_layer_returns_correct_wrapper(config, layer, expected_type):
    injector = LoRAManager(model=TinyVAE(),   config=config)
    lora_layer = injector.create_lora_layer(layer)

    assert isinstance(lora_layer, expected_type)
    assert lora_layer.base_layer is layer


def test_create_lora_layer_rejects_unsupported_layer(config):
    injector = LoRAManager(model=TinyVAE(),   config=config)

    with pytest.raises(TypeError, match="Unsupported layer type"):
        injector.create_lora_layer(nn.ReLU())


def test_deepcopy_keeps_original_model_unchanged(config):
    original_model = TinyVAE()
    adapted_model = copy.deepcopy(original_model)

    injector = LoRAManager(model=adapted_model,   config=config)
    injector.inject_adapters()

    assert isinstance(original_model.encoder.conv[0], nn.Conv2d)
    assert isinstance(original_model.encoder.conv[2], nn.Conv2d)
    assert isinstance(original_model.encoder.fc_mean, nn.Linear)
    assert isinstance(original_model.encoder.fc_log_variance, nn.Linear)
    assert isinstance(original_model.decoder.fc, nn.Linear)
    assert isinstance(original_model.decoder.conv_transpose[0], nn.ConvTranspose2d)
    assert isinstance(original_model.decoder.conv_transpose[2], nn.ConvTranspose2d)

    assert isinstance(adapted_model.encoder.conv[0], LoRAConv2d)
    assert isinstance(adapted_model.encoder.conv[2], LoRAConv2d)
    assert isinstance(adapted_model.encoder.fc_mean, LoRALinear)
    assert isinstance(adapted_model.encoder.fc_log_variance, LoRALinear)
    assert isinstance(adapted_model.decoder.fc, LoRALinear)
    assert isinstance(adapted_model.decoder.conv_transpose[0], LoRAConvTranspose2d)
    assert isinstance(adapted_model.decoder.conv_transpose[2], LoRAConvTranspose2d)

@pytest.fixture
def vae_config():
    return {"image_shape": [1, 28, 28], "encoder_params": {"latent_dim": 15, "out_channels": [32, 64, 128, 128], "kernels": [3, 3, 3, 3], "strides": [2, 2, 2, 1], "paddings": [1, 1, 1, 1]}, "decoder_distribution_name": "bernoulli", "symmetric_decoder": True, "lora": {"layers_to_inject": {"encoder": ["fc", "conv"], "decoder": ["fc", "convtranspose"]}, "rank": 2, "alpha": 4}}


def test_lora_injection_on_vae(vae_config):
    torch.manual_seed(0)
    model = Vae(image_shape=vae_config["image_shape"], encoder_params=vae_config["encoder_params"], decoder_distribution_name=vae_config["decoder_distribution_name"], symmetric_decoder=vae_config["symmetric_decoder"])
    model.eval()
    x = torch.rand(2, *vae_config["image_shape"])

    original_layers = select_layers(model, vae_config["lora"]["layers_to_inject"])

    print(original_layers)

    assert original_layers

    output_before_injection = model.reconstruct_images(x)

    injector = LoRAManager(model=model, config=vae_config)
    injector.inject_adapters()
    model.eval()

    output_after_injection = model.reconstruct_images(x)

    for name, original_layer in original_layers.items():
        injected_layer = model.get_submodule(name)

        if isinstance(original_layer, nn.Linear):
            assert isinstance(injected_layer, LoRALinear)
        elif isinstance(original_layer, nn.Conv2d):
            assert isinstance(injected_layer, LoRAConv2d)
        elif isinstance(original_layer, nn.ConvTranspose2d):
            assert isinstance(injected_layer, LoRAConvTranspose2d)

        assert injected_layer.base_layer is original_layer

    torch.testing.assert_close(output_before_injection, output_after_injection)
