import pytest
import torch

from domain_knowledge_analysis import utils
from domain_knowledge_analysis.adapters import LoRAManager
from domain_knowledge_analysis.training import CheckpointManager


def lora_config():
    return {
        "dataset": {"name": "mnist", "shape": [1, 8, 8]},
        "model": {
            "name": "vae",
            "encoder": {
                "latent_dim": 3,
                "out_channels": [4, 8],
                "kernels": [3, 3],
                "strides": [2, 2],
                "paddings": [1, 1],
            },
            "decoder": {"symmetrical": True},
        },
        "loss": {"log_prob_function": "bernoulli", "beta": 1.0},
        "lora": {
            "rank": 2,
            "alpha": 4,
            "layers_to_inject": {
                "encoder": ["fc", "conv"],
                "decoder": ["fc", "convtranspose"],
            },
        },
    }


@pytest.fixture
def saved_lora_checkpoint(tmp_path):
    config = lora_config()
    torch.manual_seed(0)
    adapted_model = utils.create_model(config)
    adapted_manager = LoRAManager(adapted_model, config)
    adapted_manager.inject_adapters()

    # Nonzero adapters make this a meaningful test of LoRA weights, not just base weights.
    with torch.no_grad():
        for layer in adapted_manager.lora_layers.values():
            layer.lora_A.fill_(0.1)
            layer.lora_B.fill_(0.1)

    checkpoint_manager = CheckpointManager(tmp_path, config)
    optimizer = torch.optim.Adam(adapted_manager.trainable_parameters(), lr=0.001)
    checkpoint_manager.save_last(adapted_model, optimizer, None, 1, {}, 0.5)
    return config, adapted_model, checkpoint_manager


def load_lora_model(config, checkpoint_manager):
    loaded_model = utils.create_model(config)
    loaded_manager = LoRAManager(loaded_model, config)
    loaded_manager.inject_adapters()
    checkpoint_manager.load_model(
        loaded_model,
        checkpoint_manager.checkpoint_path("last.pt"),
        torch.device("cpu"),
    )
    return loaded_model, loaded_manager


def test_loaded_lora_checkpoint_preserves_outputs(saved_lora_checkpoint):
    config, adapted_model, checkpoint_manager = saved_lora_checkpoint
    loaded_model, _ = load_lora_model(config, checkpoint_manager)

    adapted_model.eval()
    loaded_model.eval()
    images = torch.rand(2, 1, 8, 8)
    latents = torch.randn(2, 3)

    for name, tensor in adapted_model.state_dict().items():
        torch.testing.assert_close(tensor, loaded_model.state_dict()[name])

    with torch.no_grad():
        torch.testing.assert_close(
            adapted_model.reconstruct_images(images),
            loaded_model.reconstruct_images(images),
        )
        torch.testing.assert_close(
            adapted_model.decoder(latents),
            loaded_model.decoder(latents),
        )

        # VAE.forward samples latent noise, so reset the RNG before each call.
        torch.manual_seed(123)
        adapted_forward = adapted_model(images)
        torch.manual_seed(123)
        loaded_forward = loaded_model(images)
        for adapted_output, loaded_output in zip(adapted_forward, loaded_forward):
            torch.testing.assert_close(adapted_output, loaded_output)


def test_loaded_lora_checkpoint_preserves_adapter_effect(saved_lora_checkpoint):
    config, _, checkpoint_manager = saved_lora_checkpoint
    loaded_model, loaded_manager = load_lora_model(config, checkpoint_manager)

    images = torch.rand(2, 1, 8, 8)
    adapted_reconstruction = loaded_model.reconstruct_images(images)
    loaded_manager.disable_adapters()
    base_reconstruction = loaded_model.reconstruct_images(images)

    assert not torch.allclose(adapted_reconstruction, base_reconstruction)


def test_lora_checkpoint_requires_injected_model(saved_lora_checkpoint):
    config, _, checkpoint_manager = saved_lora_checkpoint
    plain_model = utils.create_model(config)

    with pytest.raises(RuntimeError, match="loading state_dict"):
        checkpoint_manager.load_model(
            plain_model,
            checkpoint_manager.checkpoint_path("last.pt"),
            torch.device("cpu"),
        )
