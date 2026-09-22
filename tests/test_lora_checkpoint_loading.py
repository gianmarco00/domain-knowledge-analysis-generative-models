from copy import deepcopy

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
def saved_checkpoints(tmp_path):
    config = lora_config()
    source_config = deepcopy(config)
    source_config.pop("lora")  # Older source checkpoints do not have this key.

    torch.manual_seed(0)
    source_model = utils.create_model(source_config)
    source_checkpoint_manager = CheckpointManager(tmp_path / "source", source_config)
    source_optimizer = torch.optim.Adam(source_model.parameters(), lr=0.001)
    source_checkpoint_manager.save_last(source_model, source_optimizer, None, 1, {}, 0.5)

    adapted_model = deepcopy(source_model)
    adapted_manager = LoRAManager(adapted_model, config)
    adapted_manager.inject_adapters()

    # Nonzero adapters make this a meaningful test of LoRA weights, not just base weights.
    with torch.no_grad():
        for layer in adapted_manager.lora_layers.values():
            layer.lora_A.fill_(0.1)
            layer.lora_B.fill_(0.1)

    adapted_checkpoint_manager = CheckpointManager(tmp_path / "adapted", config)
    adapted_optimizer = torch.optim.Adam(adapted_manager.trainable_parameters(), lr=0.001)
    adapted_checkpoint_manager.save_last(adapted_model, adapted_optimizer, None, 1, {}, 0.5)
    return (
        config,
        source_model,
        adapted_model,
        source_checkpoint_manager.checkpoint_path("last.pt"),
        adapted_checkpoint_manager.checkpoint_path("last.pt"),
    )


def load_like_score_adaptation(config, source_path, adapted_path, log_dir):
    loaded_model = utils.create_model(config)
    checkpoint_manager = CheckpointManager(log_dir, config)
    checkpoint_manager.load_model(loaded_model, source_path, torch.device("cpu"))
    assert "lora" not in checkpoint_manager.model_config

    images = torch.rand(2, 1, 8, 8)
    source_reconstruction = loaded_model.reconstruct_images(images)
    loaded_manager = LoRAManager(loaded_model, config)
    loaded_manager.inject_adapters()
    torch.testing.assert_close(
        source_reconstruction, loaded_model.reconstruct_images(images)
    )
    checkpoint_manager.load_model(
        loaded_model,
        adapted_path,
        torch.device("cpu"),
    )
    return loaded_model, loaded_manager


def test_loaded_lora_checkpoint_preserves_outputs(saved_checkpoints, tmp_path):
    config, _, adapted_model, source_path, adapted_path = saved_checkpoints
    loaded_model, _ = load_like_score_adaptation(
        config, source_path, adapted_path, tmp_path / "loading"
    )

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


def test_loaded_lora_checkpoint_preserves_adapter_effect(saved_checkpoints, tmp_path):
    config, source_model, _, source_path, adapted_path = saved_checkpoints
    loaded_model, loaded_manager = load_like_score_adaptation(
        config, source_path, adapted_path, tmp_path / "loading"
    )

    images = torch.rand(2, 1, 8, 8)
    adapted_reconstruction = loaded_model.reconstruct_images(images)
    loaded_manager.disable_adapters()
    base_reconstruction = loaded_model.reconstruct_images(images)

    assert not torch.allclose(adapted_reconstruction, base_reconstruction)
    torch.testing.assert_close(base_reconstruction, source_model.reconstruct_images(images))


def test_lora_checkpoint_rejects_different_alpha(saved_checkpoints, tmp_path):
    config, _, _, source_path, adapted_path = saved_checkpoints
    different_alpha_config = deepcopy(config)
    different_alpha_config["lora"]["alpha"] = 8

    with pytest.raises(ValueError, match="LoRA alpha differs"):
        load_like_score_adaptation(
            different_alpha_config, source_path, adapted_path, tmp_path / "loading"
        )
