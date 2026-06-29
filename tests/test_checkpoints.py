import torch
from pathlib import Path
import pytest

from domain_knowledge_analysis.training import CheckpointManager

@pytest.fixture
def config(tmp_path):
    return {
        "experiment": {
            "name": "vae_mnist_test",
        },
        "paths": {
            "dataset_dir": str(tmp_path / "datasets"),
            "runs_dir": str(tmp_path / "runs"),
        },
        "dataset": {
            "name": "mnist",
            "train": True,
            "download": False,
            "train_split": 0.8,
            "shape": [1, 28, 28],
        },
        "dataloader": {
            "batch_size": 4,
            "shuffle_train": True,
            "shuffle_validation": False,
            "num_workers": 0,
        },
        "training": {
            "epochs": 2,
            "learning_rate": 0.001,
        },
        "model": {
            "name": "vae",
            "encoder": {
                "out_channels": [24, 24, 24, 24],
                "kernels": [3, 3, 3, 3],
                "strides": [1, 2, 2, 1],
                "paddings": [1, 1, 1, 1],
                "latent_dim": 12
            },
        },
        "optimizer": {
            "name": "adam",
        },

        "seed": 42
    }


def test_checkpoint_manager_creates_checkpoint_directory(tmp_path):
    checkpoint_manager = CheckpointManager(tmp_path)

    assert checkpoint_manager.checkpoint_dir.exists()
    assert checkpoint_manager.checkpoint_dir.is_dir()


def test_save_last_creates_checkpoint_file(tmp_path):
    checkpoint_manager = CheckpointManager(tmp_path)

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = {"train_loss": [1.0], "validation_loss": [1.2]}

    checkpoint_manager.save_last(
        model=model,
        optimizer=optimizer,
        epoch=0,
        history=history,
    )

    assert checkpoint_manager.checkpoint_path("last.pt").exists()


def test_saved_checkpoint_contains_required_keys(tmp_path):
    checkpoint_manager = CheckpointManager(tmp_path)

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = {"train_loss": [1.0], "validation_loss": [1.2]}

    checkpoint_manager.save_last(
        model=model,
        optimizer=optimizer,
        epoch=3,
        history=history,
    )

    checkpoint = torch.load(
        checkpoint_manager.checkpoint_path("last.pt"),
        map_location="cpu",
    )

    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert "epoch" in checkpoint
    assert "history" in checkpoint

    assert checkpoint["epoch"] == 3
    assert checkpoint["history"] == history


def test_load_model_restores_model_weights(tmp_path, config):
    checkpoint_manager = CheckpointManager(tmp_path, config)

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = {"train_loss": [1.0], "validation_loss": [1.2]}

    original_parameters = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    checkpoint_manager.save_last(
        model=model,
        optimizer=optimizer,
        epoch=0,
        history=history,
    )

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10.0)

    modified_parameters = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    assert any(
        not torch.allclose(original, modified)
        for original, modified in zip(original_parameters, modified_parameters)
    )

    checkpoint_manager.load_model(
        model=model,
        checkpoint_path=Path(checkpoint_manager.checkpoint_dir / "last.pt"),
        device=torch.device("cpu"),
        optimizer=optimizer,
    )

    loaded_parameters = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    assert all(
        torch.allclose(original, loaded)
        for original, loaded in zip(original_parameters, loaded_parameters)
    )