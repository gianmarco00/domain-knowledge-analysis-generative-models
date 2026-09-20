from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

from domain_knowledge_analysis.utils import dataset_utils


@pytest.fixture
def config(tmp_path):
    return {
        "paths": {
            "dataset_dir": str(tmp_path / "datasets"),
        },
        "dataset": {
            "name": "mnist",
            "train_split": 0.75,
        },
        "dataloader": {
            "batch_size": 2,
            "shuffle_train": False,
            "shuffle_validation": False,
            "num_workers": 0,
        },
        "seed": 42,
    }


@pytest.fixture
def transformation_config():
    return {
        "type": "rotation",
        "intensity": 45.0,
    }


def create_small_dataset():
    images = torch.zeros(8, 1, 5, 5)
    images[:, :, 1:4, 1:4] = 1.0
    labels = torch.arange(8)
    return TensorDataset(images, labels)


def test_fixed_rotation_preserves_shape_and_uses_zero_fill():
    image = torch.ones(1, 5, 5)
    rotation = dataset_utils.FixedRotation(degrees=45)

    rotated = rotation(image)

    assert rotated.shape == image.shape
    assert rotated[0, 0, 0].item() < 0.1
    assert rotated[0, 2, 2].item() == pytest.approx(1.0)


def test_create_dataset_transformation_rejects_unknown_type():
    with pytest.raises(ValueError, match="Unsupported dataset transformation"):
        dataset_utils.create_dataset_transformation(
            transformation_type="unknown",
            intensity=1.0,
        )


def test_materialize_transformed_dataset_preserves_labels():
    dataset = create_small_dataset()
    transformation = dataset_utils.FixedRotation(degrees=15)

    transformed_dataset = dataset_utils.materialize_transformed_dataset(
        dataset=dataset,
        transformation=transformation,
    )

    images, labels = transformed_dataset.tensors
    assert images.shape == (8, 1, 5, 5)
    assert torch.equal(labels, torch.arange(8))


def test_transformed_datasets_are_saved_and_reused(
    monkeypatch,
    config,
    transformation_config,
):
    dataset = create_small_dataset()
    creation_calls = 0

    def create_source_splits(config, dataset_name):
        nonlocal creation_calls
        creation_calls += 1
        return (
            TensorDataset(dataset.tensors[0][:6], dataset.tensors[1][:6]),
            TensorDataset(dataset.tensors[0][6:], dataset.tensors[1][6:]),
        )

    monkeypatch.setattr(
        dataset_utils,
        "create_train_validation_datasets",
        create_source_splits,
    )

    first_train, first_validation = (
        dataset_utils.create_transformed_train_validation_datasets(
            config=config,
            dataset_name="mnist",
            transformation_config=transformation_config,
        )
    )
    second_train, second_validation = (
        dataset_utils.create_transformed_train_validation_datasets(
            config=config,
            dataset_name="mnist",
            transformation_config=transformation_config,
        )
    )

    cache_files = sorted(
        Path(config["paths"]["dataset_dir"])
        .joinpath("transformed")
        .rglob("*.pt")
    )

    assert creation_calls == 1
    assert [path.name for path in cache_files] == ["train.pt", "validation.pt"]
    assert torch.equal(first_train.tensors[0], second_train.tensors[0])
    assert torch.equal(first_train.tensors[1], second_train.tensors[1])
    assert torch.equal(first_validation.tensors[0], second_validation.tensors[0])
    assert torch.equal(first_validation.tensors[1], second_validation.tensors[1])


def test_training_dataloaders_use_cached_transformed_datasets(
    monkeypatch,
    config,
    transformation_config,
):
    dataset = create_small_dataset()

    def create_transformed_splits(config, dataset_name, transformation_config):
        return (
            TensorDataset(dataset.tensors[0][:6], dataset.tensors[1][:6]),
            TensorDataset(dataset.tensors[0][6:], dataset.tensors[1][6:]),
        )

    monkeypatch.setattr(
        dataset_utils,
        "create_transformed_train_validation_datasets",
        create_transformed_splits,
    )

    train_dataloader, validation_dataloader = (
        dataset_utils.create_training_dataloaders(
            config=config,
            transformation_config=transformation_config,
        )
    )

    assert len(train_dataloader.dataset) == 6
    assert len(validation_dataloader.dataset) == 2
    assert next(iter(train_dataloader))[0].shape == (2, 1, 5, 5)


def test_training_dataloaders_without_transformation_remain_unchanged(
    monkeypatch,
    config,
):
    dataset = create_small_dataset()

    def create_source_splits(config, dataset_name):
        return (
            TensorDataset(dataset.tensors[0][:6], dataset.tensors[1][:6]),
            TensorDataset(dataset.tensors[0][6:], dataset.tensors[1][6:]),
        )

    monkeypatch.setattr(
        dataset_utils,
        "create_train_validation_datasets",
        create_source_splits,
    )

    train_dataloader, validation_dataloader = (
        dataset_utils.create_training_dataloaders(config=config)
    )

    assert len(train_dataloader.dataset) == 6
    assert len(validation_dataloader.dataset) == 2
