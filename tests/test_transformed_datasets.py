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
        "scoring": {
            "dataloader": {
                "batch_size": 3,
                "num_workers": 0,
            },
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
        dataset_utils.create_transformed_datasets(
            config=config,
            dataset_name="mnist",
            transformation_config=transformation_config,
        )
    )
    second_train, second_validation = (
        dataset_utils.create_transformed_datasets(
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
        "create_transformed_datasets",
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


def test_testing_dataloader_uses_original_test_split_without_transformation(
    monkeypatch,
    config,
):
    class FakeMNIST:
        def __init__(self, root, train, download, transform):
            self.train = train
            self.images = torch.full((8, 1, 5, 5), 1.0 if train else 0.25)
            self.labels = torch.arange(8)

        def __len__(self):
            return len(self.images)

        def __getitem__(self, index):
            return self.images[index], self.labels[index]

    monkeypatch.setitem(dataset_utils.TORCHVISION_DATASETS, "mnist", FakeMNIST)
    config["dataset"]["shape"] = [1, 5, 5]
    config["model"] = {"name": "vae"}
    config["loss"] = {"log_prob_function": "bernoulli"}

    test_dataloader = dataset_utils.create_testing_dataloaders(config=config)

    assert test_dataloader.dataset.train is False
    assert test_dataloader.batch_size == 3
    images, labels = next(iter(test_dataloader))
    assert torch.equal(images, torch.full((3, 1, 5, 5), 0.25))
    assert torch.equal(labels, torch.arange(3))


def test_testing_dataloader_transforms_test_split_and_reuses_cache(
    monkeypatch,
    config,
    transformation_config,
):
    test_dataset = create_small_dataset()
    calls = []

    def create_source_dataset(config, dataset_name, train):
        calls.append((dataset_name, train))
        return test_dataset

    monkeypatch.setattr(dataset_utils, "create_dataset", create_source_dataset)

    first_dataloader = dataset_utils.create_testing_dataloaders(
        config=config,
        transformation_config=transformation_config,
    )
    second_dataloader = dataset_utils.create_testing_dataloaders(
        config=config,
        transformation_config=transformation_config,
    )

    first_images, first_labels = first_dataloader.dataset.tensors
    second_images, second_labels = second_dataloader.dataset.tensors
    expected_images = torch.stack(
        [dataset_utils.FixedRotation(45.0)(image) for image in test_dataset.tensors[0]]
    )
    cache_files = sorted(
        Path(config["paths"]["dataset_dir"])
        .joinpath("transformed")
        .rglob("*.pt")
    )

    assert calls == [("mnist", False)]
    assert [path.name for path in cache_files] == ["test.pt"]
    assert first_dataloader.batch_size == 3
    assert torch.equal(first_images, expected_images)
    assert not torch.equal(first_images, test_dataset.tensors[0])
    assert torch.equal(first_labels, test_dataset.tensors[1])
    assert torch.equal(second_images, first_images)
    assert torch.equal(second_labels, first_labels)


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
