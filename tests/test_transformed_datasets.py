from pathlib import Path
import math

import pytest
import torch
from torch.utils.data import TensorDataset

from domain_knowledge_analysis import utils
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


@pytest.mark.parametrize(
    "transformation_type, intensity",
    [
        ("horizontal_translation", -2),
        ("horizontal_translation", 2),
        ("vertical_translation", -2),
        ("vertical_translation", 2),
        ("rotation", -15),
        ("rotation", 15),
        ("isotropic_scale", 1 / 1.1),
        ("isotropic_scale", 1.1),
        ("aspect_ratio_deformation", 1 / 1.1),
        ("aspect_ratio_deformation", 1.1),
        ("diagonal_shear", -0.12),
        ("diagonal_shear", 0.12),
        ("stroke_thickness", -0.25),
        ("stroke_thickness", 0.25),
        ("stroke_thickness", -1),
        ("stroke_thickness", 1),
    ],
)
def test_regularizer_transformations_preserve_batched_image_shape(transformation_type, intensity):
    images = torch.zeros(2, 1, 9, 9)
    images[:, :, 2:7, 3:6] = 1.0
    transformation = dataset_utils.create_dataset_transformation(transformation_type, intensity)

    transformed_images = transformation(images)

    assert transformed_images.shape == images.shape
    assert torch.isfinite(transformed_images).all()


def test_fixed_translations_use_the_requested_direction_and_distance():
    image = torch.zeros(1, 7, 7)
    image[0, 3, 3] = 1.0

    translated_right = dataset_utils.FixedHorizontalTranslation(2)(image)
    translated_down = dataset_utils.FixedVerticalTranslation(2)(image)

    assert translated_right[0, 3, 5].item() == pytest.approx(1.0)
    assert translated_down[0, 5, 3].item() == pytest.approx(1.0)


def test_fixed_stroke_thickness_dilates_and_erodes_by_one_pixel():
    point = torch.zeros(1, 5, 5)
    point[0, 2, 2] = 1.0
    square = torch.zeros(1, 5, 5)
    square[0, 1:4, 1:4] = 1.0

    dilated = dataset_utils.FixedStrokeThickness(1)(point)
    eroded = dataset_utils.FixedStrokeThickness(-1)(square)

    assert dilated.sum().item() == pytest.approx(9.0)
    assert eroded.sum().item() == pytest.approx(1.0)
    assert eroded[0, 2, 2].item() == pytest.approx(1.0)


def test_fixed_stroke_thickness_supports_weaker_fractional_effects():
    point = torch.zeros(1, 5, 5)
    point[0, 2, 2] = 1.0
    square = torch.zeros(1, 5, 5)
    square[0, 1:4, 1:4] = 1.0

    weakly_dilated = dataset_utils.FixedStrokeThickness(0.25)(point)
    weakly_eroded = dataset_utils.FixedStrokeThickness(-0.25)(square)

    assert weakly_dilated.sum().item() == pytest.approx(3.0)
    assert weakly_eroded.sum().item() == pytest.approx(7.0)
    assert weakly_dilated[0, 2, 1].item() == pytest.approx(0.25)
    assert weakly_eroded[0, 1, 1].item() == pytest.approx(0.75)


def test_regularizer_uses_parameter_distance_from_identity():
    config = {
        "lora": {
            "regularizer": {
                "transformations": [
                    ["rotation", [-15, 15]],
                    ["isotropic_scale", [1 / 1.1, 1.1]],
                    ["diagonal_shear", [-0.12, 0.12]],
                    ["stroke_thickness", [-0.25, 0.25]],
                ],
            },
        },
    }

    transformation_functions, intensities = utils.create_regularizer_transformations(config)

    assert len(transformation_functions) == 8
    assert intensities.tolist() == pytest.approx([15, 15, math.log(1.1), math.log(1.1), 0.12, 0.12, 0.25, 0.25])


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
