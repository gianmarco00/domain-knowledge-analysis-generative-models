from copy import deepcopy
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
        ("vertical_shear", 0.12),
        ("gaussian_blur", 0.7),
        ("contrast", 1.2),
        ("stroke_thickness", -0.25),
        ("stroke_thickness", 0.25),
        ("stroke_thickness", -1),
        ("stroke_thickness", 1),
        ("random_direction_0", 0.05),
        ("horizontal_flip", 0.5),
        ("vertical_flip", 0.5),
        ("gamma_correction", 1.2),
        ("binarization", 0.5),
        ("pixelation", 0.5),
        ("horizontal_motion_blur", 0.5),
        ("vertical_motion_blur", 0.5),
        ("horizontal_wave", 2),
        ("vertical_wave", 2),
        ("perspective", 0.2),
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


def test_fixed_translations_support_fractional_pixels():
    image = torch.zeros(1, 5, 5)
    image[0, 2, 2] = 1.0

    translated_right = dataset_utils.FixedHorizontalTranslation(0.5)(image)
    translated_down = dataset_utils.FixedVerticalTranslation(0.5)(image)

    assert translated_right[0, 2, 2:4].tolist() == pytest.approx([0.5, 0.5])
    assert translated_down[0, 2:4, 2].tolist() == pytest.approx([0.5, 0.5])


@pytest.mark.parametrize(
    "transformation_type, identity_intensity",
    [
        ("horizontal_translation", 0),
        ("vertical_translation", 0),
        ("rotation", 0),
        ("isotropic_scale", 1),
        ("aspect_ratio_deformation", 1),
        ("diagonal_shear", 0),
        ("vertical_shear", 0),
        ("stroke_thickness", 0),
        ("gaussian_blur", 0),
        ("contrast", 1),
        ("random_direction_0", 0),
        ("horizontal_flip", 0),
        ("vertical_flip", 0),
        ("gamma_correction", 1),
        ("binarization", 0),
        ("pixelation", 0),
        ("horizontal_motion_blur", 0),
        ("vertical_motion_blur", 0),
        ("horizontal_wave", 0),
        ("vertical_wave", 0),
        ("perspective", 0),
    ],
)
def test_regularizer_transformations_are_exact_at_identity(transformation_type, identity_intensity):
    images = torch.rand(2, 1, 9, 9)
    transformation = dataset_utils.create_dataset_transformation(transformation_type, identity_intensity)

    assert torch.equal(transformation(images), images)


def test_aspect_ratio_deformation_is_continuous():
    image = torch.zeros(1, 9, 9)
    image[:, 2:7, 3:6] = 1.0

    transformed = dataset_utils.FixedAspectRatioDeformation(1.01)(image)

    assert transformed.shape == image.shape
    assert not torch.equal(transformed, image)


@pytest.mark.parametrize("transformation_type", ["horizontal_flip", "vertical_flip", "binarization", "pixelation", "horizontal_motion_blur", "vertical_motion_blur"])
def test_bounded_transformations_follow_a_continuous_blend_path(transformation_type):
    image = torch.rand(2, 1, 9, 9)
    endpoint = dataset_utils.create_dataset_transformation(transformation_type, 1)(image)
    midpoint = dataset_utils.create_dataset_transformation(transformation_type, 0.5)(image)

    torch.testing.assert_close(midpoint, torch.lerp(image, endpoint, 0.5))


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
                    {"name": "rotation", "intensity": 15},
                    {"name": "isotropic_scale", "intensity": 1.1},
                    {"name": "diagonal_shear", "intensity": 0.12},
                    {"name": "stroke_thickness", "intensity": 0.25},
                    {"name": "random_direction_0", "intensity": 0.05, "direction_seed": 10},
                ],
            },
        },
    }

    transformation_functions, intensities = utils.create_regularizer_transformations(config)

    assert len(transformation_functions) == 5
    assert intensities.tolist() == pytest.approx([15, math.log(1.1), 0.12, 0.25, 0.05])


def target_rms_config(target_input_rms=0.02):
    return {
        "lora": {
            "regularizer": {
                "target_input_rms": target_input_rms,
                "calibration_batch_size": 3,
                "transformations": [
                    {"name": "horizontal_translation", "intensity": 2},
                    {"name": "vertical_translation", "intensity": -2},
                    {"name": "rotation", "intensity": 15},
                    {"name": "isotropic_scale", "intensity": 1.1},
                    {"name": "aspect_ratio_deformation", "intensity": 1 / 1.1},
                    {"name": "diagonal_shear", "intensity": 0.12},
                    {"name": "vertical_shear", "intensity": -0.12},
                    {"name": "stroke_thickness", "intensity": 0.5},
                    {"name": "gaussian_blur", "intensity": 0.7},
                    {"name": "contrast", "intensity": 1.2},
                    {"name": "random_direction_0", "intensity": 0.05, "direction_seed": 7},
                ],
            },
        },
    }


def relative_strength_config(transformation_strength=0.5):
    config = target_rms_config()
    config["lora"]["regularizer"].pop("target_input_rms")
    config["lora"]["regularizer"]["transformation_strength"] = transformation_strength
    config["lora"]["regularizer"]["transformations"] = config["lora"]["regularizer"]["transformations"][:-1]
    config["lora"]["regularizer"]["transformations"].extend([
        {"name": "horizontal_flip", "intensity": 0.5},
        {"name": "vertical_flip", "intensity": 0.5},
        {"name": "gamma_correction", "intensity": 1.2},
        {"name": "binarization", "intensity": 0.5},
        {"name": "pixelation", "intensity": 0.5},
        {"name": "horizontal_motion_blur", "intensity": 0.5},
        {"name": "vertical_motion_blur", "intensity": 0.5},
        {"name": "horizontal_wave", "intensity": 2},
        {"name": "vertical_wave", "intensity": -2},
        {"name": "perspective", "intensity": -0.2},
    ])
    return config


def test_regularizer_transformations_match_shared_target_rms_and_are_deterministic():
    torch.manual_seed(4)
    anchors = torch.rand(8, 1, 9, 9)
    first_config = target_rms_config()
    second_config = deepcopy(first_config)

    first_transformations, _ = utils.create_regularizer_transformations(first_config, anchor_points=anchors)
    second_transformations, _ = utils.create_regularizer_transformations(second_config, anchor_points=anchors)
    first_metadata = first_config["lora"]["regularizer"]["calibration"]["transformations"]
    second_metadata = second_config["lora"]["regularizer"]["calibration"]["transformations"]

    assert [item["achieved_input_rms"] for item in first_metadata] == pytest.approx([0.02] * len(first_metadata), rel=0.01, abs=1e-8)
    assert [item["resolved_intensity"] for item in first_metadata] == pytest.approx([item["resolved_intensity"] for item in second_metadata])
    assert first_metadata[1]["resolved_intensity"] < 0
    assert first_metadata[4]["resolved_intensity"] < 1
    assert first_metadata[6]["resolved_intensity"] < 0
    assert first_metadata[-1]["resolved_intensity"] == pytest.approx(0.02)
    assert first_metadata[-1]["effective_direction_seed"] == 7

    for first, second in zip(first_transformations, second_transformations):
        torch.testing.assert_close(first(anchors), second(anchors))


@pytest.mark.parametrize("target_input_rms", [0, 1e-6, float("nan")])
def test_shared_target_rms_must_exceed_input_scale_floor(target_input_rms):
    anchors = torch.rand(2, 1, 5, 5)

    with pytest.raises(ValueError, match="greater than 1e-06"):
        utils.create_regularizer_transformations(target_rms_config(target_input_rms), anchor_points=anchors)


def test_shared_target_rms_reports_unreachable_transformation():
    anchors = torch.rand(2, 1, 5, 5)
    config = target_rms_config(10.0)
    config["lora"]["regularizer"]["transformations"] = [{"name": "stroke_thickness", "intensity": 0.5}]

    with pytest.raises(ValueError, match=r"stroke_thickness.*measured reachable RMS range"):
        utils.create_regularizer_transformations(config, anchor_points=anchors)


def test_relative_strength_is_a_fraction_of_each_transformation_maximum():
    torch.manual_seed(4)
    anchors = torch.rand(8, 1, 9, 9)
    config = relative_strength_config(0.6)

    utils.create_regularizer_transformations(config, anchor_points=anchors)
    metadata = config["lora"]["regularizer"]["calibration"]["transformations"]

    assert [item["relative_input_rms"] for item in metadata] == pytest.approx([0.6] * len(metadata), rel=0.01, abs=1e-8)
    assert all(item["maximum_input_rms"] > item["achieved_input_rms"] for item in metadata)
    assert metadata[1]["resolved_intensity"] < 0
    assert metadata[4]["resolved_intensity"] < 1
    assert metadata[6]["resolved_intensity"] < 0


def test_maximum_relative_strength_uses_earliest_actual_maximum():
    torch.manual_seed(4)
    anchors = torch.rand(8, 1, 9, 9)
    config = relative_strength_config(1.0)
    config["lora"]["regularizer"]["transformations"] = [{"name": "rotation", "intensity": 15}]
    repeated_config = deepcopy(config)

    transformations, _ = utils.create_regularizer_transformations(config, anchor_points=anchors)
    utils.create_regularizer_transformations(repeated_config, anchor_points=anchors)
    metadata = config["lora"]["regularizer"]["calibration"]["transformations"][0]
    repeated_metadata = repeated_config["lora"]["regularizer"]["calibration"]["transformations"][0]
    achieved_rms = utils.measure_input_rms(transformations[0], anchors, 3)

    assert 0 < metadata["resolved_intensity"] <= 180
    assert metadata["resolved_intensity"] == pytest.approx(repeated_metadata["resolved_intensity"])
    assert achieved_rms == pytest.approx(metadata["maximum_input_rms"], rel=0.01)
    assert metadata["relative_input_rms"] == pytest.approx(1.0, rel=0.01)


@pytest.mark.parametrize("transformation_strength", [0, -0.1, 1.1, float("nan")])
def test_relative_strength_must_be_in_unit_interval(transformation_strength):
    anchors = torch.rand(2, 1, 5, 5)

    with pytest.raises(ValueError, match=r"in \(0, 1\]"):
        utils.create_regularizer_transformations(relative_strength_config(transformation_strength), anchor_points=anchors)


def test_relative_strength_rejects_unbounded_random_directions():
    anchors = torch.rand(2, 1, 5, 5)
    config = relative_strength_config()
    config["lora"]["regularizer"]["transformations"] = [{"name": "random_direction_0", "intensity": 0.05, "direction_seed": 7}]

    with pytest.raises(ValueError, match="no finite maximum"):
        utils.create_regularizer_transformations(config, anchor_points=anchors)


def test_random_direction_is_fixed_and_direction_index_changes_it():
    images = torch.zeros(2, 1, 9, 9)
    first = dataset_utils.FixedRandomDirectionPerturbation(0, 0.05, 123)
    same = dataset_utils.FixedRandomDirectionPerturbation(0, 0.05, 123)
    different = dataset_utils.FixedRandomDirectionPerturbation(1, 0.05, 123)

    assert torch.equal(first(images), same(images))
    assert not torch.equal(first(images), different(images))


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
