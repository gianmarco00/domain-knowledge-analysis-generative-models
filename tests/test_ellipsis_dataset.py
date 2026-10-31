from copy import deepcopy

import torch

from domain_knowledge_analysis import utils
from domain_knowledge_analysis.utils.dataset_utils import create_ellipsis_dataset


def test_ellipses_vary_in_position_size_tilt_and_thickness():
    images, labels = create_ellipsis_dataset(train=True, seed=42).tensors
    repeated, _ = create_ellipsis_dataset(train=True, seed=42).tensors
    test, _ = create_ellipsis_dataset(train=False, seed=42).tensors

    assert images.shape == (5000, 1, 28, 28)
    assert test.shape == (1000, 1, 28, 28)
    assert torch.equal(images, repeated)
    assert not torch.equal(images[:1000], test)
    assert torch.all(labels == 0)
    assert images.min() == 0 and images.max() == 1
    assert ((images > 0) & (images < 1)).any()
    assert not images[:, :, 0, :].any() and not images[:, :, -1, :].any()
    assert not images[:, :, :, 0].any() and not images[:, :, :, -1].any()

    mask = images > 0
    widths = mask.any(dim=2).sum(dim=-1).squeeze(1)
    heights = mask.any(dim=3).sum(dim=-1).squeeze(1)
    assert widths.min() <= 11 and widths.max() >= 22
    assert heights.min() <= 6 and heights.max() >= 11

    # Image moments measure position and tilt without needing generator internals.
    pixels = images[:, 0]
    coordinates = torch.arange(28, dtype=torch.float)
    mass = pixels.sum(dim=(1, 2))
    center_x = (pixels.sum(dim=1) * coordinates).sum(dim=1) / mass
    center_y = (pixels.sum(dim=2) * coordinates).sum(dim=1) / mass
    assert center_x.min() < 8 and center_x.max() > 19
    assert center_y.min() < 5 and center_y.max() > 22

    dx = coordinates[None, None, :] - center_x[:, None, None]
    dy = coordinates[None, :, None] - center_y[:, None, None]
    covariance = (pixels * dx * dy).sum(dim=(1, 2))
    variance_x = (pixels * dx.square()).sum(dim=(1, 2))
    variance_y = (pixels * dy.square()).sum(dim=(1, 2))
    angles = 0.5 * torch.atan2(2 * covariance, variance_x - variance_y)
    assert (angles < -0.02).sum() > 500 and (angles > 0.02).sum() > 500
    assert angles.abs().max() < 0.06  # Allow for pixel rounding around a 5% slope.

    center_pixels = pixels[torch.arange(len(images)), center_y.round().long(), center_x.round().long()]
    assert (center_pixels < 0.1).sum() > 100  # Open holes.
    assert (center_pixels > 0.9).sum() > 100  # Filled ellipses.
    assert ((center_pixels > 0.1) & (center_pixels < 0.9)).sum() > 100


def test_ellipsis_dataloaders_and_transformed_splits(tmp_path):
    config = deepcopy(utils.load_config(utils.get_repo_root() / "config" / "vae_lines.yaml"))
    config["dataset"]["name"] = "ellipsis_dataset"
    config["paths"]["dataset_dir"] = str(tmp_path)

    old_cache = (tmp_path / "transformed" / "ellipsis_dataset" / "rotation_intensity_90"
                 / "version_5_ellipsis_6" / f'seed_{config["seed"]}_split_{config["dataset"]["train_split"]:g}')
    old_cache.mkdir(parents=True)
    (old_cache / "train.pt").write_bytes(b"old ellipses")
    (old_cache / "validation.pt").write_bytes(b"old ellipses")

    train, validation = utils.create_training_dataloaders(config)
    test = utils.create_testing_dataloaders(config)
    transformed_train, transformed_validation = utils.create_training_dataloaders(
        config, {"type": "rotation", "intensity": 90}
    )
    transformed_test = utils.create_testing_dataloaders(config, {"type": "rotation", "intensity": 90})

    assert (len(train.dataset), len(validation.dataset), len(test.dataset)) == (4000, 1000, 1000)
    assert (len(transformed_train.dataset), len(transformed_validation.dataset), len(transformed_test.dataset)) == (4000, 1000, 1000)
    assert set(train.dataset.indices).isdisjoint(validation.dataset.indices)
    image, label = train.dataset[0]
    transformed_image, transformed_label = transformed_train.dataset[0]
    assert torch.equal(label, transformed_label)
    assert torch.allclose(transformed_image, torch.rot90(image, 1, (-2, -1)), atol=1e-5)
    rotated_mask = transformed_image > 1e-5
    original_mask = image > 1e-5
    assert rotated_mask.any(dim=1).sum() == original_mask.any(dim=2).sum()
    assert rotated_mask.any(dim=2).sum() == original_mask.any(dim=1).sum()
    assert torch.equal(utils.create_dataset(config, "ellipse_dataset", train=False)[0][0], test.dataset[0][0])
