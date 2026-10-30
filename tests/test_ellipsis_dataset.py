from copy import deepcopy

import torch

from domain_knowledge_analysis import utils
from domain_knowledge_analysis.utils.dataset_utils import create_ellipsis_dataset


def test_ellipses_are_centered_grayscale_and_vary_in_size_and_thickness():
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
    assert torch.equal(images, images.flip(-1))
    assert torch.equal(images, images.flip(-2))
    assert not images[:, :, 13:15, 13:15].any()

    mask = images > 0
    widths = mask.any(dim=2).sum(dim=-1).squeeze(1)
    heights = mask.any(dim=3).sum(dim=-1).squeeze(1)
    assert widths.min() >= 20 and widths.max() <= 24
    assert heights.min() >= 10 and heights.max() <= 12
    assert widths.unique().numel() > 1 and heights.unique().numel() > 1

    # Images with the same outer dimensions can still have different thicknesses.
    smallest = images[(widths == widths.min()) & (heights == heights.min())]
    ink = smallest.sum(dim=(1, 2, 3))
    assert ink.max() - ink.min() > 10

    # Even a thick, small ellipse has a center that narrows toward its top edge.
    thickest = smallest[ink.argmax(), 0]
    center_width = (thickest[13, 8:20] < 0.2).sum()
    upper_width = (thickest[12, 8:20] < 0.2).sum()
    assert upper_width < center_width


def test_ellipsis_dataloaders_and_transformed_splits(tmp_path):
    config = deepcopy(utils.load_config(utils.get_repo_root() / "config" / "vae_lines.yaml"))
    config["dataset"]["name"] = "ellipsis_dataset"
    config["paths"]["dataset_dir"] = str(tmp_path)

    old_cache = (tmp_path / "transformed" / "ellipsis_dataset" / "rotation_intensity_90"
                 / "version_5_ellipsis_5" / f'seed_{config["seed"]}_split_{config["dataset"]["train_split"]:g}')
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
