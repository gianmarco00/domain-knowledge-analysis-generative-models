from copy import deepcopy

import torch

from domain_knowledge_analysis import utils
from domain_knowledge_analysis.utils.dataset_utils import create_lines_dataset


def test_lines_are_binary_reproducible_and_have_varied_thickness():
    horizontal, labels = create_lines_dataset(vertical=False, train=True, seed=42).tensors
    vertical, vertical_labels = create_lines_dataset(vertical=True, train=True, seed=42).tensors
    repeated, _ = create_lines_dataset(vertical=False, train=True, seed=42).tensors
    test, _ = create_lines_dataset(vertical=False, train=False, seed=42).tensors

    assert horizontal.shape == vertical.shape == (5000, 1, 28, 28)
    assert test.shape == (1000, 1, 28, 28)
    assert torch.equal(horizontal, repeated)
    assert not torch.equal(horizontal[:1000], test)
    assert torch.all(labels == 0) and torch.all(vertical_labels == 1)

    for images, expected_thicknesses in [(horizontal, {1, 2}), (vertical.transpose(-1, -2), {4, 5})]:
        assert set(images.unique().tolist()) == {0.0, 1.0}
        assert not images[:, :, [0, -1], :].any()
        assert not images[:, :, :, [0, -1]].any()

        column_widths = images.sum(dim=2)[:, 0]
        lengths = (column_widths > 0).sum(dim=1)
        assert torch.all((lengths >= 16) & (lengths <= 24))
        thicknesses = column_widths.masked_fill(column_widths == 0, 28).min(dim=1).values
        assert set(thicknesses.tolist()) == expected_thicknesses
        assert torch.all(column_widths.max(dim=1).values <= thicknesses + 1)

        # Consecutive columns overlap, including both endpoints.
        overlap = (images[:, :, :, 1:] * images[:, :, :, :-1]).sum(dim=2)[:, 0]
        adjacent = (column_widths[:, 1:] > 0) & (column_widths[:, :-1] > 0)
        assert torch.all(overlap[adjacent] > 0)
        start = (column_widths > 0).float().argmax(dim=1)
        end = start + lengths - 1
        indices = torch.arange(len(images))
        assert torch.equal(images[indices, 0, :, start], images[indices, 0, :, start + 1])
        assert torch.equal(column_widths[indices, end], thicknesses)


def test_lines_have_tilts_in_both_directions():
    images, _ = create_lines_dataset(vertical=False, train=True, seed=42).tensors
    widths = images[:, 0].sum(dim=1)
    centers = (images[:, 0] * torch.arange(28)[None, :, None]).sum(dim=1) / widths.clamp_min(1)
    starts = (widths > 0).float().argmax(dim=1)
    ends = 27 - (widths > 0).flip(dims=[1]).float().argmax(dim=1)
    indices = torch.arange(len(images))
    displacement = centers[indices, ends] - centers[indices, starts]
    assert (displacement >= 2).sum() > 100
    assert (displacement <= -2).sum() > 100
    assert (displacement == 0).sum() > 100


def test_lines_loaders_transformations_and_small_vae(tmp_path):
    config = utils.load_config(utils.get_repo_root() / "config" / "vae_lines.yaml")
    config = deepcopy(config)
    config["paths"]["dataset_dir"] = str(tmp_path)

    old_cache = tmp_path / "transformed" / "lines_dataset" / "rotation_intensity_90" / "version_5" / "seed_42_split_0.8"
    old_cache.mkdir(parents=True)
    (old_cache / "train.pt").write_bytes(b"old dataset")
    (old_cache / "validation.pt").write_bytes(b"old dataset")

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

    model = utils.create_model(config)
    images, _ = next(iter(train))
    logits, mean, log_variance = model(images)
    loss = utils.create_loss(config)(images, logits, mean, log_variance)
    loss.backward()
    assert logits.shape == images.shape
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_lines_ablation_uses_the_source_model_and_checkpoint(tmp_path):
    from domain_knowledge_analysis.experiments.ablation_utils import generate_model_runs

    root = utils.get_repo_root()
    source = utils.load_config(root / "config" / "vae_lines.yaml")
    ablation_config = utils.load_config(root / "config" / "ablation_vae_lines.yaml")
    base = utils.load_config(root / ablation_config["base_config"])
    ablation = ablation_config["ablation"]

    assert source["model"] == base["model"]
    assert source["loss"] == base["loss"]
    assert source["dataset"] == base["dataset"]
    assert base["lora"]["transformed_dataset"]["intensity"] == 90.0
    runs = generate_model_runs(base, ablation, tmp_path)
    expected_runs = len(ablation["eta_values"]) * len(ablation["m_values"]) * len(ablation["s_values"]) + 2
    assert len(runs) == expected_runs
    assert {run["m"] for run in runs} == {0, *ablation["m_values"]}
