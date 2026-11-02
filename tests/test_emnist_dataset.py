import struct
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from domain_knowledge_analysis import utils
from domain_knowledge_analysis.utils import dataset_utils


@pytest.fixture
def emnist_config(tmp_path):
    """Small real IDX files exercise torchvision loading without a download."""
    raw = tmp_path / "EMNIST" / "raw"
    raw.mkdir(parents=True)
    for split, repeats in [("train", 4), ("test", 2)]:
        labels = torch.arange(1, 27, dtype=torch.uint8).repeat(repeats)
        images = torch.zeros(len(labels), 28, 28, dtype=torch.uint8)
        images[:, 3, 17] = labels
        images[:, 8, 20] = 255
        prefix = raw / f"emnist-letters-{split}"
        prefix.with_name(prefix.name + "-images-idx3-ubyte").write_bytes(
            struct.pack(">IIII", 2051, len(images), 28, 28) + images.numpy().tobytes()
        )
        prefix.with_name(prefix.name + "-labels-idx1-ubyte").write_bytes(
            struct.pack(">II", 2049, len(labels)) + labels.numpy().tobytes()
        )

    config = utils.load_config(utils.get_repo_root() / "config" / "vae_mnist.yaml")
    config["paths"]["dataset_dir"] = str(tmp_path)
    config["dataset"]["name"] = "emnist"
    return config


@pytest.mark.parametrize("train, repeats", [(True, 4), (False, 2)])
def test_emnist_filters_relabels_and_corrects_orientation(emnist_config, train, repeats):
    dataset = utils.create_dataset(emnist_config, "EMNIST", train=train)

    assert dataset.classes == ["B", "G", "I", "L", "O", "Q", "S", "T", "Y", "Z"]
    assert dataset.class_to_idx == {letter: label for label, letter in enumerate(dataset.classes)}
    assert len(dataset) == 10 * repeats
    assert torch.equal(dataset.targets, torch.arange(10).repeat(repeats))
    assert torch.equal(torch.bincount(dataset.targets), torch.full((10,), repeats))

    # Asymmetric markers detect rotation/reflection errors as well as relabeling.
    for index, letter in enumerate(dataset.classes):
        image, label = dataset[index]
        expected = torch.zeros(1, 28, 28)
        expected[0, 17, 3] = (ord(letter) - ord("A") + 1) / 255
        expected[0, 20, 8] = 1
        assert torch.equal(image, expected)
        assert label == index
        assert image.dtype == torch.float32
        assert image.is_contiguous()


def test_emnist_loaders_preserve_reproducible_disjoint_splits(emnist_config):
    train, validation = utils.create_training_dataloaders(emnist_config)
    test = utils.create_testing_dataloaders(emnist_config)
    repeated_train, repeated_validation = utils.create_training_dataloaders(emnist_config)
    assert (len(train.dataset), len(validation.dataset), len(test.dataset)) == (32, 8, 20)
    assert set(train.dataset.indices).isdisjoint(validation.dataset.indices)
    assert set(train.dataset.indices) | set(validation.dataset.indices) == set(range(40))
    assert train.dataset.indices == repeated_train.dataset.indices
    assert validation.dataset.indices == repeated_validation.dataset.indices
    images, labels = next(iter(train))
    assert images.shape == (32, 1, 28, 28)
    assert set(labels.tolist()) == set(range(10))

    source, targets = utils.create_scoring_dataloaders(emnist_config, "emnist", ["EMNIST"])
    assert len(source.dataset) == len(targets["EMNIST"].dataset) == 20
    calibration = utils.create_calibration_dataloader(emnist_config, "emnist")
    assert calibration.dataset.indices == validation.dataset.indices


def test_emnist_transforms_and_cache_preserve_images_and_labels(emnist_config):
    transformation = {"type": "rotation", "intensity": 90}
    train, validation = utils.create_training_dataloaders(emnist_config)
    transformed_train, transformed_validation = utils.create_training_dataloaders(emnist_config, transformation)
    test = utils.create_testing_dataloaders(emnist_config)
    transformed_test = utils.create_testing_dataloaders(emnist_config, transformation)
    cached_train, cached_validation = utils.create_training_dataloaders(emnist_config, transformation)
    cached_test = utils.create_testing_dataloaders(emnist_config, transformation)

    for original, transformed, cached in [
        (train, transformed_train, cached_train),
        (validation, transformed_validation, cached_validation),
        (test, transformed_test, cached_test),
    ]:
        assert len(original.dataset) == len(transformed.dataset) == len(cached.dataset)
        for index in range(len(original.dataset)):
            image, label = original.dataset[index]
            transformed_image, transformed_label = transformed.dataset[index]
            cached_image, cached_label = cached.dataset[index]
            assert torch.allclose(transformed_image, torch.rot90(image, 1, (-2, -1)), atol=1e-5)
            assert torch.equal(transformed_image, cached_image)
            assert label == transformed_label == cached_label


def test_emnist_supports_continuous_bernoulli_preprocessing(emnist_config):
    original = utils.create_dataset(emnist_config, "emnist", train=True)
    emnist_config["loss"]["log_prob_function"] = "continuous_bernoulli"
    dequantized = utils.create_dataset(emnist_config, "emnist", train=True)
    repeated = utils.create_dataset(emnist_config, "emnist", train=True)
    for index in (0, 17, 39):
        image, label = dequantized[index]
        assert 0 < image.min() <= image.max() < 1
        assert torch.equal(image, repeated[index][0])
        assert label == original[index][1]
        assert not torch.equal(image, original[index][0])


@pytest.mark.parametrize("legacy_rotation", [False, True])
@pytest.mark.parametrize("likelihood", ["bernoulli", "continuous_bernoulli"])
def test_lora_target_selects_emnist_without_rotating_or_changing_source(
    emnist_config, monkeypatch, legacy_rotation, likelihood,
):
    config = deepcopy(emnist_config)
    config["dataset"]["name"] = "mnist"
    config["loss"]["log_prob_function"] = likelihood
    target = {"name": "EMNIST", "train_split": 0.75, "shape": [1, 28, 28]}
    if legacy_rotation:
        target.update(type="rotation", intensity=45)
    config["lora"]["transformed_dataset"] = target
    original_config = deepcopy(config)

    def reject_transformation(*args, **kwargs):
        pytest.fail("A named EMNIST target must not transform MNIST or create a transformed cache.")

    monkeypatch.setattr(dataset_utils, "create_transformed_datasets", reject_transformation)
    train, validation = utils.create_training_dataloaders(config, target)
    test = utils.create_testing_dataloaders(config, target)
    expected_pool = utils.create_dataset(config, "emnist", train=True)
    expected_test = utils.create_dataset(config, "emnist", train=False)
    assert (len(train.dataset), len(validation.dataset), len(test.dataset)) == (30, 10, 20)
    assert set(train.dataset.indices).isdisjoint(validation.dataset.indices)
    for subset in (train.dataset, validation.dataset):
        for index, pool_index in enumerate(subset.indices):
            image, label = subset[index]
            expected_image, expected_label = expected_pool[pool_index]
            assert torch.equal(image, expected_image)
            assert label == expected_label
    for index in range(len(test.dataset)):
        image, label = test.dataset[index]
        assert torch.equal(image, expected_test[index][0])
        assert label == expected_test[index][1]
    assert config == original_config
    assert not (Path(config["paths"]["dataset_dir"]) / "transformed").exists()


def test_direct_target_rejects_incompatible_shape(emnist_config):
    target = {"name": "emnist", "shape": [1, 32, 32]}
    with pytest.raises(ValueError, match="target dataset shape"):
        utils.create_training_dataloaders(emnist_config, target)
    with pytest.raises(ValueError, match="target dataset shape"):
        utils.create_testing_dataloaders(emnist_config, target)


def test_unknown_target_without_transformation_is_rejected(emnist_config):
    target = {"name": "emnits"}
    with pytest.raises(ValueError, match="Unknown target dataset"):
        utils.create_training_dataloaders(emnist_config, target)
    with pytest.raises(ValueError, match="Unknown target dataset"):
        utils.create_testing_dataloaders(emnist_config, target)


def test_emnist_lora_config_preserves_existing_source_and_training_settings():
    config_dir = utils.get_repo_root() / "config"
    original = utils.load_config(config_dir / "lora_vae_mnist.yaml")
    emnist = utils.load_config(config_dir / "lora_vae_mnist_emnist.yaml")
    assert emnist["dataset"]["name"] == "mnist"
    assert emnist["lora"]["transformed_dataset"]["name"] == "emnist"
    assert "type" not in emnist["lora"]["transformed_dataset"]
    emnist["experiment"] = original["experiment"]
    emnist["lora"]["transformed_dataset"] = original["lora"]["transformed_dataset"]
    assert emnist == original


@pytest.mark.parametrize("target_name", ["emnist", "rotated_mnist"])
def test_lora_yaml_target_trains_scores_and_loads_checkpoint(
    emnist_config, monkeypatch, tmp_path, target_name,
):
    import yaml
    from domain_knowledge_analysis.experiments import experiment as module
    from domain_knowledge_analysis.training import CheckpointManager, Trainer

    # A distinguishable MNIST source and the fixture's EMNIST target use real
    # torchvision IDX loaders throughout the training and scoring paths.
    raw = Path(emnist_config["paths"]["dataset_dir"]) / "MNIST" / "raw"
    raw.mkdir(parents=True)
    for split, repeats in [("train", 4), ("t10k", 2)]:
        labels = torch.arange(10, dtype=torch.uint8).repeat(repeats)
        images = torch.zeros(len(labels), 28, 28, dtype=torch.uint8)
        images[:, 22, 4:12] = 255
        (raw / f"{split}-images-idx3-ubyte").write_bytes(
            struct.pack(">IIII", 2051, len(images), 28, 28) + images.numpy().tobytes()
        )
        (raw / f"{split}-labels-idx1-ubyte").write_bytes(
            struct.pack(">II", 2049, len(labels)) + labels.numpy().tobytes()
        )

    config = utils.load_config(utils.get_repo_root() / "config" / "lora_vae_mnist_emnist.yaml")
    config["paths"].update(dataset_dir=emnist_config["paths"]["dataset_dir"], run_dir=str(tmp_path / "adapted"))
    config["training"]["epochs"] = 1
    config["dataloader"]["batch_size"] = 16
    config["model"]["encoder"].update(
        latent_dim=3, out_channels=[4, 8], kernels=[3, 3], strides=[2, 2], paddings=[1, 1],
    )
    config["lora"].update(rank=2, regularizer=None)
    if target_name == "rotated_mnist":
        config["lora"]["transformed_dataset"] = utils.load_config(
            utils.get_repo_root() / "config" / "lora_vae_mnist.yaml"
        )["lora"]["transformed_dataset"]
    source_config = deepcopy(config)
    source_config.update(lora=None, pretrained_model=None)
    source_model = utils.create_model(source_config)
    checkpoint_manager = CheckpointManager(tmp_path / "source", source_config)
    checkpoint_manager.save_last(
        source_model, torch.optim.Adam(source_model.parameters()), None, 1, {}, 0.5,
    )
    config["pretrained_model"] = str(checkpoint_manager.checkpoint_path("last.pt"))
    config_path = tmp_path / "lora.yaml"
    config_path.write_text(yaml.safe_dump(config))

    trainers, scorers = [], []
    class RecordingTrainer(Trainer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            trainers.append(self)
    class RecordingScorer(module.AdaptationScorer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            scorers.append(self)

    monkeypatch.setattr(module, "Trainer", RecordingTrainer)
    monkeypatch.setattr(module, "AdaptationScorer", RecordingScorer)
    monkeypatch.setattr(module.utils, "get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(module.Experiment, "print_tensorboard_instructions", lambda *args: None)
    experiment = module.Experiment(config_path)
    try:
        experiment.score_adaptation()
    finally:
        experiment.logger.close()

    assert len(trainers) == len(scorers) == 1
    assert (len(trainers[0].train_dataloader.dataset), len(trainers[0].validate_dataloader.dataset)) == (32, 8)
    assert torch.isfinite(torch.tensor(trainers[0].history["train_loss"])).all()
    assert experiment.source_dataset_name == experiment.config["dataset"]["name"] == "mnist"
    source_test = utils.create_dataset(config, "mnist", train=False)
    expected_test = utils.create_dataset(config, "emnist", train=False) if target_name == "emnist" else source_test
    for index in range(len(expected_test)):
        expected_image, expected_label = expected_test[index]
        if target_name == "rotated_mnist":
            expected_image = dataset_utils.FixedRotation(45)(expected_image)
        image, label = scorers[0].target_dataloader.dataset[index]
        assert torch.equal(image, expected_image)
        assert label == expected_label
        assert torch.equal(scorers[0].source_dataloader.dataset[index][0], source_test[index][0])
    assert (experiment.log_dir / "results" / "adaptation_reconstructions.png").exists()

    # Scoring a saved adapter uses its recorded target even if the current YAML
    # describes a different target. The source must remain original MNIST.
    config["lora"]["pretrained_model"] = str(experiment.checkpoint_manager.checkpoint_path("last.pt"))
    config["lora"]["transformed_dataset"] = {"name": "emnist" if target_name != "emnist" else "mnist"}
    config["paths"]["run_dir"] = str(tmp_path / "reloaded")
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(module.Experiment, "adapt", lambda *args: pytest.fail("Saved adapters must not retrain."))
    reloaded = module.Experiment(config_path)
    try:
        reloaded.score_adaptation()
    finally:
        reloaded.logger.close()
    assert len(trainers) == 1 and len(scorers) == 2
    assert reloaded.source_dataset_name == "mnist"
    assert reloaded.lora_config["transformed_dataset"]["name"] == target_name
    for index in range(len(expected_test)):
        assert torch.equal(scorers[1].target_dataloader.dataset[index][0], scorers[0].target_dataloader.dataset[index][0])
