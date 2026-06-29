import random
from pathlib import Path

import pytest
import torch
import yaml

from domain_knowledge_analysis.training import utils
from domain_knowledge_analysis.models import Vae
from domain_knowledge_analysis.losses import vae_loss


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


class DummyImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, train, download, transform):
        self.root = root
        self.train = train
        self.download = download
        self.transform = transform
        self.data = torch.rand(20, 1, 28, 28)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]

        label = 0
        return image, label


def test_load_config_reads_yaml(tmp_path, config):
    config_path = tmp_path / "config.yaml"

    with open(config_path, "w") as file:
        yaml.safe_dump(config, file)

    loaded_config = utils.load_config(config_path)

    assert loaded_config["experiment"]["name"] == "vae_mnist_test"
    assert loaded_config["training"]["learning_rate"] == 0.001
    assert loaded_config["dataset"]["shape"] == [1, 28, 28]


def test_set_seed_makes_torch_randomness_reproducible():
    utils.set_seed(123)
    first = torch.rand(3)

    utils.set_seed(123)
    second = torch.rand(3)

    assert torch.allclose(first, second)


def test_set_seed_makes_python_randomness_reproducible():
    utils.set_seed(123)
    first = random.random()

    utils.set_seed(123)
    second = random.random()

    assert first == second


def test_build_run_name_contains_experiment_name_and_learning_rate(config):
    run_name = utils.build_run_name(config)

    assert "vae_mnist_test" in run_name
    assert "0.001" in run_name


def test_get_device_returns_torch_device():
    device = utils.get_device()

    assert isinstance(device, torch.device)
    assert device.type in ["cuda", "mps", "cpu"]


def test_create_transform_for_mnist(config):
    dataset_name = config["dataset"]["name"]
    transform = utils.create_transform(dataset_name)

    assert transform is not None


def test_create_model_returns_vae(config):
    model = utils.create_model(config)

    assert isinstance(model, Vae)


def test_create_model_raises_for_unknown_model(config):
    config["model"]["name"] = "unknown_model"

    with pytest.raises(ValueError, match="Unsupported model"):
        utils.create_model(config)


def test_create_optimizer_returns_adam(config):
    model = utils.create_model(config)
    optimizer = utils.create_optimizer(config, model)

    assert isinstance(optimizer, torch.optim.Adam)


def test_create_optimizer_raises_for_unknown_optimizer(config):
    model = utils.create_model(config)
    config["optimizer"]["name"] = "sgd_but_not_supported_yet"

    with pytest.raises(ValueError, match="Unsupported optimizer"):
        utils.create_optimizer(config, model)


def test_create_loss_returns_vae_loss(config):
    loss = utils.create_loss(config)

    assert loss is vae_loss


def test_create_loss_raises_for_unknown_model(config):
    config["model"]["name"] = "unknown_model"

    with pytest.raises(ValueError, match="Unsupported loss"):
        utils.create_loss(config)


def test_create_log_dir_uses_runs_dir(config):
    log_dir = utils.create_log_dir(config)

    assert isinstance(log_dir, Path)
    assert "runs" in str(log_dir)
    assert "vae_mnist_test" in str(log_dir)


def test_create_dataset_returns_dataset_with_expected_shape(monkeypatch, config):
    monkeypatch.setattr(utils.datasets, "MNIST", DummyImageDataset)

    dataset_name = config["dataset"]["name"]

    transform = utils.create_transform(dataset_name)
    dataset = utils.create_dataset(config, dataset_name, transform)

    image, label = dataset[0]

    assert len(dataset) == 20
    assert image.shape == torch.Size([1, 28, 28])
    assert label == 0


def test_create_dataset_raises_if_shape_is_wrong(monkeypatch, config):
    monkeypatch.setattr(utils.datasets, "MNIST", DummyImageDataset)

    config["dataset"]["shape"] = [3, 28, 28]
    dataset_name = config["dataset"]["name"]

    transform = utils.create_transform(dataset_name)

    with pytest.raises(ValueError, match="wrong data shape"):
        utils.create_dataset(config, dataset_name, transform)


def test_create_training_dataloaders_returns_train_and_validation_loaders(monkeypatch, config):
    monkeypatch.setattr(utils.datasets, "MNIST", DummyImageDataset)

    train_dataloader, validation_dataloader = utils.create_training_dataloaders(config)

    assert len(train_dataloader.dataset) == 16
    assert len(validation_dataloader.dataset) == 4

    x, y = next(iter(train_dataloader))

    assert x.shape == torch.Size([4, 1, 28, 28])
    assert y.shape == torch.Size([4])

def test_create_log_dir_creates_directory(config):
    log_dir = utils.create_log_dir(config)

    assert log_dir.exists()
    assert log_dir.is_dir()

def test_create_dataset_does_not_download_if_mnist_already_exists(monkeypatch, config):
    monkeypatch.setattr(utils.datasets, "MNIST", DummyImageDataset)

    dataset_name = config["dataset"]["name"]

    dataset_dir = Path(config["paths"]["dataset_dir"])
    processed_dir = dataset_dir / "MNIST" / "processed"
    processed_dir.mkdir(parents=True)

    (processed_dir / "training.pt").touch()
    (processed_dir / "test.pt").touch()

    dataset = utils.create_dataset(
        config=config,
        dataset_name=dataset_name,
        train=True,
    )

    assert dataset.download is False

def test_create_dataset_downloads_if_mnist_test_file_is_missing(monkeypatch, config):
    monkeypatch.setattr(utils.datasets, "MNIST", DummyImageDataset)

    dataset_name = config["dataset"]["name"]

    dataset_dir = Path(config["paths"]["dataset_dir"])
    processed_dir = dataset_dir / "MNIST" / "processed"
    processed_dir.mkdir(parents=True)

    (processed_dir / "training.pt").touch()

    dataset = utils.create_dataset(
        config=config,
        dataset_name=dataset_name,
        train=False,
    )

    assert dataset.download is True

def test_create_dataset_downloads_if_mnist_is_missing(monkeypatch, config):
    monkeypatch.setattr(utils.datasets, "MNIST", DummyImageDataset)

    dataset_name = config["dataset"]["name"]

    transform = utils.create_transform(dataset_name)
    dataset = utils.create_dataset(config, dataset_name, transform)

    assert dataset.download is True

def test_create_scoring_dataloaders_creates_test_dataloaders(monkeypatch, config):
    monkeypatch.setattr(utils.datasets, "MNIST", DummyImageDataset)
    monkeypatch.setattr(utils.datasets, "FashionMNIST", DummyImageDataset)

    config["scoring"] = {
        "dataloader": {
            "batch_size": 3,
            "num_workers": 0,
        }
    }

    in_dataloader, out_dataloader = utils.create_scoring_dataloaders(
        config=config,
        in_distribution_dataset_name="mnist",
        out_distribution_dataset_name="fashion_mnist",
    )

    assert in_dataloader.batch_size == 3
    assert out_dataloader.batch_size == 3

    assert in_dataloader.dataset.train is False
    assert out_dataloader.dataset.train is False

def test_create_scoring_dataloaders_return_batches_with_expected_shape(monkeypatch, config):
    monkeypatch.setattr(utils.datasets, "MNIST", DummyImageDataset)
    monkeypatch.setattr(utils.datasets, "FashionMNIST", DummyImageDataset)

    config["scoring"] = {
        "dataloader": {
            "batch_size": 4,
            "num_workers": 0,
        }
    }

    in_dataloader, out_dataloader = utils.create_scoring_dataloaders(
        config=config,
        in_distribution_dataset_name="mnist",
        out_distribution_dataset_name="fashion_mnist",
    )

    in_batch = next(iter(in_dataloader))[0]
    out_batch = next(iter(out_dataloader))[0]

    assert in_batch.shape == (4, 1, 28, 28)
    assert out_batch.shape == (4, 1, 28, 28)
