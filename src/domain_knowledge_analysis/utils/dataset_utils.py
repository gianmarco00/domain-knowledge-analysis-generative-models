from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset, Dataset
from torchvision import datasets, transforms

import medmnist
from medmnist import INFO

from .utils import get_repo_root

DATASET_ALIASES = {
    "fashionmnist": "fashion_mnist",
    "fmnist": "fashion_mnist",

    "organ_amnist": "organamnist",
    "organ-amnist": "organamnist",

    "pneumonia_mnist": "pneumoniamnist",
    "pneumonia-mnist": "pneumoniamnist",
}


TORCHVISION_DATASETS = {
    "mnist": datasets.MNIST,
    "fashion_mnist": datasets.FashionMNIST,
    "kmnist": datasets.KMNIST,
}


MEDMNIST_DATASETS = {
    "organamnist",
    "pneumoniamnist",
}
    

def create_dataset(config, dataset_name, train):
    dataset_name = dataset_name.lower()
    dataset_name = DATASET_ALIASES.get(
        dataset_name,
        dataset_name,
    )

    transform = create_transform(
        dataset_name=dataset_name
    )

    dataset_dir = config["paths"]["dataset_dir"]
    repo_root = get_repo_root()
    dataset_dir = Path(repo_root / dataset_dir)

    expected_shape = tuple(
        config["dataset"]["shape"]
    )

    if dataset_name in TORCHVISION_DATASETS:
        dataset = create_torchvision_dataset(
            dataset_class=TORCHVISION_DATASETS[dataset_name],
            dataset_dir=dataset_dir,
            train=train,
            transform=transform,
        )

    elif dataset_name in MEDMNIST_DATASETS:
        dataset = create_medmnist_dataset(
            dataset_name=dataset_name,
            dataset_dir=dataset_dir,
            train=train,
            transform=transform,
        )

    else:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}"
        )
    
    if config["model"]["name"] == "vae" and config["loss"]["log_prob_function"].lower() == "continuous_bernoulli":
        dequantization_seed = config["seed"] + (0 if train else 1_000_000)
        dataset = UniformDequantizedDataset(dataset=dataset, seed=dequantization_seed)

    actual_shape = tuple(dataset[0][0].shape)

    if actual_shape != expected_shape:
        raise ValueError(
            f"Config file contains wrong data shape. "
            f"Expected {expected_shape}, got {actual_shape}."
        )

    return dataset


def create_train_validation_datasets(config, dataset_name):
    dataset = create_dataset(
        config=config,
        dataset_name=dataset_name,
        train=True,
    )

    train_split = config["dataset"]["train_split"]
    train_size = int(train_split * len(dataset))
    validation_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(config["seed"])

    train_dataset, validation_dataset = random_split(
        dataset,
        [train_size, validation_size],
        generator=generator,
    )

    return train_dataset, validation_dataset


def create_training_dataloaders(config):

    dataset_name = config["dataset"]["name"].lower()

    train_dataset, validation_dataset = create_train_validation_datasets(
        config=config,
        dataset_name=dataset_name,
    )

    dataloader_config = config["dataloader"]

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=dataloader_config["batch_size"],
        shuffle=dataloader_config["shuffle_train"],
        num_workers=dataloader_config["num_workers"],
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=dataloader_config["batch_size"],
        shuffle=dataloader_config["shuffle_validation"],
        num_workers=dataloader_config["num_workers"],
    )

    return train_dataloader, validation_dataloader


def create_scoring_dataloaders(
    config,
    in_distribution_dataset_name,
    out_distribution_dataset_names,
):

    in_distribution_dataset = create_dataset(
        config=config,
        dataset_name=in_distribution_dataset_name,
        train=False,
    )

    out_distribution_datasets = {}

    for dataset_name in out_distribution_dataset_names:
        out_distribution_dataset = create_dataset(
            config=config,
            dataset_name=dataset_name,
            train=False,
        )
        out_distribution_datasets.update({dataset_name : out_distribution_dataset})

    dataloader_config = config["scoring"]["dataloader"]

    in_distribution_dataloader = DataLoader(
        in_distribution_dataset,
        batch_size=dataloader_config["batch_size"],
        shuffle=False,
        num_workers=dataloader_config["num_workers"],
    )

    out_distribution_dataloaders = {}

    for dataset_name, out_distribution_dataset in out_distribution_datasets.items():

        out_distribution_dataloader = DataLoader(
            out_distribution_dataset,
            batch_size=dataloader_config["batch_size"],
            shuffle=False,
            num_workers=dataloader_config["num_workers"],
        )
        out_distribution_dataloaders.update({dataset_name: out_distribution_dataloader})

    return in_distribution_dataloader, out_distribution_dataloaders


def create_calibration_dataloader(config, dataset_name):
    _, validation_dataset = create_train_validation_datasets(
        config=config,
        dataset_name=dataset_name,
    )

    dataloader_config = config["scoring"]["dataloader"]

    dataloader = DataLoader(
        validation_dataset,
        batch_size=dataloader_config["batch_size"],
        shuffle=False,
        num_workers=dataloader_config["num_workers"],
    )

    return dataloader


def create_transform(dataset_name):
    supported_datasets = (
        set(TORCHVISION_DATASETS)
        | MEDMNIST_DATASETS
    )

    if dataset_name not in supported_datasets:
        raise ValueError(
            f"Unsupported transform for dataset: {dataset_name}"
        )

    return transforms.ToTensor()

def create_torchvision_dataset(
    dataset_class,
    dataset_dir,
    train,
    transform,
):
    processed_dir = (
        dataset_dir
        / dataset_class.__name__
        / "processed"
    )

    download_needed = not (
        (processed_dir / "training.pt").exists()
        and (processed_dir / "test.pt").exists()
    )

    return dataset_class(
        root=dataset_dir,
        train=train,
        download=download_needed,
        transform=transform,
    )

def create_medmnist_dataset(
    dataset_name,
    dataset_dir,
    train,
    transform,
):
    dataset_class = getattr(
        medmnist,
        INFO[dataset_name]["python_class"],
    )

    if train:
        train_dataset = dataset_class(
            split="train",
            root=dataset_dir,
            download=True,
            transform=transform,
        )

        validation_dataset = dataset_class(
            split="val",
            root=dataset_dir,
            download=True,
            transform=transform,
        )

        return ConcatDataset([
            train_dataset,
            validation_dataset,
        ])

    return dataset_class(
        split="test",
        root=dataset_dir,
        download=True,
        transform=transform,
    )


class UniformDequantizedDataset(Dataset):
    """Apply fixed uniform dequantization: (pixel + U[0, 1)) / 256."""

    def __init__(self, dataset, seed):
        self.dataset = dataset
        self.seed = seed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        image = sample[0]

        generator = torch.Generator()
        generator.manual_seed(self.seed + index)

        noise = torch.rand(
            image.shape,
            generator=generator,
            dtype=image.dtype,
        )

        # Prevent an exactly-zero floating-point draw.
        noise = noise.clamp_min(torch.finfo(image.dtype).eps)

        image = (image * 255.0 + noise) / 256.0

        return (image, *sample[1:])