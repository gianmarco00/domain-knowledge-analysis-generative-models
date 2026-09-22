from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset, Dataset, TensorDataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

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


TRANSFORMED_DATASET_CACHE_VERSION = 1


class FixedRotation:
    """Rotate a tensor image by a fixed angle without changing its shape."""

    def __init__(self, degrees):
        self.degrees = float(degrees)

    def __call__(self, image):
        return TF.rotate(
            image,
            angle=self.degrees,
            interpolation=InterpolationMode.BILINEAR,
            expand=False,
            fill=0,
        )


def create_dataset_transformation(transformation_type, intensity):
    transformation_type = transformation_type.lower()

    if transformation_type == "rotation":
        return FixedRotation(degrees=intensity)

    raise ValueError(
        f"Unsupported dataset transformation: {transformation_type}"
    )
    

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


def create_training_dataloaders(config, transformation_config=None):

    dataset_name = config["dataset"]["name"].lower()

    if transformation_config is None:
        train_dataset, validation_dataset = create_train_validation_datasets(
            config=config,
            dataset_name=dataset_name,
        )
    else:
        train_dataset, validation_dataset = create_transformed_datasets(
            config=config,
            dataset_name=dataset_name,
            transformation_config=transformation_config,
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


def create_testing_dataloaders(config, transformation_config=None):
    dataset_name = config["dataset"]["name"].lower()

    if transformation_config is None:
        test_dataset = create_dataset(
            config=config,
            dataset_name=dataset_name,
            train=False,
        )
    else:
        test_dataset = create_transformed_datasets(
            config=config,
            dataset_name=dataset_name,
            transformation_config=transformation_config,
            split="test",
        )

    dataloader_config = config["scoring"]["dataloader"]
    return DataLoader(
        test_dataset,
        batch_size=dataloader_config["batch_size"],
        shuffle=False,
        num_workers=dataloader_config["num_workers"],
    )


def materialize_transformed_dataset(dataset, transformation):
    images = []
    labels = []

    with torch.no_grad():
        for image, label in dataset:
            transformed_image = transformation(image)
            images.append(transformed_image.detach().cpu())
            labels.append(torch.as_tensor(label).detach().cpu())

    if not images:
        raise ValueError("Cannot materialize an empty dataset.")

    return TensorDataset(
        torch.stack(images),
        torch.stack(labels),
    )


def create_transformed_datasets(
    config,
    dataset_name,
    transformation_config,
    split="train_validation",
):
    dataset_name = dataset_name.lower()
    dataset_name = DATASET_ALIASES.get(
        dataset_name,
        dataset_name,
    )

    transformation_type = transformation_config["type"].lower()
    intensity = float(transformation_config["intensity"])

    dataset_dir = get_repo_root() / Path(config["paths"]["dataset_dir"])
    train_split = float(config["dataset"]["train_split"])
    seed = int(config["seed"])

    cache_dir = (
        dataset_dir
        / "transformed"
        / dataset_name
        / f"{transformation_type}_intensity_{intensity:g}"
        / f"version_{TRANSFORMED_DATASET_CACHE_VERSION}"
        / f"seed_{seed}_split_{train_split:g}"
    )

    train_path = cache_dir / "train.pt"
    validation_path = cache_dir / "validation.pt"
    test_path = cache_dir / "test.pt"

    metadata = {
        "cache_version": TRANSFORMED_DATASET_CACHE_VERSION,
        "dataset_name": dataset_name,
        "transformation_type": transformation_type,
        "intensity": intensity,
        "interpolation": "bilinear",
        "fill": 0,
        "expand": False,
        "seed": seed,
        "train_split": train_split,
    }

    train_metadata = {**metadata, "split": "train"}
    validation_metadata = {**metadata, "split": "validation"}
    test_metadata = {**metadata, "split": "test"}
    transformation = create_dataset_transformation(
        transformation_type=transformation_type,
        intensity=intensity,
    )

    if split == "test":
        if test_path.exists():
            return load_tensor_dataset(test_path, test_metadata)

        test_dataset = create_dataset(
            config=config,
            dataset_name=dataset_name,
            train=False,
        )
        transformed_test_dataset = materialize_transformed_dataset(
            dataset=test_dataset,
            transformation=transformation,
        )
        save_tensor_dataset(
            dataset=transformed_test_dataset,
            path=test_path,
            metadata=test_metadata,
        )
        return transformed_test_dataset

    if split != "train_validation":
        raise ValueError(f"Unsupported transformed dataset split: {split}")

    if train_path.exists() and validation_path.exists():
        return (
            load_tensor_dataset(train_path, train_metadata),
            load_tensor_dataset(validation_path, validation_metadata),
        )

    train_dataset, validation_dataset = create_train_validation_datasets(
        config=config,
        dataset_name=dataset_name,
    )

    transformed_train_dataset = materialize_transformed_dataset(
        dataset=train_dataset,
        transformation=transformation,
    )
    transformed_validation_dataset = materialize_transformed_dataset(
        dataset=validation_dataset,
        transformation=transformation,
    )

    save_tensor_dataset(
        dataset=transformed_train_dataset,
        path=train_path,
        metadata=train_metadata,
    )
    save_tensor_dataset(
        dataset=transformed_validation_dataset,
        path=validation_path,
        metadata=validation_metadata,
    )

    return transformed_train_dataset, transformed_validation_dataset


# Preserve direct callers of the original train/validation helper.
create_transformed_train_validation_datasets = create_transformed_datasets


def save_tensor_dataset(dataset, path, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")

    torch.save(
        {
            "images": dataset.tensors[0],
            "labels": dataset.tensors[1],
            "metadata": metadata,
        },
        temporary_path,
    )
    temporary_path.replace(path)


def load_tensor_dataset(path, expected_metadata):
    cached_dataset = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if cached_dataset["metadata"] != expected_metadata:
        raise ValueError(
            f"Cached transformed dataset metadata does not match: {path}"
        )

    return TensorDataset(
        cached_dataset["images"],
        cached_dataset["labels"],
    )


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
