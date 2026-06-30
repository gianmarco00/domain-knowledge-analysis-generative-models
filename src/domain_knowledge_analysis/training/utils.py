from pathlib import Path
from datetime import datetime
import random

import yaml
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from domain_knowledge_analysis.models import Vae
from domain_knowledge_analysis.losses import vae_loss



def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_run_name(config):
    experiment_name = config["experiment"]["name"]
    learning_rate = config["training"]["learning_rate"]
    timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")

    return f"{experiment_name}_lr_{learning_rate}_{timestamp}"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def create_transform(dataset_name):

    if dataset_name in ["mnist", "fashionmnist", "fashion_mnist", "fmnist"]:
        return transforms.ToTensor()
    
    raise ValueError(f"Unsupported transform for dataset: {dataset_name}")

def get_repo_root():

    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if (parent / "pyproject.toml").exists():
            repo_root = parent
            return repo_root
 
    raise FileNotFoundError("Could not find repository root. Missing pyproject.toml.")
    

def create_dataset(config, dataset_name, train):
    dataset_name = dataset_name.lower()

    transform = create_transform(dataset_name=dataset_name)

    dataset_dir = config["paths"]["dataset_dir"]
    repo_root = get_repo_root()
    dataset_dir = Path(repo_root / dataset_dir)

    expected_shape = tuple(config["dataset"]["shape"])

    if dataset_name == "mnist":
        dataset_class = datasets.MNIST
        processed_folder = "MNIST"

    elif dataset_name in ["fashion_mnist", "fmnist"]:
        dataset_class = datasets.FashionMNIST
        processed_folder = "FashionMNIST"

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    processed_dir = dataset_dir / processed_folder / "processed"

    download_needed = not (
        (processed_dir / "training.pt").exists()
        and (processed_dir / "test.pt").exists()
    )

    dataset = dataset_class(
        root=dataset_dir,
        train=train,
        download=download_needed,
        transform=transform,
    )

    actual_shape = tuple(dataset[0][0].shape)

    if actual_shape != expected_shape:
        raise ValueError(
            f"Config file contains wrong data shape. "
            f"Expected {expected_shape}, got {actual_shape}."
        )

    return dataset


def create_training_dataloaders(config):

    dataset_name = config["dataset"]["name"].lower()
    dataset = create_dataset(
        config, 
        dataset_name, 
        train=True
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
    out_distribution_dataset_name,
):

    in_distribution_dataset = create_dataset(
        config=config,
        dataset_name=in_distribution_dataset_name,
        train=False,
    )

    out_distribution_dataset = create_dataset(
        config=config,
        dataset_name=out_distribution_dataset_name,
        train=False,
    )

    dataloader_config = config["scoring"]["dataloader"]

    in_distribution_dataloader = DataLoader(
        in_distribution_dataset,
        batch_size=dataloader_config["batch_size"],
        shuffle=False,
        num_workers=dataloader_config["num_workers"],
    )

    out_distribution_dataloader = DataLoader(
        out_distribution_dataset,
        batch_size=dataloader_config["batch_size"],
        shuffle=False,
        num_workers=dataloader_config["num_workers"],
    )

    return in_distribution_dataloader, out_distribution_dataloader

def create_calibration_dataloader(config, dataset_name):
    dataset = create_dataset(
        config=config,
        dataset_name=dataset_name,
        train=True,
    )

    dataloader_config = config["scoring"]["dataloader"]

    dataloader = DataLoader(
        dataset,
        batch_size=dataloader_config["batch_size"],
        shuffle=False,
        num_workers=dataloader_config["num_workers"],
    )

    return dataloader


def create_model(config):
    model_name = config["model"]["name"].lower()
    image_shape = tuple(config["dataset"]["shape"])

    if model_name == "vae":
        encoder_params = config["model"]["encoder"]
        return Vae(image_shape=image_shape, encoder_params=encoder_params)

    raise ValueError(f"Unsupported model: {model_name}")


def create_optimizer(config, model):
    optimizer_name = config["optimizer"]["name"].lower()
    learning_rate = config["training"]["learning_rate"]

    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def create_log_dir(config):
    runs_dir = Path(config["paths"]["runs_dir"])
    repo_root = get_repo_root()
    runs_dir = repo_root / runs_dir

    run_name = build_run_name(config)
    log_dir = runs_dir / run_name

    if log_dir.exists() and not log_dir.is_dir():
        raise NotADirectoryError(f"Log path exists but is not a directory: {log_dir}")

    log_dir.mkdir(parents=True, exist_ok=True)

    return log_dir

def create_loss(config):
    model_name = config["model"]["name"].lower()

    if model_name == "vae":
        return vae_loss

    raise ValueError(f"Unsupported loss for model: {model_name}")

