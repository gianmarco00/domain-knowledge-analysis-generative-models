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


def create_transform(config):

    dataset_name = config["dataset"]["name"].lower()

    if dataset_name == "mnist":
        return transforms.ToTensor()
    
    raise ValueError(f"Unsupported transform for dataset: {dataset_name}")


def create_dataset(config, transform):
    dataset_name = config["dataset"]["name"].lower()
    dataset_dir = config["paths"]["dataset_dir"]
    expected_shape = tuple(config["dataset"]["shape"])

    if dataset_name == "mnist":
        dataset = datasets.MNIST(
            root=dataset_dir,
            train=config["dataset"]["train"],
            download=config["dataset"]["download"],
            transform=transform,
        )

        actual_shape = tuple(dataset[0][0].shape)

        if actual_shape != expected_shape:
            raise ValueError(
                f"Config file contains wrong data shape. "
                f"Expected {expected_shape}, got {actual_shape}."
            )

        return dataset

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def create_dataloaders(config):
    transform = create_transform(config)
    dataset = create_dataset(config, transform)

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
    run_name = build_run_name(config)

    return runs_dir / run_name

def create_loss(config):
    model_name = config["model"]["name"].lower()

    if model_name == "vae":
        return vae_loss

    raise ValueError(f"Unsupported loss for model: {model_name}")

