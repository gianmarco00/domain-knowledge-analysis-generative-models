from pathlib import Path
from datetime import datetime
import random

import yaml
import torch

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
    timestamp = datetime.now().strftime("%d_%b_%H%M").lower()

    return f"{experiment_name}_lr_{learning_rate}_{timestamp}"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def get_repo_root():

    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if (parent / "pyproject.toml").exists():
            repo_root = parent
            return repo_root
 
    raise FileNotFoundError("Could not find repository root. Missing pyproject.toml.")
    


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

