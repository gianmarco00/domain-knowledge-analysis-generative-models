from pathlib import Path
from datetime import datetime
from functools import partial

import random

import yaml
import torch

from domain_knowledge_analysis.models import Vae
from domain_knowledge_analysis.losses import vae_loss
from domain_knowledge_analysis.math import continuous_bernoulli_log_prob_from_logits, bernoulli_log_prob_from_logits



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
        decoder_distribution_name = config["loss"]["log_prob_function"].lower()
        return Vae(image_shape=image_shape, encoder_params=encoder_params, decoder_distribution_name=decoder_distribution_name)

    raise ValueError(f"Unsupported model: {model_name}")


def create_optimizer(config, model):
    optimizer_name = config["optimizer"]["name"].lower()
    learning_rate = config["training"]["learning_rate"]

    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def create_lr_scheduler(config, optimizer):
    lr_scheduler_name = config["lr_scheduler"]["name"].lower()
    lr_scheduler_threshold = config["lr_scheduler"]["treshold"]
    lr_scheduler_threshold_mode = config["lr_scheduler"]["mode"]

    if lr_scheduler_name is None:
        return None

    if lr_scheduler_name == "reduce_lr_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=lr_scheduler_threshold, threshold_mode=lr_scheduler_threshold_mode)

    raise ValueError(f"Unsupported scheduler {lr_scheduler_name}")


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

def create_random_generator(seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator

def sample_random_latents(n_images, latent_dim, generator):
    return torch.randn(n_images, latent_dim, generator=generator, device="cpu")

def create_vae_decoder_distribution(config):  
    decoder_distribution_name = config["loss"]["log_prob_function"].lower()

    if decoder_distribution_name == "continuous_bernoulli":
        return continuous_bernoulli_log_prob_from_logits
    elif decoder_distribution_name == "bernoulli":
        return bernoulli_log_prob_from_logits
    else:
        raise ValueError(f"Unsupported decoder distribution: {decoder_distribution_name}")

def create_loss(config):
    model_name = config["model"]["name"].lower()

    if model_name == "vae":
        log_prob_function = create_vae_decoder_distribution(config)
        return partial(vae_loss, log_prob_function=log_prob_function)
    
    raise ValueError(f"Unsupported loss for model: {model_name}")

