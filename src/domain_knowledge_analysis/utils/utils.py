from pathlib import Path
from datetime import datetime
from functools import partial

import math
import random

import yaml
import torch

from domain_knowledge_analysis.models import Vae
from domain_knowledge_analysis.losses import VAELoss
from domain_knowledge_analysis.math import continuous_bernoulli_log_prob_from_logits, bernoulli_log_prob_from_logits


INPUT_SCALE_FLOOR = 1e-6
RMS_RELATIVE_TOLERANCE = 0.01
RMS_ABSOLUTE_TOLERANCE = 1e-8
MAXIMUM_RELATIVE_TOLERANCE = 1e-6
CALIBRATION_SEARCH_STEPS = 64
MAXIMUM_REFINEMENT_STEPS = 4
BOUNDED_BLEND_TRANSFORMATIONS = {"horizontal_flip", "vertical_flip", "binarization", "pixelation", "horizontal_motion_blur", "vertical_motion_blur"}


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

    if config["model"]["name"] == "vae":
        latent_dims = config["model"]["encoder"]["latent_dim"]
        beta = config["loss"]["beta"]
        loss = "CB" if config["loss"]["log_prob_function"] == "continuous_bernoulli" else "B"

        if "adapt" in experiment_name:
            rank = config["lora"]["rank"]
            alpha = config["lora"]["alpha"]
            finetuning_dataset = config["lora"]["transformed_dataset"]["name"]

            if config["lora"]["regularizer"] is not None:
                from .dataset_utils import TRANSFORMATION_IMPLEMENTATION_VERSION

                regularizer_config = config["lora"]["regularizer"]
                eta = regularizer_config["eta"]
                num_anchor_points = regularizer_config["num_anchor_points"]
                k = len(regularizer_config["transformations"])
                target_input_rms = regularizer_config.get("target_input_rms")
                transformation_strength = regularizer_config.get("transformation_strength")
                rms_name = f"_RMS_{float(target_input_rms):g}" if target_input_rms is not None else ""
                strength_name = f"_S_{float(transformation_strength):g}" if transformation_strength is not None else ""
                return f"REG_{experiment_name}_finetuned_on_{finetuning_dataset}_{timestamp}_eta_{eta}_K_{k}_AP_{num_anchor_points}{rms_name}{strength_name}_TV_{TRANSFORMATION_IMPLEMENTATION_VERSION}_rank_{rank}_alpha_{alpha}"
            
            return f"{experiment_name}_finetuned_on_{finetuning_dataset}_{timestamp}_rank_{rank}_alpha_{alpha}"
        
        return f"{experiment_name}_lr_{learning_rate}_{timestamp}_{loss}_beta_{beta}_LD_{latent_dims}"

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

        decoder_config = config["model"].get("decoder", {})
        symmetric_decoder = decoder_config.get("symmetrical", True)
        
        return Vae(image_shape=image_shape, encoder_params=encoder_params, decoder_distribution_name=decoder_distribution_name, symmetric_decoder=symmetric_decoder)

    raise ValueError(f"Unsupported model: {model_name}")


def create_optimizer(config, parameters):
    optimizer_name = config["optimizer"]["name"].lower()
    learning_rate = config["training"]["learning_rate"]

    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate)

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def create_lr_scheduler(config, optimizer):
    lr_scheduler_name = config["lr_scheduler"]["name"]
    lr_scheduler_threshold = config["lr_scheduler"]["threshold"]
    lr_scheduler_threshold_mode = config["lr_scheduler"]["threshold_mode"]
    lr_scheduler_min_lr = config["lr_scheduler"]["min_lr"]
    lr_scheduler_factor = config["lr_scheduler"]["factor"]
    lr_scheduler_patience = config["lr_scheduler"]["patience"]
    lr_scheduler_cooldown = config["lr_scheduler"]["cooldown"]
    lr_scheduler_mode = config["lr_scheduler"]["mode"]
    lr_scheduler_start_epoch = config["lr_scheduler"]["start_epoch"]

    if lr_scheduler_name is None:
        return None

    if lr_scheduler_name == "reduce_lr_on_plateau":
        return (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=lr_scheduler_threshold, threshold_mode=lr_scheduler_threshold_mode, min_lr=lr_scheduler_min_lr, factor=lr_scheduler_factor, patience=lr_scheduler_patience, cooldown=lr_scheduler_cooldown, mode=lr_scheduler_mode), lr_scheduler_start_epoch)

    raise ValueError(f"Unsupported scheduler {lr_scheduler_name}")


def create_log_dir(config):
    
    repo_root = get_repo_root()

    if config["paths"].get("run_dir") is not None:
        log_dir = repo_root / Path(config["paths"]["run_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    experiment_name = config["experiment"]["name"]
    pretrained_model_path = repo_root / Path(config["pretrained_model"]) if config["pretrained_model"] else None
    lora_pretrained_model_path = repo_root / Path(config["lora"]["pretrained_model"]) if config["lora"] and config["lora"]["pretrained_model"] else None

    if pretrained_model_path is not None and config["lora"] is None:
        return pretrained_model_path.parent.parent
    
    if lora_pretrained_model_path is not None and config["lora"] is not None:
        return lora_pretrained_model_path.parent.parent

    runs_dir = config["paths"]["runs_dir"]
    runs_dir = repo_root / runs_dir

    if config["lora"] is not None:
        
        if "adapt" not in experiment_name:
            raise ValueError("LoRA configuration is present but the experiment name does not contain 'adapt'.")
        
        runs_dir = pretrained_model_path.parent.parent

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

def measure_transformation(transformation, anchor_points, batch_size, return_per_anchor=False):
    square_sum = 0.0
    output_variation_sum = 0.0
    num_values = 0
    per_anchor_rms = []

    with torch.no_grad():
        for start in range(0, len(anchor_points), batch_size):
            anchors = anchor_points[start:start + batch_size]
            transformed = transformation(anchors)
            delta = transformed - anchors
            square_sum += delta.square().sum().item()
            output_mean = transformed.flatten(start_dim=1).mean(dim=1).view(-1, 1, 1, 1)
            output_variation_sum += (transformed - output_mean).square().sum().item()
            num_values += delta.numel()

            if return_per_anchor:
                per_anchor_rms.append(delta.flatten(start_dim=1).square().mean(dim=1).sqrt().cpu())

    input_rms = math.sqrt(square_sum / num_values)
    output_variation = math.sqrt(output_variation_sum / num_values)
    per_anchor_rms = torch.cat(per_anchor_rms) if return_per_anchor else None
    return input_rms, output_variation, per_anchor_rms


def measure_input_rms(transformation, anchor_points, batch_size, return_per_anchor=False):
    input_rms, _, per_anchor_rms = measure_transformation(transformation, anchor_points, batch_size, return_per_anchor)
    return (input_rms, per_anchor_rms) if return_per_anchor else input_rms


def transformation_path(transformation_type, configured_intensity, strength):
    factor_transformations = {"isotropic_scale", "aspect_ratio_deformation", "contrast", "gamma_correction"}

    if transformation_type in factor_transformations:
        direction = 1 if configured_intensity > 1 else -1
        return math.exp(direction * strength)

    if transformation_type in BOUNDED_BLEND_TRANSFORMATIONS:
        return strength

    if transformation_type == "gaussian_blur":
        return strength

    direction = 1 if configured_intensity > 0 else -1
    return direction * strength


def parameter_strength(transformation_type, intensity):
    if not math.isfinite(intensity):
        raise ValueError(f"Transformation {transformation_type} has a nonfinite intensity.")

    if transformation_type in {"isotropic_scale", "aspect_ratio_deformation", "contrast", "gamma_correction"}:
        if intensity <= 0 or intensity == 1:
            raise ValueError(f"Transformation {transformation_type} needs a positive, non-identity factor to define its direction.")
        return abs(math.log(intensity))

    if transformation_type in BOUNDED_BLEND_TRANSFORMATIONS:
        if not 0 < intensity <= 1:
            raise ValueError(f"Transformation {transformation_type} needs an intensity in (0, 1].")
        return intensity

    if transformation_type == "gaussian_blur":
        if intensity < 0:
            raise ValueError("Transformation gaussian_blur needs a nonnegative intensity.")
        return intensity

    if intensity == 0:
        raise ValueError(f"Transformation {transformation_type} needs a nonzero intensity to define its direction.")

    return abs(intensity)


def transformation_strength_limit(transformation_type, anchor_points):
    """End the unique path before periodic repetition or spatial collapse.

    Geometric paths retain at least two sampled pixels, blur uses the largest non-repeating reflected kernel, and contrast retains one 8-bit intensity level.
    """
    height, width = anchor_points.shape[-2:]
    minimum_size = min(height, width)

    if transformation_type == "rotation":
        return 180.0
    if transformation_type == "horizontal_translation":
        return float(width - 1)
    if transformation_type == "vertical_translation":
        return float(height - 1)
    if transformation_type == "diagonal_shear":
        return 2 * (width - 1) / max(height - 1, 1)
    if transformation_type == "vertical_shear":
        return 2 * (height - 1) / max(width - 1, 1)
    if transformation_type in {"isotropic_scale", "aspect_ratio_deformation"}:
        return math.log(max(minimum_size / 2, 1.0))
    if transformation_type in {"contrast", "gamma_correction"}:
        return math.log(255.0)
    if transformation_type == "stroke_thickness":
        return 1.0
    if transformation_type == "gaussian_blur":
        return (minimum_size - 1) / 3
    if transformation_type in BOUNDED_BLEND_TRANSFORMATIONS:
        return 1.0
    if transformation_type == "horizontal_wave":
        return float(width - 1)
    if transformation_type == "vertical_wave":
        return float(height - 1)
    if transformation_type == "perspective":
        return 1 - 2 / minimum_size

    raise ValueError(f"Unsupported dataset transformation: {transformation_type}")


def per_anchor_rms_summary(per_anchor_rms):
    quantiles = torch.quantile(per_anchor_rms, torch.tensor([0.25, 0.5, 0.75]))
    return {
        "min": per_anchor_rms.min().item(),
        "q25": quantiles[0].item(),
        "median": quantiles[1].item(),
        "q75": quantiles[2].item(),
        "max": per_anchor_rms.max().item(),
        "mean": per_anchor_rms.mean().item(),
        "std": per_anchor_rms.std(unbiased=False).item(),
    }


def calibrate_transformation(transformation_config, anchor_points, target_input_rms, batch_size, search_limit=None):
    from .dataset_utils import create_dataset_transformation

    transformation_type = transformation_config["name"].lower()
    configured_intensity = float(transformation_config["intensity"])
    direction_seed = int(transformation_config.get("direction_seed", 0))
    initial_strength = parameter_strength(transformation_type, configured_intensity)

    if transformation_type.startswith("random_direction_"):
        resolved_intensity = target_input_rms if configured_intensity > 0 else -target_input_rms
        transformation = create_dataset_transformation(transformation_type, resolved_intensity, direction_seed)
        achieved_rms, per_anchor_rms = measure_input_rms(transformation, anchor_points, batch_size, return_per_anchor=True)
        search_limit = target_input_rms
    else:
        search_limit = search_limit or transformation_strength_limit(transformation_type, anchor_points)
        search_strengths = [search_limit * index / CALIBRATION_SEARCH_STEPS for index in range(CALIBRATION_SEARCH_STEPS + 1)]

        if 0 < initial_strength < search_limit:
            search_strengths.append(initial_strength)

        search_strengths = sorted(set(search_strengths))
        previous_strength, previous_rms = 0.0, 0.0
        measured_rms = [previous_rms]
        bracket = None

        for strength in search_strengths[1:]:
            intensity = transformation_path(transformation_type, configured_intensity, strength)
            transformation = create_dataset_transformation(transformation_type, intensity, direction_seed)
            input_rms = measure_input_rms(transformation, anchor_points, batch_size)

            if not math.isfinite(input_rms):
                raise ValueError(f"Transformation {transformation_type} produced a nonfinite RMS displacement.")

            measured_rms.append(input_rms)

            if (previous_rms - target_input_rms) * (input_rms - target_input_rms) <= 0:
                bracket = [previous_strength, previous_rms, strength, input_rms]
                break

            previous_strength, previous_rms = strength, input_rms

        if bracket is None:
            reachable_min, reachable_max = min(measured_rms), max(measured_rms)
            raise ValueError(f"Transformation {transformation_type} cannot reach target_input_rms={target_input_rms:g}; measured reachable RMS range is [{reachable_min:g}, {reachable_max:g}].")

        lower_strength, lower_rms, upper_strength, upper_rms = bracket
        best_strength, best_rms = min([(lower_strength, lower_rms), (upper_strength, upper_rms)], key=lambda value: abs(value[1] - target_input_rms))

        for _ in range(60):
            middle_strength = (lower_strength + upper_strength) / 2

            if middle_strength == lower_strength or middle_strength == upper_strength:
                break

            intensity = transformation_path(transformation_type, configured_intensity, middle_strength)
            transformation = create_dataset_transformation(transformation_type, intensity, direction_seed)
            middle_rms = measure_input_rms(transformation, anchor_points, batch_size)

            if abs(middle_rms - target_input_rms) < abs(best_rms - target_input_rms):
                best_strength, best_rms = middle_strength, middle_rms

            if math.isclose(best_rms, target_input_rms, rel_tol=RMS_RELATIVE_TOLERANCE, abs_tol=RMS_ABSOLUTE_TOLERANCE):
                break

            if (lower_rms - target_input_rms) * (middle_rms - target_input_rms) <= 0:
                upper_strength, upper_rms = middle_strength, middle_rms
            else:
                lower_strength, lower_rms = middle_strength, middle_rms

        resolved_intensity = transformation_path(transformation_type, configured_intensity, best_strength)
        transformation = create_dataset_transformation(transformation_type, resolved_intensity, direction_seed)
        achieved_rms, per_anchor_rms = measure_input_rms(transformation, anchor_points, batch_size, return_per_anchor=True)

    if not math.isfinite(achieved_rms) or achieved_rms <= INPUT_SCALE_FLOOR:
        raise ValueError(f"Transformation {transformation_type} produced a zero, nonfinite or numerically unresolved perturbation.")

    if not math.isclose(achieved_rms, target_input_rms, rel_tol=RMS_RELATIVE_TOLERANCE, abs_tol=RMS_ABSOLUTE_TOLERANCE):
        raise ValueError(f"Transformation {transformation_type} resolved to RMS {achieved_rms:g}, not target_input_rms={target_input_rms:g}.")

    metadata = {
        "name": transformation_type,
        "configured_intensity": configured_intensity,
        "resolved_intensity": transformation.intensity,
        "resolved_strength": transformation.finite_difference_step,
        "target_input_rms": target_input_rms,
        "achieved_input_rms": achieved_rms,
        "direction": 1 if transformation.intensity > (1 if transformation_type in {"isotropic_scale", "aspect_ratio_deformation", "contrast", "gamma_correction"} else 0) else -1,
        "search_strength_limit": search_limit,
        "per_anchor_input_rms": per_anchor_rms_summary(per_anchor_rms),
    }

    if transformation_type.startswith("random_direction_"):
        metadata["direction_seed"] = direction_seed
        metadata["effective_direction_seed"] = direction_seed + transformation.direction_index

    return transformation, metadata


def find_maximum_strength(transformation_config, anchor_points, batch_size):
    from .dataset_utils import create_dataset_transformation

    transformation_type = transformation_config["name"].lower()
    configured_intensity = float(transformation_config["intensity"])
    direction_seed = int(transformation_config.get("direction_seed", 0))

    if transformation_type.startswith("random_direction_"):
        raise ValueError(f"Transformation {transformation_type} has no finite maximum because its additive path is unbounded; use an explicit intensity instead.")

    parameter_strength(transformation_type, configured_intensity)
    search_limit = transformation_strength_limit(transformation_type, anchor_points)
    measurements = {}

    def evaluate(strength):
        if strength not in measurements:
            intensity = transformation_path(transformation_type, configured_intensity, strength)
            transformation = create_dataset_transformation(transformation_type, intensity, direction_seed)
            input_rms, output_variation, _ = measure_transformation(transformation, anchor_points, batch_size)

            if not math.isfinite(input_rms) or not math.isfinite(output_variation):
                raise ValueError(f"Transformation {transformation_type} produced a nonfinite calibration measurement.")

            measurements[strength] = input_rms, output_variation

        return measurements[strength]

    if transformation_type in BOUNDED_BLEND_TRANSFORMATIONS:
        maximum_input_rms, output_variation = evaluate(search_limit)

        if output_variation <= INPUT_SCALE_FLOOR:
            raise ValueError(f"Transformation {transformation_type} has a numerically degenerate endpoint.")

        return search_limit, maximum_input_rms

    search_strengths = [search_limit * index / CALIBRATION_SEARCH_STEPS for index in range(CALIBRATION_SEARCH_STEPS + 1)]

    for strength in search_strengths:
        evaluate(strength)

    for _ in range(MAXIMUM_REFINEMENT_STEPS):
        valid_strengths = [strength for strength in search_strengths if strength > 0 and measurements[strength][1] > INPUT_SCALE_FLOOR]

        if not valid_strengths:
            raise ValueError(f"Transformation {transformation_type} has no numerically non-degenerate result in its valid range.")

        best_rms = max(measurements[strength][0] for strength in valid_strengths)
        maximum_tolerance = max(RMS_ABSOLUTE_TOLERANCE, best_rms * MAXIMUM_RELATIVE_TOLERANCE)
        best_strength = min(strength for strength in valid_strengths if measurements[strength][0] >= best_rms - maximum_tolerance)
        best_index = search_strengths.index(best_strength)
        lower_strength = search_strengths[max(0, best_index - 1)]
        upper_strength = search_strengths[min(len(search_strengths) - 1, best_index + 1)]
        search_strengths = [lower_strength + (upper_strength - lower_strength) * index / 8 for index in range(9)]

        for strength in search_strengths:
            evaluate(strength)

    valid_measurements = [(strength, input_rms) for strength, (input_rms, output_variation) in measurements.items() if strength > 0 and output_variation > INPUT_SCALE_FLOOR]
    maximum_input_rms = max(input_rms for _, input_rms in valid_measurements)
    maximum_tolerance = max(RMS_ABSOLUTE_TOLERANCE, maximum_input_rms * MAXIMUM_RELATIVE_TOLERANCE)
    maximum_strength = min(strength for strength, input_rms in valid_measurements if input_rms >= maximum_input_rms - maximum_tolerance)
    return maximum_strength, measurements[maximum_strength][0]


def calibrate_relative_transformation(transformation_config, anchor_points, relative_strength, batch_size):
    maximum_strength, maximum_input_rms = find_maximum_strength(transformation_config, anchor_points, batch_size)
    target_input_rms = relative_strength * maximum_input_rms
    transformation, metadata = calibrate_transformation(transformation_config, anchor_points, target_input_rms, batch_size, maximum_strength)
    metadata["transformation_strength"] = relative_strength
    metadata["maximum_strength"] = maximum_strength
    metadata["maximum_intensity"] = transformation_path(metadata["name"], metadata["configured_intensity"], maximum_strength)
    metadata["maximum_input_rms"] = maximum_input_rms
    metadata["relative_input_rms"] = metadata["achieved_input_rms"] / maximum_input_rms
    return transformation, metadata


def create_regularizer_transformations(config, device=None, anchor_points=None):
    from .dataset_utils import TRANSFORMATION_IMPLEMENTATION_VERSION, create_dataset_transformation

    regularizer_config = config["lora"]["regularizer"]
    transformation_configs = regularizer_config["transformations"]
    target_input_rms = regularizer_config.get("target_input_rms")
    relative_strength = regularizer_config.get("transformation_strength")
    transformation_functions = []
    intensities = []
    regularizer_config["transformation_implementation_version"] = TRANSFORMATION_IMPLEMENTATION_VERSION

    if target_input_rms is not None and relative_strength is not None:
        raise ValueError("Use either target_input_rms or transformation_strength, not both.")

    if target_input_rms is not None or relative_strength is not None:
        calibration_mode = "target_input_rms" if target_input_rms is not None else "transformation_strength"

        if anchor_points is None or len(anchor_points) == 0:
            raise ValueError(f"Anchor points are required to calibrate {calibration_mode}.")

        calibration_batch_size = int(regularizer_config.get("calibration_batch_size", 100))

        if calibration_batch_size <= 0:
            raise ValueError("calibration_batch_size must be positive.")

        if target_input_rms is not None:
            target_input_rms = float(target_input_rms)
            regularizer_config["target_input_rms"] = target_input_rms

            if not math.isfinite(target_input_rms) or target_input_rms <= INPUT_SCALE_FLOOR:
                raise ValueError(f"target_input_rms must be finite and greater than {INPUT_SCALE_FLOOR:g}.")
        else:
            relative_strength = float(relative_strength)
            regularizer_config["transformation_strength"] = relative_strength

            if not math.isfinite(relative_strength) or not 0 < relative_strength <= 1:
                raise ValueError("transformation_strength must be finite and in (0, 1].")

        resolved_transformations = []

        for transformation_config in transformation_configs:
            if target_input_rms is not None:
                transformation, metadata = calibrate_transformation(transformation_config, anchor_points, target_input_rms, calibration_batch_size)
            else:
                transformation, metadata = calibrate_relative_transformation(transformation_config, anchor_points, relative_strength, calibration_batch_size)

            transformation_functions.append(transformation)
            intensities.append(transformation.finite_difference_step)
            resolved_transformations.append(metadata)

        regularizer_config["calibration"] = {
            calibration_mode: target_input_rms if target_input_rms is not None else relative_strength,
            "relative_tolerance": RMS_RELATIVE_TOLERANCE,
            "absolute_tolerance": RMS_ABSOLUTE_TOLERANCE,
            "maximum_relative_tolerance": MAXIMUM_RELATIVE_TOLERANCE,
            "maximum_definition": "Earliest maximum anchor RMS on the non-redundant, numerically non-degenerate transformation path.",
            "transformation_implementation_version": TRANSFORMATION_IMPLEMENTATION_VERSION,
            "transformations": resolved_transformations,
        }
    else:
        regularizer_config.pop("calibration", None)

        for transformation_config in transformation_configs:
            transformation_type = transformation_config["name"]
            intensity = float(transformation_config["intensity"])
            direction_seed = int(transformation_config.get("direction_seed", 0))
            transformation = create_dataset_transformation(transformation_type, intensity, direction_seed)

            if not math.isfinite(transformation.finite_difference_step) or transformation.finite_difference_step == 0:
                raise ValueError("A regularizer transformation must differ from the identity transformation.")

            transformation_functions.append(transformation)
            intensities.append(transformation.finite_difference_step)

    if not transformation_functions:
        raise ValueError("At least one regularizer transformation is required.")

    return transformation_functions, torch.tensor(intensities, dtype=torch.float32, device=device)

def select_anchor_points(config, dataset_name, device=None):
    from .dataset_utils import create_train_validation_datasets

    num_anchor_points = config["lora"]["regularizer"]["num_anchor_points"]
    num_anchor_points = int(num_anchor_points)

    if num_anchor_points <= 0:
        raise ValueError("The number of anchor points must be positive.")

    train_dataset, _ = create_train_validation_datasets(config, dataset_name)
    indices_by_class = {}

    for index in range(len(train_dataset)):
        _, label = train_dataset[index]
        label = torch.as_tensor(label)

        if label.numel() != 1:
            raise ValueError("Anchor-point selection requires one class label per image.")

        class_label = int(label.item())
        indices_by_class.setdefault(class_label, []).append(index)

    if not indices_by_class:
        raise ValueError("Cannot select anchor points from an empty training dataset.")

    num_classes = len(indices_by_class)

    if num_anchor_points % num_classes != 0:
        raise ValueError(f"num_anchor_points ({num_anchor_points}) must be divisible by the number of classes ({num_classes}).")

    points_per_class = num_anchor_points // num_classes
    generator = create_random_generator(config["seed"])
    selected_indices = []

    for class_label in sorted(indices_by_class):
        class_indices = indices_by_class[class_label]

        if len(class_indices) < points_per_class:
            raise ValueError(f"Class {class_label} has only {len(class_indices)} samples, but {points_per_class} anchor points were requested.")

        selected_positions = torch.randperm(len(class_indices), generator=generator)[:points_per_class]
        selected_indices.extend(class_indices[position] for position in selected_positions.tolist())

    anchor_points = torch.stack([train_dataset[index][0] for index in selected_indices])
    return anchor_points.to(device) if device is not None else anchor_points

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
        beta = config["loss"]["beta"]
        log_prob_function = create_vae_decoder_distribution(config)
        vae_loss = VAELoss(log_prob_function=log_prob_function, beta=beta)
        return vae_loss
    
    raise ValueError(f"Unsupported loss for model: {model_name}")
