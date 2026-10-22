from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from statistics import fmean

import csv
import yaml


def transformation_name(transformation_families):
    return "+".join(family["name"] for family in transformation_families) if transformation_families else "none"


def select_named_transformations(ablation_config, transformation_names):
    transformations_by_name = {family["name"]: family for family in ablation_config["semantic_transformations"]}
    missing_names = [name for name in transformation_names if name not in transformations_by_name]

    if missing_names:
        raise ValueError(f"Unknown semantic transformations: {', '.join(missing_names)}.")

    return [deepcopy(transformations_by_name[name]) for name in transformation_names]


def select_transformation_prefix(transformation_families, m):
    m = int(m)

    if not 0 < m <= len(transformation_families):
        raise ValueError(f"M must be between 1 and {len(transformation_families)}, received {m}.")

    return deepcopy(transformation_families[:m])


def create_training_config(base_config, seed, eta, transformation_families, transformation_strength=None, regularizer_mode=None):
    from domain_knowledge_analysis.utils.dataset_utils import TRANSFORMATION_IMPLEMENTATION_VERSION

    config = deepcopy(base_config)
    config["seed"] = int(seed)
    config["lora"]["pretrained_model"] = None

    if float(eta) == 0 or (not transformation_families and regularizer_mode != "source_loss"):
        config["lora"]["regularizer"] = None
        return config

    regularizer_config = deepcopy(base_config["lora"]["regularizer"])
    regularizer_config["eta"] = float(eta)

    if regularizer_mode == "source_loss":
        regularizer_config["mode"] = regularizer_mode
        regularizer_config["transformations"] = []
        regularizer_config.pop("transformation_strength", None)
        regularizer_config.pop("target_input_rms", None)
        regularizer_config.pop("calibration", None)
        regularizer_config.pop("transformation_implementation_version", None)
        config["lora"]["regularizer"] = regularizer_config
        return config

    regularizer_config["transformations"] = deepcopy(transformation_families)
    regularizer_config["transformation_implementation_version"] = TRANSFORMATION_IMPLEMENTATION_VERSION

    if transformation_strength is not None:
        regularizer_config["transformation_strength"] = float(transformation_strength)
        regularizer_config.pop("target_input_rms", None)

    if regularizer_mode is not None:
        regularizer_config["mode"] = regularizer_mode

    config["lora"]["regularizer"] = regularizer_config
    return config


def training_config_id(config):
    training_keys = ["seed", "pretrained_model", "dataset", "dataloader", "training", "loss", "model", "optimizer", "lr_scheduler", "lora"]
    training_config = {key: config[key] for key in training_keys}
    config_text = yaml.safe_dump(training_config, sort_keys=True)
    return sha256(config_text.encode()).hexdigest()[:10]


def save_yaml(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as file:
        yaml.safe_dump(data, file, sort_keys=False)


def generate_model_runs(base_config, ablation_config, output_dir):
    output_dir = Path(output_dir)
    regularizer_mode = ablation_config["regularizer_mode"]
    semantic_transformations = ablation_config["semantic_transformations"]
    runs = {}

    def add_run(seed, eta, subset, strength=0.0, mode=regularizer_mode, extra_run=False):
        is_regularized = float(eta) != 0
        config = create_training_config(base_config, seed, eta, subset, strength if subset else None, mode if is_regularized else None)
        model_id = training_config_id(config)

        if model_id in runs:
            return

        m = len(subset)
        model_name = f"{mode}_seed_{seed}_eta_{float(eta):g}_M_{m}_s_{float(strength):g}_{model_id}"
        model_dir = output_dir / "models" / model_name
        config["paths"]["run_dir"] = str(model_dir)
        config_path = model_dir / "config.yaml"
        save_yaml(config, config_path)

        runs[model_id] = {
            "model_id": model_id,
            "model_dir": model_dir,
            "config_path": config_path,
            "seed": int(seed),
            "eta": float(eta),
            "mode": mode if is_regularized else "ordinary_lora",
            "m": m,
            "s": float(strength) if subset else 0.0,
            "subset": transformation_name(subset),
            "extra_run": bool(extra_run),
        }

    for seed in ablation_config["seeds"]:
        add_run(seed, 0.0, [])

        for eta in ablation_config["eta_values"]:
            for m in ablation_config["m_values"]:
                subset = select_transformation_prefix(semantic_transformations, m)

                for strength in ablation_config["s_values"]:
                    add_run(seed, eta, subset, strength)

    for extra_run in ablation_config.get("extra_runs", []):
        mode = extra_run.get("mode", regularizer_mode)
        subset = [] if mode == "source_loss" else select_named_transformations(ablation_config, extra_run["transformations"])
        strength = 0.0 if mode == "source_loss" else extra_run.get("s", extra_run.get("transformation_strength"))

        if strength is None:
            raise ValueError("Every custom ablation run must define s.")

        add_run(extra_run["seed"], extra_run["eta"], subset, strength, mode, extra_run=True)

    return list(runs.values())


def save_rows(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_best_models(results, target_gain_fraction=0.99):
    target_gain_fraction = float(target_gain_fraction)

    if not 0 < target_gain_fraction <= 1:
        raise ValueError("target_gain_fraction must be in (0, 1].")

    ordinary_rows = [row for row in results if float(row["eta"]) == 0]
    regularized_rows = [row for row in results if float(row["eta"]) != 0 and row.get("mode") != "source_loss"]

    if not ordinary_rows or not regularized_rows:
        raise ValueError("Best-model selection needs ordinary LoRA and regularized LoRA results.")

    ordinary_target_gain = fmean(float(row["validation_target_gain"]) for row in ordinary_rows)
    required_target_gain = target_gain_fraction * ordinary_target_gain
    best_target = max(regularized_rows, key=lambda row: (float(row["validation_target_gain"]), -float(row["validation_source_degradation"])))
    best_source = min(regularized_rows, key=lambda row: (float(row["validation_source_degradation"]), -float(row["validation_target_gain"])))
    eligible_rows = [row for row in regularized_rows if float(row["validation_target_gain"]) >= required_target_gain]
    best_overall = min(eligible_rows, key=lambda row: (float(row["validation_source_degradation"]), -float(row["validation_target_gain"]))) if eligible_rows else None
    best_target_gain = float(best_target["validation_target_gain"])
    selection_summary = {
        "target_gain_fraction": target_gain_fraction,
        "ordinary_target_gain": ordinary_target_gain,
        "required_target_gain": required_target_gain,
        "best_regularized_target_gain": best_target_gain,
        "best_regularized_target_fraction": best_target_gain / ordinary_target_gain if ordinary_target_gain != 0 else None,
        "num_eligible_models": len(eligible_rows),
    }
    return {"best_target_gain": best_target, "best_source_degradation": best_source, "best_overall": best_overall}, selection_summary
