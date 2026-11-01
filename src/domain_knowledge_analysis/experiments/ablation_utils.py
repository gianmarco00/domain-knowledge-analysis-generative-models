from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from statistics import fmean

import csv
import math
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
    if regularizer_config is None:
        raise ValueError("The ablation base config must define lora.regularizer for regularized runs; it is currently null.")
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

    def add_run(seed, eta, subset, strength=0.0, mode=regularizer_mode, extra_run=False, local_sweep=False):
        is_regularized = float(eta) != 0
        config = create_training_config(base_config, seed, eta, subset, strength if subset else None, mode if is_regularized else None)
        model_id = training_config_id(config)

        if model_id in runs:
            if local_sweep:
                runs[model_id]["local_sweep"] = True
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
            "local_sweep": bool(local_sweep),
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

    local_sweep = ablation_config.get("local_sweep")
    if local_sweep is not None:
        if not isinstance(local_sweep, dict):
            raise ValueError("local_sweep must be a mapping with eta_values, m_values and s_values.")
        for key in ("eta_values", "m_values", "s_values"):
            if not isinstance(local_sweep.get(key), list) or not local_sweep[key]:
                raise ValueError(f"local_sweep.{key} must be a non-empty list.")
        for eta in local_sweep["eta_values"]:
            if not math.isfinite(float(eta)) or float(eta) <= 0:
                raise ValueError("local_sweep eta values must be finite and positive.")
        for m in local_sweep["m_values"]:
            if not math.isfinite(float(m)) or not float(m).is_integer():
                raise ValueError("local_sweep M values must be integers.")
            select_transformation_prefix(semantic_transformations, m)
        for strength in local_sweep["s_values"]:
            if not math.isfinite(float(strength)) or float(strength) < 0:
                raise ValueError("local_sweep s values must be finite and nonnegative.")
        for seed in ablation_config["seeds"]:
            for eta in local_sweep["eta_values"]:
                for m in local_sweep["m_values"]:
                    subset = select_transformation_prefix(semantic_transformations, m)
                    for strength in local_sweep["s_values"]:
                        add_run(seed, eta, subset, strength, local_sweep=True)

    return list(runs.values())


def save_rows(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as file:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def select_manual_model(results, choice):
    """Resolve an explicit regularized run; never silently pick between matches."""
    if choice is None:
        return None
    if not isinstance(choice, dict):
        raise ValueError("best_model must be null or a mapping containing eta, m and s.")
    required = {"eta", "m", "s"}
    allowed = required | {"seed", "mode", "subset", "model_id"}
    missing = required - choice.keys()
    unknown = choice.keys() - allowed
    if missing or unknown:
        raise ValueError(f"Invalid best_model: missing fields {sorted(missing)}; unknown fields {sorted(unknown)}.")
    try:
        numeric = {key: float(choice[key]) for key in ("eta", "m", "s")}
        if "seed" in choice:
            numeric["seed"] = float(choice["seed"])
        if not all(math.isfinite(value) for value in numeric.values()):
            raise ValueError
        if numeric["eta"] <= 0 or numeric["m"] <= 0 or not numeric["m"].is_integer() or numeric["s"] < 0:
            raise ValueError
        if "seed" in numeric and not numeric["seed"].is_integer():
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        raise ValueError("best_model requires finite eta > 0, integer m > 0, s >= 0 and an optional integer seed.") from None

    matches = []
    for row in results:
        if float(row["eta"]) == 0 or row.get("mode") == "source_loss":
            continue
        if all(float(row[key]) == value for key, value in numeric.items()) and all(str(row.get(key, "")) == str(choice[key]) for key in ("mode", "subset", "model_id") if key in choice):
            matches.append(row)
    if not matches:
        raise ValueError(f"No regularized model matches best_model {choice}; choose values from the configured runs.")
    if len(matches) > 1:
        raise ValueError(f"best_model {choice} matches {len(matches)} runs; add seed, mode, subset or model_id to select exactly one.")
    return matches[0]


def validate_fid_increase(value):
    try:
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        raise ValueError("max_fid_increase_fraction must be a finite nonnegative number (0.12 means 12%).") from None
    return value


def select_best_models(results, target_gain_fraction=0.99, max_fid_increase_fraction=0.12, manual_choice=None):
    target_gain_fraction = float(target_gain_fraction)
    max_fid_increase_fraction = validate_fid_increase(max_fid_increase_fraction)

    if not 0 < target_gain_fraction <= 1:
        raise ValueError("target_gain_fraction must be in (0, 1].")

    ordinary_rows = [row for row in results if float(row["eta"]) == 0]
    regularized_rows = [row for row in results if float(row["eta"]) != 0 and row.get("mode") != "source_loss"]
    if not ordinary_rows or not regularized_rows:
        raise ValueError("Best-model selection needs ordinary LoRA and regularized LoRA results.")

    fid_limits = {}
    for row in ordinary_rows:
        seed = row.get("seed")
        if seed in fid_limits:
            raise ValueError(f"Multiple ordinary LoRA baselines for seed {seed}.")
        try:
            fid = float(row["test_target_fid"])
            if not math.isfinite(fid) or fid < 0:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"Ordinary LoRA for seed {seed} needs a finite nonnegative test_target_fid.") from None
        fid_limits[seed] = fid * (1 + max_fid_increase_fraction)

    fid_eligible = []
    for row in regularized_rows:
        seed = row.get("seed")
        if seed not in fid_limits:
            raise ValueError(f"No ordinary LoRA FID baseline for seed {seed}.")
        try:
            fid = float(row["test_target_fid"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(fid) and 0 <= fid <= fid_limits[seed]:
            fid_eligible.append(row)

    ordinary_target_gain = fmean(float(row["validation_target_gain"]) for row in ordinary_rows)
    required_target_gain = target_gain_fraction * ordinary_target_gain
    best_target = max(fid_eligible, key=lambda row: (float(row["validation_target_gain"]), -float(row["validation_source_degradation"])), default=None)
    best_source = min(fid_eligible, key=lambda row: (float(row["validation_source_degradation"]), -float(row["validation_target_gain"])), default=None)
    eligible_rows = [row for row in fid_eligible if float(row["validation_target_gain"]) >= required_target_gain]
    automatic_best = min(eligible_rows, key=lambda row: (float(row["validation_source_degradation"]), -float(row["validation_target_gain"])), default=None)
    manual_best = select_manual_model(results, manual_choice)
    best_overall = manual_best if manual_best is not None else automatic_best
    best_target_gain = float(best_target["validation_target_gain"]) if best_target is not None else None
    selection_summary = {
        "target_gain_fraction": target_gain_fraction,
        "ordinary_target_gain": ordinary_target_gain,
        "required_target_gain": required_target_gain,
        "max_fid_increase_fraction": max_fid_increase_fraction,
        "max_fid_by_seed": {str(seed): limit for seed, limit in fid_limits.items()},
        "num_fid_eligible_models": len(fid_eligible),
        "best_regularized_target_gain": best_target_gain,
        "best_regularized_target_fraction": best_target_gain / ordinary_target_gain if best_target_gain is not None and ordinary_target_gain != 0 else None,
        "num_eligible_models": len(eligible_rows),
        "manual_best_model": manual_choice,
        "automatic_best_overall_model_id": automatic_best.get("model_id") if automatic_best else None,
    }
    return {"best_target_gain": best_target, "best_source_degradation": best_source, "best_overall": best_overall}, selection_summary
