from copy import deepcopy
from hashlib import sha256
from itertools import combinations
from math import comb
from pathlib import Path
from statistics import fmean

import csv
import random
import yaml


def expand_random_perturbations(random_perturbation_config):
    num_transformations = int(random_perturbation_config["num_transformations"])
    intensity = float(random_perturbation_config["intensity"])
    direction_seed = int(random_perturbation_config["direction_seed"])

    return [
        {
            "name": f"random_direction_{direction_index}",
            "intensity": intensity,
            "direction_seed": direction_seed,
        }
        for direction_index in range(num_transformations)
    ]


def create_transformation_families(ablation_config, mode):
    semantic_transformations = deepcopy(ablation_config["semantic_transformations"])
    random_transformations = expand_random_perturbations(ablation_config["random_perturbation"])

    transformations_by_mode = {
        "semantic_perturbations": semantic_transformations,
        "random_perturbations": random_transformations,
        "both": semantic_transformations + random_transformations,
    }

    return transformations_by_mode[mode]


def select_transformation_subsets(transformation_families, m, require_rotation, max_subsets, selection_seed, mode):
    if require_rotation:
        rotation = next(family for family in transformation_families if family["name"] == "rotation")
        remaining_families = [family for family in transformation_families if family["name"] != "rotation"]
        fixed_families = [rotation]
    else:
        remaining_families = transformation_families
        fixed_families = []

    subset_size = m - len(fixed_families)
    num_subsets = comb(len(remaining_families), subset_size)

    if num_subsets <= max_subsets:
        return [[*fixed_families, *subset] for subset in combinations(remaining_families, subset_size)]

    generator = random.Random(f"{selection_seed}:{mode}:{m}")
    balanced_names = [family["name"] for family in remaining_families]
    family_counts = {name: 0 for name in balanced_names}
    selected_subsets = []
    selected_names = set()

    while len(selected_subsets) < max_subsets:
        ranked_families = sorted(remaining_families, key=lambda family: (family_counts[family["name"]], generator.random()))
        selected_subset = [*fixed_families, *ranked_families[:subset_size]]
        subset_names = tuple(sorted(family["name"] for family in selected_subset))

        if subset_names in selected_names:
            continue

        selected_subsets.append(selected_subset)
        selected_names.add(subset_names)

        for family in selected_subset[len(fixed_families):]:
            family_counts[family["name"]] += 1

    return selected_subsets


def transformation_name(transformation_families):
    return "+".join(family["name"] for family in transformation_families) if transformation_families else "none"


def select_named_transformations(ablation_config, transformation_names):
    transformation_families = create_transformation_families(ablation_config, "both")
    transformations_by_name = {family["name"]: family for family in transformation_families}
    return [deepcopy(transformations_by_name[name]) for name in transformation_names]


def create_training_config(base_config, seed, eta, transformation_families):
    from domain_knowledge_analysis.utils.dataset_utils import TRANSFORMATION_IMPLEMENTATION_VERSION

    config = deepcopy(base_config)
    config["seed"] = int(seed)
    config["lora"]["pretrained_model"] = None

    if float(eta) == 0 or not transformation_families:
        config["lora"]["regularizer"] = None
        return config

    regularizer_config = deepcopy(base_config["lora"]["regularizer"])
    regularizer_config["eta"] = float(eta)
    regularizer_config["transformations"] = deepcopy(transformation_families)
    regularizer_config["transformation_implementation_version"] = TRANSFORMATION_IMPLEMENTATION_VERSION
    config["lora"]["regularizer"] = regularizer_config

    return config


def training_config_id(config):
    training_keys = [
        "seed",
        "pretrained_model",
        "dataset",
        "dataloader",
        "training",
        "loss",
        "model",
        "optimizer",
        "lr_scheduler",
        "lora",
    ]

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
    eta_values = sorted(set([0.0, *[float(eta) for eta in ablation_config["eta_values"]]]))
    max_subsets = int(ablation_config["max_subsets_per_m"])
    selection_seed = int(ablation_config["subset_selection_seed"])
    runs = {}

    def add_run(seed, eta, mode, subset, subset_index, extra_run=False):
        config = create_training_config(base_config, seed, eta, subset)
        model_id = training_config_id(config)

        if model_id in runs:
            return

        m = len(subset)
        model_name = f"{mode}_seed_{seed}_eta_{float(eta):g}_M_{m}_{model_id}"
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
            "mode": mode,
            "m": m,
            "subset_index": int(subset_index),
            "subset": transformation_name(subset),
            "extra_run": bool(extra_run),
        }

    for seed in ablation_config["seeds"]:
        add_run(seed, 0.0, "ordinary_lora", [], 0)

        for eta in eta_values:
            if eta == 0:
                continue

            for mode in ablation_config["transformation_modes"]:
                transformation_families = create_transformation_families(ablation_config, mode)
                require_rotation = mode in {"semantic_perturbations", "both"}

                for m in ablation_config["m_values"][mode]:
                    subsets = select_transformation_subsets(transformation_families, int(m), require_rotation, max_subsets, selection_seed, mode)

                    for subset_index, subset in enumerate(subsets):
                        add_run(seed, eta, mode, subset, subset_index)

    for extra_run in ablation_config.get("extra_runs", []):
        subset = select_named_transformations(ablation_config, extra_run["transformations"])
        add_run(extra_run["seed"], extra_run["eta"], extra_run["mode"], subset, 0, extra_run=True)

    return list(runs.values())


def save_rows(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_eta(results, eta_values, selection_mode, selection_m, target_gain_fraction):
    eta_values = sorted(set([0.0, *[float(eta) for eta in eta_values]]))
    ordinary_rows = [row for row in results if row["eta"] == 0]
    ordinary_target_gain = fmean(float(row["validation_target_gain"]) for row in ordinary_rows)
    required_target_gain = float(target_gain_fraction) * ordinary_target_gain
    summary = []

    for eta in eta_values:
        if eta == 0:
            rows = ordinary_rows
        else:
            rows = [row for row in results if row["eta"] == eta and row["mode"] == selection_mode and row["m"] == selection_m]

        summary.append({
            "eta": eta,
            "validation_target_gain": fmean(float(row["validation_target_gain"]) for row in rows),
            "validation_source_degradation": fmean(float(row["validation_source_degradation"]) for row in rows),
        })

    eligible = [row for row in summary if row["validation_target_gain"] >= required_target_gain]
    selected = min(eligible, key=lambda row: row["validation_source_degradation"])

    for row in summary:
        row["required_target_gain"] = required_target_gain
        row["selected"] = row["eta"] == selected["eta"]

    return selected["eta"], summary


def select_model(results, target_gain_fraction):
    ordinary_rows = [row for row in results if row["eta"] == 0]
    ordinary_target_gain = fmean(float(row["validation_target_gain"]) for row in ordinary_rows)
    required_target_gain = float(target_gain_fraction) * ordinary_target_gain
    groups = {}

    for row in results:
        if row["eta"] == 0:
            continue

        key = row["mode"], int(row["m"]), float(row["eta"])
        groups.setdefault(key, []).append(row)

    summary = []

    for (mode, m, eta), rows in sorted(groups.items()):
        summary.append({
            "mode": mode,
            "m": m,
            "eta": eta,
            "validation_target_gain": fmean(float(row["validation_target_gain"]) for row in rows),
            "validation_source_degradation": fmean(float(row["validation_source_degradation"]) for row in rows),
            "required_target_gain": required_target_gain,
        })

    eligible = [row for row in summary if row["validation_target_gain"] >= required_target_gain]
    selected = min(eligible, key=lambda row: row["validation_source_degradation"])

    for row in summary:
        row["selected"] = row is selected

    return selected, summary
