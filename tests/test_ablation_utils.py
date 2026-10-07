from collections import Counter
from pathlib import Path

import pytest
import yaml

from domain_knowledge_analysis.experiments.ablation_utils import create_transformation_families, expand_random_perturbations, generate_model_runs, select_eta, select_transformation_subsets


def ablation_config():
    return {
        "semantic_transformations": [
            {"name": "rotation", "intensity": 15},
            {"name": "horizontal_translation", "intensity": 2},
            {"name": "vertical_translation", "intensity": 2},
            {"name": "isotropic_scale", "intensity": 1.1},
            {"name": "aspect_ratio_deformation", "intensity": 1.1},
            {"name": "diagonal_shear", "intensity": 0.12},
            {"name": "vertical_shear", "intensity": 0.12},
            {"name": "stroke_thickness", "intensity": 0.5},
            {"name": "gaussian_blur", "intensity": 0.7},
            {"name": "contrast", "intensity": 1.2},
        ],
        "random_perturbation": {
            "name": "perturbation",
            "num_transformations": 10,
            "intensity": 0.05,
            "direction_seed": 5000,
        },
    }


@pytest.mark.parametrize("num_transformations", [9, 10])
def test_random_perturbation_expands_to_requested_number_of_families(num_transformations):
    config = ablation_config()["random_perturbation"]
    config["num_transformations"] = num_transformations
    families = expand_random_perturbations(config)

    assert len(families) == num_transformations
    assert families[0]["name"] == "random_direction_0"
    assert families[-1]["name"] == f"random_direction_{num_transformations - 1}"
    assert all(family["direction_seed"] == 5000 for family in families)


def test_transformation_modes_have_the_expected_number_of_families():
    config = ablation_config()

    assert len(create_transformation_families(config, "semantic_perturbations")) == 10
    assert len(create_transformation_families(config, "random_perturbations")) == 10
    assert len(create_transformation_families(config, "both")) == 20


def test_semantic_m5_selects_five_unique_subsets_that_contain_rotation():
    families = create_transformation_families(ablation_config(), "semantic_perturbations")
    subsets = select_transformation_subsets(families, 5, True, 5, 123, "semantic_perturbations")
    names = [tuple(family["name"] for family in subset) for subset in subsets]

    assert len(subsets) == 5
    assert len(set(names)) == 5
    assert all("rotation" in subset_names for subset_names in names)

    counts = Counter(name for subset_names in names for name in subset_names if name != "rotation")
    assert set(counts) == {family["name"] for family in families if family["name"] != "rotation"}
    assert max(counts.values()) - min(counts.values()) <= 1


def test_random_m5_selects_five_unique_subsets_without_rotation():
    families = create_transformation_families(ablation_config(), "random_perturbations")
    subsets = select_transformation_subsets(families, 5, False, 5, 123, "random_perturbations")
    names = [tuple(family["name"] for family in subset) for subset in subsets]

    assert len(subsets) == 5
    assert len(set(names)) == 5
    assert all("rotation" not in subset_names for subset_names in names)

    counts = Counter(name for subset_names in names for name in subset_names)
    assert set(counts) == {family["name"] for family in families}
    assert max(counts.values()) - min(counts.values()) <= 1


def test_full_modes_return_one_subset():
    config = ablation_config()

    for mode, m, require_rotation in [("semantic_perturbations", 10, True), ("random_perturbations", 10, False), ("both", 20, True)]:
        families = create_transformation_families(config, mode)
        subsets = select_transformation_subsets(families, m, require_rotation, 5, 123, mode)

        assert len(subsets) == 1
        assert len(subsets[0]) == m


def test_ablation_config_generates_expected_unique_models(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    source_checkpoint = "runs/vae_mnist_lr_0.001_29_jul_1606_B_beta_1.7_LD_16/checkpoints/best.pt"

    with open(repo_root / "config" / "lora_vae_mnist.yaml") as file:
        base_config = yaml.safe_load(file)

    with open(repo_root / "config" / "ablation_vae_mnist.yaml") as file:
        config = yaml.safe_load(file)["ablation"]

    runs = generate_model_runs(base_config, config, tmp_path)
    num_seeds = len(config["seeds"])
    num_positive_eta = len([eta for eta in config["eta_values"] if float(eta) != 0])
    subsets_per_mode = {}

    for mode in config["transformation_modes"]:
        families = create_transformation_families(config, mode)
        require_rotation = mode in {"semantic_perturbations", "both"}
        subsets_per_mode[mode] = sum(
            len(select_transformation_subsets(families, int(m), require_rotation, config["max_subsets_per_m"], config["subset_selection_seed"], mode))
            for m in config["m_values"][mode]
        )

    assert len(runs) == num_seeds * (1 + sum(subsets_per_mode.values()) * num_positive_eta)
    assert len([run for run in runs if run["mode"] == "ordinary_lora"]) == num_seeds

    for mode, num_subsets in subsets_per_mode.items():
        assert len([run for run in runs if run["mode"] == mode]) == num_seeds * num_subsets * num_positive_eta

    assert all(run["config_path"].is_file() for run in runs)
    assert base_config["pretrained_model"] == source_checkpoint

    for run in runs:
        with open(run["config_path"]) as file:
            generated_config = yaml.safe_load(file)

        assert generated_config["pretrained_model"] == source_checkpoint

    semantic_m5_subsets = {run["subset"] for run in runs if run["mode"] == "semantic_perturbations" and run["m"] == 5}
    random_m5_subsets = {run["subset"] for run in runs if run["mode"] == "random_perturbations" and run["m"] == 5}

    assert len(semantic_m5_subsets) == 5
    assert len(random_m5_subsets) == 5


def test_eta_selection_uses_95_percent_target_rule_and_lowest_source_degradation():
    results = []

    for seed in [0, 1]:
        results.extend([
            {"seed": seed, "eta": 0.0, "mode": "ordinary_lora", "m": 0, "validation_target_gain": 0.10, "validation_source_degradation": 0.05},
            {"seed": seed, "eta": 0.1, "mode": "both", "m": 20, "validation_target_gain": 0.098, "validation_source_degradation": 0.03},
            {"seed": seed, "eta": 1.0, "mode": "both", "m": 20, "validation_target_gain": 0.095, "validation_source_degradation": 0.01},
            {"seed": seed, "eta": 10.0, "mode": "both", "m": 20, "validation_target_gain": 0.09, "validation_source_degradation": 0.0},
        ])

    eta_star, summary = select_eta(results, [0.0, 0.1, 1.0, 10.0], "both", 20, 0.95)

    assert eta_star == 1.0
    assert next(row for row in summary if row["eta"] == 1.0)["selected"]
