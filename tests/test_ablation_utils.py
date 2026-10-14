from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from domain_knowledge_analysis.experiments.ablation_utils import generate_model_runs, select_best_models, select_named_transformations, select_transformation_prefix, training_config_id


def load_configs():
    repo_root = Path(__file__).resolve().parents[1]

    with open(repo_root / "config" / "lora_vae_mnist.yaml") as file:
        base_config = yaml.safe_load(file)

    with open(repo_root / "config" / "ablation_vae_mnist.yaml") as file:
        ablation_config = yaml.safe_load(file)["ablation"]

    return base_config, ablation_config


def test_transformation_prefix_uses_yaml_order_without_forcing_rotation():
    _, config = load_configs()
    first_five = select_transformation_prefix(config["semantic_transformations"], 5)
    first_ten = select_transformation_prefix(config["semantic_transformations"], 10)

    assert [transformation["name"] for transformation in first_five] == ["horizontal_translation", "vertical_translation", "isotropic_scale", "aspect_ratio_deformation", "diagonal_shear"]
    assert "rotation" not in [transformation["name"] for transformation in first_five]
    assert "rotation" in [transformation["name"] for transformation in first_ten]


@pytest.mark.parametrize("m", [0, 21])
def test_transformation_prefix_rejects_invalid_m(m):
    _, config = load_configs()

    with pytest.raises(ValueError, match="M must be between"):
        select_transformation_prefix(config["semantic_transformations"], m)


def test_named_transformations_keep_requested_order_and_values():
    _, config = load_configs()
    names = ["vertical_translation", "rotation", "contrast"]
    transformations = select_named_transformations(config, names)

    assert [transformation["name"] for transformation in transformations] == names
    assert transformations[0]["intensity"] == 2
    assert transformations[1]["intensity"] == 15
    assert transformations[2]["intensity"] == 1.2


def test_ablation_config_generates_45_regularized_models_and_one_baseline(tmp_path):
    base_config, config = load_configs()
    source_checkpoint = base_config["pretrained_model"]
    runs = generate_model_runs(base_config, config, tmp_path)
    regularized_runs = [run for run in runs if run["eta"] != 0]
    ordinary_runs = [run for run in runs if run["eta"] == 0]

    assert len(regularized_runs) == 45
    assert len(ordinary_runs) == 1
    assert len({run["model_id"] for run in runs}) == 46
    assert {(run["eta"], run["m"], run["s"]) for run in regularized_runs} == {(eta, m, s) for eta in config["eta_values"] for m in config["m_values"] for s in config["s_values"]}
    assert all(run["mode"] == "output_gram" for run in regularized_runs)
    assert all(run["config_path"].is_file() for run in runs)
    assert base_config["pretrained_model"] == source_checkpoint

    for run in regularized_runs:
        with open(run["config_path"]) as file:
            generated_config = yaml.safe_load(file)

        regularizer_config = generated_config["lora"]["regularizer"]
        assert regularizer_config["mode"] == "output_gram"
        assert regularizer_config["transformation_strength"] == run["s"]
        assert len(regularizer_config["transformations"]) == run["m"]


def test_custom_runs_remain_supported(tmp_path):
    base_config, config = load_configs()
    config = deepcopy(config)
    config["extra_runs"] = [{"seed": 42, "eta": 7.0, "mode": "encoder_gram", "s": 0.25, "transformations": ["rotation", "contrast"]}]
    runs = generate_model_runs(base_config, config, tmp_path)
    custom_run = next(run for run in runs if run["extra_run"])

    assert custom_run["mode"] == "encoder_gram"
    assert custom_run["m"] == 2
    assert custom_run["s"] == 0.25
    assert custom_run["subset"] == "rotation+contrast"


def test_transformation_strength_changes_ablation_identity():
    first_config, _ = load_configs()
    second_config = deepcopy(first_config)
    second_config["lora"]["regularizer"]["transformation_strength"] = 0.6

    assert training_config_id(first_config) != training_config_id(second_config)


def test_regularizer_mode_changes_ablation_identity():
    first_config, _ = load_configs()
    second_config = deepcopy(first_config)
    second_config["lora"]["regularizer"]["mode"] = "encoder_response"

    assert training_config_id(first_config) != training_config_id(second_config)


def test_best_models_use_validation_metrics_and_the_99_percent_lora_rule():
    results = [
        {"model_id": "lora", "eta": 0.0, "validation_target_gain": 0.10, "validation_source_degradation": 0.05},
        {"model_id": "target", "eta": 10.0, "validation_target_gain": 0.101, "validation_source_degradation": 0.04},
        {"model_id": "overall", "eta": 100.0, "validation_target_gain": 0.099, "validation_source_degradation": 0.02},
        {"model_id": "source", "eta": 1000.0, "validation_target_gain": 0.098, "validation_source_degradation": 0.01},
    ]

    selected, required_target_gain = select_best_models(results)

    assert required_target_gain == pytest.approx(0.099)
    assert selected["best_target_gain"]["model_id"] == "target"
    assert selected["best_source_degradation"]["model_id"] == "source"
    assert selected["best_overall"]["model_id"] == "overall"


def test_best_model_selection_fails_when_no_model_reaches_the_target_rule():
    results = [
        {"eta": 0.0, "validation_target_gain": 0.10, "validation_source_degradation": 0.05},
        {"eta": 10.0, "validation_target_gain": 0.09, "validation_source_degradation": 0.01},
    ]

    with pytest.raises(ValueError, match="No regularized model reaches 99%"):
        select_best_models(results)
