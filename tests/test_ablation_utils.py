from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from domain_knowledge_analysis.experiments.ablation_utils import create_training_config, generate_model_runs, select_best_models, select_manual_model, select_named_transformations, select_transformation_prefix, training_config_id


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


def test_ablation_config_generates_the_grid_source_loss_and_one_baseline(tmp_path):
    base_config, config = load_configs()
    source_checkpoint = base_config["pretrained_model"]
    runs = generate_model_runs(base_config, config, tmp_path)
    regularized_runs = [run for run in runs if run["mode"] == config["regularizer_mode"]]
    source_loss_run = next(run for run in runs if run["mode"] == "source_loss")
    ordinary_runs = [run for run in runs if run["eta"] == 0]

    assert len(regularized_runs) == 45
    assert len(ordinary_runs) == 1
    assert len({run["model_id"] for run in runs}) == 47
    assert {(run["eta"], run["m"], run["s"]) for run in regularized_runs} == {(eta, m, s) for eta in config["eta_values"] for m in config["m_values"] for s in config["s_values"]}
    assert all(run["mode"] == config["regularizer_mode"] for run in regularized_runs)
    assert all(run["config_path"].is_file() for run in runs)
    assert base_config["pretrained_model"] == source_checkpoint
    assert (source_loss_run["eta"], source_loss_run["m"], source_loss_run["s"]) == (1.0, 0, 0.0)

    with open(source_loss_run["config_path"]) as file:
        source_loss_config = yaml.safe_load(file)["lora"]["regularizer"]

    assert source_loss_config["mode"] == "source_loss"
    assert source_loss_config["transformations"] == []
    assert "transformation_strength" not in source_loss_config

    for run in regularized_runs:
        with open(run["config_path"]) as file:
            generated_config = yaml.safe_load(file)

        regularizer_config = generated_config["lora"]["regularizer"]
        assert regularizer_config["mode"] == config["regularizer_mode"]
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
        {"model_id": "source_loss", "mode": "source_loss", "eta": 1.0, "validation_target_gain": 0.2, "validation_source_degradation": 0.0},
    ]

    for row in results:
        row["test_target_fid"] = 20.0
    selected, selection_summary = select_best_models(results)

    assert selection_summary["required_target_gain"] == pytest.approx(0.099)
    assert selection_summary["num_eligible_models"] == 2
    assert selected["best_target_gain"]["model_id"] == "target"
    assert selected["best_source_degradation"]["model_id"] == "source"
    assert selected["best_overall"]["model_id"] == "overall"


def test_best_model_selection_records_when_no_model_reaches_the_target_rule():
    results = [
        {"eta": 0.0, "validation_target_gain": 0.10, "validation_source_degradation": 0.05},
        {"eta": 10.0, "validation_target_gain": 0.09, "validation_source_degradation": 0.01},
    ]

    for row in results:
        row["test_target_fid"] = 20.0
    selected, selection_summary = select_best_models(results)

    assert selected["best_overall"] is None
    assert selected["best_target_gain"] == results[1]
    assert selection_summary["num_eligible_models"] == 0
    assert selection_summary["best_regularized_target_fraction"] == pytest.approx(0.9)


def test_regularized_ablation_explains_missing_base_regularizer():
    base, config = load_configs()
    base["lora"]["regularizer"] = None
    with pytest.raises(ValueError, match="base config must define lora.regularizer"):
        create_training_config(base, 42, 10, config["semantic_transformations"][:5], 0.1)


def test_fid_limit_uses_each_seed_baseline_and_includes_boundary():
    results = [
        {"model_id": "lora1", "seed": 1, "eta": 0, "test_target_fid": 20, "validation_target_gain": 0.1, "validation_source_degradation": 0.1},
        {"model_id": "lora2", "seed": 2, "eta": 0, "test_target_fid": 10, "validation_target_gain": 0.1, "validation_source_degradation": 0.1},
        {"model_id": "boundary", "seed": 1, "eta": 10, "m": 5, "s": 0.1, "test_target_fid": 22.4, "validation_target_gain": 0.1, "validation_source_degradation": 0.03},
        {"model_id": "too_high", "seed": 1, "eta": 100, "m": 5, "s": 0.1, "test_target_fid": 22.4001, "validation_target_gain": 0.2, "validation_source_degradation": 0.001},
        {"model_id": "other_seed", "seed": 2, "eta": 10, "m": 5, "s": 0.1, "test_target_fid": 15, "validation_target_gain": 0.2, "validation_source_degradation": 0.001},
    ]
    selected, summary = select_best_models(results)
    assert selected["best_overall"]["model_id"] == "boundary"
    assert selected["best_target_gain"]["model_id"] == "boundary"
    assert selected["best_source_degradation"]["model_id"] == "boundary"
    assert summary["num_fid_eligible_models"] == 1
    assert summary["max_fid_by_seed"] == {"1": pytest.approx(22.4), "2": pytest.approx(11.2)}


def test_no_fid_eligible_model_does_not_become_fallback():
    results = [
        {"eta": 0, "seed": 1, "test_target_fid": 20, "validation_target_gain": 0.1},
        {"eta": 10, "seed": 1, "test_target_fid": 23, "validation_target_gain": 0.2, "validation_source_degradation": 0.01},
    ]
    selected, summary = select_best_models(results)
    assert all(model is None for model in selected.values())
    assert summary["num_fid_eligible_models"] == 0


def test_manual_best_model_overrides_fid_and_gain_rules_and_rejects_ambiguity():
    results = [
        {"model_id": "baseline", "eta": 0, "seed": 1, "test_target_fid": 20, "validation_target_gain": 0.1},
        {"model_id": "automatic", "eta": 10, "m": 5, "s": 0.1, "seed": 1, "mode": "output_gram", "test_target_fid": 20, "validation_target_gain": 0.1, "validation_source_degradation": 0.02},
        {"model_id": "manual", "eta": 100, "m": 5, "s": 0.5, "seed": 1, "mode": "output_gram", "test_target_fid": 30, "validation_target_gain": 0.05, "validation_source_degradation": 0.01},
    ]
    choice = {"eta": 100, "m": 5, "s": 0.5}
    selected, summary = select_best_models(results, manual_choice=choice)
    assert selected["best_overall"]["model_id"] == "manual"
    assert summary["automatic_best_overall_model_id"] == "automatic"
    assert summary["manual_best_model"] == choice
    assert select_manual_model(results, {**choice, "seed": 1})["model_id"] == "manual"
    duplicate = {**results[-1], "model_id": "different", "seed": 2}
    with pytest.raises(ValueError, match="matches 2 runs"):
        select_manual_model(results + [duplicate], choice)
    assert select_manual_model(results + [duplicate], {**choice, "seed": 2})["model_id"] == "different"
    with pytest.raises(ValueError, match="No regularized model matches"):
        select_manual_model(results, {**choice, "s": 0.75})
    with pytest.raises(ValueError, match="Invalid best_model"):
        select_manual_model(results, {"eta": 100, "m": 5})


@pytest.mark.parametrize("fraction", [-0.1, float("nan"), float("inf"), "invalid"])
def test_fid_tolerance_must_be_finite_and_nonnegative(fraction):
    with pytest.raises(ValueError, match="max_fid_increase_fraction"):
        select_best_models([], max_fid_increase_fraction=fraction)


def test_best_overall_selects_m20_when_eligible_with_less_validation_degradation():
    baseline = {"model_id": "lora", "seed": 42, "eta": 0, "test_target_fid": 20,
                "validation_target_gain": 0.1, "validation_source_degradation": 0.05}
    m10 = {"model_id": "m10", "seed": 42, "eta": 100000, "m": 10, "s": 0.5,
           "test_target_fid": 21, "validation_target_gain": 0.0995, "validation_source_degradation": 0.03}
    m20 = {"model_id": "m20", "seed": 42, "eta": 100000, "m": 20, "s": 0.5,
           "test_target_fid": 19, "validation_target_gain": 0.0996, "validation_source_degradation": 0.028}

    for results in ([baseline, m10, m20], [baseline, m20, m10]):
        best, summary = select_best_models(results)
        assert best["best_overall"]["model_id"] == "m20"
        assert summary["num_eligible_models"] == 2


def test_best_overall_uses_validation_even_when_test_graph_favors_m20():
    baseline = {"model_id": "lora", "seed": 42, "eta": 0, "test_target_fid": 20,
                "validation_target_gain": 0.1, "validation_source_degradation": 0.05}
    m10 = {"model_id": "m10", "seed": 42, "eta": 100000, "m": 10, "s": 0.5,
           "test_target_fid": 21, "validation_target_gain": 0.0995, "validation_source_degradation": 0.03,
           "test_target_gain": 0.099, "test_source_degradation": 0.029}
    m20 = {"model_id": "m20", "seed": 42, "eta": 100000, "m": 20, "s": 0.5,
           "test_target_fid": 19, "validation_target_gain": 0.0989, "validation_source_degradation": 0.028,
           "test_target_gain": 0.100, "test_source_degradation": 0.028}

    best, summary = select_best_models([baseline, m10, m20])
    assert best["best_overall"]["model_id"] == "m10"
    assert summary["num_eligible_models"] == 1
    assert m20["validation_target_gain"] < summary["required_target_gain"]


def test_mnist_selection_matches_reported_m10_m20_metrics():
    rows = [
        {"model_id": "08409eae5f", "seed": 42, "eta": 0, "test_target_fid": 20.320091247558594,
         "validation_target_gain": 0.016977336555719376, "validation_source_degradation": 0.020937145431836446},
        {"model_id": "1fc9741ddb", "seed": 42, "eta": 10000, "m": 10, "s": 0.5, "test_target_fid": 21.31299591064453,
         "validation_target_gain": 0.016812034219503405, "validation_source_degradation": 0.01491932608683904},
        {"model_id": "3c793eb1b5", "seed": 42, "eta": 10000, "m": 20, "s": 0.5, "test_target_fid": 20.52738380432129,
         "validation_target_gain": 0.01695359694957733, "validation_source_degradation": 0.016346378922462462},
        {"model_id": "ce081f0cd4", "seed": 42, "eta": 100000, "m": 20, "s": 0.5, "test_target_fid": 21.891908645629883,
         "validation_target_gain": 0.016698507865269982, "validation_source_degradation": 0.011953310658534368},
    ]

    best, summary = select_best_models(rows)
    assert summary["required_target_gain"] == pytest.approx(0.016807563190162182)
    assert summary["max_fid_by_seed"]["42"] == pytest.approx(22.758502197265628)
    assert summary["num_fid_eligible_models"] == 3
    assert summary["num_eligible_models"] == 2
    assert best["best_overall"]["model_id"] == "1fc9741ddb"
