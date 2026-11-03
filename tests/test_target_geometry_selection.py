from copy import deepcopy

import pytest

from domain_knowledge_analysis.experiments.target_geometry_ablation import select_target_geometry_models


def selection_rows():
    def row(model_id, view, mse, fid, validation=None, seed=42):
        return {"model_id": model_id, "view": view, "seed": seed, "eta": 0. if view == "ordinary_lora" else 100.,
                "geometry_batch_size": 64, "dual_learning_rate": .01,
                "validation_adapted_target_mse": mse if validation is None else validation,
                "test_adapted_target_mse": mse, "test_target_fid": fid}
    return [row("baseline", "ordinary_lora", 8., 16.),
            row("full-reconstruction", "full", 4., 14., validation=2.),
            row("full-generation", "full", 6., 6., validation=4.),
            row("target-best-mse", "target_only", 4., 12.),  # (50% + 25%)/2 = 37.5%
            row("target-balanced", "target_only", 6., 8.),  # (25% + 50%)/2 = 37.5%; lower FID wins.
            row("target-best-fid", "target_only", 12., 4.)]  # (-50% + 75%)/2 = 12.5%.


def test_balanced_percent_improvements_lower_fid_tie_and_independent_generation_selection():
    selections, score = select_target_geometry_models(selection_rows())
    assert score == 37.5
    assert {key: row["model_id"] for key, row in selections.items()} == {
        "full_reconstruction": "full-reconstruction", "full_generation": "full-generation",
        "target_only_reconstruction": "target-balanced", "target_only_generation": "target-best-fid"}


def test_balanced_selection_normalizes_to_each_seeds_baseline_before_comparing():
    rows = selection_rows()
    baseline = {**rows[0], "model_id": "baseline-43", "seed": 43,
                "test_adapted_target_mse": 80., "test_target_fid": 160.}
    improved = {**rows[4], "model_id": "target-43", "seed": 43,
                "test_adapted_target_mse": 20., "test_target_fid": 40.}
    selections, score = select_target_geometry_models([*rows, baseline, improved])
    assert score == 75.
    assert selections["target_only_reconstruction"]["model_id"] == "target-43"
    # Generation selects absolute FID, independently of normalized improvements.
    assert selections["target_only_generation"]["model_id"] == "target-best-fid"


def test_selection_is_deterministic_and_rejects_invalid_or_missing_comparisons():
    rows = selection_rows()
    assert select_target_geometry_models(rows) == select_target_geometry_models(list(reversed(rows)))
    with pytest.raises(ValueError, match="baseline"):
        select_target_geometry_models(rows[1:])
    with pytest.raises(ValueError, match="full and target-only"):
        select_target_geometry_models([row for row in rows if row["view"] != "target_only"])
    invalid = deepcopy(rows)
    invalid[0]["test_target_fid"] = 0.
    with pytest.raises(ValueError, match="finite and positive"):
        select_target_geometry_models(invalid)
