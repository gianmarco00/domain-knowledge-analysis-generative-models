"""Setup for the opt-in target_geometry regularizer."""

from contextlib import nullcontext

from domain_knowledge_analysis import utils
from domain_knowledge_analysis.regularizers.target_geometry import GeometryBatchSampler, TargetGeometryRegularizer
from domain_knowledge_analysis.training.target_geometry_trainer import TargetGeometryTrainer
from domain_knowledge_analysis.utils.dataset_utils import create_train_validation_datasets
from .ablation_utils import save_yaml


def create_target_geometry_regularizer(experiment):
    config = experiment.config["lora"]["regularizer"]
    allowed = {"mode", "eta", "view", "geometry_batch_size", "calibration_batches",
               "validation_batches", "dual_learning_rate", "energy_floor"}
    unknown = config.keys() - allowed
    if unknown:
        raise ValueError(f"Unknown target_geometry settings: {sorted(unknown)}.")
    # Use the SOURCE checkpoint's split/seed, so source training images are
    # never accidentally reclassified as held out when the adaptation seed varies.
    _, held_out_source = create_train_validation_datasets(experiment.source_config, experiment.source_dataset_name)
    batch_size = config.get("geometry_batch_size", 64)
    cache = nullcontext({"calibration": None})
    if experiment.device.type == "cuda":
        from domain_knowledge_analysis.utils.target_geometry_runtime import (
            resident_tensor_dataset, calibration_cache, calibration_cache_key,
        )

        held_out_source, source_digest = resident_tensor_dataset(held_out_source, experiment.device)
        if source_digest is not None and experiment.pretrained_model_path is not None:
            key = calibration_cache_key(experiment, source_digest, batch_size, config)
            cache = calibration_cache(utils.get_repo_root() / ".cache" / "target_geometry" / "calibration", key)
    source_sampler = GeometryBatchSampler(held_out_source, batch_size, experiment.config["seed"] + 17001)
    with cache as cached:
        regularizer = TargetGeometryRegularizer(
            experiment.model, source_sampler,
            calibration_batches=config.get("calibration_batches", 128),
            eta=config["eta"], dual_learning_rate=config.get("dual_learning_rate", 0.01),
            view=config.get("view", "full"), energy_floor=config.get("energy_floor", 1e-12),
            calibration=cached["calibration"],
        )
        cached["calibration"] = regularizer.calibration
    calibration = {**regularizer.calibration, "source_split_seed": experiment.source_config["seed"],
                   "calibration_seed": experiment.config["seed"] + 17001,
                   "source_checkpoint": experiment.config["pretrained_model"],
                   "tolerance_rule": "measured mean; no added slack or uncertainty inflation"}
    save_yaml(calibration, experiment.log_dir / "target_geometry_calibration.yaml")
    print(f"Target geometry: c_s={regularizer.source_tolerance:.6g}, "
          f"SE={regularizer.calibration['standard_error']:.3g}, "
          f"B={batch_size}, view={regularizer.view}, eta={regularizer.eta:g}")
    return regularizer, {
        "geometry_batch_size": batch_size,
        "validation_batches": config.get("validation_batches", 64),
        "geometry_seed": experiment.config["seed"] + 27001,
    }


__all__ = ["create_target_geometry_regularizer", "TargetGeometryTrainer"]
