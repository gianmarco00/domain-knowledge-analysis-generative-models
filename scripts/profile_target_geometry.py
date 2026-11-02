"""Reproducible timing/profile runs using the real source checkpoints and data.

Optionally compare a Git revision's regularizer/trainer with the working tree.
No experiment checkpoints or training configurations are modified.
"""

import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import tempfile
import time

import torch
from torch.utils.data import DataLoader, Subset

from domain_knowledge_analysis import utils
from domain_knowledge_analysis.adapters import LoRAManager
from domain_knowledge_analysis.regularizers import target_geometry
from domain_knowledge_analysis.training import target_geometry_trainer
from domain_knowledge_analysis.utils.dataset_utils import create_train_validation_datasets


def load_reference(ref, directory):
    modules = []
    for package, name in (("regularizers", "target_geometry"), ("training", "target_geometry_trainer")):
        source = subprocess.run(["git", "show", f"{ref}:src/domain_knowledge_analysis/{package}/{name}.py"], check=True, capture_output=True).stdout
        path = directory / f"reference_{name}.py"
        path.write_bytes(source)
        spec = importlib.util.spec_from_file_location(f"domain_knowledge_analysis.{package}.profile_reference_{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    modules[1].GeometryBatchSampler = modules[0].GeometryBatchSampler
    return modules


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def timed(function, device):
    synchronize(device)
    start = time.perf_counter()
    result = function()
    synchronize(device)
    return time.perf_counter() - start, result


def profile_case(dataset, batch_size, geometry, training, args, label):
    config = utils.load_config(f"config/lora_vae_{dataset}_target_geometry.yaml")
    utils.set_seed(config["seed"])
    device = torch.device(args.device)
    checkpoint = torch.load(config["pretrained_model"], map_location="cpu", weights_only=True)
    model = utils.create_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    manager = LoRAManager(model, config)
    manager.inject_adapters()
    _, source = create_train_validation_datasets(checkpoint["config"], config["dataset"]["name"])
    train_loader, validation_loader = utils.create_training_dataloaders(config, config["lora"]["transformed_dataset"])
    sampler = geometry.GeometryBatchSampler(source, batch_size, config["seed"] + 17001)
    calibration_time, regularizer = timed(lambda: geometry.TargetGeometryRegularizer(
        model, sampler, calibration_batches=args.calibration_batches,
        eta=config["lora"]["regularizer"]["eta"],
        dual_learning_rate=config["lora"]["regularizer"]["dual_learning_rate"],
        view=config["lora"]["regularizer"]["view"],
        energy_floor=config["lora"]["regularizer"]["energy_floor"]), device)
    trainer = training.TargetGeometryTrainer(
        model=model, train_dataloader=train_loader, validate_dataloader=validation_loader,
        optimizer=utils.create_optimizer(config, manager.trainable_parameters()),
        lr_scheduler=None, lr_scheduler_start_epoch=0, loss=utils.create_loss(checkpoint["config"]),
        epochs=1, device=device, regularizer=regularizer, regularizer_eta=regularizer.eta,
        geometry_batch_size=batch_size, validation_batches=args.validation_batches,
        geometry_seed=config["seed"] + 27001)
    # Sampler timing includes image decoding/conversion and device transfer.
    sampling = {}
    for name, data in (("source", source), ("target", train_loader.dataset)):
        probe = geometry.GeometryBatchSampler(data, batch_size, 123)
        elapsed, _ = timed(lambda: [probe.sample(device) for _ in range(100)], device)
        sampling[name + "_batch_ms"] = elapsed * 10

    def step(stages=None):
        stages = defaultdict(float) if stages is None else stages
        def stage(name, function):
            elapsed, value = timed(function, device)
            stages[name] += elapsed
            return value
        trainer.optimizer.zero_grad()
        images = train_loader.dataset.tensors[0][:config["dataloader"]["batch_size"]].to(device)
        base = stage("vae_forward_loss", lambda: trainer.loss(images, *model(images)))
        batch = stage("geometry_sample", lambda: trainer.geometry_sampler.sample(device))
        with torch.set_grad_enabled(not hasattr(trainer, "_active_geometry") or regularizer.multiplier != 0):
            values = stage("geometry_forward", lambda: regularizer.distortions(batch))
        if hasattr(trainer, "_active_geometry"):
            host = stage("metrics_transfer", lambda: trainer._host_values({"vae_loss": base, **values}))
            worst = trainer._active_geometry(values, host)
            dual_value = host["worst_distortion"]
        else:
            worst, dual_value = values["worst_distortion"], values["worst_distortion"]
        loss = base + regularizer.primal_loss(worst)
        stage("backward", loss.backward)
        stage("optimizer", trainer.optimizer.step)
        stage("dual_update", lambda: regularizer.update_dual(dual_value))
        return loss

    model.train()
    for _ in range(args.warmup):
        step()
    stages = defaultdict(float)
    for _ in range(args.steps):
        step(stages)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as profiler:
        step()
    table = profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=25)
    trace_path = args.output / f"{label}_{dataset}_B{batch_size}_operators.txt"
    trace_path.write_text(table)

    # Restore the same starting model/RNG for actual full-epoch comparisons.
    # Warmup timings above deliberately exercise nonzero LoRA adapters.
    utils.set_seed(config["seed"])
    source_model = utils.create_model(config).to(device)
    source_model.load_state_dict(checkpoint["model_state_dict"])
    source_manager = LoRAManager(source_model, config)
    source_manager.inject_adapters()
    trainer.model.load_state_dict(source_model.state_dict())
    trainer.optimizer = utils.create_optimizer(config, manager.trainable_parameters())
    regularizer.dual_fraction = 0.
    trainer.geometry_sampler.generator.manual_seed(config["seed"] + 27001)
    epochs = []
    for epoch in range(args.epochs):
        train_time, train_result = timed(trainer.train_epoch, device)
        validation_time, validation_result = timed(trainer.validate_epoch, device)
        record = {"train_seconds": train_time, "validation_seconds": validation_time,
                  "train_loss": train_result[0], "validation_loss": validation_result[0],
                  "dual_fraction": regularizer.dual_fraction}
        epochs.append(record)
        print(json.dumps({"implementation": label, "dataset": dataset, "B": batch_size, "epoch": epoch + 1, **record}), flush=True)
    return {"dataset": dataset, "geometry_batch_size": batch_size, "implementation": label,
            "calibration_seconds": calibration_time, "source_tolerance": regularizer.source_tolerance,
            "sampling": sampling, "stage_ms": {key: 1000 * value / args.steps for key, value in stages.items()},
            "epochs": epochs, "operator_profile": str(trace_path)}


def audit_equivalence(reference, args):
    """Compare actual checkpoint models/Adam states after coupled noisy steps."""
    rows = []
    for dataset in args.datasets:
        config = utils.load_config(f"config/lora_vae_{dataset}_target_geometry.yaml")
        checkpoint = torch.load(config["pretrained_model"], map_location="cpu", weights_only=True)
        _, source_validation = create_train_validation_datasets(checkpoint["config"], config["dataset"]["name"])
        train_loader, validation_loader = utils.create_training_dataloaders(config, config["lora"]["transformed_dataset"])
        count = min(len(train_loader.dataset), args.audit_steps * config["dataloader"]["batch_size"])
        for batch_size in args.batch_sizes:
            trainers = []
            for geometry, training in (reference, (target_geometry, target_geometry_trainer)):
                utils.set_seed(config["seed"])
                model = utils.create_model(config).to(args.device)
                model.load_state_dict(checkpoint["model_state_dict"])
                manager = LoRAManager(model, config)
                manager.inject_adapters()
                regularizer = geometry.TargetGeometryRegularizer(model,
                    geometry.GeometryBatchSampler(source_validation, batch_size, config["seed"] + 17001),
                    calibration_batches=args.calibration_batches,
                    eta=config["lora"]["regularizer"]["eta"],
                    dual_learning_rate=config["lora"]["regularizer"]["dual_learning_rate"],
                    view=config["lora"]["regularizer"]["view"],
                    energy_floor=config["lora"]["regularizer"]["energy_floor"])
                trainers.append(training.TargetGeometryTrainer(model=model,
                    train_dataloader=DataLoader(Subset(train_loader.dataset, range(count)),
                        batch_size=config["dataloader"]["batch_size"], shuffle=True),
                    validate_dataloader=validation_loader,
                    optimizer=utils.create_optimizer(config, manager.trainable_parameters()),
                    lr_scheduler=None, lr_scheduler_start_epoch=0, loss=utils.create_loss(checkpoint["config"]),
                    epochs=1, device=args.device, regularizer=regularizer, regularizer_eta=regularizer.eta,
                    geometry_batch_size=batch_size, validation_batches=args.validation_batches,
                    geometry_seed=config["seed"] + 27001))
            before, after = trainers
            rng = torch.random.get_rng_state()
            accelerator_rng = None
            if torch.device(args.device).type == "cuda":
                accelerator_rng = torch.cuda.get_rng_state_all()
            elif torch.device(args.device).type == "mps":
                accelerator_rng = torch.mps.get_rng_state()
            expected, expected_components = before.train_epoch()
            torch.random.set_rng_state(rng)
            if torch.device(args.device).type == "cuda":
                torch.cuda.set_rng_state_all(accelerator_rng)
            elif torch.device(args.device).type == "mps":
                torch.mps.set_rng_state(accelerator_rng)
            actual, actual_components = after.train_epoch()
            errors = {"parameters": 0., "gradients": 0., "adam_state": 0.}
            for expected_parameter, actual_parameter in zip(before.model.parameters(), after.model.parameters()):
                for key, left, right in (("parameters", expected_parameter, actual_parameter),
                                         ("gradients", expected_parameter.grad, actual_parameter.grad)):
                    if left is None or right is None:
                        assert left is right, "Gradient presence differs."
                        continue
                    errors[key] = max(errors[key], (left.detach() - right.detach()).abs().max().item())
                    torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-7)
                for key, left in before.optimizer.state[expected_parameter].items():
                    right = after.optimizer.state[actual_parameter][key]
                    errors["adam_state"] = max(errors["adam_state"], (left - right).abs().max().item())
                    torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-7)
            assert before.regularizer.source_tolerance == after.regularizer.source_tolerance
            assert before.regularizer.dual_fraction == after.regularizer.dual_fraction
            torch.testing.assert_close(before.geometry_sampler.generator.get_state(), after.geometry_sampler.generator.get_state())
            row = {"dataset": dataset, "geometry_batch_size": batch_size,
                   "optimizer_steps": len(before.train_dataloader), "max_abs_errors": errors,
                   "loss_abs_error": abs(actual - expected),
                   "metric_max_abs_error": max(abs(actual_components[key] - expected_components[key]) for key in expected_components),
                   "source_tolerance_exact": True, "dual_exact": True, "sampler_state_exact": True}
            rows.append(row)
            print(json.dumps(row), flush=True)
    (args.output / "equivalence.json").write_text(json.dumps(rows, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-ref", help="Also benchmark the regularizer/trainer at this Git ref.")
    parser.add_argument("--datasets", nargs="+", choices=("mnist", "ellipsis"), default=["mnist", "ellipsis"])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[64, 128])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--calibration-batches", type=int, default=128)
    parser.add_argument("--validation-batches", type=int, default=64)
    parser.add_argument("--audit-only", action="store_true", help="Compare real-model weights, gradients and Adam state instead of timing.")
    parser.add_argument("--audit-steps", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.threads < 1 or args.steps < 1 or args.warmup < 0 or args.epochs < 0 or args.audit_steps < 1:
        parser.error("threads/steps/audit-steps must be positive; warmup/epochs must be nonnegative.")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    metadata = {"torch": torch.__version__, "python": platform.python_version(), "platform": platform.platform(),
                "device": args.device, "threads": args.threads, "reference_ref": args.reference_ref,
                "steps": args.steps, "warmup": args.warmup, "epochs": args.epochs,
                "calibration_batches": args.calibration_batches, "validation_batches": args.validation_batches}
    results = []
    with tempfile.TemporaryDirectory(prefix="target-geometry-profile-") as temporary:
        if args.audit_only:
            if not args.reference_ref:
                parser.error("--audit-only requires --reference-ref.")
            audit_equivalence(load_reference(args.reference_ref, Path(temporary)), args)
            return
        implementations = []
        if args.reference_ref:
            implementations.append(("reference", *load_reference(args.reference_ref, Path(temporary))))
        implementations.append(("current", target_geometry, target_geometry_trainer))
        for dataset in args.datasets:
            for batch_size in args.batch_sizes:
                for label, geometry, training in implementations:
                    print(f"Profiling {label}: {dataset}, B={batch_size}", flush=True)
                    results.append(profile_case(dataset, batch_size, geometry, training, args, label))
                    (args.output / "timings.json").write_text(json.dumps({"metadata": metadata, "results": results}, indent=2))


if __name__ == "__main__":
    main()
