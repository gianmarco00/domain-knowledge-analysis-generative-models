"""Offline numerical investigation; never used by the training entry points.

Loads a failure snapshot with weights_only=True. Optional probes run on copies;
they do not modify the supplied snapshot, configs or production optimizer.
"""

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import struct
from types import SimpleNamespace

import torch

from domain_knowledge_analysis import utils
from domain_knowledge_analysis.adapters import LoRAManager
from domain_knowledge_analysis.math.gaussian import kl_divergence
from domain_knowledge_analysis.regularizers.target_geometry import TargetGeometryRegularizer


@torch.no_grad()
def adam_step_with_rms_state(optimizer):
    """Research prototype: identical Adam equations, storing sqrt(v) via hypot.

    Only the real-valued, unweighted Adam settings in this captured run are
    supported. No clipping, loss scaling, damping, or extra hyperparameter.
    The different state key is intentional: materializing v can itself overflow.
    """
    for group in optimizer.param_groups:
        if (group.get("weight_decay", 0) or group.get("amsgrad", False)
                or group.get("maximize", False) or group.get("differentiable", False)):
            raise ValueError("This diagnostic prototype supports ordinary unweighted Adam only.")
        beta1, beta2 = group["betas"]
        for parameter in group["params"]:
            gradient = parameter.grad
            if gradient is None:
                continue
            if parameter.is_complex() or gradient.is_sparse:
                raise ValueError("This diagnostic prototype requires dense real gradients.")
            state = optimizer.state[parameter]
            if not state:
                state["step"] = torch.tensor(0.)
                state["exp_avg"] = torch.zeros_like(parameter)
                state["exp_avg_sq_sqrt"] = torch.zeros_like(parameter)
            if "exp_avg_sq_sqrt" not in state:
                state["exp_avg_sq_sqrt"] = state.pop("exp_avg_sq").sqrt()
            state["step"].add_(1)
            step = float(state["step"])
            mean, rms = state["exp_avg"], state["exp_avg_sq_sqrt"]
            mean.mul_(beta1).add_(gradient, alpha=1 - beta1)
            torch.hypot(rms * math.sqrt(beta2), gradient * math.sqrt(1 - beta2), out=rms)
            denominator = rms / math.sqrt(1 - beta2**step) + group["eps"]
            parameter.addcdiv_(mean, denominator, value=-group["lr"] / (1 - beta1**step))


def build_model(snapshot, dtype, device="cpu"):
    model = utils.create_model(snapshot["config"])
    adapters = LoRAManager(model, snapshot["config"])
    adapters.inject_adapters()
    model.load_state_dict(snapshot["model_state_dict"])
    model.to(device=device, dtype=dtype)
    return model, adapters


def optimizer_for(snapshot, adapters):
    optimizer = utils.create_optimizer(snapshot["config"], adapters.trainable_parameters())
    optimizer.load_state_dict(deepcopy(snapshot["optimizer_state_dict"]))
    return optimizer


def finite_optimizer(optimizer):
    return all(bool(torch.isfinite(value).all()) for state in optimizer.state.values()
               for value in state.values() if isinstance(value, torch.Tensor))


@torch.no_grad()
def reverse_saved_adam_step(optimizer, preceding=False):
    """Approximately invert the last update; FP32 rounding is not invertible.

    The preceding=True estimate additionally assumes a dominant gradient spike
    followed by a negligible gradient, an assumption tested against the actual
    moment ratios and recovered input batches in recover_history().
    """
    for group in optimizer.param_groups:
        if group.get("weight_decay", 0) or group.get("amsgrad", False):
            raise ValueError("Update reversal only supports this run's unweighted Adam.")
        beta1, beta2 = group["betas"]
        for parameter in group["params"]:
            state = optimizer.state[parameter]
            step = float(state["step"]) - int(preceding)
            mean = state["exp_avg"] / beta1 if preceding else state["exp_avg"]
            variance = state["exp_avg_sq"] / beta2 if preceding else state["exp_avg_sq"]
            denominator = variance.sqrt() / math.sqrt(1 - beta2**step) + group["eps"]
            parameter.addcdiv_(mean, denominator, value=group["lr"] / (1 - beta1**step))


def recover_history(snapshot):
    """Recover the shuffle seed and verify all pixels of the saved training batch.

    The current snapshot has a 5056-byte CPU MT19937 state. Rewinding the two
    words used by RandomSampler's seed draw is valid away from a state twist.
    Exact RNG and image comparisons reject incompatible formats/dataset orders.
    """
    from domain_knowledge_analysis.utils.dataset_utils import (
        create_train_validation_datasets, resolve_target_dataset,
    )

    saved_rng = snapshot["pre_forward_rng_state"]["cpu"]
    if saved_rng.numel() != 5056:
        raise ValueError("Unsupported CPU RNG layout for shuffle recovery.")
    raw = bytearray(saved_rng.tolist())
    _, left, _, next_word = struct.unpack_from("<QiiQ", raw)
    if next_word < 2:
        raise ValueError("Seed recovery crosses a CPU RNG twist boundary.")
    struct.pack_into("<i", raw, 8, left + 2)
    struct.pack_into("<Q", raw, 16, next_word - 2)
    generator = torch.Generator()
    generator.set_state(torch.tensor(list(raw), dtype=torch.uint8))
    seed = int(torch.empty((), dtype=torch.int64).random_(generator=generator))
    if not torch.equal(generator.get_state(), saved_rng):
        raise ValueError("Recovered seed draw does not reproduce the saved RNG.")
    config = snapshot["config"]
    target_config, name, _ = resolve_target_dataset(config, config["lora"]["transformed_dataset"])
    dataset, _ = create_train_validation_datasets(target_config, name)
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))
    size, batch = config["dataloader"]["batch_size"], snapshot["report"]["batch"]

    def images(offset):
        indices = order[(batch - 1 - offset) * size:(batch - offset) * size]
        return torch.stack([dataset[int(index)][0] for index in indices])

    if not torch.equal(images(0), snapshot["training_images"]):
        raise ValueError("Recovered shuffle does not match every saved training pixel.")
    model, adapters = build_model(snapshot, torch.float64)
    optimizer = optimizer_for(snapshot, adapters)
    reverse_saved_adam_step(optimizer)
    with torch.no_grad():
        _, previous_logvar = model.encoder(images(1).double())
        _, fixed_probe_logvar = model.encoder(images(0).double())
    reverse_saved_adam_step(optimizer, preceding=True)
    mean, logvar = model.encoder(images(2).double())
    kl = config["loss"]["beta"] * kl_divergence(mean, logvar).mean()
    kl.backward()
    alignments, ages = {}, {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        state = optimizer.state[parameter]
        moment, variance = state["exp_avg"], state["exp_avg_sq"]
        if moment.abs().max() < 1e10:
            continue
        # One spike: m=.1*g, v=.001*g^2; one later decay gives ratio 8.108108.
        mask = moment.abs() > moment.abs().max() * 1e-4
        ratio = moment[mask].square() / variance[mask]
        ages[name] = float((torch.log(ratio / 10) / math.log(.9**2 / .999)).median())
        predicted = .09 * parameter.grad
        alignments[name] = {
            "cosine": float(torch.nn.functional.cosine_similarity(predicted.flatten(), moment.flatten(), dim=0)),
            "relative_error": float((predicted - moment).norm() / moment.norm()),
        }
    return {"shuffle_seed": seed, "rng_and_training_pixels_match_exactly": True,
            "approx_before_last_update_logvar_max_on_previous_batch": previous_logvar.max().item(),
            "approx_before_last_update_logvar_max_on_failing_batch": fixed_probe_logvar.max().item(),
            "approx_before_spike_logvar_max_on_spike_batch": logvar.max().item(),
            "approx_spike_weighted_KL_mean": kl.item(),
            "dominant_spike_age_estimates": ages, "predicted_moment_alignments": alignments}


def precision_probe(snapshot, device="cpu", controlled_updates=0):
    config = snapshot["config"]
    model, adapters = build_model(snapshot, torch.float64, device)
    images, geometry_images = (snapshot[key].to(device=device, dtype=torch.float64)
                               for key in ("training_images", "geometry_images"))
    regularizer_state = snapshot["regularizer"]
    calibration = regularizer_state["calibration"]
    regularizer = TargetGeometryRegularizer(
        model, SimpleNamespace(batch_size=calibration["geometry_batch_size"]),
        calibration_batches=calibration["calibration_batches"], calibration=calibration,
        **{key: regularizer_state[key] for key in ("eta", "dual_learning_rate", "view", "energy_floor")},
    )
    regularizer.source_decoder.load_state_dict(snapshot["source_decoder_state_dict"])
    regularizer.dual_fraction = regularizer_state["dual_fraction"]
    loss_function = utils.create_loss(config)
    generator = torch.Generator(device=device)
    if torch.device(device).type == "cuda":
        generator.set_state(snapshot["pre_forward_rng_state"]["cuda"])
        noise_note = "FP32 noise regenerated from captured CUDA RNG; kernel/version matching still matters."
    else:
        generator.manual_seed(config["seed"])
        noise_note = "Fixed CPU FP32 noise: a controlled probe, not the original CUDA draw."
    epsilon = torch.randn(snapshot["training_forward"]["mean"].shape, device=device,
                          dtype=torch.float32, generator=generator).double()

    def objective():
        mean, logvar = model.encoder(images)
        logits = model.decoder(mean + (.5 * logvar).exp() * epsilon)
        base = loss_function(images, logits, mean, logvar)
        values = regularizer.distortions(geometry_images)
        return base + regularizer.primal_loss(values["worst_distortion"]), logvar, values

    value, _, first_values = objective()
    value.backward()
    parameters = adapters.trainable_parameters()
    report = {"noise": noise_note, "FP64_primal": value.item(),
              "FP64_gradients_finite": all(bool(torch.isfinite(p.grad).all()) for p in parameters),
              "FP64_max_abs_gradient": max(p.grad.abs().max().item() for p in parameters)}
    ordinary_model, ordinary_adapters = build_model(snapshot, torch.float64, device)
    ordinary = optimizer_for(snapshot, ordinary_adapters)
    for parameter, reference in zip(ordinary_adapters.trainable_parameters(), parameters):
        parameter.grad = reference.grad.detach().clone()
    ordinary.step()
    report["ordinary_FP64_Adam_state_finite"] = finite_optimizer(ordinary)
    optimizer = optimizer_for(snapshot, adapters)
    adam_step_with_rms_state(optimizer)
    regularizer.update_dual(first_values["worst_distortion"].detach())
    with torch.no_grad():
        _, after = model.encoder(images)
    report["RMS_Adam_state_finite"] = finite_optimizer(optimizer)
    report["RMS_Adam_next_same_batch_logvar_max"] = after.max().item()
    report["controlled_same_batch_updates"] = []
    for step in range(controlled_updates):
        optimizer.zero_grad()
        value, logvar, values = objective()
        value.backward()
        finite = bool(torch.isfinite(value)) and all(bool(torch.isfinite(p.grad).all()) for p in parameters)
        row = {"step": step + 1, "objective": value.item(), "max_logvar": logvar.max().item(),
               "finite_loss_and_gradients": finite}
        report["controlled_same_batch_updates"].append(row)
        if not finite:
            break
        adam_step_with_rms_state(optimizer)
        regularizer.update_dual(values["worst_distortion"].detach())
        row["finite_optimizer_state"] = finite_optimizer(optimizer)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--recover-history", action="store_true")
    parser.add_argument("--precision-probe", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--controlled-updates", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    snapshot = torch.load(args.snapshot, map_location="cpu", weights_only=True)
    report = {"captured_failure": snapshot["report"]}
    if args.recover_history:
        report["reconstructed_history"] = recover_history(snapshot)
    if args.precision_probe:
        report["precision_probe"] = precision_probe(snapshot, args.device, args.controlled_updates)
    result = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(result + "\n")
    else:
        print(result)


if __name__ == "__main__":
    main()
