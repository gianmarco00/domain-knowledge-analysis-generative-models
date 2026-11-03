"""Numerical provenance without changing the target-geometry updates."""

from copy import deepcopy
import traceback
import warnings

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from domain_knowledge_analysis.regularizers.target_geometry import centered_gram, image_mean


def tensor_summary(value):
    value = value.detach()
    finite = torch.isfinite(value)
    usable = value[finite]
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "nan_count": int(torch.isnan(value).sum()),
            "inf_count": int(torch.isinf(value).sum()),
            "zero_count": int((value == 0).sum()),
            "finite_min": float(usable.min()) if usable.numel() else None,
            "finite_max": float(usable.max()) if usable.numel() else None}


def invalid_named_tensors(tensors):
    """One synchronization per device; inspect names only on failure.

    Adam can keep step counters on CPU while its moments are on CUDA.
    """
    groups, invalid = {}, []
    for name, value in tensors:
        if value is not None:
            groups.setdefault(value.device, []).append((name, value))
    for tensors in groups.values():
        flags = torch.stack([torch.isfinite(value.detach()).all() for _, value in tensors])
        if not bool(flags.all()):
            invalid.extend(name for (name, _), valid in zip(tensors, flags.cpu().tolist()) if not valid)
    return invalid


def capture_rng(device):
    device = torch.device(device)
    index = (device.index if device.index is not None else torch.cuda.current_device()) if device.type == "cuda" else None
    return {"cpu": torch.random.get_rng_state(), "cuda_device": index,
            "cuda": torch.cuda.get_rng_state(index) if index is not None else None}


class OptimizerStepSnapshot:
    """Reusable device buffers for the last pre-Adam parameters and moments.

    No CPU transfer of CUDA parameters/state, no optimizer mutation, no RNG use.
    Keeping the inputs to the update lets a failed optimizer step be traced too.
    """

    def __init__(self):
        self.entries = []
        self.groups = []
        self.signature = None

    @torch.no_grad()
    def capture(self, optimizer):
        signature = (id(optimizer), len(optimizer.state))
        if signature != self.signature:
            self.entries, groups = [], {}
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    state = optimizer.state.get(parameter, {})
                    buffer = parameter.detach().clone()
                    saved_state = {key: value.detach().clone() if isinstance(value, torch.Tensor) else deepcopy(value)
                                   for key, value in state.items()}
                    self.entries.append((parameter, buffer, saved_state))
                    pairs = [(parameter, buffer)] + [(state[key], saved_state[key]) for key in state
                                                     if isinstance(state[key], torch.Tensor)]
                    for value, saved in pairs:
                        sources, destinations = groups.setdefault((value.device, value.dtype), ([], []))
                        sources.append(value)
                        destinations.append(saved)
            self.groups = list(groups.values())
            self.signature = signature
        for sources, destinations in self.groups:
            torch._foreach_copy_(destinations, sources)
        for parameter, _, state in self.entries:
            for key, value in optimizer.state.get(parameter, {}).items():
                if not isinstance(value, torch.Tensor):
                    state[key] = deepcopy(value)

    def payload(self, names):
        names = {id(parameter): name for name, parameter in names}
        return {names.get(id(parameter), str(index)): {"parameter": value, "optimizer_state": state}
                for index, (parameter, value, state) in enumerate(self.entries)}

    def replay_optimizer(self, optimizer):
        replay = deepcopy(optimizer)
        copied = [parameter for group in replay.param_groups for parameter in group["params"]]
        with torch.no_grad():
            for (original, value, state), parameter in zip(self.entries, copied):
                parameter.copy_(value)
                parameter.grad = original.grad.detach().clone() if original.grad is not None else None
                replay.state[parameter] = deepcopy(state)
        return replay


class _FirstNonfiniteOperation(RuntimeError):
    pass


def _tensor_leaves(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _tensor_leaves(child)
    elif isinstance(value, dict):
        for child in value.values():
            yield from _tensor_leaves(child)


class _NonfiniteTrace(TorchDispatchMode):
    """ATen-level checking, including backward divisions; failure replay only."""

    def __init__(self):
        super().__init__()
        self.phase = "unknown"
        self.finding = None

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        inputs = [value for value in _tensor_leaves((args, kwargs)) if value.is_floating_point()]
        # In-place Adam operations overwrite their inputs. Preserve those values
        # before executing, so a failed addcdiv_ does not misreport its NaN result
        # as an already-NaN input. Copies occur in the failure replay only.
        originals = {id(value): value.detach().clone() for value in inputs} if func._schema.is_mutable else {}
        output = func(*args, **(kwargs or {}))
        # Uninitialized allocation contents are not computation results.
        if func._schema.name in {"aten::empty", "aten::empty_like", "aten::empty_strided",
                                 "aten::new_empty", "aten::new_empty_strided", "aten::resize_"}:
            return output
        values = [value for value in _tensor_leaves(output) if value.is_floating_point()]
        if any(not torch.isfinite(value).all() for value in values):
            self.finding = {"status": "first_nonfinite_operation", "phase": self.phase,
                            "operator": str(func),
                            "inputs": [tensor_summary(originals.get(id(value), value)) for value in inputs],
                            "scalar_args": {str(index): value for index, value in enumerate(args)
                                            if isinstance(value, (int, float, bool, str))},
                            "outputs": [tensor_summary(value) for value in values],
                            "python_stack": traceback.format_stack(limit=16)}
            raise _FirstNonfiniteOperation(str(func))
        return output


def trace_failure(callback, rng):
    """Trace a copied computation; preserve live RNG/state and catch Inf as well as NaN."""
    trace = _NonfiniteTrace()
    devices = [rng["cuda_device"]] if rng["cuda_device"] is not None else []
    with torch.random.fork_rng(devices=devices), warnings.catch_warnings(record=True) as caught:
        torch.random.set_rng_state(rng["cpu"])
        if devices:
            torch.cuda.set_rng_state(rng["cuda"], devices[0])
        try:
            with torch.enable_grad(), torch.autograd.set_detect_anomaly(True, check_nan=False), trace:
                callback(trace)
            report = {"status": "replay_finite", "detail": "The bad operation did not reproduce on this replay."}
        except _FirstNonfiniteOperation:
            report = trace.finding
        except Exception as error:
            report = {"status": "replay_error", "error": f"{type(error).__name__}: {error}"}
        report["autograd_warnings"] = [str(item.message) for item in caught]
    return report


def trace_primal_failure(regularizer, loss, images, geometry_images, rng):
    """Use the exact pre-forward RNG and copied modules; never touch live gradients."""
    try:
        replay, replay_loss = deepcopy(regularizer), deepcopy(loss)
    except Exception as error:
        return {"status": "replay_error", "error": f"Cannot copy computation: {error}"}

    def compute(trace):
        replay.model.train()
        trace.phase = "vae_forward"
        base = replay_loss(images, *replay.model(images))
        trace.phase = "geometry_forward"
        with torch.set_grad_enabled(replay.multiplier != 0):
            values = replay.distortions(geometry_images, check_finite=False)
        worst = values["worst_distortion"]
        if "target_distortion" in values and "source_chart_distortion" in values:
            target, source = values["target_distortion"], values["source_chart_distortion"]
            if target != source:
                worst = target if target > source else source
        objective = base + replay.primal_loss(worst)
        trace.phase = "primal_backward"
        objective.backward()

    return trace_failure(compute, rng)


@torch.no_grad()
def diagnose_geometry(regularizer, images):
    """Replay only deterministic geometry, preserving model modes and all RNGs.

    Leaf hooks identify the first nonfinite activation in each executed view.
    This is run only after a failure, not in ordinary training iterations.
    """
    report, handles = {}, []
    modes = [(module, module.training) for module in regularizer.model.modules()]

    def watch(prefix, model):
        def hook(name):
            def inspect_output(_module, _inputs, output):
                values = output if isinstance(output, (tuple, list)) else (output,)
                for index, value in enumerate(values):
                    if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                        key = f"{prefix}_first_nonfinite_activation"
                        report.setdefault(key, {"module": name, "output_index": index, **tensor_summary(value)})
                        break
            return inspect_output
        for name, module in model.named_modules():
            if not list(module.children()):
                handles.append(module.register_forward_hook(hook(name or type(module).__name__)))

    try:
        regularizer.model.eval()
        report["input"] = tensor_summary(images)
        gram = centered_gram(images)
        report["input_gram_energy"] = float(gram.square().sum())
        watch("encoder", regularizer.model.encoder)
        codes, log_variance = regularizer.model.encoder(images)
        report["posterior_mean"] = tensor_summary(codes)
        report["posterior_log_variance"] = tensor_summary(log_variance)
        for name, decoder in (("target", regularizer.model.decoder), ("source_chart", regularizer.source_decoder)):
            if (name == "target" and regularizer.view == "source_only"
                    or name == "source_chart" and regularizer.view == "target_only"):
                continue
            watch(name, decoder)
            logits = decoder(codes)
            report[f"{name}_decoder_logits"] = tensor_summary(logits)
            # Reuse these exact logits instead of running a second decoder.
            outputs = image_mean(lambda _: logits, codes, regularizer.distribution)
            report[f"{name}_image_mean"] = tensor_summary(outputs)
            report[f"{name}_output_gram"] = tensor_summary(centered_gram(outputs))
    except (ValueError, RuntimeError) as error:
        report["diagnostic_forward_error"] = str(error)
    finally:
        for handle in handles:
            handle.remove()
        for module, training in modes:
            module.training = training
    return report
