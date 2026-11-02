"""Execution caches used only by the target-geometry experiment path."""

from collections import OrderedDict
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
import fcntl
import json
import os
import tempfile

import torch
from torch.utils.data import Subset, TensorDataset

from domain_knowledge_analysis.regularizers.target_geometry import GeometryBatchSampler


# Bounded per worker: keep MNIST/ellipse image banks across sweep runs without
# retaining models, graphs, optimizers or an unbounded number of CUDA tensors.
_RESIDENT = OrderedDict()
_RESIDENT_BYTES = 512 * 1024**2


def _tensor_digest(tensor):
    tensor = tensor.detach().cpu().contiguous()
    digest = sha256(f"{tensor.dtype}:{tuple(tensor.shape)}".encode())
    digest.update(memoryview(tensor.view(torch.uint8).numpy()))
    return digest.hexdigest()


def resident_tensor_dataset(dataset, device):
    """Transfer known immutable image storage once; preserve arbitrary transforms.

    Only exact TensorDataset/Subset and the repository's plain MNIST ToTensor
    path qualify. No stochastic or overridden __getitem__ is materialized.
    The returned digest identifies the actual selected images, not their path.
    """
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    images, indices = GeometryBatchSampler._tensor_storage(dataset)
    normalize_bytes = False
    if images is None:
        from torchvision.datasets import MNIST
        from torchvision.transforms import ToTensor

        base, indices = dataset, None
        while type(base) is Subset:
            subset = torch.as_tensor(base.indices, dtype=torch.long)
            indices = subset if indices is None else subset[indices]
            base = base.dataset
        if (type(base) is not MNIST or type(base.transform) is not ToTensor
                or base.target_transform is not None):
            return dataset, None
        images = base.data.unsqueeze(1)
        normalize_bytes = True
        if indices is not None:
            indices = torch.where(indices < 0, indices + len(base), indices)
    if indices is not None:
        images = images.index_select(0, indices.to(images.device))
    digest = _tensor_digest(images)
    key = (digest, str(device), torch.get_default_dtype() if normalize_bytes else images.dtype)
    if key in _RESIDENT:
        _RESIDENT.move_to_end(key)
        return _RESIDENT[key], digest
    images = images.to(device)
    if normalize_bytes:
        # Exactly ToTensor's uint8 conversion/division, performed on the GPU.
        images = images.to(torch.get_default_dtype()).div_(255)
    result = TensorDataset(images)
    size = images.numel() * images.element_size()
    while _RESIDENT and sum(value.tensors[0].numel() * value.tensors[0].element_size()
                            for value in _RESIDENT.values()) + size > _RESIDENT_BYTES:
        _RESIDENT.popitem(last=False)
    if size <= _RESIDENT_BYTES:
        _RESIDENT[key] = result
    return result, digest


class TensorBatchLoader:
    """Batch GPU tensors directly, without per-image Python collation/workers.

    Shuffle uses a private generator on the storage device. It is uniform and
    keeps the final partial VAE batch; it never advances the geometry RNG.
    """

    def __init__(self, dataset, batch_size, shuffle, seed):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = torch.Generator(device=dataset.tensors[0].device).manual_seed(seed)

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        tensors = self.dataset.tensors
        order = (torch.randperm(len(self.dataset), device=tensors[0].device, generator=self.generator)
                 if self.shuffle else None)
        for start in range(0, len(self.dataset), self.batch_size):
            end = start + self.batch_size
            if order is None:
                yield tuple(tensor[start:end] for tensor in tensors)
            else:
                yield tuple(tensor.index_select(0, order[start:end]) for tensor in tensors)


def cuda_target_loaders(loaders, config, device):
    """Replace only eligible loaders; unknown dataset transforms keep their path."""
    result = []
    for index, loader in enumerate(loaders):
        dataset, digest = resident_tensor_dataset(loader.dataset, device)
        if digest is None:
            result.append(loader)
        else:
            shuffle = config["dataloader"]["shuffle_train" if index == 0 else "shuffle_validation"]
            result.append(TensorBatchLoader(dataset, loader.batch_size, shuffle, config["seed"] + 37001 + index))
    return tuple(result)


def calibration_cache_key(experiment, source_digest, batch_size, config):
    """Exclude eta, view and dual rate: none participates in source calibration."""
    checkpoint = experiment.pretrained_model_path
    digest = sha256()
    with Path(checkpoint).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    device = torch.device(experiment.device)
    identity = {
        "version": 1, "checkpoint": digest.hexdigest(), "images": source_digest,
        "source_config": experiment.source_config,
        "seed": experiment.config["seed"] + 17001, "batch_size": batch_size,
        "calibration_batches": config.get("calibration_batches", 128),
        "energy_floor": config.get("energy_floor", 1e-12),
        "distribution": experiment.model.decoder_distribution_name,
        "dtype": str(next(experiment.model.parameters()).dtype),
        "torch": str(torch.__version__), "device": str(device),
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
    }
    if device.type == "cuda":
        identity["gpu"] = torch.cuda.get_device_name(device)
        identity["capability"] = torch.cuda.get_device_capability(device)
    # Editing either the calibration formula or deterministic image conversion
    # invalidates the disk cache, even if a version bump is forgotten.
    from domain_knowledge_analysis.regularizers import target_geometry
    identity["implementation"] = sha256(Path(target_geometry.__file__).read_bytes()
                                          + Path(__file__).read_bytes()).hexdigest()
    return sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()


@contextmanager
def calibration_cache(directory, key):
    """Serialize same-key calibration across ablation workers; publish atomically."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{key}.json"
    with (directory / f"{key}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cached = json.loads(path.read_text()) if path.exists() else None
        result = {"calibration": cached}
        yield result
        if cached is None and result["calibration"] is not None:
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", dir=directory, delete=False) as stream:
                    temporary = stream.name
                    json.dump(result["calibration"], stream, allow_nan=False)
                os.replace(temporary, path)
            finally:
                if temporary is not None and os.path.exists(temporary):
                    os.unlink(temporary)
