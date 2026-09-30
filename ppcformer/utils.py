"""Checkpoint I/O, initialization, reproducibility, and full-image evaluation."""

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import tempfile
import warnings

import numpy as np
import torch
from torch import nn

from .metrics import metric_suite


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def atomic_output(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=str(path.parent))
    os.close(fd)
    try:
        yield Path(temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, payload):
    with atomic_output(path) as temporary:
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def save_checkpoint(path, payload):
    with atomic_output(path) as temporary:
        torch.save(payload, temporary)


def runtime_info(device):
    device = torch.device(device)
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
    }


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def load_payload(path, trusted=False):
    """Never silently fall back to unrestricted pickle deserialization."""
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    version = re.match(r"(\d+)\.(\d+)", str(torch.__version__))
    if not trusted and (version is None or tuple(map(int, version.groups())) < (2, 6)):
        raise RuntimeError("Checkpoint tools require PyTorch >= 2.6; please upgrade PyTorch.")
    if trusted:
        warnings.warn("Loading a trusted checkpoint with Python pickle; use only files you trust.")
    try:
        return torch.load(path, map_location="cpu", weights_only=not trusted)
    except Exception as error:
        if not trusted:
            raise RuntimeError(
                "Restricted checkpoint loading failed. Use a weights-only checkpoint, or use "
                "--trusted-checkpoint ONLY for your own trusted training checkpoint."
            ) from error
        raise


def load_weights(model, payload, dataset=None):
    if not isinstance(payload, Mapping):
        raise TypeError("Checkpoint must contain a state_dict or be a tensor state_dict")
    if dataset is not None and payload.get("dataset", dataset) != dataset:
        raise ValueError("Checkpoint dataset does not match the requested dataset")
    state = payload.get("state_dict", payload)
    if not isinstance(state, Mapping) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise TypeError("Invalid tensor state_dict")
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    expected = model.state_dict()
    # Loading wavelength-dependent buffers from a different sensor would silently
    # replace the requested geometry. Reject that before loading any parameters.
    for name, value in expected.items():
        if name in {"pattern", "phase_to_band", "phase_descriptors", "WB_Conv.weight"} or name.endswith((".phase_descriptors", ".relation_features")):
            if name not in state or not torch.equal(state[name].cpu(), value.cpu()):
                raise ValueError("Checkpoint sensor geometry or fixed WB filter mismatch: " + name)
    model.load_state_dict(state, strict=True)
    return model


def weight_payload(model, dataset, source=None):
    payload = {
        "format_version": 1,
        "model": "PPCFormer",
        "dataset": dataset,
        "model_config": asdict(model.config),
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
    }
    if source is not None:
        payload["source_sha256"] = source
    return payload


def initialize_model(model, seed):
    if any(p.is_cuda for p in model.parameters()):
        raise ValueError("Initialize the model before moving it to CUDA")
    for name, module in model.named_modules():
        parameters = tuple(module.parameters(recurse=False))
        if not parameters or not any(p.requires_grad for p in parameters):
            continue
        if not hasattr(module, "reset_parameters"):
            continue
        digest = hashlib.sha256(("%d:%s" % (seed, name)).encode("utf-8")).digest()
        module_seed = int.from_bytes(digest[:8], "little") & 0x7FFFFFFFFFFFFFFF
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(module_seed)
            module.reset_parameters()
    for module in (model.reconstruction_head, model.cell_context,
                   model.mosaic_stem_prior, model.measurement_refinement):
        module.zero_initialize_output()
    for block in model.blocks:
        nn.init.zeros_(block.attention.relation_mlp[-1].weight)
    return model


def validation_schedule(epochs):
    """Monitor every 20%; select best PSNR from the last 40%, sampled every 4%."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    early = max(1, math.ceil(epochs / 5))
    start = max(1, math.ceil(3 * epochs / 5))
    late = max(1, math.ceil(epochs / 25))
    return early, start, late


def should_validate(epoch, epochs):
    early, start, late = validation_schedule(epochs)
    return epoch == epochs or epoch % early == 0 or (epoch >= start and (epoch - start) % late == 0)


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(_):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def capture_rng(generator):
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [state[0], state[1].tolist(), int(state[2]), int(state[3]), float(state[4])],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader": generator.get_state(),
    }


def restore_rng(state, generator):
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Exact RNG restoration requires the same CUDA device count")
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader"])


@torch.inference_mode()
def evaluate(model, loader, device, on_prediction=None):
    model.eval()
    metrics = metric_suite()
    rows = []
    for index, (raw, sparse, target) in enumerate(loader):
        if raw.shape[0] != 1:
            raise ValueError("Evaluate full images individually with batch_size=1")
        raw, sparse, target = (x.to(device, non_blocking=True) for x in (raw, sparse, target))
        prediction = model(raw, sparse)
        row = {name: float(metric(prediction, target)) for name, metric in metrics.items()}
        row["file"] = loader.dataset.files[index].name
        rows.append(row)
        if on_prediction is not None:
            on_prediction(row, prediction)
    if not rows:
        raise ValueError("Evaluation split is empty")
    mean = {name: sum(row[name] for row in rows) / len(rows) for name in metrics}
    return mean, rows
