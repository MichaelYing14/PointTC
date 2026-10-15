"""Configuration, device selection and explicit checkpoint loading."""

import json
import random
from pathlib import Path

import numpy as np
import torch

from .models import symbol_split


MODEL_DEFAULTS = {
    "num_points": 2048,
    "num_symbols": 300,
    "qam_order": 64,
    "revise": True,
}


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("method") not in ("pointtc", "ca", "ca_ra"):
        raise ValueError("method must be pointtc, ca or ca_ra")
    if config.get("channel") not in ("awgn", "rayleigh"):
        raise ValueError("channel must be awgn or rayleigh")
    for key, value in MODEL_DEFAULTS.items():
        selected = config.get(key, value)
        if key == "num_symbols":
            symbol_split(selected)
        elif selected != value:
            raise ValueError(f"This release supports only {key}={value}")
        config[key] = selected
    return config


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(value):
    device = torch.device(value)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Use cpu or cuda[:index]")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use --device cpu or fix the GPU environment")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {device.index} is unavailable")
    return device


def load_checkpoint(model, path, method, channel, allow_ca_to_ra=False,
                    allow_channel_transfer=False):
    """Load release/training weights; never silently ignore incompatible keys.

    PointTC+CA -> PointTC+CA+RA is the only partial initialization allowed. Its new RA
    parameters retain their initial values; every other parameter must match.
    Accepts the public weight and checkpoints produced by train.py.
    Source and target must use the same symbol budget, including for finetuning.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError("Expected a release checkpoint containing model_state_dict and config")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint lacks model config; use the public weight or a train.py checkpoint")
    source_method, source_channel = config.get("method"), config.get("channel")
    partial = allow_ca_to_ra and source_method == "ca" and method == "ca_ra"
    if source_method != method and not partial:
        raise ValueError(f"Checkpoint method {source_method!r} does not match {method!r}")
    if source_channel != channel:
        if not (allow_channel_transfer and source_method == method
                and source_channel == "awgn" and channel == "rayleigh"):
            raise ValueError(f"Checkpoint channel {source_channel!r} does not match {channel!r}")
    for key, value in MODEL_DEFAULTS.items():
        expected_value = model.num_symbols if key == "num_symbols" else value
        if config.get(key) != expected_value:
            raise ValueError(f"Checkpoint must declare {key}={expected_value}; got {config.get(key)!r}")
    symbol_split(config["num_symbols"])
    state = payload["model_state_dict"]
    # Check compatibility before modifying model parameters.
    expected = model.state_dict()
    missing = set(expected) - set(state)
    unexpected = set(state) - set(expected)
    allowed_missing = {key for key in expected if key.startswith("rate_refine.")} if partial else set()
    if missing != allowed_missing or unexpected:
        raise ValueError(f"Incompatible state keys: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    for key, value in state.items():
        if not isinstance(value, torch.Tensor) or value.shape != expected[key].shape:
            raise ValueError(f"Incompatible checkpoint tensor: {key}")
    model.load_state_dict(state, strict=not partial)
    return payload


def training_loss(outputs, method, rate_weight=0.2, num_symbols=300):
    symbol_split(num_symbols)
    cd = outputs[1]
    symbols = outputs[2] if method == "ca_ra" else cd.new_tensor(float(num_symbols))
    loss = 1000.0 * cd
    if method == "ca_ra":
        loss = loss + rate_weight * symbols / float(num_symbols)
    return loss, cd, symbols
