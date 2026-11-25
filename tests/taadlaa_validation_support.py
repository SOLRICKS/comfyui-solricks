"""Load one local node package, using its own sibling modules and weights."""

import hashlib
import importlib
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch


def load_standalone(directory, tag):
    root = Path(directory).resolve()
    if not (root / "VideoTAADLAA.py").is_file():
        raise FileNotFoundError(f"Missing {root / 'VideoTAADLAA.py'}")
    # Avoid importing unrelated custom nodes from the package's __init__.py.
    package_name = f"_solricks_validation_{tag}"
    package = types.ModuleType(package_name)
    package.__path__ = [str(root)]
    sys.modules[package_name] = package
    # Use the node's own standalone fallback even in a ComfyUI Python install.
    # Do not initialize the ComfyUI server or select its VRAM policy.
    missing = object()
    comfy_keys = (
        "comfy",
        "comfy.model_management",
        "comfy.model_patcher",
        "comfy.utils",
    )
    previous = {key: sys.modules.get(key, missing) for key in comfy_keys}
    try:
        for key in comfy_keys:
            sys.modules[key] = None
        return importlib.import_module(f"{package_name}.VideoTAADLAA")
    finally:
        for key, value in previous.items():
            if value is missing:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


@contextmanager
def select_device(module, device):
    # Only the device selector is replaced. Kernels, inference, and transfers are real.
    with patch.object(
        module.VideoTAADLAA, "_get_device", return_value=torch.device(device)
    ):
        yield


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
