"""Activate only the vendored vLLM, without installing it into site-packages."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
BASE = PACKAGE.parent
SOURCE = BASE / "vendor" / "vllm"
RUNTIME = BASE / ".runtime"
_active = False


def activate():
    """Set import paths for this process and spawned workers; fail on fallback."""
    global _active
    if _active:
        return SOURCE
    existing = sys.modules.get("vllm")
    if existing is not None:
        origin = getattr(existing, "__file__", None)
        if origin is None or not Path(origin).resolve().is_relative_to(SOURCE):
            raise RuntimeError(
                "An external vLLM is already imported. Start a fresh process "
                "and import judi_vllm before vllm.")
    lock = json.loads((BASE / "UPSTREAM.json").read_text())
    manifest = RUNTIME / "manifest.json"
    if not manifest.is_file():
        raise RuntimeError("Prepare the local runtime: python -m judi_vllm.prepare")
    state = json.loads(manifest.read_text())
    if state.get("source_commit") != lock["commit"]:
        raise RuntimeError("Runtime/source version mismatch; rerun judi_vllm.prepare")
    for name in lock["wheel"]["generated_files"]:
        target = SOURCE / name
        if not target.is_file():
            raise RuntimeError(f"Missing {name}; rerun python -m judi_vllm.prepare")
        expected = state.get("generated_files", {}).get(name)
        if expected is not None:
            import hashlib
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            if digest != expected:
                raise RuntimeError(f"Generated vLLM file checksum mismatch: {name}")
    if not (RUNTIME / "vllm-0.8.3.dist-info" / "METADATA").is_file():
        raise RuntimeError("Missing local vLLM metadata; rerun judi_vllm.prepare")
    os.environ.setdefault("VLLM_USE_V1", "0")
    if os.environ["VLLM_USE_V1"] != "0":
        raise RuntimeError("JuDi 0.8.3 requires VLLM_USE_V1=0")
    # Set paths before importing vLLM, including its distribution metadata.
    # The project root is needed for the importable worker factory under spawn.
    paths = [str(SOURCE), str(RUNTIME), str(BASE), str(BASE.parent)]
    sys.path[:] = paths + [p for p in sys.path if p not in paths]
    inherited = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    os.environ["PYTHONPATH"] = os.pathsep.join(
        paths + [p for p in inherited if p and p not in paths])
    _active = True
    return SOURCE


def describe():
    """Load and check key Python/native origins, without allocating a model."""
    activate()
    import importlib
    import importlib.metadata
    import torch
    import vllm

    if vllm.__version__ != "0.8.3":
        raise RuntimeError(f"Unexpected vLLM version: {vllm.__version__}")
    origins = {}
    for name in ("vllm", "vllm.spec_decode.spec_decode_worker", "vllm._C",
                 "vllm.vllm_flash_attn._vllm_fa2_C"):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(SOURCE):
            raise RuntimeError(f"External vLLM module loaded: {name}: {path}")
        origins[name] = str(path)
    metadata = Path(importlib.metadata.distribution("vllm").locate_file(""))
    if metadata.resolve() != RUNTIME:
        raise RuntimeError(f"External vLLM metadata loaded: {metadata}")
    return {"vllm": vllm.__version__, "torch": torch.__version__,
            "cuda": torch.version.cuda, "metadata": str(metadata), "origins": origins}
