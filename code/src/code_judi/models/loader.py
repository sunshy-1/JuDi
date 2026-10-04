"""Model and tokenizer loading for arbitrary target/draft combinations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LoadedModels:
    target: Any
    draft: Any
    tokenizer: Any


def parse_dtype(value: str | torch.dtype | None) -> str | torch.dtype:
    if value is None or value == "auto":
        return "auto"
    if isinstance(value, torch.dtype):
        return value
    aliases = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if value not in aliases:
        raise ValueError(f"Unsupported torch dtype: {value}")
    return aliases[value]


def _device_map(value: Optional[str]) -> Optional[str]:
    if value is None or value.lower() in {"none", "single"}:
        return None
    return value


def _load_model(path: str, dtype: str | torch.dtype, device_map: Optional[str], trust_remote_code: bool) -> Any:
    kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": trust_remote_code,
    }
    mapped = _device_map(device_map)
    placement = mapped if mapped in {"auto", "balanced", "balanced_low_0", "sequential"} else None
    if placement is not None:
        kwargs["device_map"] = placement
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    if mapped is not None and placement is None:
        model.to(mapped)
    model.eval()
    return model


def model_device(model: Any, fallback: str = "cuda:0" if torch.cuda.is_available() else "cpu") -> torch.device:
    device = getattr(model, "device", None)
    if device is not None and device.type != "meta":
        return torch.device(device)
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device(fallback)


def load_models(
    target_path: str,
    draft_path: str,
    *,
    dtype: str | torch.dtype = "auto",
    target_device_map: Optional[str] = "auto",
    draft_device_map: Optional[str] = "cuda:0",
    trust_remote_code: bool = True,
) -> LoadedModels:
    parsed_dtype = parse_dtype(dtype)
    target = _load_model(target_path, parsed_dtype, target_device_map, trust_remote_code)
    draft = _load_model(draft_path, parsed_dtype, draft_device_map, trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(target_path, padding_side="left", trust_remote_code=trust_remote_code)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    for model in (target, draft):
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id
    return LoadedModels(target=target, draft=draft, tokenizer=tokenizer)
