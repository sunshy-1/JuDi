"""Loading of the classifier/scaler checkpoint used by the last-layer method."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Tuple


def _load(path: Path) -> Any:
    errors = []
    try:
        import joblib

        return joblib.load(path)
    except Exception as exc:  # pragma: no cover - fallback depends on checkpoint format
        errors.append(exc)
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except Exception as exc:  # pragma: no cover
        errors.append(exc)
    raise RuntimeError(f"Could not load checkpoint {path}: {errors[-1]}")


def load_autojudge_checkpoint(path: str | Path, feature_dim: int | None = None) -> Tuple[Any, Any]:
    payload = _load(Path(path))
    if isinstance(payload, dict) and "model" in payload:
        head, scaler = payload["model"], payload.get("scaler")
    elif isinstance(payload, (tuple, list)) and len(payload) == 2:
        head, scaler = payload
    else:
        raise ValueError(
            "AutoJudge checkpoint must contain {'model': ..., 'scaler': ...} "
            "or be a (model, scaler) pair"
        )
    if scaler is None:
        raise ValueError(f"Checkpoint {path} does not contain a scaler")
    checkpoint_dim = getattr(head, "n_features_in_", None)
    if checkpoint_dim is None and hasattr(head, "coef_"):
        checkpoint_dim = head.coef_.shape[-1]
    expected_dim = int(checkpoint_dim or feature_dim or len(scaler.mean_))
    if feature_dim is not None and expected_dim != feature_dim:
        raise ValueError(
            f"Checkpoint {path} expects {expected_dim} features, but the configured "
            f"target/draft models produce {feature_dim}. Use the checkpoint trained for this pair."
        )
    if hasattr(scaler, "mean_"):
        if scaler.mean_.shape[0] < expected_dim or scaler.scale_.shape[0] < expected_dim:
            raise ValueError(
                f"Checkpoint feature dimension {scaler.mean_.shape[0]} is smaller than "
                f"the expected feature dimension {expected_dim}"
            )
        scaler.mean_ = scaler.mean_[:expected_dim]
        scaler.scale_ = scaler.scale_[:expected_dim]
        if hasattr(scaler, "var_"):
            scaler.var_ = scaler.var_[:expected_dim]
        if hasattr(scaler, "n_features_in_"):
            scaler.n_features_in_ = expected_dim
    return head, scaler
