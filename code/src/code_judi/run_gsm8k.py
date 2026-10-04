"""Command-line GSM8K runner for both AutoJudge methods."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from code_judi.data import encode_gsm8k_sample, load_gsm8k
from code_judi.decoding import run_judi, run_autojudge
from code_judi.evaluation import evaluate_gsm8k_result, summarize_results
from code_judi.models import load_autojudge_checkpoint, load_models


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILES = PROJECT_ROOT / "configs" / "profiles.json"
DEFAULT_THRESHOLDS = PROJECT_ROOT / "configs" / "thresholds.json"


def _path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _load_profiles(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_thresholds(value: str | list[str]) -> list[float]:
    """Parse comma- or whitespace-separated threshold values."""
    raw = value if isinstance(value, list) else [value]
    values = [part for item in raw for part in item.replace(",", " ").split() if part]
    if not values:
        raise ValueError("--thresholds must contain at least one numeric value")
    try:
        return [float(part) for part in values]
    except ValueError as exc:
        raise ValueError(f"Invalid threshold list: {value!r}") from exc


def _load_default_thresholds(method: str, path: Path = DEFAULT_THRESHOLDS) -> list[float]:
    with path.open("r", encoding="utf-8") as handle:
        values = json.load(handle).get(method)
    if not values:
        raise ValueError(f"No default thresholds configured for method {method!r}")
    return [float(value) for value in values]


def _threshold_name(value: float) -> str:
    """Return a readable, filesystem-safe threshold suffix."""
    return str(value).replace("+", "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AutoJudge on GSM8K")
    parser.add_argument("--method", choices=("AutoJudge", "JuDi"), default="AutoJudge")
    parser.add_argument("--profile", default="llama3-8b-1b", help="Model combination in configs/profiles.json")
    parser.add_argument("--profiles", default=str(DEFAULT_PROFILES))
    parser.add_argument("--target-model")
    parser.add_argument("--draft-model")
    parser.add_argument("--data")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--target-device-map")
    parser.add_argument("--draft-device-map")
    parser.add_argument("--draft-length", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--threshold", type=float, help="Single classifier threshold for AutoJudge")
    parser.add_argument("--judi-threshold", type=float, help="Single KL threshold for JuDi")
    parser.add_argument(
        "--thresholds",
        nargs="+",
        help="Values to scan, separated by commas or spaces; defaults come from configs/thresholds.json",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.9, help="Fixed target confidence threshold (must be 0.9)")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _select_thresholds(args: argparse.Namespace, default_path: Path = DEFAULT_THRESHOLDS) -> list[float]:
    if args.confidence_threshold != 0.9:
        raise ValueError("The target confidence threshold is fixed at 0.9")
    explicit_method_threshold = args.threshold if args.method == "AutoJudge" else args.judi_threshold
    unrelated_threshold = args.judi_threshold if args.method == "AutoJudge" else args.threshold
    if unrelated_threshold is not None:
        option = "--judi-threshold" if args.method == "AutoJudge" else "--threshold"
        raise ValueError(f"{option} is only valid with --method {'JuDi' if args.method == 'AutoJudge' else 'AutoJudge'}")
    if args.thresholds is not None and explicit_method_threshold is not None:
        raise ValueError("Use either the single-threshold option or --thresholds, not both")
    if args.thresholds is not None:
        thresholds = _parse_thresholds(args.thresholds)
    elif explicit_method_threshold is not None:
        thresholds = [float(explicit_method_threshold)]
    else:
        thresholds = _load_default_thresholds(args.method, default_path)
    if not thresholds:
        raise ValueError("At least one threshold is required")
    return thresholds


def _config(args: argparse.Namespace) -> Dict[str, Any]:
    profiles = _load_profiles(_path(args.profiles))
    if args.profile not in profiles:
        raise KeyError(f"Unknown profile {args.profile!r}; available profiles: {', '.join(sorted(profiles))}")
    profile = dict(profiles[args.profile])
    profile["target_model"] = args.target_model or profile["target_model"]
    profile["draft_model"] = args.draft_model or profile["draft_model"]
    profile["data"] = args.data or profile.get("data", "models/datasets/gsm8k/test.jsonl")
    profile["checkpoint"] = args.checkpoint or profile.get("checkpoint")
    profile["output_dir"] = args.output_dir or profile.get("output_dir", f"outputs/gsm8k/{args.profile}/{args.method}")
    profile["target_device_map"] = args.target_device_map or profile.get("target_device_map", "auto")
    profile["draft_device_map"] = args.draft_device_map or profile.get("draft_device_map", "cuda:0")
    return profile


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _run_single_threshold(
    *,
    args: argparse.Namespace,
    config: Dict[str, Any],
    loaded: Any,
    selected: list[Any],
    output_dir: Path,
    threshold: float,
    head: Any,
    scaler: Any,
    end: int,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "method": args.method,
        "profile": args.profile,
        "target_model": config["target_model"],
        "draft_model": config["draft_model"],
        "data": config["data"],
        "checkpoint": config.get("checkpoint"),
        "draft_length": args.draft_length,
        "max_new_tokens": args.max_new_tokens,
        "classifier_threshold": threshold if args.method == "AutoJudge" else None,
        "judi_threshold": threshold if args.method == "JuDi" else None,
        "threshold_type": "classifier" if args.method == "AutoJudge" else "judi",
        "confidence_threshold": 0.9,
        "start": args.start,
        "end": end,
    }
    _write_json(output_dir / "run_config.json", run_config)

    results = []
    started = time.perf_counter()
    for position, sample in enumerate(selected, start=1):
        prompt_ids = encode_gsm8k_sample(sample, loaded.tokenizer, args.enable_thinking)
        prompt_length = prompt_ids.shape[1]
        if args.method == "AutoJudge":
            stats = run_autojudge(
                prompt_ids,
                loaded.target,
                loaded.draft,
                loaded.tokenizer,
                scaler,
                head,
                draft_length=args.draft_length,
                max_new_tokens=args.max_new_tokens,
                threshold=threshold,
                confidence_threshold=0.9,
            )
        else:
            stats = run_judi(
                prompt_ids,
                loaded.target,
                loaded.draft,
                loaded.tokenizer,
                draft_length=args.draft_length,
                max_new_tokens=args.max_new_tokens,
                judi_threshold=threshold,
                confidence_threshold=0.9,
            )
        generated_ids = stats.pop("generated_ids")
        response = loaded.tokenizer.decode(generated_ids[0, prompt_length:], skip_special_tokens=True)
        stats["generated_tokens"] = int(generated_ids.shape[1] - prompt_length)
        results.append(evaluate_gsm8k_result(sample, response, stats))
        print(
            f"[{position}/{len(selected)}] threshold={threshold:g} "
            f"sample={sample.index} tokens={stats['generated_tokens']} steps={stats['steps']}",
            flush=True,
        )

    elapsed = time.perf_counter() - started
    summary = summarize_results(results)
    summary.update({"elapsed_seconds": elapsed, "generated_tokens": sum(x["generated_tokens"] for x in results)})
    summary["tokens_per_second"] = summary["generated_tokens"] / elapsed if elapsed else 0.0
    summary["threshold"] = threshold
    summary["method"] = args.method
    summary["profile"] = args.profile
    summary["confidence_threshold"] = 0.9
    summary["classifier_threshold"] = threshold if args.method == "AutoJudge" else None
    summary["judi_threshold"] = threshold if args.method == "JuDi" else None
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    args = parse_args()
    thresholds = _select_thresholds(args)
    config = _config(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    target_path = _path(config["target_model"])
    draft_path = _path(config["draft_model"])
    data_path = _path(config["data"])
    output_dir = _path(config["output_dir"])
    if not target_path.exists():
        raise FileNotFoundError(f"Target model does not exist: {target_path}")
    if not draft_path.exists():
        raise FileNotFoundError(f"Draft model does not exist: {draft_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"GSM8K data does not exist: {data_path}")

    loaded = load_models(
        str(target_path),
        str(draft_path),
        dtype=args.dtype,
        target_device_map=config["target_device_map"],
        draft_device_map=config["draft_device_map"],
    )
    feature_dim = int(getattr(loaded.draft.config, "hidden_size")) + int(getattr(loaded.target.config, "hidden_size"))
    head = scaler = None
    if args.method == "AutoJudge":
        if not config.get("checkpoint"):
            raise ValueError("The AutoJudge method requires a profile checkpoint or --checkpoint")
        checkpoint_path = _path(config["checkpoint"])
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Classifier checkpoint does not exist: {checkpoint_path}")
        head, scaler = load_autojudge_checkpoint(checkpoint_path, feature_dim=feature_dim)

    samples = load_gsm8k(data_path)
    end = len(samples) if args.end < 0 else min(args.end, len(samples))
    selected = samples[max(0, args.start) : end]
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "scan_config.json",
        {
            "method": args.method,
            "profile": args.profile,
            "thresholds": thresholds,
            "confidence_threshold": 0.9,
            "samples": len(selected),
        },
    )
    scan_rows = []
    for threshold in thresholds:
        threshold_dir = output_dir / (
            f"autojudge_threshold_{_threshold_name(threshold)}"
            if args.method == "AutoJudge"
            else f"judi_threshold_{_threshold_name(threshold)}"
        )
        summary = _run_single_threshold(
            args=args,
            config=config,
            loaded=loaded,
            selected=selected,
            output_dir=threshold_dir,
            threshold=threshold,
            head=head,
            scaler=scaler,
            end=end,
        )
        scan_rows.append(summary)
    with (output_dir / "scan_summary.jsonl").open("w", encoding="utf-8") as handle:
        for row in scan_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Completed {len(scan_rows)} threshold run(s); summaries: {output_dir / 'scan_summary.jsonl'}")


if __name__ == "__main__":
    main()
