"""Run AutoJudge or JuDi on LiveCodeBench v5."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from code_judi.data import encode_lcb_sample, load_lcb
from code_judi.decoding import run_autojudge, run_judi
from code_judi.evaluation import evaluate_lcb_result
from code_judi.models import load_autojudge_checkpoint, load_models


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILES = PROJECT_ROOT / "configs" / "profiles.json"
DEFAULT_THRESHOLDS = PROJECT_ROOT / "configs" / "thresholds.json"


def _path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _thresholds(args: argparse.Namespace) -> List[float]:
    if args.thresholds:
        values = [part for item in args.thresholds for part in item.replace(",", " ").split()]
        return [float(value) for value in values]
    single = args.threshold if args.method == "AutoJudge" else args.judi_threshold
    if single is not None:
        return [float(single)]
    with _path(args.thresholds_config).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    key = "AutoJudge" if args.method == "AutoJudge" else "JuDi"
    return [float(value) for value in payload.get("lcb", payload.get(key, []))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AutoJudge or JuDi on LiveCodeBench v5")
    parser.add_argument("--method", choices=("AutoJudge", "JuDi"), default="AutoJudge")
    parser.add_argument("--profile", default="llama3-8b-1b")
    parser.add_argument("--profiles", default=str(DEFAULT_PROFILES))
    parser.add_argument("--target-model")
    parser.add_argument("--draft-model")
    parser.add_argument("--data")
    parser.add_argument("--checkpoint")
    parser.add_argument("--fold-map")
    parser.add_argument("--output-dir")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--target-device-map")
    parser.add_argument("--draft-device-map")
    parser.add_argument("--draft-length", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--judi-threshold", type=float)
    parser.add_argument("--thresholds", nargs="+")
    parser.add_argument("--thresholds-config", default=str(DEFAULT_THRESHOLDS))
    parser.add_argument("--confidence-threshold", type=float, default=0.9)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _config(args: argparse.Namespace) -> Dict[str, Any]:
    with _path(args.profiles).open("r", encoding="utf-8") as handle:
        profiles = json.load(handle)
    if args.profile not in profiles:
        raise KeyError(f"Unknown profile {args.profile!r}")
    profile = dict(profiles[args.profile])
    profile["target_model"] = args.target_model or profile["target_model"]
    profile["draft_model"] = args.draft_model or profile["draft_model"]
    profile["data"] = args.data or profile.get("lcb_data", "models/datasets/lcb/test5.jsonl")
    profile["checkpoint"] = args.checkpoint or profile.get("lcb_checkpoint")
    profile["fold_map"] = args.fold_map or profile.get("lcb_fold_map")
    profile["output_dir"] = args.output_dir or profile.get("lcb_output_dir", f"outputs/lcb/{args.profile}/{args.method}")
    profile["target_device_map"] = args.target_device_map or profile.get("target_device_map", "auto")
    profile["draft_device_map"] = args.draft_device_map or profile.get("draft_device_map", "cuda:0")
    return profile


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_fold_map(path: Path | None) -> Dict[str, int]:
    if path is None or not path.exists():
        return {}
    payload = torch.load(path, map_location="cpu")
    mapping: Dict[str, int] = {}
    for fold, values in payload.items() if isinstance(payload, dict) else enumerate(payload):
        if isinstance(fold, str):
            match = re.search(r"(\d+)$", fold)
            if match is None:
                raise ValueError(f"Could not parse LCB fold key: {fold!r}")
            fold_id = int(match.group(1)) - 1
        else:
            fold_id = int(fold)
        for question_id in values.get("question_id", []):
            mapping[str(question_id)] = fold_id
    return mapping


def _metric_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    accepted = [value for row in rows for value in row["accepted_lengths"]]
    drafted = [value for row in rows for value in row["draft_lengths"]]
    generated = sum(row["generated_tokens"] for row in rows)
    return {
        "samples": len(rows),
        "accuracy": sum(bool(row["passed"]) for row in rows) / len(rows) if rows else 0.0,
        "average_generated_tokens": generated / len(rows) if rows else 0.0,
        "mean_accepted_tokens": sum(accepted) / len(accepted) if accepted else 0.0,
        "mean_draft_tokens": sum(drafted) / len(drafted) if drafted else 0.0,
        "MAT": sum(accepted) / len(accepted) if accepted else 0.0,
        "DT": sum(drafted) / len(drafted) if drafted else 0.0,
        "acceptance_ratio": sum(accepted) / sum(drafted) if drafted and sum(drafted) else 0.0,
        "generated_tokens": generated,
    }


def _summary(rows: List[Dict[str, Any]], elapsed: float, threshold: float, args: argparse.Namespace) -> Dict[str, Any]:
    summary = _metric_summary(rows)
    fold_rows: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        fold = row.get("fold")
        if fold is not None:
            fold_rows.setdefault(str(fold), []).append(row)
    fold_summaries = {fold: _metric_summary(group) for fold, group in sorted(fold_rows.items(), key=lambda item: int(item[0]))}
    if fold_summaries:
        summary["folds"] = fold_summaries
        summary["fold_average_accuracy"] = sum(item["accuracy"] for item in fold_summaries.values()) / len(fold_summaries)
        summary["fold_average_MAT"] = sum(item["MAT"] for item in fold_summaries.values()) / len(fold_summaries)
        summary["fold_average_DT"] = sum(item["DT"] for item in fold_summaries.values()) / len(fold_summaries)
    summary.update({
        "tokens_per_second": summary["generated_tokens"] / elapsed if elapsed else 0.0,
        "method": args.method, "profile": args.profile, "threshold": threshold,
        "confidence_threshold": args.confidence_threshold,
    })
    return summary


def main() -> None:
    args = parse_args()
    if args.confidence_threshold != 0.9:
        raise ValueError("The target confidence threshold is fixed at 0.9")
    thresholds = _thresholds(args)
    config = _config(args)
    data_path = _path(config["data"])
    target_path, draft_path = _path(config["target_model"]), _path(config["draft_model"])
    for path, label in ((data_path, "LCB data"), (target_path, "target model"), (draft_path, "draft model")):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    loaded = load_models(str(target_path), str(draft_path), dtype=args.dtype, target_device_map=config["target_device_map"], draft_device_map=config["draft_device_map"])
    problems = load_lcb(data_path, limit=args.limit)
    print(f"Loaded {len(problems)} LCB problems from {data_path}", flush=True)
    end = len(problems) if args.end < 0 else min(args.end, len(problems))
    problems = problems[max(0, args.start):end]
    fold_map = _load_fold_map(_path(config["fold_map"]) if config.get("fold_map") else None)
    if args.method == "AutoJudge" and not fold_map:
        raise ValueError("AutoJudge LCB evaluation requires a non-empty fold map")
    missing_folds = [problem.question_id for problem in problems if problem.question_id not in fold_map]
    if args.method == "AutoJudge" and missing_folds:
        raise KeyError(f"{len(missing_folds)} selected LCB problems are missing from the fold map; first={missing_folds[0]!r}")
    fold_counts: Dict[str, int] = {}
    for problem in problems:
        fold = fold_map.get(problem.question_id)
        if fold is not None:
            fold_counts[str(fold + 1)] = fold_counts.get(str(fold + 1), 0) + 1
    print(f"Selected {len(problems)} LCB problems; fold_counts={fold_counts}", flush=True)
    output_root = _path(config["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    feature_dim = int(getattr(loaded.draft.config, "hidden_size")) + int(getattr(loaded.target.config, "hidden_size"))
    checkpoint_cache: Dict[int, Any] = {}
    checkpoint_config = _path(config["checkpoint"]) if config.get("checkpoint") else None
    for threshold in thresholds:
        rows: List[Dict[str, Any]] = []
        started = time.perf_counter()
        for index, problem in enumerate(problems):
            prompt_ids = encode_lcb_sample(problem, loaded.tokenizer)
            prompt_length = prompt_ids.shape[1]
            head = scaler = None
            fold = fold_map.get(problem.question_id)
            if args.method == "AutoJudge":
                if checkpoint_config is None:
                    raise ValueError("The AutoJudge method requires --checkpoint or profile lcb_checkpoint")
                checkpoint_path = checkpoint_config
                cache_key = -1
                if checkpoint_config.is_dir():
                    checkpoint_path = checkpoint_config / f"lcb_llama_1b8b_offical_head_{fold + 1}.pkl"
                    cache_key = fold
                if cache_key not in checkpoint_cache:
                    checkpoint_cache[cache_key] = load_autojudge_checkpoint(checkpoint_path, feature_dim=feature_dim)
                head, scaler = checkpoint_cache[cache_key]
                stats = run_autojudge(prompt_ids, loaded.target, loaded.draft, loaded.tokenizer, scaler, head, draft_length=args.draft_length, max_new_tokens=args.max_new_tokens, threshold=threshold, confidence_threshold=0.9)
            else:
                stats = run_judi(prompt_ids, loaded.target, loaded.draft, loaded.tokenizer, draft_length=args.draft_length, max_new_tokens=args.max_new_tokens, judi_threshold=threshold, confidence_threshold=0.9)
            generated_ids = stats.pop("generated_ids")
            stats["generated_tokens"] = int(generated_ids.shape[1] - prompt_length)
            stats["index"] = index
            response = loaded.tokenizer.decode(generated_ids[0, prompt_length:], skip_special_tokens=True)
            row = evaluate_lcb_result(problem, response, stats, timeout=args.timeout)
            if fold is not None:
                row["fold"] = fold + 1
            if args.method == "AutoJudge" and checkpoint_config is not None and checkpoint_config.is_dir():
                row["checkpoint"] = str(checkpoint_path)
            rows.append(row)
            fold_label = f" fold={fold + 1}" if fold is not None else ""
            print(f"[{index + 1}/{len(problems)}] threshold={threshold:g} id={problem.question_id}{fold_label} passed={rows[-1]['passed']}", flush=True)
        threshold_dir = output_root / f"{'autojudge' if args.method == 'AutoJudge' else 'judi'}_threshold_{threshold:g}"
        threshold_dir.mkdir(parents=True, exist_ok=True)
        with (threshold_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary = _summary(rows, time.perf_counter() - started, threshold, args)
        _write(threshold_dir / "summary.json", summary)
        run_config = dict(vars(args))
        run_config.update({"data": str(data_path), "fold_map": str(_path(config["fold_map"])) if config.get("fold_map") else None})
        _write(threshold_dir / "run_config.json", run_config)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    _write(output_root / "scan_config.json", {
        "method": args.method,
        "profile": args.profile,
        "thresholds": thresholds,
        "samples": len(problems),
        "data": str(data_path),
        "fold_map": str(_path(config["fold_map"])) if config.get("fold_map") else None,
        "fold_counts": fold_counts,
    })


if __name__ == "__main__":
    main()
