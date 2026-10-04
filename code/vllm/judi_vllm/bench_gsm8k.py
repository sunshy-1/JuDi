"""GSM8K test accuracy and generation throughput for vLLM 0.8.3 / JuDi.

The shell scan runs each engine in a fresh process. Loading, tokenization,
warmup, scoring and result writes are outside the generation timer. Throughput
includes prefill, decoding, vLLM scheduling and detokenization; it counts only
emitted output token IDs (including an emitted EOS), never proposed draft tokens.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

from tqdm import tqdm


# bench_gsm8k.py lives in <project>/vllm/judi_vllm/ after the open-source
# layout rename; the project root is therefore two parents above this file.
ROOT = Path(__file__).resolve().parents[2]
PROMPT_PREFIX = "Given the following problem, reason and give a final answer to the problem.\n"
PROMPT_SUFFIX = ('Your response should end with "The final answer is [answer]" '
                 'where [answer] is the response to the problem.')
NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
FINAL_MARKER = re.compile(r"####|the final answer is", re.IGNORECASE)


def extract_answer(text: str) -> float | None:
    """Prefer the explicit final answer; otherwise use the last numeric value."""
    text = text.replace(",", "").replace("\u2212", "-")
    markers = list(FINAL_MARKER.finditer(text))
    if markers:
        match = NUMBER.search(text[markers[-1].end():])
    else:
        boxes = re.findall(r"\\boxed\{([^{}]*)\}", text)
        numbers = list(NUMBER.finditer(boxes[-1] if boxes else text))
        match = numbers[-1] if numbers else None
    if match is None:
        return None
    value = float(match.group())
    return value if math.isfinite(value) else None


def load_samples(path: Path, start: int, end: int) -> tuple[list[dict], int]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    stop = len(rows) if end == -1 else end
    if not 0 <= start < stop <= len(rows):
        raise ValueError(f"Invalid sample range [{start}, {end}) for {len(rows)} samples")
    selected = []
    for index in range(start, stop):
        row = rows[index]
        question, answer = row["question"], row["answer"]
        if not isinstance(question, str) or not isinstance(answer, str) or "####" not in answer:
            raise ValueError(f"Sample {index} must contain a question and GSM8K '####' answer")
        gold = extract_answer(answer.rsplit("####", 1)[-1])
        if gold is None:
            raise ValueError(f"Cannot parse gold answer for sample {index}")
        selected.append(dict(index=index, question=question, answer_text=answer, answer=gold))
    return selected, len(rows)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def visible_gpu_count() -> int:
    value = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not value or value.strip() in {"-1", "none", "None"}:
        return 1
    return max(1, len([item for item in value.split(",") if item.strip()]))


def run(args: argparse.Namespace) -> None:
    if args.method == "judi" and (args.threshold is None or not math.isfinite(args.threshold)):
        raise ValueError("JuDi requires a finite --threshold")
    if args.method == "standard" and args.threshold is not None:
        raise ValueError("The native standard baseline does not take --threshold")
    for name in ("draft_length", "max_new_tokens", "batch_size", "max_model_len",
                 "draft_tensor_parallel_size", "warmup_rounds"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.tensor_parallel_size < 0:
        raise ValueError("--tensor-parallel-size must be zero (auto) or positive")
    if args.tensor_parallel_size == 0:
        args.tensor_parallel_size = visible_gpu_count()
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("--gpu-memory-utilization must be in (0, 1]")
    for model in (args.target_model, args.draft_model):
        if not (Path(model) / "config.json").is_file():
            raise FileNotFoundError(f"Local model config.json missing: {model}")
    samples, dataset_size = load_samples(args.data, args.start, args.end)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    from .runtime import activate
    activate()
    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    prompts = []
    for sample in samples:
        content = PROMPT_PREFIX + sample["question"] + "\n" + PROMPT_SUFFIX
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True)
        if len(ids) + args.max_new_tokens > args.max_model_len:
            raise ValueError(f"Sample {sample['index']}: prompt ({len(ids)}) + output budget "
                             f"({args.max_new_tokens}) exceeds --max-model-len; increase it")
        prompts.append({"prompt_token_ids": ids})

    common = dict(
        model=str(args.target_model), dtype="bfloat16", seed=args.seed,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len, max_num_seqs=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=not args.cuda_graphs, generation_config="vllm",
        enable_prefix_caching=False, enable_chunked_prefill=False,
        disable_log_stats=True, disable_custom_all_reduce=args.disable_custom_all_reduce,
    )
    config = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "dataset_size": dataset_size, "num_samples": len(samples),
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "prompt_ids_sha256": hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
        "prompt_prefix": PROMPT_PREFIX, "prompt_suffix": PROMPT_SUFFIX,
        "answer_extraction": "last final-answer marker, then boxed number, else last number",
        "answer_tolerance": {"rel_tol": 1e-5, "abs_tol": 1e-5},
        "confidence_threshold": 0.9 if args.method == "judi" else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python": sys.executable, "vllm_version": vllm.__version__,
        "vllm_path": vllm.__file__, "torch_version": torch.__version__,
        "engine_options": common,
        "timing": "sum of synchronized llm.generate wall times after warmup; includes prefill",
        "token_count": "len(output.token_ids), including emitted EOS; excludes prompt and drafts",
    }
    write_json(args.output_dir / "config.json", config)
    label = "standard" if args.method == "standard" else f"judi_{args.threshold:.2f}"
    print(f"[{label}] {len(samples)}/{dataset_size} samples; batch={args.batch_size}; "
          f"draft_length={args.draft_length}; GPU={config['cuda_visible_devices']}", flush=True)
    if args.method == "standard":
        # Do not import/apply the JuDi patch: -1 would still pay its KL cost.
        llm = LLM(**common, speculative_config={
            "model": str(args.draft_model), "num_speculative_tokens": args.draft_length,
            "draft_tensor_parallel_size": args.draft_tensor_parallel_size,
        })
        # Match the reference JuDi EOS policy, including the native baseline.
        llm.llm_engine.generation_config_fields = {"eos_token_id": tokenizer.eos_token_id}
    else:
        from .patch import create_judi_llm
        llm = create_judi_llm(
            **common, draft_model=str(args.draft_model), judi_threshold=args.threshold,
            confidence_threshold=0.9, draft_length=args.draft_length,
            draft_tensor_parallel_size=args.draft_tensor_parallel_size)

    params = SamplingParams(temperature=0, max_tokens=args.max_new_tokens, seed=args.seed)
    # Warm both full and final partial batch shapes, without prefix-cache reuse.
    warmup_sizes = {min(args.batch_size, len(samples))}
    if len(samples) % args.batch_size:
        warmup_sizes.add(len(samples) % args.batch_size)
    print(f"[{label}] Warming up ({args.warmup_rounds} rounds per batch shape)...", flush=True)
    for _ in range(args.warmup_rounds):
        for size in sorted(warmup_sizes):
            llm.generate(prompts[:size], params, use_tqdm=False)
    torch.cuda.synchronize()

    correct = generated_tokens = truncated = completed = 0
    generation_seconds = 0.0
    progress = tqdm(total=len(samples), desc=label, unit="sample", dynamic_ncols=True)
    try:
        with (args.output_dir / "results.jsonl").open("x", encoding="utf-8") as handle:
            for offset in range(0, len(samples), args.batch_size):
                batch = samples[offset:offset + args.batch_size]
                inputs = prompts[offset:offset + args.batch_size]
                torch.cuda.synchronize()
                started = time.perf_counter()
                outputs = llm.generate(inputs, params, use_tqdm=False)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - started
                if len(outputs) != len(batch):
                    raise RuntimeError("vLLM returned an unexpected number of outputs")
                generation_seconds += seconds
                for sample, prompt, output in zip(batch, inputs, outputs):
                    if list(output.prompt_token_ids) != prompt["prompt_token_ids"]:
                        raise RuntimeError("vLLM output/prompt order mismatch")
                    completion = output.outputs[0]
                    prediction = extract_answer(completion.text)
                    is_correct = prediction is not None and math.isclose(
                        prediction, sample["answer"], rel_tol=1e-5, abs_tol=1e-5)
                    tokens = len(completion.token_ids)
                    record = {
                        **sample, "response": completion.text, "prediction": prediction,
                        "correct": is_correct, "generated_tokens": tokens,
                        "token_ids": list(completion.token_ids),
                        "finish_reason": completion.finish_reason, "stop_reason": completion.stop_reason,
                        "batch_index": offset // args.batch_size,
                        "batch_size": len(batch), "batch_seconds": seconds,
                    }
                    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    correct += is_correct
                    generated_tokens += tokens
                    truncated += completion.finish_reason == "length"
                    completed += 1
                handle.flush()
                progress.update(len(batch))
                progress.set_postfix_str(
                    f"acc={correct / completed:.4f} token/s={generated_tokens / generation_seconds:.2f}"
                )
    finally:
        progress.close()

    summary = {
        "method": args.method, "threshold": args.threshold, "num_samples": len(samples),
        "correct": correct, "accuracy": correct / len(samples),
        "generated_tokens": generated_tokens, "generation_seconds": generation_seconds,
        "tokens_per_second": generated_tokens / generation_seconds,
        "truncated_samples": truncated, "batch_size": args.batch_size,
        "results_dir": str(args.output_dir.resolve()),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(f"[{label}] Finished: {correct}/{len(samples)} correct; "
          f"acc={summary['accuracy']:.4f}; token/s={summary['tokens_per_second']:.2f}; "
          f"truncated={truncated}\nResults: {args.output_dir}", flush=True)


def summarize(output_dir: Path) -> None:
    paths = [output_dir / "standard" / "summary.json"]
    paths += sorted(output_dir.glob("judi_*/summary.json"),
                    key=lambda p: float(p.parent.name.removeprefix("judi_")))
    if not paths[0].is_file() or len(paths) < 2:
        raise ValueError("Expected a completed standard baseline and JuDi runs")
    if any(not (p / "summary.json").is_file()
           for p in output_dir.glob("judi_*") if p.is_dir()):
        raise ValueError("A JuDi run is incomplete; inspect its log before summarizing")
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    configs = [json.loads((p.parent / "config.json").read_text(encoding="utf-8")) for p in paths]
    # Never silently compare different samples, prompts or generation settings.
    for key in ("data_sha256", "prompt_ids_sha256", "start", "end", "num_samples",
                "target_model", "draft_model", "batch_size", "draft_length",
                "max_new_tokens", "warmup_rounds", "seed", "engine_options"):
        if any(c[key] != configs[0][key] for c in configs[1:]):
            raise ValueError(f"Inconsistent experiment setting: {key}")
    write_json(output_dir / "summary.json", rows)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [f"{'method':<12} {'threshold':>9} {'correct/total':>15} {'acc (%)':>10} "
             f"{'token/s':>12} {'truncated':>10}"]
    for row in rows:
        threshold = "-" if row["threshold"] is None else f"{row['threshold']:.2f}"
        count = f"{row['correct']}/{row['num_samples']}"
        lines.append(f"{row['method']:<12} {threshold:>9} {count:>15} "
                     f"{100 * row['accuracy']:>10.2f} {row['tokens_per_second']:>12.2f} "
                     f"{row['truncated_samples']:>10}")
    table = "\n".join(lines) + "\n"
    (output_dir / "summary.txt").write_text(table, encoding="utf-8")
    print("\n" + table + f"Results: {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    bench = subparsers.add_parser("run", help="Evaluate one method / threshold")
    bench.add_argument("--method", choices=("standard", "judi"), required=True)
    bench.add_argument("--threshold", type=float)
    bench.add_argument("--target-model", type=Path,
                       default=ROOT / "models/target/Llama-3.1-8B-Instruct")
    bench.add_argument("--draft-model", type=Path,
                       default=ROOT / "models/draft/Llama-3.2-1B-Instruct")
    bench.add_argument("--data", type=Path, default=ROOT / "models/datasets/gsm8k/test.jsonl")
    bench.add_argument("--output-dir", type=Path, required=True,
                       help="New directory; for the shell scan set OUTPUT_DIR instead")
    bench.add_argument("--start", type=int, default=0)
    bench.add_argument("--end", type=int, default=-1, help="Exclusive end index; -1 = full test set")
    bench.add_argument("--draft-length", type=int, default=40)
    bench.add_argument("--max-new-tokens", type=int, default=2048)
    bench.add_argument("--batch-size", type=int, default=1)
    bench.add_argument("--max-model-len", type=int, default=4096)
    bench.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    bench.add_argument("--tensor-parallel-size", type=int, default=0,
                       help="Target TP degree; 0 uses the number of visible GPUs")
    bench.add_argument("--draft-tensor-parallel-size", type=int, default=1)
    bench.add_argument("--disable-custom-all-reduce", action="store_true")
    bench.add_argument("--cuda-graphs", action="store_true",
                       help="Enable graphs for both engines (default: eager MQA scorer)")
    bench.add_argument("--warmup-rounds", type=int, default=2)
    bench.add_argument("--seed", type=int, default=42)
    report = subparsers.add_parser("summarize", help="Write the scan's JSON, CSV and text table")
    report.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    else:
        summarize(args.output_dir)


if __name__ == "__main__":
    main()
