"""GSM8K accuracy and speculative-decoding statistics."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List

from code_judi.data.gsm8k import extract_numeric_answer


def evaluate_gsm8k_result(sample: Any, response: str, stats: Dict[str, Any]) -> Dict[str, Any]:
    prediction = extract_numeric_answer(response)
    answer = extract_numeric_answer(sample.answer)
    return {
        "index": sample.index,
        "question": sample.question,
        "response": response,
        "answer": answer,
        "prediction": prediction,
        "correct": answer is not None and prediction is not None and math.isclose(answer, prediction, rel_tol=1e-5, abs_tol=1e-5),
        "generated_tokens": int(stats["generated_tokens"]),
        "steps": int(stats["steps"]),
        "draft_lengths": stats["draft_lengths"],
        "accepted_lengths": stats["accepted_lengths"],
    }


def summarize_results(results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(results)
    accepted: List[int] = [x for row in rows for x in row["accepted_lengths"]]
    drafted: List[int] = [x for row in rows for x in row["draft_lengths"]]
    mean_accepted_tokens = sum(accepted) / len(accepted) if accepted else 0.0
    mean_draft_tokens = sum(drafted) / len(drafted) if drafted else 0.0
    return {
        "samples": len(rows),
        "accuracy": sum(row["correct"] for row in rows) / len(rows) if rows else 0.0,
        "average_generated_tokens": sum(row["generated_tokens"] for row in rows) / len(rows) if rows else 0.0,
        "mean_accepted_tokens": mean_accepted_tokens,
        "mean_draft_tokens": mean_draft_tokens,
        "MAT": mean_accepted_tokens,
        "DT": mean_draft_tokens,
        "acceptance_ratio": sum(accepted) / sum(drafted) if drafted and sum(drafted) else 0.0,
    }
