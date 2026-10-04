"""LiveCodeBench pass@1 evaluation in a child process."""

from __future__ import annotations

import json
import multiprocessing
from typing import Any, Dict

from code_judi.data.lcb import LCBProblem, extract_code


def _grade_child(sample: Dict[str, str], code: str, timeout: int, result: Any) -> None:
    from code_judi.evaluation.lcb_testing import run_test

    try:
        verdict, metadata = run_test(sample, test=code, timeout=timeout)
    except BaseException as exc:
        verdict, metadata = [-4], {
            "error_code": -4,
            "error_message": f"{type(exc).__name__}: {exc}",
        }
    result.append((verdict, metadata))


def evaluate_lcb_result(problem: LCBProblem, response: str, stats: Dict[str, Any], timeout: int = 6) -> Dict[str, Any]:
    code = extract_code(response)
    manager = multiprocessing.Manager()
    result = manager.list()
    process = multiprocessing.Process(
        target=_grade_child,
        args=(problem.evaluation_sample(), code, timeout, result),
    )
    process.start()
    input_count = len(json.loads(problem.evaluation_sample()["input_output"])["inputs"])
    process.join(timeout=(timeout + 1) * input_count + 5)
    if process.is_alive():
        process.kill()
        process.join()
    verdict = None
    metadata: Dict[str, Any] = {"error_message": "Evaluation process timed out or exited without a result"}
    if result:
        verdict, metadata = result[0]
    manager.shutdown()
    passed = bool(verdict) and all(value is True or value == 1 for value in verdict)
    return {
        "index": stats["index"],
        "question_id": problem.question_id,
        "contest_id": problem.contest_id,
        "difficulty": problem.difficulty,
        "response": response,
        "code": code,
        "passed": passed,
        "evaluation": metadata,
        "generated_tokens": int(stats["generated_tokens"]),
        "steps": int(stats["steps"]),
        "draft_lengths": stats["draft_lengths"],
        "accepted_lengths": stats["accepted_lengths"],
    }
