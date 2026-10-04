"""LiveCodeBench v5 JSONL loading and prompt formatting."""

from __future__ import annotations

import base64
import json
import pickle
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch


SYSTEM_PROMPT = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program that "
    "matches the specification and passes all tests."
)
FORMAT_WITH_STARTER = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
FORMAT_WITHOUT_STARTER = (
    "Read the inputs from stdin solve the problem and write the answer to stdout "
    "(do not directly test on the sample inputs). Enclose your code within delimiters "
    "as follows. Ensure that when the Python program runs, it reads the inputs, runs "
    "the algorithm and writes output to STDOUT."
)


@dataclass(frozen=True)
class LCBTest:
    input: str
    output: str
    testtype: str


@dataclass(frozen=True)
class LCBProblem:
    question_title: str
    question_content: str
    platform: str
    question_id: str
    contest_id: str
    contest_date: str
    starter_code: str
    difficulty: str
    public_test_cases: List[LCBTest]
    private_test_cases: List[LCBTest]
    metadata: Dict[str, Any]

    def evaluation_sample(self) -> Dict[str, str]:
        tests = self.public_test_cases + self.private_test_cases
        return {
            "input_output": json.dumps({
                "inputs": [test.input for test in tests],
                "outputs": [test.output for test in tests],
                "fn_name": self.metadata.get("func_name"),
            }, ensure_ascii=False)
        }


def _decode_tests(value: Any, *, private: bool = False) -> List[LCBTest]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            if not private:
                raise
            compressed = zlib.decompress(base64.b64decode(value.encode("utf-8")))
            value = json.loads(pickle.loads(compressed))
    return [LCBTest(**item) for item in value]


def _problem_from_record(record: Dict[str, Any]) -> LCBProblem:
    metadata = record.get("metadata") or {}
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    return LCBProblem(
        question_title=record["question_title"],
        question_content=record["question_content"],
        platform=record["platform"],
        question_id=str(record["question_id"]),
        contest_id=str(record["contest_id"]),
        contest_date=record["contest_date"],
        starter_code=record.get("starter_code") or "",
        difficulty=record["difficulty"],
        public_test_cases=_decode_tests(record["public_test_cases"]),
        private_test_cases=_decode_tests(record["private_test_cases"], private=True),
        metadata=metadata,
    )


def load_lcb(path: str | Path, *, limit: int | None = None) -> List[LCBProblem]:
    """Load LCB JSONL or a HuggingFace Arrow dataset export.

    AJ-V2 evaluates the full release_v5 Arrow split (880 problems). The
    repository also keeps a smaller ``test5.jsonl`` compatibility slice, so
    the loader accepts both formats without changing the problem representation.
    """
    dataset_path = Path(path)
    if dataset_path.suffix == ".arrow":
        try:
            from datasets import Dataset
        except ImportError as exc:  # pragma: no cover - environment dependency
            raise RuntimeError("Loading LCB Arrow data requires the 'datasets' package") from exc
        dataset = Dataset.from_file(str(dataset_path))
        if limit is not None:
            dataset = dataset.select(range(min(limit, len(dataset))))
        return [_problem_from_record(dataset[index]) for index in range(len(dataset))]

    problems: List[LCBProblem] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            problems.append(_problem_from_record(json.loads(line)))
            if limit is not None and len(problems) >= limit:
                break
    return problems


def format_lcb_prompt(problem: LCBProblem) -> str:
    prompt = f"### Question:\n{problem.question_content}\n\n"
    if problem.starter_code:
        prompt += f"### Format: {FORMAT_WITH_STARTER}\n```python\n{problem.starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {FORMAT_WITHOUT_STARTER}\n```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


def encode_lcb_sample(problem: LCBProblem, tokenizer: Any) -> torch.Tensor:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_lcb_prompt(problem)},
    ]
    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors="pt",
        truncation=False, padding=False,
    )
    if isinstance(encoded, torch.Tensor):
        return encoded
    input_ids = getattr(encoded, "input_ids", None)
    if input_ids is None and isinstance(encoded, dict):
        input_ids = encoded.get("input_ids")
    if input_ids is None:
        raise TypeError("Tokenizer chat template did not return input_ids")
    return input_ids if isinstance(input_ids, torch.Tensor) else torch.as_tensor(input_ids)


def extract_code(generation: str) -> str:
    lines = generation.splitlines()
    fences = [index for index, line in enumerate(lines) if "```" in line]
    if len(fences) < 2:
        return ""
    return "\n".join(lines[fences[-2] + 1:fences[-1]])
