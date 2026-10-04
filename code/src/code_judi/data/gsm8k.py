"""GSM8K loading and prompt formatting."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch


PROMPT_PREFIX = "Given the following problem, reason and give a final answer to the problem.\n"
PROMPT_SUFFIX = (
    'Your response should end with "The final answer is [answer]" '
    "where [answer] is the response to the problem."
)


@dataclass(frozen=True)
class GSM8KSample:
    question: str
    answer: str
    index: int


def _answer_from_record(record: Dict[str, Any]) -> str:
    answer = str(record.get("answer", ""))
    marker = "#### "
    return answer[answer.rfind(marker) + len(marker) :] if marker in answer else answer


def load_gsm8k(path: str | Path) -> List[GSM8KSample]:
    samples: List[GSM8KSample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if line.strip():
                record = json.loads(line)
                samples.append(
                    GSM8KSample(
                        question=str(record["question"]),
                        answer=_answer_from_record(record),
                        index=index,
                    )
                )
    return samples


def _chat_messages(question: str) -> List[Dict[str, str]]:
    prompt = f"{PROMPT_PREFIX}{question}\n{PROMPT_SUFFIX}"
    return [{"role": "user", "content": prompt}]


def _apply_chat_template(tokenizer: Any, messages: Iterable[Dict[str, str]], enable_thinking: bool) -> torch.Tensor:
    kwargs = {"add_generation_prompt": True, "return_tensors": "pt"}
    if enable_thinking:
        kwargs["enable_thinking"] = True
    try:
        encoded = tokenizer.apply_chat_template(list(messages), **kwargs)
    except (TypeError, ValueError):
        kwargs.pop("enable_thinking", None)
        encoded = tokenizer.apply_chat_template(list(messages), **kwargs)
    if isinstance(encoded, torch.Tensor):
        return encoded
    input_ids = getattr(encoded, "input_ids", None)
    if input_ids is None and isinstance(encoded, dict):
        input_ids = encoded.get("input_ids")
    if input_ids is None:
        raise TypeError("Tokenizer chat template did not return input_ids")
    return input_ids if isinstance(input_ids, torch.Tensor) else torch.as_tensor(input_ids)


def encode_gsm8k_sample(sample: GSM8KSample, tokenizer: Any, enable_thinking: bool = False) -> torch.Tensor:
    """Return one left-unpadded prompt tensor with shape ``[1, sequence]``."""
    return _apply_chat_template(tokenizer, _chat_messages(sample.question), enable_thinking)


def extract_numeric_answer(text: str) -> float | None:
    """Extract the number after the final-answer marker."""
    normalized = text.lower().replace("<|eot_id|>", "").replace("the final answer is", "=")
    value = normalized.rsplit("=", 1)[-1].strip()
    value = re.sub(r"[^0-9.\-]", "", value).strip(".")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
