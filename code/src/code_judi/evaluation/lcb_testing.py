"""Small LiveCodeBench-compatible code grading implementation."""

from __future__ import annotations

import ast
import contextlib
import io
import json
import signal
import sys
from decimal import Decimal
from types import ModuleType
from typing import Any, Dict, List, Tuple
from unittest.mock import mock_open, patch


IMPORTS = """from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string, re, datetime, collections, heapq, bisect, copy, math, random, statistics, itertools, functools, operator, io, sys, json\nsys.setrecursionlimit(50000)\n"""


class _Timeout(Exception):
    pass


def _alarm_handler(signum: int, frame: Any) -> None:
    raise _Timeout()


def _compile(code: str, timeout: int) -> Any:
    signal.alarm(timeout)
    try:
        module = ModuleType("lcb_solution")
        exec(code, module.__dict__)
        return module.Solution() if "class Solution" in code else module
    finally:
        signal.alarm(0)


def _grade_call(code: str, inputs: List[str], outputs: List[str], fn_name: str, timeout: int) -> Tuple[List[Any], Dict[str, Any]]:
    compiled = _compile(IMPORTS + "\n" + code, timeout)
    method = getattr(compiled, fn_name, None)
    if method is None:
        return [], {"error_code": -4, "error_message": f"Missing function {fn_name}"}
    parsed_inputs = [[json.loads(line) for line in value.split("\n")] for value in inputs]
    parsed_outputs = [json.loads(value) for value in outputs]
    all_results: List[Any] = []
    for args, expected in zip(parsed_inputs, parsed_outputs):
        signal.alarm(timeout)
        try:
            prediction = method(*args)
            if isinstance(prediction, tuple):
                prediction = list(prediction)
            if prediction != expected:
                all_results.append(False)
                return all_results, {"error_code": -2, "error_message": "Wrong answer", "expected": expected, "output": prediction}
            all_results.append(True)
        except _Timeout:
            all_results.append(-3)
            return all_results, {"error_code": -3, "error_message": "Time limit exceeded"}
        except Exception as exc:
            all_results.append(-4)
            return all_results, {"error_code": -4, "error_message": repr(exc)}
        finally:
            signal.alarm(0)
    return all_results, {"execution_time": None}


def _clean_main(code: str) -> str:
    try:
        tree = ast.parse(code)
        if tree.body and isinstance(tree.body[-1], ast.If) and ast.unparse(tree.body[-1].test).strip() == "__name__ == '__main__'":
            return ast.unparse(tree.body[:-1]) + "\n" + ast.unparse(tree.body[-1].body)
    except Exception:
        pass
    return code


def _wrap_stdio(code: str) -> str:
    code = _clean_main(code)
    try:
        tree = ast.parse(code)
        imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        rest = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
        function = ast.FunctionDef(
            name="wrapped_function",
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=rest or [ast.Pass()], decorator_list=[],
        )
        return IMPORTS + "\n" + ast.unparse(imports) + "\n" + ast.unparse(function)
    except Exception:
        return IMPORTS + "\n\ndef wrapped_function():\n" + "\n".join("    " + line for line in code.splitlines())


def _grade_stdio(code: str, inputs: List[str], outputs: List[str], timeout: int) -> Tuple[List[Any], Dict[str, Any]]:
    compiled = _compile(_wrap_stdio(code), timeout)
    method = getattr(compiled, "wrapped_function", None)
    if method is None:
        return [-4], {"error_code": -4, "error_message": "Could not compile solution"}
    for raw_input, expected in zip(inputs, outputs):
        capture = io.StringIO()
        input_lines = raw_input.splitlines(True)
        input_stream = io.StringIO(raw_input)
        input_stream.readline = lambda *args: input_lines.pop(0) if input_lines else ""
        input_stream.readlines = lambda *args: raw_input.split("\n")
        input_stream.read = lambda *args: raw_input
        signal.alarm(timeout)
        try:
            with patch("builtins.open", mock_open(read_data=raw_input)), patch("sys.stdin", input_stream), contextlib.redirect_stdout(capture):
                method()
            predicted = [line.strip() for line in capture.getvalue().strip().split("\n")]
            expected_lines = [line.strip() for line in expected.strip().split("\n")]
            if predicted != expected_lines:
                try:
                    decimal_pred = [Decimal(value) for line in predicted for value in line.split()]
                    decimal_expected = [Decimal(value) for line in expected_lines for value in line.split()]
                    if decimal_pred == decimal_expected:
                        continue
                except Exception:
                    pass
                return [False], {"error_code": -2, "error_message": "Wrong answer", "expected": expected, "output": capture.getvalue()}
        except _Timeout:
            return [-3], {"error_code": -3, "error_message": "Time limit exceeded"}
        except Exception as exc:
            return [-4], {"error_code": -4, "error_message": repr(exc)}
        finally:
            signal.alarm(0)
    return [True], {"execution_time": None}


def run_test(sample: Dict[str, str], test: str, timeout: int = 6) -> Tuple[List[Any], Dict[str, Any]]:
    payload = json.loads(sample["input_output"])
    from code_judi.data.lcb import extract_code

    test = extract_code(test) if "```" in test else test
    signal.signal(signal.SIGALRM, _alarm_handler)
    if payload.get("fn_name"):
        return _grade_call(test, payload["inputs"], payload["outputs"], payload["fn_name"], timeout)
    return _grade_stdio(test, payload["inputs"], payload["outputs"], timeout)
