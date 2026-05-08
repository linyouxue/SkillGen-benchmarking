"""Helpers for preparing and evaluating LiveCodeBench tasks inside SkillGen."""

from __future__ import annotations

import ast
import base64
import json
import os
import pickle
import platform
import re
import signal
import subprocess
import sys
import zlib
from decimal import Decimal
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import Any
from unittest.mock import mock_open, patch


_LCB_DATASET_ID = "livecodebench/code_generation_lite"
_REPO_ROOT = Path(__file__).resolve().parent
_IMPORTS = (
    "from string import *\n"
    "from re import *\n"
    "from datetime import *\n"
    "from collections import *\n"
    "from heapq import *\n"
    "from bisect import *\n"
    "from copy import *\n"
    "from math import *\n"
    "from random import *\n"
    "from statistics import *\n"
    "from itertools import *\n"
    "from functools import *\n"
    "from operator import *\n"
    "from io import *\n"
    "from sys import *\n"
    "from json import *\n"
    "from builtins import *\n"
    "from typing import *\n"
    "import string\n"
    "import re\n"
    "import datetime\n"
    "import collections\n"
    "import heapq\n"
    "import bisect\n"
    "import copy\n"
    "import math\n"
    "import random\n"
    "import statistics\n"
    "import itertools\n"
    "import functools\n"
    "import operator\n"
    "import io\n"
    "import sys\n"
    "import json\n"
    "sys.setrecursionlimit(50000)\n"
)
_DEFAULT_TIMEOUT_SECONDS = 6


def _build_version_map() -> dict[str, list[str]]:
    version_map = {
        "release_v1": ["test.jsonl"],
        "release_v2": ["test.jsonl", "test2.jsonl"],
        "release_v3": ["test.jsonl", "test2.jsonl", "test3.jsonl"],
        "release_v4": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"],
        "release_v5": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl"],
        "release_v6": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"],
    }
    version_map["release_latest"] = list(version_map["release_v6"])

    versions = ["v1", "v2", "v3", "v4", "v5", "v6"]
    for idx, version in enumerate(versions, start=1):
        version_map[version] = [f"test{idx}.jsonl" if idx > 1 else "test.jsonl"]
    for start in range(1, len(versions) + 1):
        for end in range(start + 1, len(versions) + 1):
            key = f"v{start}_v{end}"
            version_map[key] = [
                f"test{idx}.jsonl" if idx > 1 else "test.jsonl"
                for idx in range(start, end + 1)
            ]
    return version_map


LIVECODEBENCH_VERSION_FILES = _build_version_map()
LIVECODEBENCH_VERSION_TAGS = tuple(LIVECODEBENCH_VERSION_FILES.keys())


def resolve_livecodebench_files(version_tag: str) -> list[str]:
    files = LIVECODEBENCH_VERSION_FILES.get(version_tag)
    if not files:
        supported = ", ".join(LIVECODEBENCH_VERSION_TAGS)
        raise ValueError(f"Unsupported LiveCodeBench version '{version_tag}'. Supported: {supported}")
    return files


def _load_json_maybe(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def load_livecodebench_records(version_tag: str = "release_latest") -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    records: list[dict[str, Any]] = []
    for file_name in resolve_livecodebench_files(version_tag):
        path = hf_hub_download(_LCB_DATASET_ID, file_name, repo_type="dataset")
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                records.append(json.loads(line))
    return records


def build_livecodebench_prompt(item: dict[str, Any]) -> str:
    title = item.get("question_title", "").strip()
    content = item.get("question_content", "").strip()
    starter_code = (item.get("starter_code") or "").strip()

    parts = [
        "Solve the following competitive programming problem in Python 3.",
        f"Title: {title}" if title else "",
        "Problem:\n" + content,
    ]
    if starter_code:
        parts.append(
            "Starter code (use it only if it helps):\n"
            f"```python\n{starter_code}\n```"
        )
    parts.append(
        "Return only the final Python 3 solution code. "
        "Do not include prose, analysis, or surrounding markdown unless a code fence is unavoidable."
    )
    return "\n\n".join(part for part in parts if part)


def convert_livecodebench_item(item: dict[str, Any], idx: int) -> dict[str, Any]:
    public_tests = _load_json_maybe(item.get("public_test_cases", "[]"))
    metadata = _load_json_maybe(item.get("metadata", "{}"))

    return {
        "instance_id": item.get("question_id") or str(idx),
        "input": build_livecodebench_prompt(item),
        "ground_truth": None,
        "metadata": {
            "benchmark": "livecodebench",
            "question_title": item.get("question_title"),
            "question_id": item.get("question_id"),
            "platform": item.get("platform"),
            "contest_id": item.get("contest_id"),
            "contest_date": item.get("contest_date"),
            "starter_code": item.get("starter_code", ""),
            "difficulty": item.get("difficulty"),
            "public_test_cases": public_tests,
            "private_test_cases": item.get("private_test_cases", "[]"),
            "lcb_metadata": metadata,
            "timeout_seconds": _DEFAULT_TIMEOUT_SECONDS,
        },
    }


def decode_private_test_cases(encoded_data: str) -> list[dict[str, Any]]:
    if not encoded_data:
        return []
    try:
        decoded = json.loads(encoded_data)
        if isinstance(decoded, list):
            return decoded
    except json.JSONDecodeError:
        pass

    decoded_bytes = base64.b64decode(encoded_data)
    decompressed = zlib.decompress(decoded_bytes)
    unpacked = pickle.loads(decompressed)
    return json.loads(unpacked)


def extract_livecodebench_code(model_output: str) -> str:
    if not model_output:
        return ""

    fenced = re.findall(r"```(?:python|py)?\s*\n(.*?)```", model_output, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced[-1].strip()

    generic_fenced = re.findall(r"```\s*\n(.*?)```", model_output, flags=re.DOTALL)
    if generic_fenced:
        return generic_fenced[-1].strip()

    return model_output.strip()


class _TimeoutException(Exception):
    pass


def _timeout_handler(signum, frame):  # pragma: no cover - subprocess-only signal handler
    raise _TimeoutException("Execution timed out")


class _Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        self._stringio = StringIO()
        self._stringio.close = lambda *_args: None
        sys.stdout = self._stringio
        return self

    def __exit__(self, *args):
        self.append(self._stringio.getvalue())
        del self._stringio
        sys.stdout = self._stdout


class _MockBuffer:
    def __init__(self, inputs: str):
        self._bytes = inputs.encode("utf-8")

    def read(self, *args):
        return self._bytes

    def readline(self, *args):
        return self._bytes.split(b"\n")[0] + b"\n"


class _MockStdinWithBuffer:
    def __init__(self, inputs: str):
        self.inputs = inputs
        self._stringio = StringIO(inputs)
        self.buffer = _MockBuffer(inputs)

    def read(self, *args):
        return self.inputs

    def readline(self, *args):
        return self._stringio.readline(*args)

    def readlines(self, *args):
        return self.inputs.split("\n")

    def __getattr__(self, name):
        return getattr(self._stringio, name)


def _truncate(value: Any, length: int = 300) -> str:
    text = value if isinstance(value, str) else str(value)
    if len(text) <= length:
        return text
    return text[: length // 2] + "...(truncated)..." + text[-length // 2 :]


def _clean_if_name(code: str) -> str:
    try:
        tree = ast.parse(code)
        last_block = tree.body[-1]
        if isinstance(last_block, ast.If) and ast.unparse(last_block.test).strip() == "__name__ == '__main__'":
            code = ast.unparse(tree.body[:-1]) + "\n" + ast.unparse(last_block.body)
    except Exception:
        return code
    return code


def _make_wrapped_function(code: str) -> str:
    try:
        tree = ast.parse(code)
        import_stmts = []
        other_stmts = []
        for stmt in tree.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                import_stmts.append(stmt)
            else:
                other_stmts.append(stmt)

        wrapper = ast.FunctionDef(
            name="wrapped_function",
            args=ast.arguments(
                posonlyargs=[],
                args=[],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=other_stmts,
            decorator_list=[],
            lineno=-1,
        )
        return _IMPORTS + "\n" + ast.unparse(import_stmts) + "\n" + ast.unparse(wrapper)
    except Exception:
        return code


def _call_stdio_method(method, raw_input: str):
    inputs_line_iterator = iter(raw_input.split("\n"))
    mock_stdin = _MockStdinWithBuffer(raw_input)

    @patch("builtins.open", mock_open(read_data=raw_input))
    @patch("sys.stdin", mock_stdin)
    @patch("sys.stdin.readline", lambda *args: next(inputs_line_iterator))
    @patch("sys.stdin.readlines", lambda *args: raw_input.split("\n"))
    @patch("sys.stdin.read", lambda *args: raw_input)
    def _inner_call(_method):
        try:
            return _method()
        except SystemExit:
            return None

    return _inner_call(method)


def _get_function(compiled_solution, fn_name: str):
    try:
        return getattr(compiled_solution, fn_name)
    except Exception:
        return None


def _compile_code(code: str, timeout_seconds: int):
    signal.alarm(timeout_seconds)
    try:
        module = ModuleType("tmp_solution", "")
        exec(code, module.__dict__)  # noqa: S102
        if "class Solution" in code:
            compiled = module.Solution()
        else:
            compiled = module
    finally:
        signal.alarm(0)
    return compiled


def _convert_line_to_decimals(line: str) -> tuple[bool, list[Decimal]]:
    try:
        return True, [Decimal(elem) for elem in line.split()]
    except Exception:
        return False, []


def _get_stripped_lines(value: str) -> list[str]:
    value = value.strip()
    return [line.strip() for line in value.split("\n")]


def _grade_call_based(code: str, all_inputs: list[str], all_outputs: list[str], fn_name: str, timeout_seconds: int):
    compiled = _compile_code(_IMPORTS + "\n\n" + code, timeout_seconds)
    method = _get_function(compiled, fn_name)
    if method is None:
        return [False], {"error_message": f"Function '{fn_name}' not found", "error_code": -4}

    parsed_inputs = [[json.loads(line) for line in inputs.split("\n")] for inputs in all_inputs]
    parsed_outputs = [json.loads(output) for output in all_outputs]

    results = []
    for expected_input, expected_output in zip(parsed_inputs, parsed_outputs):
        signal.alarm(timeout_seconds)
        try:
            prediction = method(*expected_input)
            signal.alarm(0)
            if isinstance(prediction, tuple):
                prediction = list(prediction)
            if prediction != expected_output:
                results.append(False)
                return results, {
                    "error_message": "Wrong Answer",
                    "error_code": -2,
                    "inputs": _truncate(expected_input),
                    "expected": _truncate(expected_output),
                    "output": _truncate(prediction),
                }
            results.append(True)
        except _TimeoutException as exc:
            signal.alarm(0)
            results.append(False)
            return results, {
                "error_message": "Time Limit Exceeded",
                "error_code": -3,
                "error": repr(exc),
                "inputs": _truncate(expected_input),
                "expected": _truncate(expected_output),
            }
        except Exception as exc:
            signal.alarm(0)
            results.append(False)
            return results, {
                "error_message": "Runtime Error",
                "error_code": -4,
                "error": repr(exc),
                "inputs": _truncate(expected_input),
                "expected": _truncate(expected_output),
            }
        finally:
            signal.alarm(0)

    return results, {"execution_time_seconds": 0.0}


def _grade_stdio(code: str, all_inputs: list[str], all_outputs: list[str], timeout_seconds: int):
    code = _make_wrapped_function(_clean_if_name(code))
    compiled = _compile_code(code, timeout_seconds)
    method = _get_function(compiled, "wrapped_function")
    if method is None:
        return [False], {"error_message": "wrapped_function not found", "error_code": -4}

    results = []
    for expected_input, expected_output in zip(all_inputs, all_outputs):
        signal.alarm(timeout_seconds)
        with _Capturing() as captured_output:
            try:
                _call_stdio_method(method, expected_input)
                signal.alarm(0)
            except _TimeoutException as exc:
                signal.alarm(0)
                results.append(False)
                return results, {
                    "error_message": "Time Limit Exceeded",
                    "error_code": -3,
                    "error": repr(exc),
                    "inputs": _truncate(expected_input),
                    "expected": _truncate(expected_output),
                }
            except Exception as exc:
                signal.alarm(0)
                results.append(False)
                return results, {
                    "error_message": "Runtime Error",
                    "error_code": -4,
                    "error": repr(exc),
                    "inputs": _truncate(expected_input),
                    "expected": _truncate(expected_output),
                }
            finally:
                signal.alarm(0)

        prediction = captured_output[0]
        prediction_lines = _get_stripped_lines(prediction)
        expected_lines = _get_stripped_lines(expected_output)
        if len(prediction_lines) != len(expected_lines):
            results.append(False)
            return results, {
                "error_message": "Wrong Answer: mismatched output length",
                "error_code": -2,
                "inputs": _truncate(expected_input),
                "expected": _truncate(expected_output),
                "output": _truncate(prediction),
            }

        mismatch = False
        for predicted_line, gt_line in zip(prediction_lines, expected_lines):
            if predicted_line == gt_line:
                continue
            pred_ok, pred_decimals = _convert_line_to_decimals(predicted_line)
            gt_ok, gt_decimals = _convert_line_to_decimals(gt_line)
            if pred_ok and gt_ok and pred_decimals == gt_decimals:
                continue
            mismatch = True
            break

        if mismatch:
            results.append(False)
            return results, {
                "error_message": "Wrong Answer",
                "error_code": -2,
                "inputs": _truncate(expected_input),
                "expected": _truncate(expected_output),
                "output": _truncate(prediction),
            }

        results.append(True)

    return results, {"execution_time_seconds": 0.0}


def _apply_reliability_guard(maximum_memory_bytes: int | None = None) -> None:
    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if platform.system() != "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    import builtins
    import shutil

    os.environ["OMP_NUM_THREADS"] = "1"
    builtins.quit = None
    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None
    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess as _subprocess

    _subprocess.Popen = None  # type: ignore
    builtins.help = None
    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None


def _run_lcb_sample(sample: dict[str, Any], code: str, timeout_seconds: int) -> dict[str, Any]:
    signal.signal(signal.SIGALRM, _timeout_handler)
    _apply_reliability_guard()

    input_output = json.loads(sample["input_output"])
    fn_name = input_output.get("fn_name")
    if fn_name:
        results, metadata = _grade_call_based(
            code=code,
            all_inputs=input_output["inputs"],
            all_outputs=input_output["outputs"],
            fn_name=fn_name,
            timeout_seconds=timeout_seconds,
        )
    else:
        results, metadata = _grade_stdio(
            code=code,
            all_inputs=input_output["inputs"],
            all_outputs=input_output["outputs"],
            timeout_seconds=timeout_seconds,
        )

    passed = bool(results) and all(result is True for result in results)
    metadata["passed_tests"] = sum(1 for result in results if result is True)
    metadata["total_tests"] = len(results)
    metadata["passed"] = passed
    return metadata


def _grade_payload_cli() -> None:
    payload = json.loads(sys.stdin.read())
    sample = payload["sample"]
    code = payload["code"]
    timeout_seconds = int(payload.get("timeout_seconds", _DEFAULT_TIMEOUT_SECONDS))
    try:
        result = _run_lcb_sample(sample=sample, code=code, timeout_seconds=timeout_seconds)
    except Exception as exc:  # pragma: no cover - subprocess safety net
        result = {
            "passed": False,
            "passed_tests": 0,
            "total_tests": 0,
            "error_code": -5,
            "error_message": f"Evaluation failed: {exc}",
        }
    sys.stdout.write(json.dumps(result))


def _build_evaluation_sample(instance_metadata: dict[str, Any]) -> dict[str, Any]:
    public_tests = instance_metadata.get("public_test_cases") or []
    private_tests = decode_private_test_cases(instance_metadata.get("private_test_cases", "[]"))
    benchmark_metadata = instance_metadata.get("lcb_metadata") or {}
    return {
        "input_output": json.dumps(
            {
                "inputs": [test["input"] for test in public_tests + private_tests],
                "outputs": [test["output"] for test in public_tests + private_tests],
                "fn_name": benchmark_metadata.get("func_name"),
            }
        )
    }


def evaluate_livecodebench_output(
    *,
    model_output: str,
    instance_metadata: dict[str, Any],
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    code = extract_livecodebench_code(model_output)
    if not code:
        return {
            "passed": False,
            "score": 0.0,
            "error_code": -1,
            "error_message": "No executable code found in model output.",
            "passed_tests": 0,
            "total_tests": 0,
            "extracted_code": "",
        }

    payload = {
        "sample": _build_evaluation_sample(instance_metadata),
        "code": code,
        "timeout_seconds": timeout_seconds or instance_metadata.get("timeout_seconds", _DEFAULT_TIMEOUT_SECONDS),
    }

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(_REPO_ROOT)
        if not existing_pythonpath
        else f"{_REPO_ROOT}{os.pathsep}{existing_pythonpath}"
    )

    command = [
        sys.executable,
        "-c",
        "from benchmarks.livecodebench_adapter import _grade_payload_cli; _grade_payload_cli()",
    ]

    try:
        with TemporaryDirectory(dir=_REPO_ROOT) as temp_dir:
            proc = subprocess.run(
                command,
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                cwd=temp_dir,
                env=env,
                timeout=int(payload["timeout_seconds"]) + 2,
                check=False,
            )
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "score": 0.0,
            "error_code": -3,
            "error_message": "Evaluation subprocess timed out.",
            "passed_tests": 0,
            "total_tests": 0,
            "extracted_code": code,
        }

    if proc.returncode != 0:
        return {
            "passed": False,
            "score": 0.0,
            "error_code": -5,
            "error_message": (proc.stderr or proc.stdout or "Evaluation subprocess failed.").strip(),
            "passed_tests": 0,
            "total_tests": 0,
            "extracted_code": code,
        }

    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {
            "passed": False,
            "score": 0.0,
            "error_code": -5,
            "error_message": "Evaluation subprocess returned malformed JSON.",
            "passed_tests": 0,
            "total_tests": 0,
            "extracted_code": code,
        }

    result["score"] = 1.0 if result.get("passed") else 0.0
    result["extracted_code"] = code
    return result


if __name__ == "__main__":  # pragma: no cover - subprocess entrypoint
    _grade_payload_cli()
