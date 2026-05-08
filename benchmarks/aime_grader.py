"""AIME grader: deterministic integer extraction + exact match.

AIME answers are always a single integer in [0, 999]. The original prompt
(see prepare_benchmarks._convert_aime) instructs the agent to write only that
integer on the last line. Previously we fell through to a generic LLM judge,
which added 2-5% judge noise on top of AIME's already-high variance. This
grader replaces the LLM judge with a rule-based extractor so that grading is
deterministic, reproducible, and judge-free.

Contract (mirrors pubmedqa_grader / mind2web_grader):

    evaluate_aime_output(model_output: str, instance_metadata: dict) -> dict

Returned dict fields:
    - passed          : bool  - predicted integer equals ground truth
    - score           : float - 1.0 / 0.0
    - predicted_int   : int | None
    - gt_int          : int
    - error_message   : str | None
    - extraction_rule : str   - "last_line" | "answer_pattern" | "boxed"
                                | "final_integer" | "none"

Extraction strategy (first rule that yields an integer in [0, 999] wins):
  1. Last non-empty line, if it consists of a single 1-3 digit integer
     (the prompt explicitly asks for this shape).
  2. \\boxed{NNN}  - common CoT format.
  3. "(final )?answer\\s*(is|:|=)?\\s*NNN" near the tail of the output.
  4. The very last standalone 1-3 digit integer anywhere in the output.
  5. None  -> fail with extraction error.

Values outside [0, 999] are rejected as invalid AIME answers. Leading zeros
are accepted ("007" -> 7) and compared as integers.
"""

from __future__ import annotations

import re
from typing import Any


_INT_ONLY_LINE = re.compile(r"^\s*0*(\d{1,3})\s*$")
_BOXED_RE = re.compile(r"\\boxed\s*\{\s*0*(\d{1,3})\s*\}", re.IGNORECASE)
_ANSWER_NEAR_END = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is|:|=)?\s*\**\s*0*(\d{1,3})\b",
    re.IGNORECASE,
)
_ANY_INT = re.compile(r"\b0*(\d{1,3})\b")


def _in_range(n: int) -> bool:
    return 0 <= n <= 999


def _extract_int(raw: str) -> tuple[int | None, str]:
    text = (raw or "").strip()
    if not text:
        return None, "none"

    # Rule 1: last non-empty line that is just an integer.
    for line in reversed(text.splitlines()):
        line = line.strip().strip(".,;:*`\"'")
        if not line:
            continue
        m = _INT_ONLY_LINE.match(line)
        if m:
            n = int(m.group(1))
            if _in_range(n):
                return n, "last_line"
        break  # only the truly-last non-empty line qualifies for rule 1

    # Rule 2: \boxed{NNN}
    boxed_hits = _BOXED_RE.findall(text)
    if boxed_hits:
        n = int(boxed_hits[-1])
        if _in_range(n):
            return n, "boxed"

    # Rule 3: "final answer is NNN" / "answer: NNN" - prefer the LAST
    # occurrence since models often restate the answer at the end.
    ans_hits = _ANSWER_NEAR_END.findall(text)
    if ans_hits:
        n = int(ans_hits[-1])
        if _in_range(n):
            return n, "answer_pattern"

    # Rule 4: fall back to the very last 1-3 digit integer.
    all_ints = _ANY_INT.findall(text)
    for tok in reversed(all_ints):
        n = int(tok)
        if _in_range(n):
            return n, "final_integer"

    return None, "none"


def _gt_int(instance_metadata: dict[str, Any]) -> int | None:
    """Prefer the normalised ground_truth carried by the pipeline; fall back
    to raw_answer in metadata. Return None if neither parses as an integer."""
    for key in ("ground_truth", "raw_answer"):
        val = (instance_metadata or {}).get(key)
        if val is None:
            continue
        s = str(val).strip()
        # strip common boxed wrapping
        if s.startswith("\\boxed"):
            m = re.search(r"\d{1,3}", s)
            if m:
                s = m.group(0)
        try:
            n = int(s)
            if _in_range(n):
                return n
        except ValueError:
            continue
    return None


def evaluate_aime_output(
    model_output: str,
    instance_metadata: dict[str, Any],
) -> dict[str, Any]:
    gt = _gt_int(instance_metadata)
    pred, rule = _extract_int(str(model_output or ""))

    if gt is None:
        return {
            "passed": False,
            "score": 0.0,
            "predicted_int": pred,
            "gt_int": None,
            "error_message": "missing or unparseable ground_truth",
            "extraction_rule": rule,
        }

    passed = pred is not None and pred == gt
    err: str | None = None
    if pred is None:
        err = "could not extract integer from agent output"
    elif not passed:
        err = f"answer mismatch (pred={pred} gt={gt})"

    return {
        "passed": bool(passed),
        "score": 1.0 if passed else 0.0,
        "predicted_int": pred,
        "gt_int": gt,
        "error_message": err,
        "extraction_rule": rule,
    }


__all__ = ["evaluate_aime_output"]
