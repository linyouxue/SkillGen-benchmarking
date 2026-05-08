"""PubMedQA grader: rule-based yes/no/maybe extraction.

Contract (mirrors mind2web_grader / spreadsheetbench_executor):

    evaluate_pubmedqa_output(model_output: str, instance_metadata: dict) -> dict

Returned dict fields:
    - passed          : bool  - predicted label matches final_decision
    - score           : float - 1.0 if passed else 0.0
    - predicted_label : str | None - one of "yes" | "no" | "maybe" | None
    - gt_label        : str        - the ground-truth final_decision
    - error_message   : str | None
    - extraction_rule : str        - which rule fired ("last_line" | "boxed" |
                                     "regex_scan" | "fallback" | "none")

Extraction strategy (in order):
  1. last non-empty line lowercased and stripped of punctuation - the prompt
     instructs the agent to write exactly this.
  2. \\boxed{...} content (occasionally produced by CoT-trained models).
  3. greedy scan for the LAST standalone yes/no/maybe token in the output
     (handles "The answer is: maybe.").
  4. None (fail).

Grading is exact match on the canonicalised label. The three labels are the
only valid answers in PubMedQA's expert-labeled split.
"""

from __future__ import annotations

import re
from typing import Any


_VALID = ("yes", "no", "maybe")

# Match a standalone yes/no/maybe token, word-bounded so "analyses" doesn't
# count as "yes" and "notable" doesn't count as "no".
_TOKEN_RE = re.compile(r"\b(yes|no|maybe)\b", re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\{\s*([^}]+?)\s*\}", re.IGNORECASE)


def _normalise(tok: str) -> str | None:
    t = tok.strip().strip(".,:;!?\"'`*()[]").lower()
    return t if t in _VALID else None


def _extract_label(raw: str) -> tuple[str | None, str]:
    text = (raw or "").strip()
    if not text:
        return None, "none"

    # Rule 1: last non-empty line.
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        cand = _normalise(line)
        if cand:
            return cand, "last_line"
        # allow "Answer: yes" / "Final decision: no." single-line forms
        m = _TOKEN_RE.findall(line)
        if m:
            cand = _normalise(m[-1])
            if cand:
                return cand, "last_line"
        break

    # Rule 2: \boxed{...}
    boxed = _BOXED_RE.findall(text)
    if boxed:
        cand = _normalise(boxed[-1])
        if cand:
            return cand, "boxed"

    # Rule 3: last standalone yes/no/maybe anywhere in the output.
    all_tokens = _TOKEN_RE.findall(text)
    if all_tokens:
        cand = _normalise(all_tokens[-1])
        if cand:
            return cand, "regex_scan"

    return None, "none"


def evaluate_pubmedqa_output(
    model_output: str,
    instance_metadata: dict[str, Any],
) -> dict[str, Any]:
    gt = (instance_metadata or {}).get("final_decision", "")
    gt = str(gt).strip().lower()

    pred, rule = _extract_label(str(model_output or ""))
    passed = pred is not None and pred == gt
    err = None
    if pred is None:
        err = "could not extract yes/no/maybe from agent output"
    elif not passed:
        err = f"label mismatch (pred={pred!r} gt={gt!r})"

    return {
        "passed": bool(passed),
        "score": 1.0 if passed else 0.0,
        "predicted_label": pred,
        "gt_label": gt,
        "error_message": err,
        "extraction_rule": rule,
    }


__all__ = ["evaluate_pubmedqa_output"]
