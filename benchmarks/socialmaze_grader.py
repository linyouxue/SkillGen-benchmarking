"""SocialMaze grader - rule-based extractors for FTS / RDP / UPI.

All three tasks produce small, structured final answers (a digit, a
decision, or two demographic labels). We keep the graders purely
rule-based (no LLM judge) so they are fast, deterministic, and free.

Dispatch is driven by `metadata["task"]`:
  - "fts"  -> final spy player id (1/2/3/4)
  - "rdp"  -> Accept / Reject
  - "upi"  -> age_group (18-34 / 35-54 / 55+) + gender (Male / Female / Non-binary)
             UPI returns age_correct and gender_correct separately; we
             compute a single passed = age_correct AND gender_correct,
             with a score = mean(age, gender).
"""

from __future__ import annotations

import json
import re
from typing import Any


# Find the Spy

_FTS_FINAL_PATTERNS = [
    re.compile(r"final\s+spy\s*:\s*player\s*([1-4])", re.IGNORECASE),
    re.compile(r"final\s+answer\s*:\s*player\s*([1-4])", re.IGNORECASE),
    re.compile(r"the\s+spy\s+is\s+player\s*([1-4])", re.IGNORECASE),
    re.compile(r"spy\s+is\s+player\s*([1-4])", re.IGNORECASE),
    re.compile(r"answer\s*:\s*player\s*([1-4])", re.IGNORECASE),
    re.compile(r"player\s*([1-4])\s+is\s+the\s+spy", re.IGNORECASE),
    re.compile(r"\bplayer\s*([1-4])\b", re.IGNORECASE),  # last-resort
]


def _extract_fts_prediction(text: str) -> tuple[str | None, str]:
    # Scan a trailing window first (model conclusions usually live near the end)
    tail = text[-600:]
    for patt in _FTS_FINAL_PATTERNS:
        m = patt.search(tail)
        if m:
            return m.group(1), patt.pattern
    # Full-text fallback
    for patt in _FTS_FINAL_PATTERNS:
        m = patt.search(text)
        if m:
            return m.group(1), patt.pattern
    return None, "no_match"


def _grade_fts(output: str, meta: dict[str, Any]) -> dict[str, Any]:
    gt = str(meta.get("spy_player") or "").strip()
    pred, rule = _extract_fts_prediction(output or "")
    passed = pred is not None and pred == gt
    return {
        "passed": bool(passed),
        "score": 1.0 if passed else 0.0,
        "predicted_label": pred,
        "gt_label": gt,
        "extraction_rule": rule,
        "error_message": None if pred is not None else "Could not extract spy prediction",
    }


# Review Decision Prediction

_RDP_FINAL_PATTERN = re.compile(r"final\s+decision\s*:\s*(accept|reject)", re.IGNORECASE)


def _extract_rdp_prediction(text: str) -> tuple[str | None, str]:
    m = _RDP_FINAL_PATTERN.search(text or "")
    if m:
        label = m.group(1).lower()
        return ("Accept" if label.startswith("acc") else "Reject"), "final_decision"

    low = (text or "").lower()
    has_acc = re.search(r"\baccept(ed)?\b", low) is not None
    has_rej = re.search(r"\breject(ed)?\b", low) is not None
    if has_acc and not has_rej:
        return "Accept", "accept_only"
    if has_rej and not has_acc:
        return "Reject", "reject_only"
    if has_rej:  # both mentioned: prefer the one appearing in the tail
        tail = low[-400:]
        if "reject" in tail and "accept" not in tail:
            return "Reject", "reject_tail"
        if "accept" in tail and "reject" not in tail:
            return "Accept", "accept_tail"
        # fall-through
    return None, "no_match"


def _grade_rdp(output: str, meta: dict[str, Any]) -> dict[str, Any]:
    gt = (meta.get("normalised_decision") or meta.get("decision") or "").strip()
    if gt.lower().startswith("acc"):
        gt = "Accept"
    elif gt.lower().startswith("rej"):
        gt = "Reject"
    pred, rule = _extract_rdp_prediction(output or "")
    passed = pred is not None and pred == gt
    return {
        "passed": bool(passed),
        "score": 1.0 if passed else 0.0,
        "predicted_label": pred,
        "gt_label": gt,
        "extraction_rule": rule,
        "error_message": None if pred is not None else "Could not extract Accept/Reject",
    }


# User Profile Inference

_UPI_AGE_PATTERNS = [
    re.compile(r"age\s*group\s*:\s*(18-34|35-54|55\+)", re.IGNORECASE),
    re.compile(r"age\s*:\s*(18-34|35-54|55\+)", re.IGNORECASE),
]

_UPI_GENDER_PATTERNS = [
    re.compile(r"gender\s*:\s*(male|female|non-?binary)", re.IGNORECASE),
]


def _extract_upi_prediction(text: str) -> tuple[str | None, str | None]:
    low = (text or "").lower()

    age = None
    for patt in _UPI_AGE_PATTERNS:
        m = patt.search(low)
        if m:
            age = m.group(1).strip()
            if age == "55+":
                age = "55+"
            break
    if age is None:
        if re.search(r"\b18-34\b", low):
            age = "18-34"
        elif re.search(r"\b35-54\b", low):
            age = "35-54"
        elif re.search(r"\b55\+\b|\bsenior\b|\bolder\b", low):
            age = "55+"

    gender = None
    for patt in _UPI_GENDER_PATTERNS:
        m = patt.search(low)
        if m:
            raw = m.group(1).strip().replace("-", "")
            if raw == "male":
                gender = "Male"
            elif raw == "female":
                gender = "Female"
            elif raw == "nonbinary":
                gender = "Non-binary"
            break
    if gender is None:
        # Fallback scan of full text (avoid "female" substring inside "male" regex)
        if re.search(r"\bnon-?binary\b", low):
            gender = "Non-binary"
        elif re.search(r"\bfemale\b|\bwoman\b|\bwomen\b", low):
            gender = "Female"
        elif re.search(r"(?<!fe)\bmale\b|\bman\b|\bmen\b", low):
            gender = "Male"

    return age, gender


def _grade_upi(output: str, meta: dict[str, Any]) -> dict[str, Any]:
    gt_age = str(meta.get("age_group") or "").strip()
    gt_gender = str(meta.get("gender") or "").strip()
    pred_age, pred_gender = _extract_upi_prediction(output or "")

    age_correct = pred_age is not None and pred_age == gt_age
    gender_correct = pred_gender is not None and pred_gender == gt_gender
    both_correct = age_correct and gender_correct

    return {
        "passed": bool(both_correct),
        "score": (int(age_correct) + int(gender_correct)) / 2.0,
        "predicted_label": json.dumps({"age_group": pred_age, "gender": pred_gender}),
        "gt_label": json.dumps({"age_group": gt_age, "gender": gt_gender}),
        "age_correct": bool(age_correct),
        "gender_correct": bool(gender_correct),
        "extraction_rule": "upi_regex",
        "error_message": None if (pred_age and pred_gender) else "Could not fully extract demographics",
    }


# Dispatch

def evaluate_socialmaze_output(
    model_output: str,
    instance_metadata: dict[str, Any],
) -> dict[str, Any]:
    meta = instance_metadata or {}
    task = str(meta.get("task") or "").lower()
    out = str(model_output or "")

    if task == "fts":
        return _grade_fts(out, meta)
    if task == "rdp":
        return _grade_rdp(out, meta)
    if task == "upi":
        return _grade_upi(out, meta)
    return {
        "passed": False,
        "score": 0.0,
        "predicted_label": None,
        "gt_label": None,
        "extraction_rule": "unknown_task",
        "error_message": f"Unknown SocialMaze task: {task!r}",
    }


__all__ = ["evaluate_socialmaze_output"]
