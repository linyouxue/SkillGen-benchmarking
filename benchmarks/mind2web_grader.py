"""Mind2Web grader: rule-based step-success evaluation.

Contract (mirrors livecodebench_adapter.evaluate_livecodebench_output):

    evaluate_mind2web_output(model_output: str, instance_metadata: dict) -> dict

Returned dict fields:
    - passed: bool             -- True iff element + op + (value if applicable) match
    - score: float             -- 1.0 if passed else 0.0
    - element_match: bool      -- predicted bnid  in  target_bnids
    - op_match: bool           -- predicted op == metadata["op"]
    - value_match: bool        -- normalised value equality (always True for CLICK)
    - extracted_prediction: dict | None
    - error_message: str | None
"""

from __future__ import annotations

from typing import Any

from .mind2web_adapter import parse_mind2web_prediction


def _norm_value(v: Any) -> str:
    return str(v or "").strip().lower()


def evaluate_mind2web_output(
    model_output: str,
    instance_metadata: dict[str, Any],
) -> dict[str, Any]:
    pred = parse_mind2web_prediction(str(model_output or ""))
    meta = instance_metadata or {}
    target_bnids = {str(x) for x in (meta.get("target_bnids") or []) if x}
    gt_op = (meta.get("op") or "").upper().strip()
    gt_value = meta.get("value") or ""

    if pred is None:
        return {
            "passed": False,
            "score": 0.0,
            "element_match": False,
            "op_match": False,
            "value_match": False,
            "extracted_prediction": None,
            "error_message": "could not parse prediction JSON",
        }

    element_match = pred["target_bnid"] in target_bnids

    pred_op = pred["op"].upper().strip()
    op_match = pred_op == gt_op

    # value matters only for TYPE / SELECT; CLICK passes with any value.
    if gt_op in ("TYPE", "SELECT"):
        value_match = _norm_value(pred.get("value")) == _norm_value(gt_value)
    else:
        value_match = True

    passed = bool(element_match and op_match and value_match)
    err = None
    if not passed:
        missing = []
        if not element_match:
            missing.append(
                f"element (pred={pred['target_bnid']!r} not in {sorted(target_bnids)!r})"
            )
        if not op_match:
            missing.append(f"op (pred={pred_op!r} gt={gt_op!r})")
        if not value_match:
            missing.append(
                f"value (pred={pred.get('value')!r} gt={gt_value!r})"
            )
        err = "; ".join(missing)

    return {
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "element_match": bool(element_match),
        "op_match": bool(op_match),
        "value_match": bool(value_match),
        "extracted_prediction": pred,
        "error_message": err,
    }


__all__ = ["evaluate_mind2web_output"]
