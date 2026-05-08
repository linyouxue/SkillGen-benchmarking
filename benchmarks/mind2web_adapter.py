"""Mind2Web adapter (static HTML action-prediction variant).

Each action (one step within a task) becomes one instance in our
{instance_id, input, ground_truth, metadata} schema:

- `input`   = Task description + previous-action history + cleaned HTML of the
              current page (truncated) + candidate element list.
- `ground_truth` = the target element's `backend_node_id` plus the operation
              type and (if any) the typed / selected value. Rendered as a short
              structured string so the judge/grader can diff exactly.
- `metadata` = pointers to target_bnid / op / value / candidate ids so the
              executor can score without re-parsing natural language.

The model is asked to output a strict JSON object:
    {"target_bnid": "<id>", "op": "CLICK" | "TYPE" | "SELECT", "value": "..."}

Grading (see `mind2web_grader.py`) is fully rule-based:
  - element_match  = predicted backend_node_id  in  ground-truth target ids
  - op_match       = predicted op == ground-truth op
  - value_match    = normalised value equality (only relevant for TYPE / SELECT)
  - step_success   = all three must match (TYPE/SELECT: value required)
"""

from __future__ import annotations

import json
import re
from typing import Any


# HTML / candidate rendering

_HTML_MAX_CHARS = 8000  # keeps prompt under ~3k tokens for typical cases
_MAX_CANDIDATES_RENDERED = 30


def _parse_attrs(attrs_str: str) -> dict[str, Any]:
    if not attrs_str:
        return {}
    try:
        return json.loads(attrs_str)
    except Exception:
        return {}


def _candidate_snippet(cand: dict[str, Any], max_attr_chars: int = 140) -> str:
    """Render one candidate element as <tag bnid=... attr=...>."""
    attrs = _parse_attrs(cand.get("attributes") or "")
    keep = {}
    for k in ("id", "class", "name", "type", "role", "aria_label",
              "placeholder", "value", "title", "text", "alt"):
        v = attrs.get(k)
        if v:
            keep[k] = (str(v)[:80])
    tag = cand.get("tag") or "*"
    bnid = cand.get("backend_node_id") or "?"
    attr_str = " ".join(f'{k}="{v}"' for k, v in keep.items())
    if len(attr_str) > max_attr_chars:
        attr_str = attr_str[:max_attr_chars] + "..."
    return f'<{tag} bnid="{bnid}"{" " + attr_str if attr_str else ""}>'


def _truncate_html(html: str) -> str:
    if not html:
        return ""
    if len(html) <= _HTML_MAX_CHARS:
        return html
    head = html[: _HTML_MAX_CHARS - 200]
    return head + f"\n... [HTML truncated; original length {len(html)}] ..."


def _history_line(act_repr: str) -> str:
    # Mind2Web's `action_reprs` entries look like
    #   [span]  Pickup -> CLICK
    #   [input]  Boston -> TYPE: Boston
    # We keep them as-is; they're already compact.
    return act_repr.strip()


# Public: convert one raw Mind2Web action -> pipeline instance

def convert_mind2web_action(
    task: dict[str, Any],
    action_idx: int,
) -> dict[str, Any] | None:
    """Convert one step (`task["actions"][action_idx]`) into our instance schema.

    Returns None if the action has no positive candidate (label missing after
    cleaning) or no cleaned_html - those are unusable for action prediction.
    """
    actions = task.get("actions") or []
    if action_idx >= len(actions):
        return None
    act = actions[action_idx]
    pos = act.get("pos_candidates") or []
    if not pos:
        return None
    html = act.get("cleaned_html") or ""
    if not html:
        return None

    target_bnid = str(pos[0].get("backend_node_id") or "").strip()
    all_target_bnids = [str(p.get("backend_node_id") or "").strip()
                        for p in pos if p.get("backend_node_id")]
    if not target_bnid:
        return None

    op = act.get("operation") or {}
    op_type = str(op.get("op") or "").strip().upper()
    op_value = op.get("value") or ""

    # Candidate list: positive target(s) first, then sampled negatives.
    neg = act.get("neg_candidates") or []
    # deterministic ordering: by backend_node_id (stable across runs)
    neg_sorted = sorted(neg, key=lambda c: str(c.get("backend_node_id") or ""))
    candidates = pos + neg_sorted[: max(0, _MAX_CANDIDATES_RENDERED - len(pos))]
    cand_lines = [f"  {i+1}. {_candidate_snippet(c)}" for i, c in enumerate(candidates)]
    cand_block = "\n".join(cand_lines) if cand_lines else "  (none)"

    # History = previous actions' action_reprs.
    action_reprs = task.get("action_reprs") or []
    history_lines = [_history_line(r) for r in action_reprs[:action_idx]]
    history_block = "\n".join(f"  {i+1}. {h}" for i, h in enumerate(history_lines)) \
        if history_lines else "  (first step)"

    task_desc = (task.get("confirmed_task") or "").strip()
    website = task.get("website") or ""
    domain = task.get("domain") or ""
    subdomain = task.get("subdomain") or ""

    prompt = "\n".join([
        "You are a web navigation agent. Given the current page and task history,",
        "predict the SINGLE next action.",
        "",
        f"Task: {task_desc}",
        f"Website: {website} ({domain} / {subdomain})",
        "",
        "Previous actions (already executed):",
        history_block,
        "",
        "Available candidate elements on the current page (bnid = backend_node_id):",
        cand_block,
        "",
        "Current page (cleaned HTML, may be truncated):",
        "```html",
        _truncate_html(html),
        "```",
        "",
        "Respond with EXACTLY one fenced JSON object (no extra prose):",
        "```json",
        '{"target_bnid": "<backend_node_id>", "op": "CLICK" | "TYPE" | "SELECT", "value": "<text or option if TYPE/SELECT else empty string>"}',
        "```",
        "",
        "Rules:",
        '- `target_bnid` MUST be one of the `bnid` values listed above.',
        "- For CLICK actions, set `value` to an empty string.",
        "- For TYPE, `value` is the text to type.",
        "- For SELECT, `value` is the option label to select.",
    ])

    ground_truth = json.dumps({
        "target_bnid": target_bnid,
        "op": op_type,
        "value": op_value,
    }, ensure_ascii=False)

    return {
        "instance_id": f"{task.get('annotation_id','')}__{action_idx}",
        "input": prompt,
        "ground_truth": ground_truth,
        "metadata": {
            "benchmark": "mind2web",
            "annotation_id": task.get("annotation_id"),
            "action_idx": action_idx,
            "action_uid": act.get("action_uid"),
            "website": website,
            "domain": domain,
            "subdomain": subdomain,
            "target_bnids": all_target_bnids,
            "op": op_type,
            "value": op_value,
            "candidate_bnids": [str(c.get("backend_node_id") or "")
                                for c in candidates],
        },
    }


# Parsing the model's output back into structured form

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_mind2web_prediction(raw: str) -> dict[str, Any] | None:
    """Extract the predicted action from a model response.

    Returns a dict with keys target_bnid / op / value, or None if unparseable.
    Tolerant to surrounding prose and to missing code fences.
    """
    if not raw:
        return None
    candidates: list[str] = []
    for m in _JSON_FENCE_RE.findall(raw):
        candidates.append(m.strip())
    candidates.append(raw.strip())
    # Also try to locate the last {...} block
    braces = re.findall(r"\{[^{}]*\}", raw, re.DOTALL)
    candidates.extend(braces[-3:])

    for cand in candidates:
        try:
            obj = json.loads(cand)
            if not isinstance(obj, dict):
                continue
            bnid = obj.get("target_bnid") or obj.get("bnid") or ""
            op = (obj.get("op") or "").upper().strip()
            val = obj.get("value") or ""
            if bnid and op:
                return {
                    "target_bnid": str(bnid).strip(),
                    "op": op,
                    "value": str(val),
                }
        except json.JSONDecodeError:
            continue
    return None


__all__ = [
    "convert_mind2web_action",
    "parse_mind2web_prediction",
]
