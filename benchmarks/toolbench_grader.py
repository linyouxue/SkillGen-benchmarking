"""ToolBench grader: LLM-as-judge approximating the official Pass Rate rubric.

Contract (mirrors mcp_bench_grader / mind2web_grader):

    evaluate_toolbench_output(model_output: str, instance_metadata: dict,
                              *, judge_model: str | None = None) -> dict

Returned dict fields:
    - passed           : bool   - overall score >= threshold
    - score            : float  - overall score in [0, 1]
    - subscores        : dict   - each of the five sub-dimensions in [0, 1]
    - raw_scores       : dict   - each sub-dimension as the judge's 1-10 integer
    - reasoning        : dict   - judge's per-dimension explanation
    - extracted_plan   : dict | None  - parsed agent JSON ({plan, final_answer})
    - tool_usage_valid : bool   - all (category, tool, api) triples in the
                                  emitted plan are present in the instance's
                                  api_list (deterministic check)
    - error_message    : str | None
    - judge_model      : str    - which LLM was used as judge
    - threshold        : float  - pass threshold used

Rubric - 5 dimensions averaged to overall  in  [0,1]:
    1. task_fulfillment       - plan + final_answer cover the user's request
    2. tool_appropriateness   - picked APIs are the right fit (not a distractor)
    3. parameter_accuracy     - parameter keys/values are plausible for each API
    4. plan_coherence         - dependency order is sound, parallelism marked
    5. answer_groundedness    - final_answer stays consistent with what the
                                planned calls can actually return (no fabricated
                                concrete numbers/URLs/IDs)

Because this is the STATIC PLANNING variant (no live RapidAPI), we do NOT
score "pass rate" in the upstream sense of "tool calls actually succeed".
We score plan quality. Threshold default 0.45 mirrors MCP-Bench.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import llm


PASS_THRESHOLD_DEFAULT = 0.45

_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_FIVE_DIMS = (
    "task_fulfillment",
    "tool_appropriateness",
    "parameter_accuracy",
    "plan_coherence",
    "answer_groundedness",
)


def _extract_plan_json(raw: str) -> dict[str, Any] | None:
    """Parse the agent's fenced JSON output. Tolerant to surrounding prose."""
    if not raw:
        return None
    candidates: list[str] = [m.strip() for m in _FENCE_RE.findall(raw)]
    candidates.append(raw.strip())
    brace = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace:
        candidates.append(brace.group(0))
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _validate_tool_triples(
    plan: dict[str, Any] | None,
    api_list: list[dict[str, Any]],
) -> tuple[bool, list[dict[str, str]]]:
    """Check every plan step's (category, tool, api) triple is in api_list.

    Returns (all_valid, invalid_steps). `invalid_steps` contains the raw
    triples we could not match. An empty plan counts as *invalid* so the
    judge penalises no-op responses via task_fulfillment (not here).
    """
    if not plan or not isinstance(plan, dict):
        return False, []
    steps = plan.get("plan") or []
    if not isinstance(steps, list) or not steps:
        return False, []

    allowed: set[tuple[str, str, str]] = {
        (
            str(a.get("category_name") or "").strip().lower(),
            str(a.get("tool_name") or "").strip().lower(),
            str(a.get("api_name") or "").strip().lower(),
        )
        for a in api_list
        if isinstance(a, dict)
    }

    invalid: list[dict[str, str]] = []
    for step in steps:
        if not isinstance(step, dict):
            invalid.append({"step": repr(step)[:120]})
            continue
        triple = (
            str(step.get("category") or "").strip().lower(),
            str(step.get("tool") or "").strip().lower(),
            str(step.get("api") or "").strip().lower(),
        )
        if triple not in allowed:
            invalid.append({
                "category": step.get("category", ""),
                "tool": step.get("tool", ""),
                "api": step.get("api", ""),
            })
    return len(invalid) == 0, invalid


_JUDGE_SYSTEM = (
    "You are an expert evaluator for ToolBench (tool-use planning). "
    "Score each of five sub-dimensions from 1 to 10 based ONLY on evidence "
    "from the agent's structured plan and final answer, compared to the "
    "user request and the list of available APIs. "
    "IMPORTANT: this is the STATIC planning variant - the agent cannot "
    "actually execute tools, so the final_answer may be a placeholder "
    "synthesis. Do NOT penalise the agent for missing real tool output "
    "data; score against plan quality, not missing data. "
    "Use the full 1-10 range: 7-8 for clearly good plans, 9-10 only for "
    "near-perfect plans, 4-6 for partially correct, 1-3 for severely "
    "flawed or missing."
)


_JUDGE_PROMPT = """\
## User request
{query}

## Available APIs (the ONLY APIs the agent was allowed to use)
{api_list_rendered}

## Agent's raw output
{agent_raw}

## Parsed plan (may be null if the agent did not emit valid JSON)
{parsed_plan}

## Deterministic triple-check
{triple_check}

### Scoring rubric (integers 1-10 per dimension)

1. **task_fulfillment** - Percentage of the user's request that the plan + \
final answer actually addresses. (9-10: 90-100% | 7-8: 70-80% | 4-6: 40-60% \
| 1-3: 10-30%.)

2. **tool_appropriateness** - Every chosen (category, tool, api) triple is a \
sensible match for its sub-task. Picking the wrong API when a clearly better \
one is available in the list, or inventing a triple not in the list, counts \
as a defect. Triples flagged as invalid above should weigh heavily here.

3. **parameter_accuracy** - Each step's `parameters` object supplies keys \
and values that are plausible given the API's description and required \
parameters. Missing a required parameter or inventing nonsense values \
counts as a defect.

4. **plan_coherence** - Steps run in a sound dependency order; `depends_on` \
is set where a later step consumes an earlier step's output; parallelisable \
steps share a `depends_on` set; no redundant duplicate calls.

5. **answer_groundedness** - The `final_answer` does not invent concrete \
numbers, URLs, IDs, prices, dates, or names that no planned call would \
return. An honest placeholder synthesis ("Here is the health status and \
project list returned by SQUAKE...") scores 7+. Only penalise when the \
answer asserts specific factual content that the planned calls could not \
plausibly produce.

Respond with EXACTLY one JSON object (no prose, no code fence):
{{
  "task_fulfillment": <int 1-10>,
  "tool_appropriateness": <int 1-10>,
  "parameter_accuracy": <int 1-10>,
  "plan_coherence": <int 1-10>,
  "answer_groundedness": <int 1-10>,
  "reasoning": {{
    "task_fulfillment": "<one sentence>",
    "tool_appropriateness": "<one sentence>",
    "parameter_accuracy": "<one sentence>",
    "plan_coherence": "<one sentence>",
    "answer_groundedness": "<one sentence>"
  }}
}}
"""


def _coerce_score(v: Any) -> int:
    try:
        iv = int(round(float(v)))
    except (TypeError, ValueError):
        return 1
    return max(1, min(10, iv))


def _get_pass_threshold() -> float:
    raw = os.environ.get("TOOLBENCH_PASS_THRESHOLD")
    if not raw:
        return PASS_THRESHOLD_DEFAULT
    try:
        v = float(raw)
    except ValueError:
        return PASS_THRESHOLD_DEFAULT
    return max(0.0, min(1.0, v))


def _render_api_list_for_judge(api_list: list[dict[str, Any]]) -> str:
    """Compact api_list rendering for the judge prompt (truncates long descs)."""
    if not api_list:
        return "(empty api_list)"
    lines: list[str] = []
    for a in api_list:
        cat = a.get("category_name", "?")
        tool = a.get("tool_name", "?")
        api_name = a.get("api_name", "?")
        desc = (a.get("api_description") or "").strip().replace("\n", " ")[:160]
        req = [p.get("name") for p in (a.get("required_parameters") or []) if isinstance(p, dict)]
        lines.append(f"- ({cat}, {tool}, {api_name}) required={req} - {desc}")
    return "\n".join(lines)


def _render_triple_check(valid: bool, invalid: list[dict[str, str]]) -> str:
    if valid:
        return "All plan steps use triples that exist in the available APIs list."
    if not invalid:
        return "The plan is empty or not parseable."
    lines = ["The following plan steps reference APIs NOT in the available list:"]
    for s in invalid[:8]:
        lines.append(f"  - {s}")
    if len(invalid) > 8:
        lines.append(f"  ... ({len(invalid) - 8} more)")
    return "\n".join(lines)


def evaluate_toolbench_output(
    model_output: str,
    instance_metadata: dict[str, Any],
    *,
    judge_model: str | None = None,
) -> dict[str, Any]:
    meta = instance_metadata or {}
    judge = judge_model or os.environ.get("TOOLBENCH_JUDGE_MODEL") or "openai/gpt-5.4-mini"
    threshold = _get_pass_threshold()

    api_list = list(meta.get("api_list") or [])
    plan = _extract_plan_json(str(model_output or ""))
    triple_valid, invalid_triples = _validate_tool_triples(plan, api_list)
    parsed_plan_str = json.dumps(plan, ensure_ascii=False, indent=2) if plan else "null"

    prompt = _JUDGE_PROMPT.format(
        query=(meta.get("query") or "").strip(),
        api_list_rendered=_render_api_list_for_judge(api_list)[:9000],
        agent_raw=str(model_output or "").strip()[:12000],
        parsed_plan=parsed_plan_str[:6000],
        triple_check=_render_triple_check(triple_valid, invalid_triples),
    )

    try:
        result = llm.chat_json(prompt, system=_JUDGE_SYSTEM, model=judge)
    except Exception as exc:
        return {
            "passed": False,
            "score": 0.0,
            "subscores": {d: 0.0 for d in _FIVE_DIMS},
            "raw_scores": {d: 0 for d in _FIVE_DIMS},
            "reasoning": {},
            "extracted_plan": plan,
            "tool_usage_valid": triple_valid,
            "invalid_triples": invalid_triples,
            "error_message": f"judge_error: {exc}",
            "judge_model": judge,
            "threshold": threshold,
        }

    raw_scores = {d: _coerce_score(result.get(d)) for d in _FIVE_DIMS}
    subscores = {d: raw_scores[d] / 10.0 for d in _FIVE_DIMS}
    overall = sum(subscores.values()) / len(subscores)
    reasoning = result.get("reasoning") or {}
    if not isinstance(reasoning, dict):
        reasoning = {}

    passed = overall >= threshold
    err: str | None = None
    if plan is None:
        err = "agent did not emit valid JSON plan"
    elif not triple_valid:
        err = f"{len(invalid_triples)} plan step(s) reference APIs not in the available list"
    elif not passed:
        weakest = min(subscores, key=subscores.get)
        err = f"overall={overall:.2f} < threshold={threshold:.2f} (weakest: {weakest})"

    return {
        "passed": bool(passed),
        "score": overall,
        "subscores": subscores,
        "raw_scores": raw_scores,
        "reasoning": reasoning,
        "extracted_plan": plan,
        "tool_usage_valid": triple_valid,
        "invalid_triples": invalid_triples,
        "error_message": err,
        "judge_model": judge,
        "threshold": threshold,
    }


__all__ = ["evaluate_toolbench_output", "PASS_THRESHOLD_DEFAULT"]
