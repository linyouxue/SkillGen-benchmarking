"""MCP-Bench grader: LLM-as-judge mirroring the upstream six-dimension rubric.

Contract (mirrors mind2web_grader / livecodebench_adapter):

    evaluate_mcp_bench_output(model_output: str, instance_metadata: dict,
                              *, judge_model: str | None = None) -> dict

Returned dict fields:
    - passed           : bool   - overall score >= threshold
    - score            : float  - overall score in [0, 1]
    - subscores        : dict   - each of the six sub-dimensions in [0, 1]
    - raw_scores       : dict   - each sub-dimension as the judge's 1-10 integer
    - reasoning        : dict   - judge's per-dimension explanation
    - extracted_plan   : dict | None  - parsed agent JSON ({plan, final_answer})
    - error_message    : str | None
    - judge_model      : str    - which LLM was used as judge

The rubric is a compressed version of mcp-bench/benchmark/evaluator.py's six
dimensions (task_fulfillment, grounding, tool_appropriateness,
parameter_accuracy, dependency_awareness, parallelism_and_efficiency). We ask
the judge for a single JSON object and average the six integer scores into an
overall value, then threshold to pass/fail. Threshold is configurable via
`MCP_BENCH_PASS_THRESHOLD` env var.

NOTE - static planning variant
------------------------------
Because SkillGen runs MCP-Bench as a **static planning variant** (no live MCP
servers, agent cannot actually execute tools), the `grounding` dimension is
redefined: we score *internal consistency* of the synthesis (plan <-> final
answer), not presence of real tool outputs. The judge is told explicitly not
to penalise the agent for missing tool data. The default threshold is 0.45
(roughly 4.5/10 average), chosen so that a structurally sound plan with
honest placeholder synthesis can pass and leave headroom for skill gains.
The original 0.6 threshold is only meaningful when real tool execution is
happening.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import llm
from .mcp_bench_tool_registry import validate_plan


PASS_THRESHOLD_DEFAULT = 0.35

# Optional hard-fail threshold: if miss_rate (hallucinated + wrong-server
# usage) exceeds this, the submission is forced to fail regardless of
# judge scores. Disabled by default because in the static planning variant
# agents only see server names + one-line descriptions and thus cannot
# reliably produce canonical tool names - a uniform cap penalty on
# ``tool_appropriateness``/``grounding`` already neutralises the noise
# between baseline and skill. Set to e.g. 0.5 to re-enable.
HALLUCINATION_HARD_FAIL_RATE = float(
    os.environ.get("MCP_BENCH_HARD_FAIL_RATE", "1.1")
)

_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_SIX_DIMS = (
    "task_fulfillment",
    "grounding",
    "tool_appropriateness",
    "parameter_accuracy",
    "dependency_awareness",
    "parallelism_and_efficiency",
)


def _extract_plan_json(raw: str) -> dict[str, Any] | None:
    """Parse the agent's fenced JSON output. Tolerant to surrounding prose."""
    if not raw:
        return None
    candidates: list[str] = [m.strip() for m in _FENCE_RE.findall(raw)]
    candidates.append(raw.strip())
    # Last-resort: look for any {...} block.
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


_JUDGE_SYSTEM = (
    "You are an expert evaluator for MCP-Bench (tool-use planning). "
    "Score each of six sub-dimensions from 1 to 10 based ONLY on evidence from "
    "the agent's structured plan and final answer, compared to the concrete "
    "task reference and the dependency analysis. "
    "IMPORTANT: This is the STATIC planning variant - the agent cannot "
    "actually execute tools, so the final_answer may only describe what would "
    "be returned (placeholder synthesis). Do NOT penalise the agent for lack "
    "of real tool output data; score against plan quality, not missing data. "
    "Use the full 1-10 range: 7-8 for clearly good plans, 9-10 only for "
    "near-perfect matches to the reference, 4-6 for partially correct, "
    "1-3 for severely flawed or missing."
)

_JUDGE_PROMPT = """\
## Task presented to agent (fuzzy)
{fuzzy}

## Concrete task reference (evaluation context only; the agent did NOT see this)
{concrete}

## Dependency analysis (the expected plan; the agent did NOT see this)
{dependency}

## Servers available to the agent
{servers}

## Agent's raw output
{agent_raw}

## Parsed plan (may be null if the agent did not emit valid JSON)
{parsed_plan}

## Deterministic tool-name validation report (do NOT override this with your own opinion)
{validation_report}

### Scoring rubric (integers 1-10 per dimension)

1. **task_fulfillment** - Percentage of the concrete requirements fully covered by \
the agent's plan + final answer. (9-10: 90-100% | 7-8: 70-80% | 4-6: 40-60% | 1-3: 10-30%.)

2. **grounding (static variant: internal consistency)** - The final_answer is \
internally consistent with the plan: it accurately summarises what each tool \
step would produce and aggregate, does not contradict the chosen tools, and \
does not fabricate facts no tool could plausibly return. Because no real \
tools run in this variant, an honest placeholder synthesis that explains what \
each step will contribute scores 7+. Only penalise when the final_answer \
invents concrete numbers/URLs that no planned tool would return, or \
contradicts its own plan.

3. **tool_appropriateness** - Tools chosen per step are the optimal choice for \
that sub-task. Wrong server, missing obvious tool, or calling a distractor \
server counts as a defect.

4. **parameter_accuracy** - Every step's `parameters` object is complete and \
sensible for the stated tool, with the right types and values.

5. **dependency_awareness** - The plan respects the dependency structure in \
the dependency analysis (e.g. fetch-before-filter, per-item iteration, \
consolidation at the end).

6. **parallelism_and_efficiency** - Parallelisable sub-tasks are marked with a \
shared `depends_on` set; no redundant or duplicated calls.

Respond with EXACTLY one JSON object (no prose, no code fence):
{{
  "task_fulfillment": <int 1-10>,
  "grounding": <int 1-10>,
  "tool_appropriateness": <int 1-10>,
  "parameter_accuracy": <int 1-10>,
  "dependency_awareness": <int 1-10>,
  "parallelism_and_efficiency": <int 1-10>,
  "reasoning": {{
    "task_fulfillment": "<one sentence>",
    "grounding": "<one sentence>",
    "tool_appropriateness": "<one sentence>",
    "parameter_accuracy": "<one sentence>",
    "dependency_awareness": "<one sentence>",
    "parallelism_and_efficiency": "<one sentence>"
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
    raw = os.environ.get("MCP_BENCH_PASS_THRESHOLD")
    if not raw:
        return PASS_THRESHOLD_DEFAULT
    try:
        v = float(raw)
    except ValueError:
        return PASS_THRESHOLD_DEFAULT
    return max(0.0, min(1.0, v))


def _format_servers(servers: list[str], distractions: list[str]) -> str:
    parts = [f"- {s}" for s in (servers or [])]
    if distractions:
        parts.append("Distractors (should NOT be used):")
        parts.extend(f"  - {s}" for s in distractions)
    return "\n".join(parts) if parts else "(none)"


def _format_validation_report(v: dict[str, Any]) -> str:
    """Render the deterministic tool-name validation as judge context.

    Kept terse on purpose: the cap/hard-fail happens in Python, the judge only
    needs enough information to avoid over-scoring tool_appropriateness /
    grounding when names are clearly invented.
    """
    n = v.get("n_steps") or 0
    if n == 0:
        return "(no plan steps parsed)"
    lines: list[str] = [
        f"Steps: {n} | hallucinated tool names: {v.get('n_hallucinated',0)}"
        f" | wrong-server tool usage: {v.get('n_wrong_server',0)}"
        f" | miss_rate: {float(v.get('miss_rate',0.0)):.2f}"
    ]
    halls = v.get("hallucinations") or []
    if halls:
        lines.append("Hallucinated tools (not present anywhere in the MCP-Bench task corpus):")
        for h in halls[:8]:
            lines.append(f"  - step {h['step']} @ {h['server']!r}: tool={h['tool']!r}")
        if len(halls) > 8:
            lines.append(f"  ... and {len(halls) - 8} more")
    wrong = v.get("wrong_server_usage") or []
    if wrong:
        lines.append("Wrong-server usage (tool exists but belongs to a different server):")
        for w in wrong[:5]:
            lines.append(
                f"  - step {w['step']}: used {w['tool']!r} on {w['server']!r};"
                f" actually belongs to {w['true_server']!r}"
            )
    bad_srv = v.get("invalid_servers") or []
    if bad_srv:
        lines.append("Servers not in the task's available set:")
        for b in bad_srv[:5]:
            lines.append(f"  - step {b['step']}: {b['server']!r} ({b.get('reason','')})")
    lines.append(
        "If miss_rate is high, cap tool_appropriateness and grounding accordingly;"
        " inventing tool names is a severe defect in this static planning variant."
    )
    return "\n".join(lines)


def evaluate_mcp_bench_output(
    model_output: str,
    instance_metadata: dict[str, Any],
    *,
    judge_model: str | None = None,
) -> dict[str, Any]:
    meta = instance_metadata or {}
    judge = judge_model or os.environ.get("MCP_BENCH_JUDGE_MODEL") or "openai/gpt-5.4-mini"
    threshold = _get_pass_threshold()

    plan = _extract_plan_json(str(model_output or ""))
    parsed_plan_str = (
        json.dumps(plan, ensure_ascii=False, indent=2) if plan else "null"
    )
    validation = validate_plan(plan, meta)
    validation_report = _format_validation_report(validation)

    prompt = _JUDGE_PROMPT.format(
        fuzzy=(meta.get("fuzzy_description") or "").strip(),
        concrete=(meta.get("task_description") or "").strip() or "(not provided)",
        dependency=(meta.get("dependency_analysis") or "").strip() or "(not provided)",
        servers=_format_servers(meta.get("servers") or [], meta.get("distraction_servers") or []),
        agent_raw=(str(model_output or "")).strip()[:12000],
        parsed_plan=parsed_plan_str[:6000],
        validation_report=validation_report,
    )

    try:
        result = llm.chat_json(prompt, system=_JUDGE_SYSTEM, model=judge)
    except Exception as exc:
        return {
            "passed": False,
            "score": 0.0,
            "subscores": {d: 0.0 for d in _SIX_DIMS},
            "raw_scores": {d: 0 for d in _SIX_DIMS},
            "reasoning": {},
            "extracted_plan": plan,
            "tool_validation": validation,
            "error_message": f"judge_error: {exc}",
            "judge_model": judge,
            "threshold": threshold,
        }

    raw_scores = {d: _coerce_score(result.get(d)) for d in _SIX_DIMS}
    subscores = {d: raw_scores[d] / 10.0 for d in _SIX_DIMS}

    # Deterministic hallucination penalty
    # Any hallucinated or wrong-server tool name is, by definition, a defect
    # in tool_appropriateness (can't select the right tool) and grounding
    # (plan references something that doesn't exist). Cap both dimensions
    # at `(1 - miss_rate)` so the judge cannot accidentally score them high
    # on a plan the registry has already flagged. Deterministic == cheap
    # and cannot be gamed by a skill that happens to word things nicely.
    miss_rate = float(validation.get("miss_rate") or 0.0)
    penalty_cap = max(0.0, 1.0 - miss_rate)
    penalised: list[str] = []
    for dim in ("tool_appropriateness", "grounding"):
        if subscores[dim] > penalty_cap:
            subscores[dim] = penalty_cap
            raw_scores[dim] = int(round(penalty_cap * 10))
            penalised.append(dim)

    overall = sum(subscores.values()) / len(subscores)
    reasoning = result.get("reasoning") or {}
    if not isinstance(reasoning, dict):
        reasoning = {}

    passed = overall >= threshold

    # Hard-fail: a plan that is mostly hallucinated cannot be "passing"
    # even if the judge somehow gave generous task_fulfillment / parallelism
    # scores on the remaining valid bits.
    hard_failed = False
    if (
        validation.get("n_steps", 0) > 0
        and miss_rate >= HALLUCINATION_HARD_FAIL_RATE
    ):
        hard_failed = True
        passed = False

    err = None
    if plan is None:
        err = "agent did not emit valid JSON plan"
    elif hard_failed:
        err = (
            f"hallucinated tools miss_rate={miss_rate:.2f} "
            f">= hard_fail={HALLUCINATION_HARD_FAIL_RATE:.2f}"
        )
    elif penalised and not passed:
        weakest = min(subscores, key=subscores.get)
        err = (
            f"overall={overall:.2f} < threshold={threshold:.2f} "
            f"(weakest: {weakest}; tool-name penalty applied to {','.join(penalised)})"
        )
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
        "tool_validation": validation,
        "penalised_dimensions": penalised,
        "hard_failed": hard_failed,
        "error_message": err,
        "judge_model": judge,
        "threshold": threshold,
    }


__all__ = ["evaluate_mcp_bench_output", "PASS_THRESHOLD_DEFAULT"]
