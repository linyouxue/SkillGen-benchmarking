"""ScienceWorld grader (offline plan variant).

Contract (mirrors pubmedqa_grader / mcp_bench_grader):

    evaluate_scienceworld_output(
        model_output: str,
        instance_metadata: dict,
        *, judge_model: str | None = None,
    ) -> dict

Returned dict fields:
    - passed         : bool    - overall score >= threshold
    - score          : float   - overall score in [0, 1]
    - subscores      : dict    - each of the four sub-dimensions in [0, 1]
    - raw_scores     : dict    - each sub-dimension as the judge's 1-10 integer
    - reasoning      : dict    - judge's per-dimension explanation
    - extracted_plan : list[str] | None  - parsed action list from the agent
    - key_coverage   : dict    - deterministic substring coverage of gold key
                                 actions (independent of the LLM judge, useful
                                 for diagnostics)
    - error_message  : str | None
    - judge_model    : str

Grading design
--------------
ScienceWorld has many valid solution paths per task (the 29 tasks x 5000+
variations share structural motifs), so exact action-sequence match is too
rigid. Instead we use a compact 4-dimension LLM-as-judge:

    1. goal_understanding    - does the plan target the task's stated goal?
    2. action_legality       - actions follow ScienceWorld's text-game grammar.
    3. key_step_coverage     - the plan includes analogues of each gold
                               score-changing action (focus, measurement,
                               state-change, answer).
    4. ordering_coherence    - actions appear in a feasible order (navigate
                               before inspect, pickup before use, focus before
                               read result).

Because key_step_coverage is the most important signal, we ALSO compute a
deterministic substring overlap score against the gold key actions and pass it
to the judge as evidence. The final overall score is the average of the four
subscores.

Threshold defaults to 0.55 (higher than mcp-bench's 0.45 because here the
reference path is concrete - no grounding penalty ambiguity). Configurable via
`SCIENCEWORLD_PASS_THRESHOLD`.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import llm


PASS_THRESHOLD_DEFAULT = 0.55

_FOUR_DIMS = (
    "goal_understanding",
    "action_legality",
    "key_step_coverage",
    "ordering_coherence",
)


# Plan extraction

# Numbered list lines: "1. go to kitchen", " 12) open door to kitchen",
# "* pick up thermometer" (rarely), also `1: look around`.
_NUMBERED_RE = re.compile(r"^\s*(?:\d+)[\.\)\:]\s*(.+?)\s*$")
_BULLET_RE = re.compile(r"^\s*[-**]\s+(.+?)\s*$")
_FENCE_RE = re.compile(r"```(?:\w+)?\s*\n(.*?)```", re.DOTALL)


def _extract_plan(raw: str) -> list[str]:
    """Parse the agent's output into a flat ordered list of action strings."""
    if not raw:
        return []
    # Prefer the last fenced code block, if any - that's what our prompt asks
    # for. Fall through to whole-text scan if no fences.
    fenced = _FENCE_RE.findall(raw)
    source = fenced[-1] if fenced else raw

    actions: list[str] = []
    for line in source.splitlines():
        m = _NUMBERED_RE.match(line)
        if m:
            actions.append(m.group(1).strip())
            continue
        m = _BULLET_RE.match(line)
        if m:
            actions.append(m.group(1).strip())

    # If we got nothing, fall back to splitting on newlines (drop empties).
    if not actions:
        actions = [
            s.strip().strip("`\"'")
            for s in source.splitlines()
            if s.strip() and not s.strip().startswith("```")
        ]

    # Drop trailing empty entries and obvious commentary.
    cleaned = []
    for a in actions:
        a = a.strip()
        if not a:
            continue
        if len(a) > 200:  # lines this long are almost certainly prose, not actions
            continue
        cleaned.append(a)
    return cleaned


# Deterministic key-action coverage

def _normalise_action(s: str) -> str:
    """Lowercase, collapse whitespace, strip trailing punctuation."""
    return re.sub(r"\s+", " ", s.lower().strip().strip(".,:;!?\"'`")).strip()


def _key_verb(action: str) -> str:
    """Return the first token (the 'verb') from a normalised action."""
    norm = _normalise_action(action)
    return norm.split(" ", 1)[0] if norm else ""


def _key_action_tokens(action: str) -> set[str]:
    """Significant tokens from a gold action, for looser overlap matching."""
    norm = _normalise_action(action)
    # Remove ultra-common stopwords that don't carry semantics here.
    stop = {"to", "on", "in", "the", "a", "an", "at", "with", "and", "of", "for", "from"}
    toks = [t for t in norm.split() if t and t not in stop]
    return set(toks)


def compute_key_coverage(
    agent_actions: list[str],
    gold_key_actions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Deterministic coverage metric: for each gold key action, check whether
    the agent's plan contains an action with the same verb AND >=50% of the
    gold's meaningful non-stop tokens.

    Returns:
      - matched: list[{gold_action, matched_agent_action | None}]
      - coverage_rate: float in [0, 1]  (matched / len(gold_keys))
    """
    if not gold_key_actions:
        return {"matched": [], "coverage_rate": 1.0}
    agent_norm = [(a, _normalise_action(a), _key_verb(a), _key_action_tokens(a)) for a in agent_actions]
    matched: list[dict[str, Any]] = []
    for gk in gold_key_actions:
        ga = gk.get("action") or ""
        g_verb = _key_verb(ga)
        g_toks = _key_action_tokens(ga)
        if not g_toks:
            matched.append({"gold_action": ga, "matched_agent_action": None})
            continue
        best = None
        for raw_a, na, verb, toks in agent_norm:
            if verb != g_verb:
                continue
            overlap = len(g_toks & toks) / max(1, len(g_toks))
            if overlap >= 0.5:
                best = raw_a
                break
        matched.append({"gold_action": ga, "matched_agent_action": best})
    hit = sum(1 for m in matched if m["matched_agent_action"])
    return {
        "matched": matched,
        "coverage_rate": hit / len(gold_key_actions),
        "hit": hit,
        "total": len(gold_key_actions),
    }


# LLM judge

_JUDGE_SYSTEM = (
    "You are an expert evaluator for ScienceWorld (text-based science "
    "environment). The agent produced an offline numbered action plan for a "
    "task. Compare it to the gold reference KEY ACTIONS (actions that changed "
    "the environment's score). Score each of four dimensions on an integer "
    "1-10 scale. Use the full 1-10 range; default to 4-5 unless evidence "
    "supports higher. 9-10 requires near-perfect coverage and ordering; 7-8 "
    "for clearly correct plans with minor gaps; 1-3 for severely flawed."
)

_JUDGE_PROMPT = """\
## Task goal
{task_description}

## Initial scene (abbreviated)
{initial_obs}

## Gold reference key actions (score-changing steps, in order)
{gold_key_actions}

## Agent's parsed plan
{agent_plan}

## Deterministic coverage (substring-level, for reference)
coverage_rate = {coverage_rate:.3f}  ({hit}/{total} key actions matched)
Matched gold key actions: {matched_briefs}

### Scoring rubric (integers 1-10 per dimension)

1. **goal_understanding** - Does the plan target the task's stated goal? \
Penalise plans that ignore the measurement / state-change / answer action \
implied by the goal.

2. **action_legality** - Do the listed actions use ScienceWorld's text-game \
grammar (canonical verbs like `go to`, `pick up`, `focus on`, `use thermometer \
on`, `activate`, `mix`, `answer`)? Penalise freeform prose or non-existent \
verbs.

3. **key_step_coverage** - Does the plan contain analogues of each gold \
key action? Use the deterministic coverage above as a floor. 9-10 requires \
>=90% coverage; 7-8 for 60-90%; 4-6 for 30-60%; 1-3 for <30%.

4. **ordering_coherence** - Do actions appear in a feasible order? (navigate \
before inspect/pickup, focus before measure, pickup object before using it, \
answer last.) Penalise blatantly impossible orders.

Respond with EXACTLY one JSON object (no prose, no code fence):
{{
  "goal_understanding": <int 1-10>,
  "action_legality": <int 1-10>,
  "key_step_coverage": <int 1-10>,
  "ordering_coherence": <int 1-10>,
  "reasoning": {{
    "goal_understanding": "<one sentence>",
    "action_legality": "<one sentence>",
    "key_step_coverage": "<one sentence>",
    "ordering_coherence": "<one sentence>"
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
    raw = os.environ.get("SCIENCEWORLD_PASS_THRESHOLD")
    if not raw:
        return PASS_THRESHOLD_DEFAULT
    try:
        v = float(raw)
    except ValueError:
        return PASS_THRESHOLD_DEFAULT
    return max(0.0, min(1.0, v))


def _format_key_actions(key_actions: list[dict[str, Any]]) -> str:
    if not key_actions:
        return "(no gold key actions)"
    lines = []
    for i, ka in enumerate(key_actions, 1):
        tag = " [DONE]" if ka.get("is_completed") else ""
        lines.append(f"{i:>2}. score={ka.get('score', 0):.3f}{tag}  {ka.get('action', '')!r}")
    return "\n".join(lines)


def _format_matched_briefs(matched: list[dict[str, Any]]) -> str:
    items = []
    for m in matched:
        g = m.get("gold_action") or ""
        a = m.get("matched_agent_action")
        a_disp = repr(a) if a else "(none)"
        items.append(f"  - {g!r} -> {a_disp}")
    return "\n" + "\n".join(items) if items else " (none)"


# Main entrypoint

def evaluate_scienceworld_output(
    model_output: str,
    instance_metadata: dict[str, Any],
    *,
    judge_model: str | None = None,
) -> dict[str, Any]:
    meta = instance_metadata or {}
    judge = judge_model or os.environ.get("SCIENCEWORLD_JUDGE_MODEL") or "openai/gpt-5.4-mini"
    threshold = _get_pass_threshold()

    agent_plan = _extract_plan(str(model_output or ""))
    gold_keys = meta.get("key_actions") or []
    coverage = compute_key_coverage(agent_plan, gold_keys)

    if not agent_plan:
        # Short-circuit: no plan extracted at all.
        return {
            "passed": False,
            "score": 0.0,
            "subscores": {d: 0.0 for d in _FOUR_DIMS},
            "raw_scores": {d: 0 for d in _FOUR_DIMS},
            "reasoning": {},
            "extracted_plan": agent_plan,
            "key_coverage": coverage,
            "error_message": "no plan extracted from agent output",
            "judge_model": judge,
            "threshold": threshold,
        }

    prompt = _JUDGE_PROMPT.format(
        task_description=(meta.get("task_description") or "(not provided)").strip(),
        initial_obs=(meta.get("initial_observation") or "(not provided)").strip()[:1500],
        gold_key_actions=_format_key_actions(gold_keys)[:3500],
        agent_plan="\n".join(f"{i+1:>2}. {a}" for i, a in enumerate(agent_plan))[:3500],
        coverage_rate=coverage.get("coverage_rate", 0.0),
        hit=coverage.get("hit", 0),
        total=coverage.get("total", len(gold_keys)),
        matched_briefs=_format_matched_briefs(coverage.get("matched") or [])[:1500],
    )

    try:
        result = llm.chat_json(prompt, system=_JUDGE_SYSTEM, model=judge)
    except Exception as exc:
        return {
            "passed": False,
            "score": 0.0,
            "subscores": {d: 0.0 for d in _FOUR_DIMS},
            "raw_scores": {d: 0 for d in _FOUR_DIMS},
            "reasoning": {},
            "extracted_plan": agent_plan,
            "key_coverage": coverage,
            "error_message": f"judge_error: {exc}",
            "judge_model": judge,
            "threshold": threshold,
        }

    raw_scores = {d: _coerce_score(result.get(d)) for d in _FOUR_DIMS}
    subscores = {d: raw_scores[d] / 10.0 for d in _FOUR_DIMS}
    overall = sum(subscores.values()) / len(subscores)
    reasoning = result.get("reasoning") or {}
    if not isinstance(reasoning, dict):
        reasoning = {}

    passed = overall >= threshold
    err = None
    if not passed:
        weakest = min(subscores, key=subscores.get)
        err = f"overall={overall:.2f} < threshold={threshold:.2f} (weakest: {weakest})"

    return {
        "passed": bool(passed),
        "score": overall,
        "subscores": subscores,
        "raw_scores": raw_scores,
        "reasoning": reasoning,
        "extracted_plan": agent_plan,
        "key_coverage": coverage,
        "error_message": err,
        "judge_model": judge,
        "threshold": threshold,
    }


__all__ = ["evaluate_scienceworld_output", "PASS_THRESHOLD_DEFAULT", "compute_key_coverage"]
