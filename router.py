"""Skill router: decide per-instance whether to apply a skill.

At verification-time and eval-time, a skill is force-injected on every instance
in the sample pool. For weaker agent models (llama-3.1-8b, mistral-small,
qwen-2.5-7b) this causes large regression rates on boundary-success instances
because the model blindly applies the skill's template even to unrelated tasks.

This module adds a cheap gate - typically openai/gpt-5.4-mini - that is shown
one task and one candidate skill and returns a binary "apply or not" decision.
When the router says "no", the caller reuses the baseline trajectory as the
skill-augmented trajectory (no regression possible).

Public surface
--------------
should_apply_skill(task_input, skill, *, model)
    Single-instance decision, returns (apply: bool, reason: str).

route_instances(instances, skill, *, model, max_workers, progress_desc)
    Batched, parallel routing. Returns {instance_id: (apply, reason)}.

Design notes
------------
* Prompt comes from prompts.SKILL_ROUTER_*.
* Failure mode is SAFE: any LLM error is interpreted as "apply=False", because
  a false-negative only loses potential repair (cost = 0), whereas a blind
  false-positive can cause real regression.
* Decisions are deterministic at temperature=0 so repeated calls for the same
  (task, skill) pair are cacheable by the caller if desired.
"""

from __future__ import annotations

from typing import Any

import llm
from models import SkillItem, TaskInstance
from prompts import SKILL_ROUTER_PROMPT, SKILL_ROUTER_SYSTEM


DEFAULT_ROUTER_MODEL = "openai/gpt-5.4-mini"
_TASK_INPUT_CHAR_LIMIT = 2000
_ROUTING_SUMMARY_CHAR_LIMIT = 500


def _routing_summary(body: str) -> str:
    """Extract a short "when to use / when NOT" summary from a skill body.

    Mirrors the logic used in eval_repo._judge_select_skill so training and
    eval see the same routing signal.
    """
    if not body:
        return "(empty skill body)"

    lines = [ln.rstrip() for ln in body.splitlines()]

    def collect_section(header_keywords: tuple[str, ...], char_cap: int = 240) -> str:
        start = None
        for i, line in enumerate(lines):
            low = line.strip().lower().lstrip("#").strip()
            if any(keyword in low for keyword in header_keywords):
                start = i + 1
                break
        if start is None:
            return ""
        collected = []
        for line in lines[start:]:
            stripped = line.strip()
            if stripped.startswith("#"):
                break
            if stripped:
                collected.append(stripped)
            if len(" ".join(collected)) >= char_cap:
                break
        return " ".join(collected)[:char_cap]

    when_use = collect_section(("when to use",))
    when_not = collect_section(("when not to use", "when not", "do not use"))
    parts: list[str] = []
    if when_use:
        parts.append(f"When to use: {when_use}")
    if when_not:
        parts.append(f"When NOT to use: {when_not}")
    if not parts:
        preview = " ".join(line.strip() for line in lines if line.strip())
        parts.append(f"Preview: {preview[:240]}")
    summary = "\n".join(parts)
    return summary[:_ROUTING_SUMMARY_CHAR_LIMIT]


def should_apply_skill(
    task_input: Any,
    skill: SkillItem,
    *,
    model: str = DEFAULT_ROUTER_MODEL,
) -> tuple[bool, str]:
    """Ask the router whether this skill should be applied to this task.

    Returns (apply, reason). Any error => (False, "router_error: ...").
    """
    if skill is None:
        return False, "no_skill"

    text = str(task_input or "")[:_TASK_INPUT_CHAR_LIMIT]
    contextual_abstract = (
        getattr(skill, "contextual_abstract", "")
        or getattr(skill, "semantic_aim", "")
        or "(none)"
    )
    prompt = SKILL_ROUTER_PROMPT.format(
        task_input=text,
        skill_id=skill.skill_id,
        contextual_abstract=contextual_abstract,
        routing_summary=_routing_summary(skill.body),
    )

    try:
        result = llm.chat_json(prompt, system=SKILL_ROUTER_SYSTEM, model=model)
    except Exception as exc:
        return False, f"router_error: {exc.__class__.__name__}"

    apply_raw = result.get("apply", False)
    if isinstance(apply_raw, str):
        apply_bool = apply_raw.strip().lower() in ("true", "yes", "1")
    else:
        apply_bool = bool(apply_raw)

    reason = str(result.get("reason", "")).strip() or ("apply" if apply_bool else "skip")
    return apply_bool, reason[:300]


def route_instances(
    instances: list[TaskInstance],
    skill: SkillItem,
    *,
    model: str = DEFAULT_ROUTER_MODEL,
    max_workers: int = 16,
    progress_desc: str | None = None,
) -> dict[str, tuple[bool, str]]:
    """Route a batch of instances against a single skill, in parallel.

    Returns {instance_id: (apply_bool, reason)}.
    """
    if not instances:
        return {}

    args_list = [(inst.input, skill) for inst in instances]

    def _call(task_input: Any, sk: SkillItem) -> tuple[bool, str]:
        return should_apply_skill(task_input, sk, model=model)

    results = llm.run_concurrent(
        _call,
        args_list,
        max_workers=max_workers,
        progress_desc=progress_desc,
    )

    return {inst.instance_id: result for inst, result in zip(instances, results)}
