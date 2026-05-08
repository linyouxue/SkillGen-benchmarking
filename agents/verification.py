"""Verification Agent: run skill vs baseline, analyse cases, synthesise revision guidance."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from artifacts import write_json
from models import (
    CandidateSkill,
    CaseAnalysis,
    SkillItem,
    TaskInstance,
    TaskType,
    Trajectory,
    VerificationFeedback,
)
from effectiveness import verify_effectiveness
from prompts import (
    CASE_ANALYSIS_PROMPT,
    CASE_ANALYSIS_SYSTEM,
    REVISION_SYNTHESIS_PROMPT,
    REVISION_SYNTHESIS_SYSTEM,
)
import llm

log = logging.getLogger("skillgen.verification")


_BUCKET_LABEL = {
    "repair": "REPAIR  (baseline FAILED -> skill PASSED)",
    "regression": "REGRESSION  (baseline PASSED -> skill FAILED)",
    "still_failing": "STILL_FAILING  (both FAILED)",
}


def _candidate_to_temp_skill(candidate: CandidateSkill) -> SkillItem:
    """Wrap a candidate as a transient SkillItem the runner can consume."""
    return SkillItem(
        skill_id=candidate.candidate_id,
        body=candidate.body,
        contextual_abstract=candidate.contextual_abstract,
        scripts=list(candidate.scripts),
        requirements=list(candidate.requirements),
        references=list(candidate.references),
        reference_docs=list(getattr(candidate, "reference_docs", []) or []),
        analysis_id=candidate.analysis_id,
    )


def _case_outcome_to_bucket(outcome: str) -> str:
    if outcome == "repair":
        return "repair"
    if outcome == "regression":
        return "regression"
    if outcome == "both_fail":
        return "still_failing"
    return outcome


def _analyse_one_case(
    case: dict,
    *,
    skill_body: str,
    model: str,
    max_tokens: int,
) -> CaseAnalysis:
    """Run the case-analyst LLM on a single case and return a CaseAnalysis."""
    bucket = _case_outcome_to_bucket(case.get("outcome") or "")
    try:
        result = llm.chat_json(
            CASE_ANALYSIS_PROMPT.format(
                bucket_label=_BUCKET_LABEL.get(bucket, bucket.upper()),
                instance_id=case.get("instance_id", "?"),
                task_input=case.get("task_input") or "",
                ground_truth=case.get("ground_truth") or "",
                baseline_success=case.get("baseline_success"),
                baseline_score=float(case.get("baseline_score") or 0.0),
                baseline_output=case.get("baseline_output") or "",
                skill_success=case.get("skill_success"),
                skill_score=float(case.get("skill_score") or 0.0),
                skill_output=case.get("skill_output") or "",
                skill_body=skill_body,
            ),
            system=CASE_ANALYSIS_SYSTEM,
            model=model,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        log.warning(
            "case analyst failed for instance=%s bucket=%s: %s",
            case.get("instance_id"), bucket, exc,
        )
        return CaseAnalysis(
            instance_id=str(case.get("instance_id") or ""),
            bucket=bucket,
            analysis=f"(analyst error: {exc})",
            skill_influence="unclear",
            micro_recommendation="(analyst unavailable - no recommendation)",
        )
    return CaseAnalysis(
        instance_id=str(case.get("instance_id") or ""),
        bucket=bucket,
        analysis=(result.get("analysis") or "").strip(),
        skill_influence=(result.get("skill_influence") or "unclear").strip(),
        micro_recommendation=(result.get("micro_recommendation") or "").strip(),
    )


def _format_case_analyses_for_synthesis(analyses: list[CaseAnalysis]) -> str:
    """Group per-case analyses by bucket for the synthesiser prompt."""
    if not analyses:
        return "(no cases)"
    groups: dict[str, list[CaseAnalysis]] = {
        "repair": [], "regression": [], "still_failing": [],
    }
    for ca in analyses:
        groups.setdefault(ca.bucket, []).append(ca)

    parts: list[str] = []
    for bucket in ("repair", "regression", "still_failing"):
        entries = groups.get(bucket, [])
        header = f"### {_BUCKET_LABEL.get(bucket, bucket.upper())}  ({len(entries)} cases)"
        if not entries:
            parts.append(header + "\n  (no cases in this bucket)")
            continue
        lines = [header]
        for ca in entries:
            lines.append(
                f"- instance_id={ca.instance_id}  |  skill_influence={ca.skill_influence}\n"
                f"  analysis: {ca.analysis}\n"
                f"  micro_recommendation: {ca.micro_recommendation}"
            )
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def _synthesise_revision_guidance(
    *,
    skill_body: str,
    diagnostic_summary: str,
    case_analyses: list[CaseAnalysis],
    model: str,
    max_tokens: int,
) -> str:
    """Single LLM call -> structured revision guidance (returned as JSON text)."""
    case_block = _format_case_analyses_for_synthesis(case_analyses)
    try:
        result = llm.chat_json(
            REVISION_SYNTHESIS_PROMPT.format(
                skill_body=skill_body,
                diagnostic_summary=diagnostic_summary,
                case_analyses_block=case_block,
            ),
            system=REVISION_SYNTHESIS_SYSTEM,
            model=model,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        log.warning("revision synthesiser failed: %s", exc)
        return json.dumps(
            {
                "overall_direction": (
                    "(synthesiser unavailable this round - preserve the current "
                    "skill unless regressions are severe; fall back to "
                    "per-case micro-recommendations below)"
                ),
                "keep": [], "remove_or_soften": [], "add_or_sharpen": [],
                "length_budget_note": "(no synthesiser output)",
            },
            indent=2,
            ensure_ascii=False,
        )
    # Return the JSON as nicely formatted text so the refiner sees a clean
    # structured document.
    return json.dumps(result, indent=2, ensure_ascii=False)


def run_verification(
    candidate: CandidateSkill,
    target_failure_instances: list[TaskInstance],
    boundary_success_instances: list[TaskInstance],
    task_type: TaskType,
    *,
    round_idx: int = 0,
    baseline_cache: dict[str, Trajectory] | None = None,
    baseline_agent_model: str = "openai/gpt-5.4-mini",
    baseline_judge_model: str = "openai/gpt-5.4-mini",
    effectiveness_agent_model: str = "openai/gpt-5.4-mini",
    effectiveness_judge_model: str = "openai/gpt-5.4-mini",
    # Per-case analyst LLM (runs once per diagnostic case).
    case_analyst_model: str = "openai/gpt-5.4-mini",
    case_analyst_max_workers: int = 8,
    case_analyst_max_tokens: int = 2048,
    # Revision synthesiser LLM (single call over all case analyses).
    revision_synthesiser_model: str = "openai/gpt-5.4-mini",
    revision_synthesiser_max_tokens: int = 4096,
    artifact_dir: str | None = None,
    artifact_prefix: str = "verification",
    progress_label: str | None = None,
    router_model: str | None = None,
    router_max_workers: int = 16,
    min_net_gain_abs: int = 1,
    min_net_gain_rel: float = 0.0,
) -> tuple[VerificationFeedback, dict[str, Trajectory]]:
    """Run effectiveness check -> per-case analysis -> revision synthesis."""

    temp_skill = _candidate_to_temp_skill(candidate)
    effectiveness, baseline_cache = verify_effectiveness(
        temp_skill,
        target_failure_instances,
        boundary_success_instances,
        task_type,
        baseline_cache=baseline_cache,
        baseline_agent_model=baseline_agent_model,
        baseline_judge_model=baseline_judge_model,
        agent_model=effectiveness_agent_model,
        judge_model=effectiveness_judge_model,
        artifact_dir=artifact_dir,
        artifact_prefix=artifact_prefix,
        progress_label=progress_label,
        router_model=router_model,
        router_max_workers=router_max_workers,
        min_net_gain_abs=min_net_gain_abs,
        min_net_gain_rel=min_net_gain_rel,
    )

    feedback = VerificationFeedback(effectiveness=effectiveness, round_idx=round_idx)

    cases = list(effectiveness.cases or [])
    if not cases:
        log.info(
            "[verification round %d] no diagnostic cases (all instances in "
            "stay_pass / stay_fail-unchanged) - skipping case analysis.",
            round_idx,
        )
        feedback.revision_guidance = json.dumps(
            {
                "overall_direction": (
                    "No repairs, regressions, or still-failing cases this "
                    "round - the skill had no measurable effect on any "
                    "instance. Preserve the current skill body."
                ),
                "keep": [], "remove_or_soften": [], "add_or_sharpen": [],
                "length_budget_note": "(no case signal)",
            },
            indent=2,
            ensure_ascii=False,
        )
        return feedback, baseline_cache

    # 1. Per-case analysis (parallel)
    log.info(
        "[verification round %d] running case analyst on %d case(s) "
        "(model=%s, workers=%d)",
        round_idx, len(cases), case_analyst_model, case_analyst_max_workers,
    )
    analyses: list[CaseAnalysis] = [None] * len(cases)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=max(1, int(case_analyst_max_workers))) as ex:
        futs = {
            ex.submit(
                _analyse_one_case,
                case,
                skill_body=candidate.body,
                model=case_analyst_model,
                max_tokens=case_analyst_max_tokens,
            ): idx
            for idx, case in enumerate(cases)
        }
        for fut in as_completed(futs):
            idx = futs[fut]
            try:
                analyses[idx] = fut.result()
            except Exception as exc:
                log.warning("case analyst task failed (idx=%d): %s", idx, exc)
                analyses[idx] = CaseAnalysis(
                    instance_id=str(cases[idx].get("instance_id") or ""),
                    bucket=_case_outcome_to_bucket(cases[idx].get("outcome") or ""),
                    analysis=f"(task error: {exc})",
                    skill_influence="unclear",
                    micro_recommendation="(analyst unavailable)",
                )
    feedback.case_analyses = [a for a in analyses if a is not None]

    # 2. Revision guidance synthesis (single call)
    log.info(
        "[verification round %d] synthesising revision guidance (model=%s) "
        "over %d case analyses",
        round_idx, revision_synthesiser_model, len(feedback.case_analyses),
    )
    feedback.revision_guidance = _synthesise_revision_guidance(
        skill_body=candidate.body,
        diagnostic_summary=effectiveness.diagnostic_summary,
        case_analyses=feedback.case_analyses,
        model=revision_synthesiser_model,
        max_tokens=revision_synthesiser_max_tokens,
    )

    # 3. Persist diagnostic artefacts for audit
    if artifact_dir:
        out_dir = Path(artifact_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            str(out_dir / f"{artifact_prefix}_case_analyses.json"),
            {
                "round_idx": round_idx,
                "case_analyst_model": case_analyst_model,
                "revision_synthesiser_model": revision_synthesiser_model,
                "case_analyses": [ca.__dict__ for ca in feedback.case_analyses],
                "revision_guidance": feedback.revision_guidance,
            },
        )

    return feedback, baseline_cache
