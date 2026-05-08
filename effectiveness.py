"""Protocol A: paired instance-level before/after effectiveness verification."""

from __future__ import annotations

import copy
import logging

from artifacts import write_json, write_trajectories
from models import (
    TaskInstance, TaskType, SkillItem, EffectivenessResult, Trajectory,
)
from trajectory import collect_trajectories, AgentConfig
from router import route_instances

log = logging.getLogger("skillgen.effectiveness")


def _build_case_example(
    inst: TaskInstance,
    baseline_traj: Trajectory,
    skill_traj: Trajectory,
    *,
    bucket: str,
    outcome: str,
) -> dict:
    """Return the full (un-truncated) case payload used by the case analyst.

    The case analyst step downstream needs the raw task input, ground truth
    and both full final outputs to reason about *why* each case flipped. We
    do not apply any length limit here - the refinement step no longer sees
    these cases directly, it only sees the distilled analyses.
    """
    return {
        "instance_id": inst.instance_id,
        "bucket": bucket,
        "outcome": outcome,
        "task_input": str(inst.input or ""),
        "ground_truth": str(inst.ground_truth or ""),
        "baseline_output": str(baseline_traj.final_output or ""),
        "skill_output": str(skill_traj.final_output or ""),
        "baseline_success": bool(baseline_traj.success),
        "skill_success": bool(skill_traj.success),
        "baseline_score": float(getattr(baseline_traj, "score", 0.0) or 0.0),
        "skill_score": float(getattr(skill_traj, "score", 0.0) or 0.0),
    }


def collect_baseline_cache(
    instances: list[TaskInstance],
    task_type: TaskType,
    *,
    config: AgentConfig | None = None,
    progress_desc: str | None = None,
) -> dict[str, Trajectory]:
    """Run baseline agent once and return a cache keyed by instance_id."""
    trajs = collect_trajectories(
        instances,
        task_type,
        config=config,
        progress_desc=progress_desc,
    )
    return {t.instance_id: t for t in trajs}


def verify_effectiveness(
    skill: SkillItem,
    target_failure_instances: list[TaskInstance],
    boundary_success_instances: list[TaskInstance],
    task_type: TaskType,
    *,
    baseline_cache: dict[str, Trajectory] | None = None,
    baseline_agent_model: str = "openai/gpt-5.4-mini",
    baseline_judge_model: str = "openai/gpt-5.4-mini",
    agent_model: str = "openai/gpt-5.4-mini",
    judge_model: str = "openai/gpt-5.4-mini",
    temperature: float = 0.0,
    artifact_dir: str | None = None,
    artifact_prefix: str = "effectiveness",
    progress_label: str | None = None,
    router_model: str | None = None,
    router_max_workers: int = 16,
    min_net_gain_abs: int = 1,
    min_net_gain_rel: float = 0.0,
) -> tuple[EffectivenessResult, dict[str, Trajectory]]:
    """
    Protocol A unified effectiveness check.
    Returns (result, baseline_cache) so callers can reuse the cache.
    """
    baseline_config = AgentConfig(
        model=baseline_agent_model,
        judge_model=baseline_judge_model,
        temperature=temperature,
    )
    # If the candidate skill ships executable scripts, verification MUST exec
    # them - otherwise tool calls return mock strings and every resource-path
    # skill looks no better than the body-only path. Baseline stays unchanged
    # (no skill = no scripts = no execution).
    skill_has_scripts = bool(getattr(skill, "scripts", None))
    skill_config = AgentConfig(
        model=agent_model,
        judge_model=judge_model,
        temperature=temperature,
        execute_scripts=skill_has_scripts,
    )
    all_instances = target_failure_instances + boundary_success_instances

    # Reuse or build baseline cache
    if baseline_cache is None:
        baseline_cache = collect_baseline_cache(
            all_instances,
            task_type,
            config=baseline_config,
            progress_desc=f"{progress_label or artifact_prefix} baseline",
        )
    else:
        missing = [i for i in all_instances if i.instance_id not in baseline_cache]
        if missing:
            extra = collect_trajectories(
                missing,
                task_type,
                config=baseline_config,
                progress_desc=f"{progress_label or artifact_prefix} baseline",
            )
            for t in extra:
                baseline_cache[t.instance_id] = t

    # Optional router: decide per-instance whether the skill should be applied.
    # If the router says "no" for an instance, we skip the skill run for it and
    # reuse the baseline trajectory as the skill trajectory (no regression
    # possible for bypassed instances). This mirrors real-world deployment:
    # a router gates whether the skill prompt is even shown to the agent.
    router_decisions: dict[str, tuple[bool, str]] = {}
    route_in_instances: list[TaskInstance] = all_instances
    route_out_instances: list[TaskInstance] = []
    if router_model:
        router_decisions = route_instances(
            all_instances,
            skill,
            model=router_model,
            max_workers=router_max_workers,
            progress_desc=f"{progress_label or artifact_prefix} router",
        )
        route_in_instances = [
            inst for inst in all_instances
            if router_decisions.get(inst.instance_id, (False, ""))[0]
        ]
        route_out_instances = [
            inst for inst in all_instances
            if not router_decisions.get(inst.instance_id, (False, ""))[0]
        ]
        target_failure_ids_for_log = {inst.instance_id for inst in target_failure_instances}
        n_target_in = sum(
            1 for inst in route_in_instances
            if inst.instance_id in target_failure_ids_for_log
        )
        n_boundary_in = len(route_in_instances) - n_target_in
        log.info(
            "Router | model=%s | route_in=%d (target=%d, boundary=%d) | route_out=%d of %d",
            router_model, len(route_in_instances), n_target_in, n_boundary_in,
            len(route_out_instances), len(all_instances),
        )

    # Only run skill-augmented on route-in instances (always fresh, since skill
    # changes each round). If the router rejected every instance, we skip the
    # LLM calls entirely and rely on baseline clones below.
    skill_map: dict[str, Trajectory] = {}
    skill_trajs: list[Trajectory] = []
    if route_in_instances:
        skill_trajs = collect_trajectories(
            route_in_instances,
            task_type,
            skill=skill,
            config=skill_config,
            progress_desc=f"{progress_label or artifact_prefix} with skill",
        )
        skill_map = {t.instance_id: t for t in skill_trajs}

    # For route-out instances, clone the baseline trajectory so downstream
    # paired statistics see (baseline_result == skill_result) - neither repair
    # nor regression - and tag metadata with the routing decision for audit.
    for inst in route_out_instances:
        base_traj = baseline_cache.get(inst.instance_id)
        if base_traj is None:
            continue
        clone = copy.copy(base_traj)
        clone.metadata = dict(base_traj.metadata or {})
        apply_decision, reason = router_decisions.get(inst.instance_id, (False, ""))
        clone.metadata["router"] = {
            "applied": False,
            "reason": reason,
            "model": router_model,
            "skill_id": skill.skill_id,
        }
        clone.agent_config = dict(base_traj.agent_config or {})
        clone.agent_config["skill_id"] = skill.skill_id
        clone.agent_config["skill_bypassed_by_router"] = True
        skill_map[inst.instance_id] = clone

    # Build paired outcomes
    repair, regression = 0, 0
    baseline_fail_count, baseline_pass_count = 0, 0
    repaired_ids: list[str] = []
    regression_ids: list[str] = []
    failed_ids_after_skill: list[str] = []
    paired_count = 0
    baseline_success_total = 0
    skill_success_total = 0
    target_failure_ids = {inst.instance_id for inst in target_failure_instances}
    per_instance_records: list[dict] = []
    target_repair_count = 0
    target_fail_count = 0
    success_guard_regression_count = 0
    success_guard_pass_count = 0
    # ALL cases in the 3 diagnostic buckets (no cap, no truncation) - consumed
    # by the verification agent's per-case analyser downstream.
    cases: list[dict] = []

    for inst in all_instances:
        bt = baseline_cache.get(inst.instance_id)
        st = skill_map.get(inst.instance_id)
        if bt is None or st is None:
            continue
        paired_count += 1
        bp, sp = bt.success, st.success
        baseline_success_total += int(bool(bp))
        skill_success_total += int(bool(sp))
        in_target_bucket = inst.instance_id in target_failure_ids

        case_bucket = "target_failure" if in_target_bucket else "success_guard"

        if not bp:
            baseline_fail_count += 1
            if in_target_bucket:
                target_fail_count += 1
            if sp:
                repair += 1
                repaired_ids.append(inst.instance_id)
                if in_target_bucket:
                    target_repair_count += 1
                outcome = "repair"
                cases.append(_build_case_example(inst, bt, st, bucket=case_bucket, outcome=outcome))
            else:
                failed_ids_after_skill.append(inst.instance_id)
                outcome = "both_fail"
                cases.append(_build_case_example(inst, bt, st, bucket=case_bucket, outcome=outcome))
        else:
            baseline_pass_count += 1
            if not in_target_bucket:
                success_guard_pass_count += 1
            if not sp:
                regression += 1
                regression_ids.append(inst.instance_id)
                if not in_target_bucket:
                    success_guard_regression_count += 1
                outcome = "regression"
                cases.append(_build_case_example(inst, bt, st, bucket=case_bucket, outcome=outcome))
            else:
                outcome = "both_pass"

        per_instance_records.append(
            {
                "instance_id": inst.instance_id,
                "bucket": (
                    "target_failure" if in_target_bucket else "success_guard"
                ),
                "baseline_success": bool(bp),
                "skill_success": bool(sp),
                "outcome": outcome,
                "baseline_output": str(bt.final_output or ""),
                "skill_output": str(st.final_output or ""),
            }
        )

    repair_rate = repair / baseline_fail_count if baseline_fail_count else 0.0
    regression_rate = regression / baseline_pass_count if baseline_pass_count else 0.0
    net_gain = repair - regression
    baseline_acc = baseline_success_total / paired_count if paired_count else 0.0
    skill_acc = skill_success_total / paired_count if paired_count else 0.0
    # Single-skill gate: the net gain must clear both an absolute floor and a
    # proportional (sample-size-scaled) floor. Requiring both guards against
    # tiny-sample flukes - a +1 net_gain on n=60 is noise, but +6 on n=120 is
    # meaningful signal.
    import math
    required_net_gain = max(
        int(min_net_gain_abs),
        int(math.ceil(float(min_net_gain_rel) * paired_count)),
    )
    # Guarantee that the gate is at least strictly positive so the name still
    # matches intuition even when both knobs are set to 0.
    required_net_gain = max(required_net_gain, 1)
    passed = net_gain >= required_net_gain
    diagnostic = (
        "mode=net_gain_gated, "
        f"paired_n={paired_count}, "
        f"n_target={len(target_failure_instances)}, n_success_guard={len(boundary_success_instances)}, "
        f"baseline_acc={baseline_acc:.1%}, skill_acc={skill_acc:.1%}, "
        f"repair={repair}/{baseline_fail_count} ({repair_rate:.1%}), "
        f"regression={regression}/{baseline_pass_count} ({regression_rate:.1%}), "
        f"net_gain={net_gain}, "
        f"required={required_net_gain} (abs={int(min_net_gain_abs)}, rel={float(min_net_gain_rel):.2f}*n), "
        f"passed={passed}"
    )

    eff = EffectivenessResult(
        passed=passed,
        n_target=len(target_failure_instances),
        n_boundary=len(boundary_success_instances),
        paired_n=paired_count,
        baseline_acc=baseline_acc,
        skill_acc=skill_acc,
        repair_count=repair,
        regression_count=regression,
        repair_rate=repair_rate,
        regression_rate=regression_rate,
        net_gain=net_gain,
        target_repair_count=target_repair_count,
        target_fail_count=target_fail_count,
        success_guard_regression_count=success_guard_regression_count,
        success_guard_pass_count=success_guard_pass_count,
        repaired_ids=repaired_ids,
        regression_ids=regression_ids,
        failed_ids_after_skill=failed_ids_after_skill,
        cases=cases,
        diagnostic_summary=diagnostic,
    )
    if artifact_dir:
        baseline_trajs = [baseline_cache[i.instance_id] for i in all_instances if i.instance_id in baseline_cache]
        baseline_path = write_trajectories(
            f"{artifact_dir}/{artifact_prefix}_baseline.jsonl",
            baseline_trajs,
        )
        skill_path = write_trajectories(
            f"{artifact_dir}/{artifact_prefix}_with_skill.jsonl",
            list(skill_map.values()),
        )
        write_json(
            f"{artifact_dir}/{artifact_prefix}_summary.json",
            {
                "skill_id": skill.skill_id,
                "artifact_prefix": artifact_prefix,
                "baseline_inference_model": baseline_agent_model,
                "baseline_judge_model": baseline_judge_model,
                "with_skill_inference_model": agent_model,
                "with_skill_judge_model": judge_model,
                "n_instances": len(all_instances),
                "target_failure_ids": [inst.instance_id for inst in target_failure_instances],
                "boundary_success_ids": [inst.instance_id for inst in boundary_success_instances],
                "baseline_path": str(baseline_path),
                "with_skill_path": str(skill_path),
                "result": {
                    "passed": eff.passed,
                    "paired_n": eff.paired_n,
                    "baseline_acc": eff.baseline_acc,
                    "skill_acc": eff.skill_acc,
                    "repair_count": eff.repair_count,
                    "regression_count": eff.regression_count,
                    "repair_rate": eff.repair_rate,
                    "regression_rate": eff.regression_rate,
                    "net_gain": eff.net_gain,
                    "target_repair_count": eff.target_repair_count,
                    "target_fail_count": eff.target_fail_count,
                    "success_guard_regression_count": eff.success_guard_regression_count,
                    "success_guard_pass_count": eff.success_guard_pass_count,
                    "repaired_ids": eff.repaired_ids,
                    "regression_ids": eff.regression_ids,
                    "failed_ids_after_skill": eff.failed_ids_after_skill,
                    "cases": eff.cases,
                    "diagnostic_summary": eff.diagnostic_summary,
                },
                "per_instance": per_instance_records,
            },
        )
    return eff, baseline_cache
