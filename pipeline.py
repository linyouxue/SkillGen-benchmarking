"""End-to-end skill-discovery pipeline: baseline -> induction -> generation -> verification."""

from __future__ import annotations

import copy
import json
import logging
import random
import threading
import yaml
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from artifacts import (
    ensure_dir, make_run_dir, read_trajectories, write_json, write_trajectories,
    save_trajectories, load_trajectories, checkpoint_exists, load_progress, save_progress,
)
from models import (
    CandidateSkill, CaseAnalysis, EffectivenessResult, SkillAnalysis,
    SkillStatus, TaskInstance, TaskType, Trajectory, VerificationFeedback,
)
from trajectory import collect_trajectories, AgentConfig
from agents.induction import run_induction, save_analysis, load_analysis
from agents.generation import (
    generate_skill,
    generate_skill_with_resources,
    refine_skill,
)
from agents.verification import run_verification
from skill_store import finalize_skill, save_skill
import llm

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

log = logging.getLogger("skillgen.pipeline")


def _feedback_from_record(payload: dict | None) -> VerificationFeedback | None:
    if not payload:
        return None
    effectiveness_payload = payload.get("effectiveness")
    effectiveness = (
        EffectivenessResult(**effectiveness_payload)
        if isinstance(effectiveness_payload, dict)
        else None
    )
    analyses = [
        CaseAnalysis(**item)
        for item in (payload.get("case_analyses") or [])
        if isinstance(item, dict)
    ]
    return VerificationFeedback(
        effectiveness=effectiveness,
        round_idx=int(payload.get("round_idx", 0)),
        case_analyses=analyses,
        revision_guidance=str(payload.get("revision_guidance") or ""),
    )


def _candidate_from_record(payload: dict | None) -> CandidateSkill | None:
    return CandidateSkill(**payload) if isinstance(payload, dict) else None


def _write_round_checkpoint(path: Path, payload: dict) -> None:
    """Atomically persist enough state to continue at the next round."""

    temp_path = path.with_suffix(path.suffix + ".tmp")
    write_json(temp_path, payload)
    temp_path.replace(path)


def _load_round_checkpoint(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid round checkpoint: {path}")
    return payload


def _atomic_write_trajectories(path: Path, trajectories: list[Trajectory]) -> None:
    """Replace a trajectory JSONL only after the complete file is durable."""

    temp_path = path.with_suffix(path.suffix + ".tmp")
    write_trajectories(temp_path, trajectories)
    temp_path.replace(path)


def _ordered_baseline_subset(
    trajectories: list[Trajectory], expected_ids: list[str]
) -> list[Trajectory]:
    """Validate and order an arbitrary completed subset of frozen D_ind slots."""

    completed_ids = [str(trajectory.instance_id) for trajectory in trajectories]
    if len(completed_ids) != len(set(completed_ids)):
        raise ValueError("baseline slot checkpoint contains duplicate instance IDs")
    expected_set = set(expected_ids)
    unexpected = sorted(set(completed_ids) - expected_set)
    if unexpected:
        raise ValueError(
            "baseline slot checkpoint contains IDs outside frozen D_ind: "
            + ", ".join(unexpected[:5])
        )
    by_id = {
        str(trajectory.instance_id): trajectory for trajectory in trajectories
    }
    return [by_id[instance_id] for instance_id in expected_ids if instance_id in by_id]


def _write_baseline_slot_checkpoint(
    run_dir: Path,
    trajectories: list[Trajectory],
    expected_ids: list[str],
    *,
    complete: bool,
) -> list[Trajectory]:
    """Atomically expose any scored D_ind subset in frozen slot order."""

    ordered = _ordered_baseline_subset(trajectories, expected_ids)
    completed_ids = [str(trajectory.instance_id) for trajectory in ordered]
    if complete != (len(completed_ids) == len(expected_ids)):
        raise ValueError("baseline slot checkpoint completion flag is inconsistent")

    # Publish trajectory data before the progress pointer.  If the process dies
    # between replacements, resume trusts and validates the trajectory subset;
    # a stale progress document can never make an unscored slot look complete.
    _atomic_write_trajectories(
        run_dir / "checkpoint_trajectories.jsonl", ordered
    )
    _atomic_write_trajectories(run_dir / "baseline_trajectories.jsonl", ordered)
    _write_round_checkpoint(
        run_dir / "checkpoint.json",
        {
            "schema_version": 1,
            "stage": "baseline_done" if complete else "baseline_collecting",
            "total_stages": 3,
            "completed_stages": ["baseline"] if complete else [],
            "expected_instance_ids": expected_ids,
            "completed_instance_ids": completed_ids,
        },
    )
    return ordered


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _progress_iter(items, *, desc: str):
    if tqdm is None:
        return items
    return tqdm(items, desc=desc, dynamic_ncols=True)


def _build_verification_sample(
    failure_trajs: list[Trajectory],
    success_trajs: list[Trajectory],
    inst_map: dict[str, TaskInstance],
    *,
    sample_size: int,
    min_sample: int,
    seed: int,
) -> tuple[list[TaskInstance], list[TaskInstance]]:
    """Sample uniformly from all baseline-evaluated instances and partition by outcome."""
    rng = random.Random(seed)
    fail_ids = {t.instance_id for t in failure_trajs if t.instance_id in inst_map}
    success_ids = {t.instance_id for t in success_trajs if t.instance_id in inst_map}

    # Only include instances for which we have a baseline trajectory, so the
    # cache lookup later is guaranteed to hit.
    candidate_ids = sorted(fail_ids | success_ids)
    if not candidate_ids:
        return [], []

    target_n = max(min_sample, sample_size)
    target_n = min(target_n, len(candidate_ids))
    sampled_ids = rng.sample(candidate_ids, target_n)

    target_insts: list[TaskInstance] = []
    guard_insts: list[TaskInstance] = []
    for iid in sampled_ids:
        inst = inst_map.get(iid)
        if inst is None:
            continue
        if iid in fail_ids:
            target_insts.append(inst)
        else:
            guard_insts.append(inst)
    return target_insts, guard_insts


def _build_baseline_cache(
    target_failures: list[TaskInstance],
    success_guard: list[TaskInstance],
    failure_trajs: list[Trajectory],
    success_trajs: list[Trajectory],
) -> dict[str, Trajectory]:
    """Reuse already-collected baseline trajectories as the verification cache."""
    want = {i.instance_id for i in target_failures} | {i.instance_id for i in success_guard}
    cache: dict[str, Trajectory] = {}
    for traj in list(failure_trajs) + list(success_trajs):
        if traj.instance_id in want and traj.instance_id not in cache:
            cache[traj.instance_id] = traj
    return cache


def run_pipeline(
    instances: list[TaskInstance],
    task_type: TaskType,
    *,
    config_path: str = "config.yaml",
    dataset_id: str | None = None,
    task_name: str | None = None,
    dataset_metadata: dict | None = None,
    generate_scripts: bool | None = None,
    resume_dir: str | None = None,
    verification_instances: list[TaskInstance] | None = None,
    exhaustive_refinement: bool = False,
):
    """End-to-end single-skill discovery pipeline.

    ``verification_instances`` and ``exhaustive_refinement`` are opt-in
    protocol controls used by the SkillsBench family-transfer study.  Their
    defaults preserve the released implementation: induction and verification
    share one instance pool and refinement stops at the first passing round.
    When a separate verification pool is supplied, only ``instances`` are
    visible to induction, while the verification pool is baseline-evaluated
    once and reused across every candidate round.
    """

    if not instances:
        raise ValueError("instances must not be empty")
    if verification_instances is not None:
        if not verification_instances:
            raise ValueError("verification_instances must not be empty")
        induction_ids = [str(instance.instance_id) for instance in instances]
        verification_ids = [
            str(instance.instance_id) for instance in verification_instances
        ]
        if len(induction_ids) != len(set(induction_ids)):
            raise ValueError("induction instance IDs must be unique")
        if len(verification_ids) != len(set(verification_ids)):
            raise ValueError("verification instance IDs must be unique")
        overlap = sorted(set(induction_ids) & set(verification_ids))
        if overlap:
            raise ValueError(
                "induction and verification instance IDs overlap: "
                + ", ".join(overlap[:5])
            )

    cfg = load_config(config_path)
    llm_cfg = cfg["llm"]
    model_cfg = cfg.get("models", {})
    cl_cfg = cfg.get("clustering", {})
    ind_cfg = cfg.get("induction", {})
    gen_cfg = cfg.get("generation", {})
    ver_cfg = cfg.get("verification", {})
    ver_analysis_cfg = cfg.get("verification_analysis", {}) or {}
    pipe_cfg = cfg["pipeline"]
    router_cfg = cfg.get("router", {}) or {}
    router_enabled = bool(router_cfg.get("enabled", False))
    router_model_name = router_cfg.get("model") if router_enabled else None
    router_max_workers = int(router_cfg.get("max_workers", pipe_cfg.get("max_workers", 16)))

    if router_enabled:
        log.info("Skill router enabled | model=%s | workers=%d",
                 router_model_name, router_max_workers)

    if generate_scripts is None:
        generate_scripts = bool(gen_cfg.get("generate_scripts", False))
    generate_references = bool(gen_cfg.get("generate_references", generate_scripts))
    # When the resource path is active, pass the sandbox library allow-list to
    # the script/reference generators. Not wired globally - each benchmark
    # declares its own list (e.g. chem uses rdkit, code uses ast).
    available_libraries = list(gen_cfg.get("available_libraries", []) or [])
    max_scripts_cfg = int(gen_cfg.get("max_scripts", 5))
    max_references_cfg = int(gen_cfg.get("max_references", 5))

    default_model = model_cfg.get("default", "openai/gpt-5.4-mini")

    def model_name(key: str, fallback: str | None = None) -> str:
        return model_cfg.get(key, fallback or default_model)

    baseline_agent_cfg = AgentConfig(
        model=model_name("baseline_agent"),
        judge_model=model_name("baseline_judge"),
        temperature=llm_cfg["temperature"],
    )

    # Each run gets its own timestamped output dir for the skill itself.
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    skill_output_base = cfg.get("skill_output", {}).get(
        "path", cfg.get("skill_repo", {}).get("path", "./skill_output")
    )
    skill_output_path = Path(skill_output_base) / run_timestamp
    skill_output_path.mkdir(parents=True, exist_ok=True)

    # Resume support
    _resuming = False
    if resume_dir is not None:
        _resume_path = Path(resume_dir)
        if not checkpoint_exists(_resume_path):
            raise FileNotFoundError(
                f"No valid checkpoint found in '{resume_dir}'."
            )
        run_dir = _resume_path
        _resuming = True
        log.info("Resuming run from '%s'", run_dir)
    else:
        run_dir = make_run_dir(pipe_cfg.get("artifact_root", "./artifacts/runs"))
        write_json(
            run_dir / "run_metadata.json",
            {
                "dataset_id": dataset_id,
                "task_name": task_name,
                "dataset_metadata": dataset_metadata or {},
                "task_type": task_type.value,
                "n_instances": len(instances),
                "n_verification_instances": (
                    len(verification_instances)
                    if verification_instances is not None
                    else None
                ),
                "separate_verification_pool": verification_instances is not None,
                "exhaustive_refinement": bool(exhaustive_refinement),
                "models": model_cfg,
                "llm": llm_cfg,
                "pipeline": pipe_cfg,
                "skill_output_path": str(skill_output_path),
            },
        )

    log.info("Skill output dir for this run: %s", skill_output_path)
    log.info("Run artifacts will be saved under %s", run_dir)
    log.info(
        "Models | baseline=%s | baseline_judge=%s | generation=%s | verification=%s",
        baseline_agent_cfg.model,
        baseline_agent_cfg.judge_model,
        model_name("generation_execute"),
        model_name("verification_agent"),
    )

    embedding_model = cfg.get("embedding", {}).get("model", "text-embedding-3-small")
    max_tokens_gen = llm_cfg.get("max_tokens_generation", 16384)

    llm.reset_token_stats()

    # Stage 1: baseline trajectories
    baseline_runs_per_instance = int(
        pipe_cfg.get("baseline_runs_per_instance", 1)
    )
    slot_checkpointing = (
        verification_instances is not None and baseline_runs_per_instance == 1
    )
    if slot_checkpointing:
        expected_baseline_ids = [str(instance.instance_id) for instance in instances]
        progress = load_progress(run_dir) if _resuming else None
        if progress and progress.get("expected_instance_ids") is not None:
            recorded_ids = [
                str(instance_id)
                for instance_id in progress.get("expected_instance_ids") or []
            ]
            if recorded_ids != expected_baseline_ids:
                raise ValueError(
                    "baseline slot checkpoint belongs to different frozen D_ind slots"
                )

        checkpoint_path = run_dir / "checkpoint_trajectories.jsonl"
        trajs = (
            load_trajectories(run_dir)
            if _resuming and checkpoint_path.is_file()
            else []
        )
        trajs = _ordered_baseline_subset(trajs, expected_baseline_ids)
        completed_ids = [str(trajectory.instance_id) for trajectory in trajs]
        completed_set = set(completed_ids)
        missing_instances = [
            instance
            for instance in instances
            if str(instance.instance_id) not in completed_set
        ]
        if completed_ids:
            log.info(
                "Stage 1/3 | Resuming frozen D_ind slots | completed=%d/%d",
                len(completed_ids), len(expected_baseline_ids),
            )
        if missing_instances:
            log.info(
                "Stage 1/3 | Collecting baseline trajectories on %d remaining "
                "instances | inference_model=%s | judge_model=%s",
                len(missing_instances),
                baseline_agent_cfg.model,
                baseline_agent_cfg.judge_model,
            )

            trajectories_by_id = {
                str(trajectory.instance_id): trajectory for trajectory in trajs
            }
            expected_baseline_set = set(expected_baseline_ids)
            checkpoint_lock = threading.Lock()

            def checkpoint_scored_slot(trajectory: Trajectory) -> None:
                instance_id = str(trajectory.instance_id)
                if instance_id not in expected_baseline_set:
                    raise ValueError(
                        "baseline collector returned an ID outside frozen D_ind: "
                        f"{instance_id!r}"
                    )
                with checkpoint_lock:
                    previous = trajectories_by_id.get(instance_id)
                    if previous is not None:
                        if previous != trajectory:
                            raise ValueError(
                                "baseline collector returned conflicting results for "
                                f"frozen slot {instance_id!r}"
                            )
                        return
                    trajectories_by_id[instance_id] = trajectory
                    current = [
                        trajectories_by_id[expected_id]
                        for expected_id in expected_baseline_ids
                        if expected_id in trajectories_by_id
                    ]
                    _write_baseline_slot_checkpoint(
                        run_dir,
                        current,
                        expected_baseline_ids,
                        complete=len(current) == len(expected_baseline_ids),
                    )

            with llm.stage_scope("baseline"):
                collected = collect_trajectories(
                    missing_instances,
                    task_type,
                    config=baseline_agent_cfg,
                    runs_per_instance=1,
                    max_workers=pipe_cfg.get("max_workers", 16),
                    progress_desc="Baseline trajectories",
                    on_trajectory=checkpoint_scored_slot,
                )
            # Test doubles and third-party collectors may not implement the
            # callback. Reconcile every returned slot idempotently.
            for trajectory in collected:
                checkpoint_scored_slot(trajectory)
            trajs = [
                trajectories_by_id[expected_id]
                for expected_id in expected_baseline_ids
                if expected_id in trajectories_by_id
            ]

        if len(trajs) != len(expected_baseline_ids):
            raise RuntimeError(
                "baseline slot collection returned without completing frozen D_ind"
            )
        if not progress or progress.get("stage") == "baseline_collecting":
            trajs = _write_baseline_slot_checkpoint(
                run_dir, trajs, expected_baseline_ids, complete=True
            )
    elif _resuming:
        log.info("Stage 1/3 | Loading trajectories from checkpoint...")
        trajs = load_trajectories(run_dir)
    else:
        log.info(
            "Stage 1/3 | Collecting baseline trajectories on %d instances | "
            "inference_model=%s | judge_model=%s",
            len(instances),
            baseline_agent_cfg.model,
            baseline_agent_cfg.judge_model,
        )
        with llm.stage_scope("baseline"):
            trajs = collect_trajectories(
                instances, task_type, config=baseline_agent_cfg,
                runs_per_instance=baseline_runs_per_instance,
                max_workers=pipe_cfg.get("max_workers", 16),
                progress_desc="Baseline trajectories",
            )
        save_trajectories(run_dir, trajs)
        write_trajectories(run_dir / "baseline_trajectories.jsonl", trajs)

    failures = [t for t in trajs if not t.success]
    successes = [t for t in trajs if t.success]
    if not _resuming:
        write_trajectories(run_dir / "baseline_failures.jsonl", failures)
        write_trajectories(run_dir / "baseline_successes.jsonl", successes)
    log.info("Baseline | %d trajectories | %d failures | %d successes",
             len(trajs), len(failures), len(successes))

    if not failures:
        log.info("No failures found. Pipeline stops early - nothing to learn.")
        return None

    inst_map = {inst.instance_id: inst for inst in instances}

    # Stage 2: Induction
    analysis_dir = ensure_dir(run_dir / "analysis")
    analysis_path = analysis_dir / "skill_analysis.json"
    if _resuming and analysis_path.exists():
        log.info("Stage 2/3 | Loading analysis from checkpoint...")
        analysis = load_analysis(analysis_dir)
    else:
        log.info("Stage 2/3 | Running multi-aspect induction...")
        with llm.stage_scope("induction"):
            analysis = run_induction(
                failures, successes, instances, task_type,
                dataset_id=dataset_id,
                task_name=task_name,
                contextual_model=model_name("induction_contextual", model_name("induction")),
                summary_model=model_name("induction_summary", model_name("clustering_summary")),
                pattern_model=model_name("induction_pattern", model_name("induction")),
                contrastive_model=model_name("induction_contrastive", model_name("induction")),
                embedding_model=embedding_model,
                clustering_method=cl_cfg.get("method", "kmeans"),
                n_failure_clusters=cl_cfg.get("n_failure_clusters", cl_cfg.get("n_clusters")),
                n_success_clusters=cl_cfg.get("n_success_clusters", cl_cfg.get("n_clusters")),
                max_failure_clusters=cl_cfg.get("max_failure_clusters",
                                                cl_cfg.get("max_clusters", 8)),
                max_success_clusters=cl_cfg.get("max_success_clusters",
                                                cl_cfg.get("max_clusters", 8)),
                min_clusters=cl_cfg.get("min_clusters", 2),
                target_cluster_size=cl_cfg.get("target_cluster_size"),
                min_cluster_size=cl_cfg.get("min_cluster_size", 1),
                max_contrastive_pairs=ind_cfg.get("max_contrastive_pairs", 16),
                max_workers=pipe_cfg.get("max_workers", 16),
                artifact_dir=analysis_dir,
            )
        save_progress(run_dir, ["analysis"], 1, stage="analysis_done")
    log.info(
        "Induction | %d failure clusters | %d success clusters | %d contrastive pairs (%d same-type)",
        len(analysis.failure_clusters),
        len(analysis.success_clusters),
        len(analysis.contrastive_pairs),
        sum(1 for p in analysis.contrastive_pairs if p.same_type),
    )

    # Stage 3: Generation / Refinement / Verification loop
    sample_size_cfg = ver_cfg.get("sample_size")
    min_sample = int(ver_cfg.get("min_sample", ver_cfg.get("min_holdout", 4)))
    seed = int(ver_cfg.get("seed", 42))

    if sample_size_cfg is not None:
        sample_size = int(sample_size_cfg)
    else:
        holdout_ratio = float(ver_cfg.get("holdout_ratio", 0.3))
        sample_size = max(min_sample, int(len(inst_map) * holdout_ratio))

    if verification_instances is None:
        target_failures, success_guard = _build_verification_sample(
            failures, successes, inst_map,
            sample_size=sample_size,
            min_sample=min_sample,
            seed=seed,
        )
        log.info(
            "Verification sample | size=%d (target_n=%d baseline-failures, guard_n=%d baseline-successes) | drawn uniformly from %d instances",
            len(target_failures) + len(success_guard),
            len(target_failures), len(success_guard), len(inst_map),
        )
        baseline_cache = _build_baseline_cache(
            target_failures, success_guard, failures, successes,
        )
    else:
        verification_agent_cfg = AgentConfig(
            model=model_name("baseline_agent"),
            judge_model=model_name("baseline_judge"),
            temperature=llm_cfg["temperature"],
        )
        verification_baseline_path = (
            run_dir / "verification_baseline_trajectories.jsonl"
        )
        if _resuming and verification_baseline_path.is_file():
            log.info(
                "Verification baseline | loading frozen disjoint pool from %s",
                verification_baseline_path,
            )
            verification_trajs = read_trajectories(verification_baseline_path)
        else:
            log.info(
                "Verification baseline | collecting one frozen no-skill trajectory "
                "for each of %d disjoint verification slots",
                len(verification_instances),
            )
            with llm.stage_scope("verification_baseline"):
                verification_trajs = collect_trajectories(
                    verification_instances,
                    task_type,
                    config=verification_agent_cfg,
                    runs_per_instance=1,
                    max_workers=pipe_cfg.get("max_workers", 16),
                    progress_desc="Verification baselines",
                )
            write_trajectories(verification_baseline_path, verification_trajs)
        verification_map = {
            str(instance.instance_id): instance
            for instance in verification_instances
        }
        verification_traj_ids = [str(traj.instance_id) for traj in verification_trajs]
        if len(verification_traj_ids) != len(set(verification_traj_ids)):
            raise RuntimeError("verification baseline contains duplicate instance IDs")
        seen_verification_ids = set(verification_traj_ids)
        missing_verification_ids = sorted(
            set(verification_map) - seen_verification_ids
        )
        extra_verification_ids = sorted(
            seen_verification_ids - set(verification_map)
        )
        if missing_verification_ids:
            raise RuntimeError(
                "verification baseline is incomplete; missing IDs: "
                + ", ".join(missing_verification_ids[:5])
            )
        if extra_verification_ids:
            raise RuntimeError(
                "verification baseline contains unexpected IDs: "
                + ", ".join(extra_verification_ids[:5])
            )
        target_failures = [
            verification_map[str(traj.instance_id)]
            for traj in verification_trajs
            if not traj.success
        ]
        success_guard = [
            verification_map[str(traj.instance_id)]
            for traj in verification_trajs
            if traj.success
        ]
        baseline_cache = {
            str(traj.instance_id): traj for traj in verification_trajs
        }
        log.info(
            "Verification sample | separate/disjoint size=%d "
            "(target_n=%d baseline-failures, guard_n=%d baseline-successes)",
            len(verification_trajs), len(target_failures), len(success_guard),
        )

    verification_dir = ensure_dir(run_dir / "verification")
    candidate_dir = gen_cfg.get("candidate_output_dir", str(run_dir / "candidates"))

    feedback: VerificationFeedback | None = None
    candidate = None
    best_candidate = None
    best_feedback: VerificationFeedback | None = None
    best_net_gain = -(10 ** 9)
    round_history: list[dict] = []

    max_rounds = int(pipe_cfg.get("max_refine_rounds", 3))
    round_checkpoint_path = run_dir / "refinement_checkpoint.json"
    start_round = 0
    stop_after_checkpoint = False
    in_progress_round: int | None = None
    if _resuming:
        round_checkpoint = _load_round_checkpoint(round_checkpoint_path)
        if round_checkpoint:
            start_round = int(round_checkpoint.get("completed_rounds", 0))
            if not 0 <= start_round <= max_rounds:
                raise ValueError(
                    "refinement checkpoint completed_rounds is outside the "
                    f"configured range: {start_round} not in [0, {max_rounds}]"
                )
            candidate = _candidate_from_record(round_checkpoint.get("candidate"))
            feedback = _feedback_from_record(round_checkpoint.get("feedback"))
            best_candidate = _candidate_from_record(
                round_checkpoint.get("best_candidate")
            )
            best_feedback = _feedback_from_record(
                round_checkpoint.get("best_feedback")
            )
            best_net_gain = int(
                round_checkpoint.get("best_net_gain", -(10 ** 9))
            )
            round_history = list(round_checkpoint.get("round_history") or [])
            stop_after_checkpoint = bool(
                round_checkpoint.get("stopped_after_passing_gate", False)
            )
            raw_in_progress = round_checkpoint.get("in_progress_round")
            if raw_in_progress is not None:
                in_progress_round = int(raw_in_progress)
                if in_progress_round != start_round or candidate is None:
                    raise ValueError(
                        "refinement checkpoint has an inconsistent in-progress round"
                    )
            log.info(
                "Refinement resume | completed_rounds=%d/%d | "
                "in_progress_round=%s | best_net_gain=%+d",
                start_round,
                max_rounds,
                in_progress_round,
                best_net_gain,
            )

    round_range = range(max_rounds if stop_after_checkpoint else start_round, max_rounds)
    for round_idx in _progress_iter(round_range, desc="Refine rounds"):
        round_dir = ensure_dir(verification_dir / f"round_{round_idx + 1}")
        reuse_checkpointed_candidate = in_progress_round == round_idx
        if reuse_checkpointed_candidate:
            log.info(
                "Round %d/%d | Reusing checkpointed candidate before verification",
                round_idx + 1,
                max_rounds,
            )
        else:
            log.info(
                "Round %d/%d | Generating candidate skill...",
                round_idx + 1,
                max_rounds,
            )

            gen_stage = (
                "generation"
                if (round_idx == 0 or candidate is None)
                else f"refinement_r{round_idx + 1}"
            )
            with llm.stage_scope(gen_stage):
                if round_idx == 0 or candidate is None:
                    if generate_scripts:
                        # Resource-bundle path: SKILL.md + scripts/ + references/.
                        # The generator runs decompose -> per-resource LLM calls
                        # (concurrent) -> compose body, and auto-prepends the
                        # `load_reference` tool so reference access stays inside
                        # the `skill_*` tool surface.
                        candidate = generate_skill_with_resources(
                            analysis,
                            decompose_model=model_name("generation_plan"),
                            script_model=model_name("generation_execute"),
                            reference_model=model_name("generation_execute"),
                            compose_model=model_name("generation_execute"),
                            available_libraries=available_libraries,
                            generate_references=generate_references,
                            max_scripts=max_scripts_cfg,
                            max_references=max_references_cfg,
                            max_failure_clusters=gen_cfg.get("max_failure_clusters_in_prompt", 6),
                            max_success_clusters=gen_cfg.get("max_success_clusters_in_prompt", 6),
                            max_contrastive_pairs=gen_cfg.get("max_contrastive_pairs_in_prompt", 8),
                            max_tokens_generation=max_tokens_gen,
                            max_tokens_per_resource=int(
                                gen_cfg.get("max_tokens_per_resource", 4096)
                            ),
                            max_workers=int(gen_cfg.get("resource_gen_workers", 8)),
                            candidate_dir=candidate_dir,
                        )
                    else:
                        candidate = generate_skill(
                            analysis,
                            plan_model=model_name("generation_plan"),
                            execute_model=model_name("generation_execute"),
                            web_search_model=model_name("generation_web_search", "gpt-5.4-mini"),
                            use_web_search=gen_cfg.get("use_web_search", True),
                            max_search_queries=gen_cfg.get("max_search_queries", 3),
                            max_failure_clusters=gen_cfg.get("max_failure_clusters_in_prompt", 6),
                            max_success_clusters=gen_cfg.get("max_success_clusters_in_prompt", 6),
                            max_contrastive_pairs=gen_cfg.get("max_contrastive_pairs_in_prompt", 8),
                            generate_scripts=generate_scripts,
                            max_tokens_generation=max_tokens_gen,
                            candidate_dir=candidate_dir,
                        )
                else:
                    candidate = refine_skill(
                        candidate, analysis, feedback,
                        model=model_name("refinement"),
                        web_search_model=model_name("refinement_web_search", "gpt-5.4-mini"),
                        use_web_search=gen_cfg.get("use_web_search", True),
                        max_search_queries=gen_cfg.get("max_search_queries", 3),
                        max_failure_clusters=gen_cfg.get("max_failure_clusters_in_prompt", 6),
                        max_success_clusters=gen_cfg.get("max_success_clusters_in_prompt", 6),
                        generate_scripts=generate_scripts,
                        max_tokens_generation=max_tokens_gen,
                        candidate_dir=candidate_dir,
                    )

            if candidate is None:
                raise RuntimeError(
                    f"candidate generation returned None for round {round_idx + 1}"
                )
            # Freeze the exact candidate before any paid verification slot.
            # If a parallel slot later fails, resume reuses this body and the
            # content-addressed rollout cache instead of regenerating a subtly
            # different candidate.
            _write_round_checkpoint(
                round_checkpoint_path,
                {
                    "schema_version": 2,
                    "completed_rounds": round_idx,
                    "in_progress_round": round_idx,
                    "max_rounds": max_rounds,
                    "exhaustive_refinement": bool(exhaustive_refinement),
                    "stopped_after_passing_gate": False,
                    "candidate": asdict(candidate),
                    "feedback": asdict(feedback) if feedback is not None else None,
                    "best_candidate": (
                        asdict(best_candidate) if best_candidate is not None else None
                    ),
                    "best_feedback": (
                        asdict(best_feedback) if best_feedback is not None else None
                    ),
                    "best_net_gain": best_net_gain,
                    "round_history": round_history,
                },
            )
            in_progress_round = round_idx

        with llm.stage_scope(f"verification_r{round_idx + 1}"):
            feedback, baseline_cache = run_verification(
                candidate, target_failures, success_guard, task_type,
                round_idx=round_idx,
                baseline_cache=baseline_cache,
                baseline_agent_model=model_name("baseline_agent"),
                baseline_judge_model=model_name("baseline_judge"),
                effectiveness_agent_model=model_name("verification_agent"),
                effectiveness_judge_model=model_name("verification_judge"),
                effectiveness_max_workers=int(pipe_cfg.get("max_workers", 16)),
                case_analyst_model=model_name(
                    "verification_case_analyst", model_name("verification_judge")
                ),
                case_analyst_max_workers=int(
                    ver_analysis_cfg.get("case_analyst_workers", 8)
                ),
                case_analyst_max_tokens=int(
                    ver_analysis_cfg.get("case_analyst_max_tokens", 2048)
                ),
                revision_synthesiser_model=model_name(
                    "verification_revision_synthesiser", model_name("verification_judge")
                ),
                revision_synthesiser_max_tokens=int(
                    ver_analysis_cfg.get("revision_synthesiser_max_tokens", 4096)
                ),
                artifact_dir=str(round_dir),
                artifact_prefix="verification",
                progress_label=f"Verify r{round_idx + 1}",
                router_model=router_model_name,
                router_max_workers=router_max_workers,
                min_net_gain_abs=int(ver_cfg.get("min_net_gain_abs", 1)),
                min_net_gain_rel=float(ver_cfg.get("min_net_gain_rel", 0.0)),
            )

        eff = feedback.effectiveness
        if eff:
            log.info(
                "Round %d | paired_n=%d | baseline_acc=%.1f%% | skill_acc=%.1f%% | "
                "repair=%d | regression=%d | net_gain=%+d | passed=%s",
                round_idx + 1, eff.paired_n, eff.baseline_acc * 100.0,
                eff.skill_acc * 100.0, eff.repair_count, eff.regression_count,
                eff.net_gain, eff.passed,
            )
            if eff.net_gain > best_net_gain:
                best_net_gain = eff.net_gain
                # refine_skill mutates CandidateSkill in place.  Snapshot the
                # winner so a later round cannot silently rewrite the selected
                # best-of-K candidate.
                best_candidate = copy.deepcopy(candidate)
                best_feedback = copy.deepcopy(feedback)
            round_history.append({
                "round_idx": round_idx,
                "passed": bool(eff.passed),
                "paired_n": int(eff.paired_n),
                "net_gain": int(eff.net_gain),
                "repair_count": int(eff.repair_count),
                "regression_count": int(eff.regression_count),
                "baseline_acc": float(eff.baseline_acc),
                "skill_acc": float(eff.skill_acc),
                "diagnostic": eff.diagnostic_summary,
            })
            stopping_after_gate = bool(eff.passed and not exhaustive_refinement)
            _write_round_checkpoint(
                round_checkpoint_path,
                {
                    "schema_version": 2,
                    "completed_rounds": round_idx + 1,
                    "in_progress_round": None,
                    "max_rounds": max_rounds,
                    "exhaustive_refinement": bool(exhaustive_refinement),
                    "stopped_after_passing_gate": stopping_after_gate,
                    "candidate": asdict(candidate),
                    "feedback": asdict(feedback),
                    "best_candidate": (
                        asdict(best_candidate) if best_candidate is not None else None
                    ),
                    "best_feedback": (
                        asdict(best_feedback) if best_feedback is not None else None
                    ),
                    "best_net_gain": best_net_gain,
                    "round_history": round_history,
                },
            )
            in_progress_round = None
            if eff.passed:
                if exhaustive_refinement:
                    log.info(
                        "Skill passes net-gain gate at round %d; continuing "
                        "because exhaustive_refinement=True.",
                        round_idx + 1,
                    )
                else:
                    log.info("Skill passes net-gain gate at round %d; stopping refinement.",
                             round_idx + 1)
                    break

    # Persist the best skill observed across all refinement rounds.
    final_candidate = best_candidate or candidate
    final_feedback = best_feedback or feedback
    if final_candidate is None:
        log.warning("No candidate produced; skipping skill persistence.")
        return None

    any_round_passed = bool(
        final_feedback
        and final_feedback.effectiveness
        and final_feedback.effectiveness.passed
    )

    skill = finalize_skill(
        final_candidate, skill_output_path,
        analysis_id=analysis.analysis_id,
        dataset_id=dataset_id,
        task_name=task_name,
    )
    if not any_round_passed:
        skill.status = SkillStatus.DEPRECATED
        log.warning(
            "No refinement round passed the verification gate; marking skill "
            "%s as DEPRECATED (best observed net_gain=%+d). Downstream eval "
            "will skip the skill condition and report net_gain=0.",
            skill.skill_id, best_net_gain,
        )
    if round_history:
        for item in round_history:
            item["selected_best"] = bool(
                final_feedback is not None
                and item["round_idx"] == final_feedback.round_idx
            )
            item["rejected_no_passing_round"] = not any_round_passed
        skill.verification_history.extend(round_history)
    save_skill(skill, skill_output_path)

    save_analysis(analysis, skill_output_path)

    try:
        stats_path = llm.dump_token_stats(run_dir / "token_usage.json")
        rows = llm.get_token_stats()
        grand_total = sum(r["total_tokens"] for r in rows)
        total_prompt = sum(r["prompt_tokens"] for r in rows)
        total_completion = sum(r["completion_tokens"] for r in rows)
        log.info(
            "Token usage | prompt=%s | completion=%s | total=%s | %d (model,stage,call_type) cells -> %s",
            f"{total_prompt:,}", f"{total_completion:,}", f"{grand_total:,}",
            len(rows), stats_path,
        )
    except Exception:
        log.exception("Failed to dump token usage stats")

    log.info("Pipeline complete | skill_id=%s | stored at %s",
             skill.skill_id, skill_output_path)
    return skill
