"""End-to-end skill-discovery pipeline: baseline -> induction -> generation -> verification."""

from __future__ import annotations

import logging
import random
import yaml
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from artifacts import (
    ensure_dir, make_run_dir, write_json, write_trajectories,
    save_trajectories, load_trajectories, checkpoint_exists, load_progress, save_progress,
)
from models import (
    SkillAnalysis, SkillStatus, TaskInstance, TaskType, Trajectory,
    VerificationFeedback,
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
):
    """End-to-end single-skill discovery pipeline."""

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
    if _resuming:
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
                runs_per_instance=pipe_cfg.get("baseline_runs_per_instance", 1),
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

    verification_dir = ensure_dir(run_dir / "verification")
    candidate_dir = gen_cfg.get("candidate_output_dir", str(run_dir / "candidates"))

    feedback: VerificationFeedback | None = None
    candidate = None
    best_candidate = None
    best_feedback: VerificationFeedback | None = None
    best_net_gain = -(10 ** 9)

    max_rounds = int(pipe_cfg.get("max_refine_rounds", 3))
    round_range = range(max_rounds)
    for round_idx in _progress_iter(round_range, desc="Refine rounds"):
        round_dir = ensure_dir(verification_dir / f"round_{round_idx + 1}")
        log.info("Round %d/%d | Generating candidate skill...", round_idx + 1, max_rounds)

        gen_stage = "generation" if (round_idx == 0 or candidate is None) else f"refinement_r{round_idx + 1}"
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

        with llm.stage_scope(f"verification_r{round_idx + 1}"):
            feedback, baseline_cache = run_verification(
                candidate, target_failures, success_guard, task_type,
                round_idx=round_idx,
                baseline_cache=baseline_cache,
                baseline_agent_model=model_name("baseline_agent"),
                baseline_judge_model=model_name("baseline_judge"),
                effectiveness_agent_model=model_name("verification_agent"),
                effectiveness_judge_model=model_name("verification_judge"),
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
                best_candidate = candidate
                best_feedback = feedback
            if eff.passed:
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
    if final_feedback and final_feedback.effectiveness:
        skill.verification_history.append({
            "round_idx": final_feedback.round_idx,
            "passed": final_feedback.effectiveness.passed,
            "net_gain": final_feedback.effectiveness.net_gain,
            "repair_count": final_feedback.effectiveness.repair_count,
            "regression_count": final_feedback.effectiveness.regression_count,
            "baseline_acc": final_feedback.effectiveness.baseline_acc,
            "skill_acc": final_feedback.effectiveness.skill_acc,
            "diagnostic": final_feedback.effectiveness.diagnostic_summary,
            "rejected_no_passing_round": not any_round_passed,
        })
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
