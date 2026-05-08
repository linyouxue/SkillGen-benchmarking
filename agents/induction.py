"""Induction Agent: cluster failures/successes and produce a SkillAnalysis artifact."""

from __future__ import annotations

import json
import random
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np

from artifacts import write_json
from clustering import (
    cluster_trajectories,
    nearest_success_for_failures,
    render_trace,
    summarise_trajectories,
)
from models import (
    ContrastivePair,
    SkillAnalysis,
    TaskInstance,
    TaskType,
    Trajectory,
    TrajectoryCluster,
)
from prompts import (
    CONTEXTUAL_ABSTRACT_PROMPT,
    CONTEXTUAL_ABSTRACT_SYSTEM,
    CONTRASTIVE_PROMPT,
    CONTRASTIVE_SYSTEM,
    FAILURE_PATTERN_PROMPT,
    FAILURE_PATTERN_SYSTEM,
    SUCCESS_PATTERN_PROMPT,
    SUCCESS_PATTERN_SYSTEM,
)
import llm


# Contextual abstract

def _format_sample_instances(instances: list[TaskInstance], *, n: int = 6) -> str:
    parts = []
    for inst in instances[:n]:
        text = str(inst.input or "")
        if len(text) > 600:
            text = text[:600] + " ..."
        parts.append(f"- ({inst.instance_id}) {text}")
    return "\n".join(parts) if parts else "(no samples)"


def _infer_contextual_abstract(
    instances: list[TaskInstance],
    *,
    task_name: str | None,
    task_type: TaskType,
    model: str,
    n_samples: int = 6,
    seed: int = 42,
) -> str:
    if not instances:
        return ""

    rng = random.Random(seed)
    samples = rng.sample(instances, min(n_samples, len(instances)))

    result = llm.chat_json(
        CONTEXTUAL_ABSTRACT_PROMPT.format(
            task_name=task_name or "(unnamed task)",
            task_type=task_type.value if hasattr(task_type, "value") else str(task_type),
            n=len(samples),
            samples=_format_sample_instances(samples, n=len(samples)),
        ),
        system=CONTEXTUAL_ABSTRACT_SYSTEM,
        model=model,
    )
    return (result.get("contextual_abstract") or "").strip()


# Cluster-level pattern synthesis

def _format_cluster_summaries(cluster: TrajectoryCluster, *, max_items: int = 8) -> str:
    parts = []
    for iid, summary in zip(cluster.member_instance_ids[:max_items],
                            cluster.member_summaries[:max_items]):
        parts.append(f"- ({iid}) {summary.strip()}")
    return "\n".join(parts) if parts else "(empty cluster)"


def _synthesize_failure_pattern(cluster: TrajectoryCluster, model: str) -> str:
    result = llm.chat_json(
        FAILURE_PATTERN_PROMPT.format(
            summaries=_format_cluster_summaries(cluster),
        ),
        system=FAILURE_PATTERN_SYSTEM,
        model=model,
    )
    lines = [
        f"Name: {result.get('pattern_name', '').strip()}",
        f"Root cause: {result.get('root_cause', '').strip()}",
        f"Typical mistake sequence: {result.get('typical_mistake_sequence', '').strip()}",
        "Observable signals: " + "; ".join(result.get("observable_signals") or []),
        f"Correct behavior: {result.get('correct_behavior', '').strip()}",
    ]
    return "\n".join(line for line in lines if line.split(":", 1)[1].strip())


def _synthesize_success_pattern(cluster: TrajectoryCluster, model: str) -> str:
    result = llm.chat_json(
        SUCCESS_PATTERN_PROMPT.format(
            summaries=_format_cluster_summaries(cluster),
        ),
        system=SUCCESS_PATTERN_SYSTEM,
        model=model,
    )
    lines = [
        f"Technique: {result.get('technique_name', '').strip()}",
        f"Procedure: {result.get('procedure', '').strip()}",
        "Observable signals: " + "; ".join(result.get("observable_signals") or []),
        f"Robustness checks: {result.get('robustness_checks', '').strip()}",
    ]
    return "\n".join(line for line in lines if line.split(":", 1)[1].strip())


def _synthesize_patterns(
    clusters: list[TrajectoryCluster],
    *,
    kind: str,
    model: str,
    max_workers: int = 8,
) -> None:
    """Fill cluster.pattern for each cluster (in place), concurrently."""
    if not clusters:
        return

    fn = _synthesize_failure_pattern if kind == "failure" else _synthesize_success_pattern
    args = [(cluster, model) for cluster in clusters]
    patterns = llm.run_concurrent(
        fn,
        args,
        max_workers=max_workers,
        progress_desc=f"Synthesizing {kind} patterns",
    )
    for cluster, pattern in zip(clusters, patterns):
        cluster.pattern = pattern


# Contrastive pair analysis

def _one_contrastive(
    failure: Trajectory,
    success: Trajectory,
    failure_input: str,
    success_input: str,
    failure_summary: str,
    success_summary: str,
    similarity: float,
    model: str,
    trace_char_limit: int = 1600,
) -> ContrastivePair:
    result = llm.chat_json(
        CONTRASTIVE_PROMPT.format(
            failure_input=failure_input[:800],
            failure_summary=failure_summary,
            failure_trace=render_trace(failure)[:trace_char_limit],
            success_input=success_input[:800],
            success_summary=success_summary,
            success_trace=render_trace(success)[:trace_char_limit],
        ),
        system=CONTRASTIVE_SYSTEM,
        model=model,
    )
    same_type_raw = result.get("same_type", False)
    if isinstance(same_type_raw, str):
        same_type = same_type_raw.strip().lower() in ("true", "yes", "1")
    else:
        same_type = bool(same_type_raw)

    analysis = (result.get("analysis") or "").strip() if same_type else ""
    return ContrastivePair(
        failure_instance_id=failure.instance_id,
        failure_trajectory_id=failure.trajectory_id,
        success_instance_id=success.instance_id,
        success_trajectory_id=success.trajectory_id,
        similarity=similarity,
        same_type=same_type,
        same_type_reason=(result.get("same_type_reason") or "").strip(),
        analysis=analysis,
    )


def _build_contrastive_pairs(
    failure_trajs: list[Trajectory],
    success_trajs: list[Trajectory],
    failure_summaries: list[str],
    success_summaries: list[str],
    failure_embeds: np.ndarray,
    success_embeds: np.ndarray,
    task_inputs: dict[str, str],
    *,
    model: str,
    max_pairs: int,
    max_workers: int = 8,
) -> list[ContrastivePair]:
    """Select up to `max_pairs` failures and pair each with its nearest success."""
    if not failure_trajs or not success_trajs or max_pairs <= 0:
        return []

    nearest = nearest_success_for_failures(
        failure_embeds, success_embeds, failure_trajs, success_trajs, top_k=1,
    )

    candidates: list[tuple[Trajectory, Trajectory, str, str, float, int]] = []
    for f_idx, neighbours in enumerate(nearest):
        if not neighbours:
            continue
        s_idx, sim = neighbours[0]
        candidates.append((
            failure_trajs[f_idx],
            success_trajs[s_idx],
            failure_summaries[f_idx],
            success_summaries[s_idx],
            sim,
            f_idx,
        ))

    candidates.sort(key=lambda c: c[4], reverse=True)
    candidates = candidates[:max_pairs]

    args = [
        (
            f,
            s,
            task_inputs.get(f.instance_id, ""),
            task_inputs.get(s.instance_id, ""),
            fs,
            ss,
            sim,
            model,
        )
        for f, s, fs, ss, sim, _ in candidates
    ]
    return llm.run_concurrent(
        _one_contrastive,
        args,
        max_workers=max_workers,
        progress_desc="Contrastive pairs",
    )


# Top-level induction entry point

def _to_task_inputs(instances: list[TaskInstance]) -> dict[str, str]:
    return {inst.instance_id: str(inst.input or "") for inst in instances}


def _to_ground_truths(instances: list[TaskInstance]) -> dict[str, str]:
    return {inst.instance_id: str(inst.ground_truth or "") for inst in instances}


def run_induction(
    failure_trajs: list[Trajectory],
    success_trajs: list[Trajectory],
    instances: list[TaskInstance],
    task_type: TaskType,
    *,
    dataset_id: str | None = None,
    task_name: str | None = None,
    contextual_model: str = "openai/gpt-5.4-mini",
    summary_model: str = "openai/gpt-5.4-mini",
    pattern_model: str = "openai/gpt-5.4-mini",
    contrastive_model: str = "openai/gpt-5.4-mini",
    embedding_model: str = "text-embedding-3-small",
    clustering_method: str = "kmeans",
    n_failure_clusters: int | None = None,
    n_success_clusters: int | None = None,
    max_failure_clusters: int = 8,
    max_success_clusters: int = 8,
    min_clusters: int = 2,
    target_cluster_size: int | None = None,
    min_cluster_size: int = 1,
    max_contrastive_pairs: int = 20,
    max_workers: int = 16,
    artifact_dir: str | Path | None = None,
) -> SkillAnalysis:
    """Full multi-aspect induction on baseline trajectories for ONE dataset."""

    task_inputs = _to_task_inputs(instances)
    ground_truths = _to_ground_truths(instances)

    # 1. Contextual abstract
    contextual_abstract = _infer_contextual_abstract(
        instances,
        task_name=task_name,
        task_type=task_type,
        model=contextual_model,
    )

    # 2. Per-trajectory summaries
    failure_summaries = summarise_trajectories(
        failure_trajs, task_inputs,
        kind="failure", model=summary_model,
        max_workers=max_workers,
        progress_desc="Summarizing failures",
    ) if failure_trajs else []

    success_summaries = summarise_trajectories(
        success_trajs, task_inputs,
        kind="success", model=summary_model,
        max_workers=max_workers,
        progress_desc="Summarizing successes",
        ground_truths=ground_truths,
    ) if success_trajs else []

    # 3. Cluster failures and successes
    failure_clusters, failure_embeds = cluster_trajectories(
        failure_trajs, failure_summaries, task_inputs,
        kind="failure",
        method=clustering_method,
        n_clusters=n_failure_clusters,
        max_clusters=max_failure_clusters,
        min_clusters=min_clusters,
        target_cluster_size=target_cluster_size,
        min_cluster_size=min_cluster_size,
        embedding_model=embedding_model,
    )
    success_clusters, success_embeds = cluster_trajectories(
        success_trajs, success_summaries, task_inputs,
        kind="success",
        method=clustering_method,
        n_clusters=n_success_clusters,
        max_clusters=max_success_clusters,
        min_clusters=min_clusters,
        target_cluster_size=target_cluster_size,
        min_cluster_size=min_cluster_size,
        embedding_model=embedding_model,
    )

    # 4. Synthesize cluster-level patterns / techniques
    _synthesize_patterns(failure_clusters, kind="failure", model=pattern_model)
    _synthesize_patterns(success_clusters, kind="success", model=pattern_model)

    # 5. Contrastive pairs
    contrastive_pairs = _build_contrastive_pairs(
        failure_trajs, success_trajs,
        failure_summaries, success_summaries,
        failure_embeds, success_embeds,
        task_inputs,
        model=contrastive_model,
        max_pairs=max_contrastive_pairs,
        max_workers=max_workers,
    )

    analysis = SkillAnalysis(
        analysis_id=str(uuid.uuid4()),
        dataset_id=dataset_id,
        task_name=task_name,
        task_type=task_type.value if hasattr(task_type, "value") else str(task_type),
        contextual_abstract=contextual_abstract,
        n_total=len(failure_trajs) + len(success_trajs),
        n_failures=len(failure_trajs),
        n_successes=len(success_trajs),
        failure_clusters=failure_clusters,
        success_clusters=success_clusters,
        contrastive_pairs=contrastive_pairs,
        metadata={
            "contextual_model": contextual_model,
            "summary_model": summary_model,
            "pattern_model": pattern_model,
            "contrastive_model": contrastive_model,
            "embedding_model": embedding_model,
            "clustering_method": clustering_method,
        },
    )

    if artifact_dir is not None:
        save_analysis(analysis, artifact_dir)
    return analysis


# Persistence

def save_analysis(analysis: SkillAnalysis, artifact_dir: str | Path) -> Path:
    """Write the full SkillAnalysis (plus a compact summary) to disk."""
    out = Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)
    full_path = out / "skill_analysis.json"
    with full_path.open("w", encoding="utf-8") as fh:
        json.dump(asdict(analysis), fh, indent=2, ensure_ascii=True)

    summary = {
        "analysis_id": analysis.analysis_id,
        "dataset_id": analysis.dataset_id,
        "task_name": analysis.task_name,
        "n_failures": analysis.n_failures,
        "n_successes": analysis.n_successes,
        "n_failure_clusters": len(analysis.failure_clusters),
        "n_success_clusters": len(analysis.success_clusters),
        "n_contrastive_pairs": len(analysis.contrastive_pairs),
        "same_type_pairs": sum(1 for p in analysis.contrastive_pairs if p.same_type),
        "contextual_abstract": analysis.contextual_abstract,
    }
    write_json(out / "skill_analysis_summary.json", summary)
    return full_path


def load_analysis(artifact_dir: str | Path) -> SkillAnalysis:
    """Load a previously persisted SkillAnalysis from disk."""
    path = Path(artifact_dir) / "skill_analysis.json"
    with path.open(encoding="utf-8") as fh:
        raw = json.load(fh)

    failure_clusters = [TrajectoryCluster(**c) for c in raw.get("failure_clusters", [])]
    success_clusters = [TrajectoryCluster(**c) for c in raw.get("success_clusters", [])]
    contrastive_pairs = [ContrastivePair(**p) for p in raw.get("contrastive_pairs", [])]
    return SkillAnalysis(
        analysis_id=raw["analysis_id"],
        dataset_id=raw.get("dataset_id"),
        task_name=raw.get("task_name"),
        task_type=raw.get("task_type", "open_ended"),
        contextual_abstract=raw.get("contextual_abstract", ""),
        n_total=raw.get("n_total", 0),
        n_failures=raw.get("n_failures", 0),
        n_successes=raw.get("n_successes", 0),
        failure_clusters=failure_clusters,
        success_clusters=success_clusters,
        contrastive_pairs=contrastive_pairs,
        metadata=raw.get("metadata", {}),
    )
