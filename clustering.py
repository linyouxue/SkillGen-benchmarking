"""Trajectory clustering: summarise, embed, cluster, and find contrastive pairs."""

from __future__ import annotations
import uuid

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

from models import Trajectory, TrajectoryCluster
from prompts import SUMMARISE_FAILURE_PROMPT, SUMMARISE_SUCCESS_PROMPT
import llm


# Rendering helpers (shared by summaries and prompts)

def _render_trace_message(message: dict) -> str:
    role = message.get("role", "")
    content = str(message.get("content", "") or "")[:500]
    if role == "assistant" and message.get("tool_calls"):
        tool_names = [
            tc.get("function", {}).get("name", "")
            for tc in message.get("tool_calls", [])
        ]
        tool_names = [name for name in tool_names if name]
        if tool_names:
            return f"[assistant] tool_calls={', '.join(tool_names)}"
    if role == "tool":
        tool_name = message.get("name", "")
        prefix = f"[tool:{tool_name}]" if tool_name else "[tool]"
        return f"{prefix} {content}"
    return f"[{role}] {content}"


def render_trace(traj: Trajectory, *, max_messages: int | None = None) -> str:
    """Render the full trace of a trajectory for prompt injection."""
    msgs = traj.messages or []
    if max_messages is not None:
        msgs = msgs[:max_messages]
    return "\n".join(_render_trace_message(m) for m in msgs)


def _task_context_for_summary(traj: Trajectory, task_input: str) -> str:
    if traj.metadata.get("benchmark") != "tau_bench":
        return task_input

    lines = [task_input]
    domain = traj.metadata.get("domain")
    task_index = traj.metadata.get("task_index")
    reward = traj.metadata.get("reward")
    if domain is not None and task_index is not None:
        lines.append(f"Benchmark context: tau-bench domain={domain} task_index={task_index}")
    if reward is not None:
        lines.append(f"Reward observed: {reward}")
    tool_calls = traj.metadata.get("tool_calls") or []
    if tool_calls:
        lines.append("Observed tool sequence: " + " -> ".join(tool_calls))
    last_observation = str(traj.metadata.get("last_observation", "") or "")
    if last_observation:
        lines.append("Last observation: " + last_observation[:500])
    return "\n".join(lines)


# Per-trajectory summaries

def _summarise_failure(traj: Trajectory, task_input: str, model: str) -> str:
    trace_text = render_trace(traj)
    return llm.chat(
        SUMMARISE_FAILURE_PROMPT.format(
            input=_task_context_for_summary(traj, task_input),
            trace=trace_text,
            error_summary=traj.error_summary or "(not available)",
        ),
        model=model,
    )


def _summarise_success(
    traj: Trajectory,
    task_input: str,
    ground_truth: str,
    model: str,
) -> str:
    trace_text = render_trace(traj)
    return llm.chat(
        SUMMARISE_SUCCESS_PROMPT.format(
            input=_task_context_for_summary(traj, task_input),
            trace=trace_text,
            ground_truth=ground_truth or "(not available)",
        ),
        model=model,
    )


def summarise_trajectories(
    trajectories: list[Trajectory],
    task_inputs: dict[str, str],
    *,
    kind: str,
    model: str,
    max_workers: int = 16,
    progress_desc: str | None = None,
    ground_truths: dict[str, str] | None = None,
) -> list[str]:
    """Summarise failure or success trajectories concurrently.

    `kind` must be "failure" or "success". For successes, `ground_truths` can
    optionally supply per-instance ground-truth text.
    """
    if kind == "failure":
        args = [
            (t, task_inputs.get(t.instance_id, ""), model)
            for t in trajectories
        ]
        return llm.run_concurrent(
            _summarise_failure,
            args,
            max_workers=max_workers,
            progress_desc=progress_desc or "Summarizing failures",
        )
    if kind == "success":
        gt = ground_truths or {}
        args = [
            (t, task_inputs.get(t.instance_id, ""), gt.get(t.instance_id, ""), model)
            for t in trajectories
        ]
        return llm.run_concurrent(
            _summarise_success,
            args,
            max_workers=max_workers,
            progress_desc=progress_desc or "Summarizing successes",
        )
    raise ValueError(f"Unknown summary kind: {kind}")


# Clustering

def _auto_n_clusters(
    embeddings: np.ndarray,
    max_k: int,
    min_k: int = 2,
    target_cluster_size: int | None = None,
) -> int:
    """Pick k using Calinski-Harabasz; use target_cluster_size if set."""
    from sklearn.metrics import calinski_harabasz_score

    n = len(embeddings)
    if n <= min_k:
        return max(1, n)
    if target_cluster_size is not None and target_cluster_size > 0:
        return max(min_k, min(max_k, max(1, n // target_cluster_size)))

    best_k, best_score = min_k, -1.0
    for k in range(min_k, min(max_k + 1, n)):
        labels = KMeans(n_clusters=k, n_init="auto", random_state=42).fit_predict(embeddings)
        score = calinski_harabasz_score(embeddings, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def _embed_for_cluster(
    summaries: list[str],
    trajectories: list[Trajectory],
    task_inputs: dict[str, str],
    embedding_model: str,
) -> np.ndarray:
    texts = [
        f"Summary: {summary}\nTask input: {task_inputs.get(t.instance_id, '')}"
        for summary, t in zip(summaries, trajectories)
    ]
    return np.array(llm.embed(texts, model=embedding_model))


def cluster_trajectories(
    trajectories: list[Trajectory],
    summaries: list[str],
    task_inputs: dict[str, str],
    *,
    kind: str,
    method: str = "kmeans",
    n_clusters: int | None = None,
    max_clusters: int = 20,
    min_clusters: int = 2,
    target_cluster_size: int | None = None,
    min_cluster_size: int = 1,
    embedding_model: str = "text-embedding-3-small",
) -> tuple[list[TrajectoryCluster], np.ndarray]:
    """Cluster trajectories by (summary + task_input) embedding.

    Returns (clusters, per-trajectory embedding matrix). The embedding matrix
    has one row per input trajectory and is handy for downstream nearest-
    neighbour retrieval (e.g. contrastive-pair selection).
    """
    if not trajectories:
        return [], np.zeros((0, 0))

    embeds = _embed_for_cluster(summaries, trajectories, task_inputs, embedding_model)

    if len(trajectories) < max(4, min_cluster_size):
        labels = np.zeros(len(trajectories), dtype=int)
    else:
        if n_clusters is None:
            n_clusters = _auto_n_clusters(
                embeds,
                max_k=max_clusters,
                min_k=min_clusters,
                target_cluster_size=target_cluster_size,
            )

        if method == "kmeans":
            labels = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42).fit_predict(embeds)
        elif method == "dbscan":
            labels = DBSCAN(eps=0.3, min_samples=min_cluster_size).fit_predict(embeds)
        elif method == "agglomerative":
            labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(embeds)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

    clusters: list[TrajectoryCluster] = []
    for cid in sorted(set(labels)):
        if cid == -1:  # DBSCAN noise
            continue
        indices = [i for i, lab in enumerate(labels) if lab == cid]
        if not indices or len(indices) < min_cluster_size:
            continue
        clusters.append(
            TrajectoryCluster(
                cluster_id=str(uuid.uuid4()),
                kind=kind,
                member_trajectory_ids=[trajectories[i].trajectory_id for i in indices],
                member_instance_ids=[trajectories[i].instance_id for i in indices],
                member_summaries=[summaries[i] for i in indices],
                pattern="",  # filled in by the induction agent
            )
        )
    return clusters, embeds


# Nearest-neighbour pairing (contrastive selection)

def nearest_success_for_failures(
    failure_embeds: np.ndarray,
    success_embeds: np.ndarray,
    failure_trajs: list[Trajectory],
    success_trajs: list[Trajectory],
    *,
    top_k: int = 1,
) -> list[list[tuple[int, float]]]:
    """For each failure row, return the indices of its top-k nearest successes.

    Similarity is cosine on the embedding rows already used for clustering.
    Returns a list aligned with `failure_trajs`, where each entry is a list
    of (success_index, cosine_similarity) pairs sorted by similarity desc.
    """
    if len(failure_embeds) == 0 or len(success_embeds) == 0:
        return [[] for _ in range(len(failure_trajs))]

    fnorm = failure_embeds / (np.linalg.norm(failure_embeds, axis=1, keepdims=True) + 1e-9)
    snorm = success_embeds / (np.linalg.norm(success_embeds, axis=1, keepdims=True) + 1e-9)
    sims = fnorm @ snorm.T  # (n_fail, n_succ)

    out: list[list[tuple[int, float]]] = []
    for i in range(len(failure_trajs)):
        row = sims[i]
        k = min(top_k, len(success_trajs))
        top_idx = np.argpartition(-row, k - 1)[:k] if k > 0 else np.array([], dtype=int)
        top_sorted = top_idx[np.argsort(-row[top_idx])]
        out.append([(int(j), float(row[j])) for j in top_sorted])
    return out
