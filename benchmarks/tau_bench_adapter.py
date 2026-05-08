"""Helpers for loading and running tau-bench tasks inside SkillGen."""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

from models import TaskDataset, TaskInstance, TaskType


_TAU_BENCH_ROOT = Path(__file__).parent / "external" / "tau-bench"


def _ensure_tau_bench_importable() -> None:
    root = str(_TAU_BENCH_ROOT)
    if not _TAU_BENCH_ROOT.exists():
        raise FileNotFoundError(
            "tau-bench repository not found. Expected clone at "
            f"'{_TAU_BENCH_ROOT}'."
        )
    if root not in sys.path:
        sys.path.insert(0, root)


def _select_task_indices(
    *,
    total_tasks: int,
    task_ids: list[int] | None,
    start_index: int,
    end_index: int,
    n: int | None,
    seed: int,
    shuffle: bool,
) -> list[int]:
    if task_ids:
        indices = [idx for idx in task_ids if 0 <= idx < total_tasks]
    else:
        stop = total_tasks if end_index == -1 else min(end_index, total_tasks)
        indices = list(range(max(0, start_index), stop))

    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(indices)

    if n is not None and n > 0 and n < len(indices):
        if shuffle:
            indices = indices[:n]
        else:
            indices = rng.sample(indices, n)
    return indices


def _format_task_input(*, domain: str, split: str, task_index: int, instruction: str) -> str:
    return (
        "Interactive benchmark task.\n"
        f"Benchmark: tau-bench\n"
        f"Domain: {domain}\n"
        f"Split: {split}\n"
        f"Task ID: {task_index}\n\n"
        "Customer instruction:\n"
        f"{instruction}"
    )


def is_tau_bench_instance(instance: TaskInstance) -> bool:
    return instance.metadata.get("benchmark") == "tau_bench"


def _load_tasks(domain: str, task_split: str):
    _ensure_tau_bench_importable()
    if domain == "retail":
        if task_split == "train":
            from tau_bench.envs.retail.tasks_train import TASKS_TRAIN as tasks
        elif task_split == "dev":
            from tau_bench.envs.retail.tasks_dev import TASKS_DEV as tasks
        elif task_split == "test":
            from tau_bench.envs.retail.tasks_test import TASKS_TEST as tasks
        else:
            raise ValueError(f"Unknown retail task split: {task_split}")
    elif domain == "airline":
        if task_split != "test":
            raise ValueError("airline only supports the test split in this tau-bench repo")
        from tau_bench.envs.airline.tasks_test import TASKS as tasks
    else:
        raise ValueError(f"Unknown tau-bench domain: {domain}")
    return tasks


def load_tau_bench_dataset(
    *,
    domain: str,
    task_split: str,
    user_strategy: str,
    user_model: str,
    user_provider: str | None,
    max_env_steps: int = 30,
    task_ids: list[int] | None = None,
    start_index: int = 0,
    end_index: int = -1,
    n: int | None = None,
    seed: int = 42,
    shuffle: bool = False,
) -> TaskDataset:
    tasks = _load_tasks(domain, task_split)
    indices = _select_task_indices(
        total_tasks=len(tasks),
        task_ids=task_ids,
        start_index=start_index,
        end_index=end_index,
        n=n,
        seed=seed,
        shuffle=shuffle,
    )

    instances: list[TaskInstance] = []
    for idx in indices:
        task = tasks[idx]
        instances.append(
            TaskInstance(
                instance_id=f"{domain}:{task_split}:{idx}",
                input=_format_task_input(
                    domain=domain,
                    split=task_split,
                    task_index=idx,
                    instruction=task.instruction,
                ),
                ground_truth=None,
                metadata={
                    "benchmark": "tau_bench",
                    "domain": domain,
                    "task_split": task_split,
                    "task_index": idx,
                    "user_strategy": user_strategy,
                    "user_model": user_model,
                    "user_provider": user_provider,
                    "max_env_steps": max_env_steps,
                    "instruction": task.instruction,
                    "expected_outputs": list(task.outputs),
                    "user_id": task.user_id,
                },
            )
        )

    return TaskDataset(
        dataset_id=f"tau_bench_{domain}_{task_split}",
        task_name=f"tau-bench-{domain}-{task_split}",
        task_type=TaskType.BINARY,
        instances=instances,
        metadata={
            "benchmark": "tau_bench",
            "domain": domain,
            "task_split": task_split,
            "user_strategy": user_strategy,
            "user_model": user_model,
            "user_provider": user_provider,
            "max_env_steps": max_env_steps,
            "n": len(instances),
            "task_ids": indices,
        },
    )


def create_tau_bench_env(instance: TaskInstance):
    _ensure_tau_bench_importable()
    from tau_bench.envs import get_env

    md = instance.metadata
    return get_env(
        md["domain"],
        user_strategy=md["user_strategy"],
        user_model=md["user_model"],
        user_provider=md.get("user_provider"),
        task_split=md["task_split"],
        task_index=md["task_index"],
    )


def get_tau_bench_types():
    _ensure_tau_bench_importable()
    from tau_bench.types import Action, RESPOND_ACTION_NAME

    return Action, RESPOND_ACTION_NAME


def summarise_tau_bench_outcome(
    *,
    instance: TaskInstance,
    reward: float,
    final_info: dict[str, Any] | None,
    tool_calls: list[str],
    terminated: bool,
    last_observation: str,
) -> tuple[str | None, dict[str, Any]]:
    md = instance.metadata
    info = final_info or {}
    reward_info = info.get("reward_info") or {}
    reward_details = reward_info.get("info") or {}

    details: list[str] = [
        f"tau-bench reward={reward:.3f}",
        f"domain={md.get('domain')}",
        f"task_index={md.get('task_index')}",
    ]
    if tool_calls:
        details.append("tool_sequence=" + " -> ".join(tool_calls))
    if not terminated:
        details.append("episode did not terminate before max_env_steps")
    if info.get("error"):
        details.append(f"runner error: {info['error']}")
    if "r_actions" in reward_details:
        details.append(
            "database actions matched expected state"
            if reward_details["r_actions"]
            else "database state diverged from expected final state"
        )
    outputs = reward_details.get("outputs")
    if isinstance(outputs, dict) and outputs:
        missing = [key for key, matched in outputs.items() if not matched]
        if missing:
            details.append("missing required outputs: " + "; ".join(missing))
    if last_observation and "###STOP###" not in last_observation:
        details.append("last_observation=" + last_observation[:240].replace("\n", " "))

    metadata = {
        "benchmark": "tau_bench",
        "domain": md.get("domain"),
        "task_split": md.get("task_split"),
        "task_index": md.get("task_index"),
        "reward": reward,
        "terminated": terminated,
        "tool_calls": tool_calls,
        "expected_outputs": list(md.get("expected_outputs", [])),
        "last_observation": last_observation,
        "final_info": info,
    }
    return " | ".join(details), metadata
