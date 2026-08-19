"""Task-level SkillsBench orchestration for the released SkillGen pipeline.

This module intentionally owns *orchestration only*.  It does not modify the
induction, retrieval, generation, refinement, verification gate, or best-of-K
selection implemented by :func:`pipeline.run_pipeline`.

The CLI is dry-run by default.  Importing this module and producing a dry plan
do not import the LLM/runtime modules and do not inspect API-key environment
variables.  Any action that can issue paid model calls requires ``--execute``.

The expected input datasets use SkillGen's regular JSON schema.  For a
single-package SkillsBench task, their instances are pre-declared rollout
replicas with unique IDs.  Construction and sealed-test IDs must be disjoint;
they are rollout-disjoint, not semantically task-disjoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


PROTOCOL_VERSION = "skillsbench-task-runner-v1"
STATUS_SCHEMA_VERSION = 1
_FINAL_CONSTRUCTION_STATUSES = {
    "not_applicable_no_failure",
    "active",
    "deprecated",
}


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    construction_dataset: Path
    sealed_test_dataset: Path
    config_path: Path
    run_root: Path
    task_package: Path | None = None
    generate_scripts: bool | None = None
    resume: bool = True
    allow_paid_retry: bool = False

    @property
    def task_root(self) -> Path:
        return self.run_root / self.task_id


@dataclass(frozen=True)
class RuntimeHooks:
    """Lazy-loaded released runtime, replaceable by offline fakes in tests."""

    load_dataset: Callable[..., Any]
    run_pipeline: Callable[..., Any]
    run_condition: Callable[..., Any]
    load_baseline_condition: Callable[..., Any]
    paired_analysis: Callable[..., Any]
    write_trajectories: Callable[..., Any]
    load_skill: Callable[..., Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_task_id(task_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", task_id):
        raise ValueError(
            "task_id must contain only letters, digits, '.', '_' or '-', "
            "and must begin with a letter or digit"
        )


def _require_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is not a file: {resolved}")
    return resolved


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_directory(path: Path) -> str:
    """Content hash a task package, excluding VCS/cache noise."""

    digest = hashlib.sha256()
    excluded_parts = {".git", "__pycache__", ".pytest_cache"}
    files = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file()
        and not any(part in excluded_parts for part in candidate.relative_to(path).parts)
        and not candidate.name.endswith((".pyc", ".pyo"))
    )
    for candidate in files:
        relative = candidate.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def build_protocol(spec: TaskSpec) -> tuple[dict[str, Any], str]:
    """Build the immutable protocol manifest and its stable SHA-256 hash."""

    _validate_task_id(spec.task_id)
    construction = _require_file(spec.construction_dataset, "construction dataset")
    sealed = _require_file(spec.sealed_test_dataset, "sealed-test dataset")
    config = _require_file(spec.config_path, "config")

    package_hash: str | None = None
    if spec.task_package is not None:
        package = spec.task_package.resolve()
        if not package.is_dir():
            raise FileNotFoundError(f"task package is not a directory: {package}")
        package_hash = _hash_directory(package)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "task_id": spec.task_id,
        "construction_dataset_sha256": _hash_file(construction),
        "sealed_test_dataset_sha256": _hash_file(sealed),
        "source_config_sha256": _hash_file(config),
        "task_package_sha256": package_hash,
        "generate_scripts_override": spec.generate_scripts,
        "split_kind": "rollout_disjoint",
        "primary_drop_blank_pairs": False,
        "upstream_default_drop_blank_sensitivity": True,
    }
    serialized = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return manifest, hashlib.sha256(serialized).hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    """Atomically replace ``path`` using a temporary file in the same folder."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        temp_name = None
    finally:
        if temp_name is not None:
            try:
                Path(temp_name).unlink()
            except FileNotFoundError:
                pass


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _initial_status(
    spec: TaskSpec, manifest: dict[str, Any], protocol_hash: str
) -> dict[str, Any]:
    return {
        "schema_version": STATUS_SCHEMA_VERSION,
        "task_id": spec.task_id,
        "protocol_hash": protocol_hash,
        "protocol": manifest,
        "stage": "prepared",
        "method_status": "pending",
        "pipeline_run_dir": None,
        "skill_repo": None,
        "skill_id": None,
        "skill_status": None,
        "baseline_counts": None,
        "sealed_baseline_path": None,
        "sealed_skill_path": None,
        "result_path": None,
        "failure": None,
        "updated_at": _utc_now(),
    }


def _save_status(path: Path, state: dict[str, Any], **updates: Any) -> dict[str, Any]:
    state = dict(state)
    state.update(updates)
    state["updated_at"] = _utc_now()
    _atomic_write_json(path, state)
    return state


def _load_or_initialize_status(
    spec: TaskSpec,
    manifest: dict[str, Any],
    protocol_hash: str,
) -> tuple[Path, dict[str, Any]]:
    status_path = spec.task_root / "status.json"
    if status_path.exists():
        if not spec.resume:
            raise FileExistsError(
                f"status already exists and --no-resume was requested: {status_path}"
            )
        state = _load_json(status_path)
        if state.get("protocol_hash") != protocol_hash:
            raise ValueError(
                "existing task state has a different protocol hash; use a new "
                "run root instead of mixing datasets/configurations"
            )
        return status_path, state

    spec.task_root.mkdir(parents=True, exist_ok=True)
    state = _initial_status(spec, manifest, protocol_hash)
    _atomic_write_json(status_path, state)
    return status_path, state


def _load_runtime_hooks() -> RuntimeHooks:
    """Import the released runtime only after the paid-action gate is open."""

    repo_root = Path(__file__).resolve().parents[1]
    repo_text = str(repo_root)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)

    from artifacts import write_trajectories
    from eval_skill import (
        load_baseline_condition,
        paired_analysis,
        run_condition,
    )
    from main import load_dataset
    from pipeline import run_pipeline
    from skill_store import load_skill

    return RuntimeHooks(
        load_dataset=load_dataset,
        run_pipeline=run_pipeline,
        run_condition=run_condition,
        load_baseline_condition=load_baseline_condition,
        paired_analysis=paired_analysis,
        write_trajectories=write_trajectories,
        load_skill=load_skill,
    )


def _write_runtime_config(spec: TaskSpec) -> tuple[Path, dict[str, Any]]:
    """Copy config while changing artifact destinations only, never algorithms."""

    import yaml

    with spec.config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"config root must be a mapping: {spec.config_path}")

    task_root = spec.task_root.resolve()
    config.setdefault("pipeline", {})["artifact_root"] = str(
        task_root / "pipeline_runs"
    )
    config.setdefault("skill_output", {})["path"] = str(task_root / "skill_output")
    config.setdefault("generation", {})["candidate_output_dir"] = str(
        task_root / "candidates"
    )

    runtime_path = task_root / "runtime_config.yaml"
    rendered = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    _atomic_write_text(runtime_path, rendered)
    return runtime_path, config


def _validate_datasets(construction: Any, sealed: Any) -> None:
    construction_ids = [str(instance.instance_id) for instance in construction.instances]
    sealed_ids = [str(instance.instance_id) for instance in sealed.instances]
    if not construction_ids:
        raise ValueError("construction dataset has no rollout replicas")
    if not sealed_ids:
        raise ValueError("sealed-test dataset has no rollout replicas")
    if len(construction_ids) != len(set(construction_ids)):
        raise ValueError("construction rollout instance IDs must be unique")
    if len(sealed_ids) != len(set(sealed_ids)):
        raise ValueError("sealed-test rollout instance IDs must be unique")
    overlap = set(construction_ids) & set(sealed_ids)
    if overlap:
        raise ValueError(
            "construction and sealed-test rollout IDs overlap: "
            f"{sorted(overlap)[:5]}"
        )
    if construction.task_type != sealed.task_type:
        raise ValueError("construction and sealed-test task_type values differ")
    construction_meta = getattr(construction, "metadata", {}) or {}
    sealed_meta = getattr(sealed, "metadata", {}) or {}
    for key in ("task_id", "task_digest", "source_version"):
        left = construction_meta.get(key)
        right = sealed_meta.get(key)
        if left is not None or right is not None:
            if left != right:
                raise ValueError(
                    f"construction and sealed-test metadata differ for {key}: "
                    f"{left!r} != {right!r}"
                )


def _matching_pipeline_runs(
    task_root: Path, task_id: str, protocol_hash: str
) -> list[Path]:
    root = task_root / "pipeline_runs"
    if not root.exists():
        return []
    matches: list[Path] = []
    for metadata_path in root.glob("*/run_metadata.json"):
        try:
            metadata = _load_json(metadata_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        dataset_metadata = metadata.get("dataset_metadata") or {}
        if (
            metadata.get("dataset_id") == task_id
            and dataset_metadata.get("protocol_hash") == protocol_hash
        ):
            matches.append(metadata_path.parent)
    return sorted(matches, key=lambda path: (path.stat().st_mtime_ns, path.name))


def _latest_pipeline_run(
    task_root: Path, task_id: str, protocol_hash: str
) -> Path | None:
    matches = _matching_pipeline_runs(task_root, task_id, protocol_hash)
    return matches[-1] if matches else None


def _is_resumable_checkpoint(run_dir: Path | None) -> bool:
    if run_dir is None:
        return False
    return (
        (run_dir / "checkpoint_trajectories.jsonl").is_file()
        and (run_dir / "checkpoint.json").is_file()
    )


def _inspect_baseline_checkpoint(run_dir: Path | None) -> dict[str, Any]:
    """Read persisted baseline outcomes using the pipeline's own truth rule."""

    if run_dir is None:
        return {
            "valid": False,
            "reason": "pipeline run directory was not found",
            "n_records": 0,
            "n_successes": 0,
            "n_failures": 0,
            "source": None,
            "instance_ids": [],
        }

    candidates = [
        run_dir / "checkpoint_trajectories.jsonl",
        run_dir / "baseline_trajectories.jsonl",
    ]
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        return {
            "valid": False,
            "reason": "baseline checkpoint is missing",
            "n_records": 0,
            "n_successes": 0,
            "n_failures": 0,
            "source": None,
            "instance_ids": [],
        }

    records: list[dict[str, Any]] = []
    try:
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("trajectory record is not an object")
                    records.append(value)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "valid": False,
            "reason": f"could not read baseline checkpoint: {exc}",
            "n_records": 0,
            "n_successes": 0,
            "n_failures": 0,
            "source": str(source),
            "instance_ids": [],
        }

    if not records:
        return {
            "valid": False,
            "reason": "baseline checkpoint is empty",
            "n_records": 0,
            "n_successes": 0,
            "n_failures": 0,
            "source": str(source),
            "instance_ids": [],
        }

    # run_pipeline uses ``not t.success``.  Mirror that exactly: null/missing
    # success is a failure signal, not evidence that all runs succeeded.
    n_successes = sum(1 for record in records if bool(record.get("success")))
    n_failures = len(records) - n_successes
    return {
        "valid": True,
        "reason": None,
        "n_records": len(records),
        "n_successes": n_successes,
        "n_failures": n_failures,
        "source": str(source),
        "instance_ids": [str(record.get("instance_id")) for record in records],
    }


def _prepared_instance_ids(dataset_path: Path) -> list[str]:
    payload = _load_json(dataset_path)
    instances = payload.get("instances") or []
    return [str(item.get("instance_id")) for item in instances if isinstance(item, dict)]


def _find_skill_repo(task_root: Path, skill_id: str) -> Path | None:
    output_root = task_root / "skill_output"
    if not output_root.exists():
        return None
    matches = list(output_root.rglob(f"{skill_id}.json"))
    if not matches:
        return None
    matches.sort(key=lambda path: (path.stat().st_mtime_ns, str(path)))
    return matches[-1].parent


def _status_text(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw).lower()


def _config_models(config: dict[str, Any]) -> tuple[str, str, int]:
    model_cfg = config.get("models") or {}
    pipeline_cfg = config.get("pipeline") or {}
    default_model = model_cfg.get("default", "openai/gpt-5.4-mini")
    base_model = model_cfg.get("baseline_agent", default_model)
    judge_model = model_cfg.get("baseline_judge", default_model)
    max_workers = int(pipeline_cfg.get("max_workers", 16))
    return str(base_model), str(judge_model), max_workers


def _ensure_sealed_baseline(
    *,
    spec: TaskSpec,
    state: dict[str, Any],
    status_path: Path,
    sealed: Any,
    hooks: RuntimeHooks,
    base_model: str,
    judge_model: str,
    max_workers: int,
) -> tuple[Any, dict[str, Any]]:
    baseline_path = spec.task_root / "sealed" / "baseline.jsonl"
    expected_ids = {str(instance.instance_id) for instance in sealed.instances}

    recorded = state.get("sealed_baseline_path")
    reusable_path = (
        Path(recorded)
        if recorded and Path(recorded).is_file()
        else baseline_path
    )
    if reusable_path.is_file():
        baseline = hooks.load_baseline_condition(
            reusable_path,
            base_model,
            expected_instance_ids=expected_ids,
        )
        state = _save_status(
            status_path,
            state,
            stage="sealed_baseline_ready",
            sealed_baseline_path=str(reusable_path.resolve()),
            failure=None,
        )
        return baseline, state

    state = _save_status(
        status_path,
        state,
        stage="evaluating_baseline",
        failure=None,
    )
    baseline = hooks.run_condition(
        sealed.instances,
        sealed.task_type,
        base_model,
        judge_model,
        skill=None,
        max_workers=max_workers,
        enable_web_search=False,
        execute_scripts=False,
    )
    _atomic_write_trajectory_file(hooks, baseline_path, baseline.trajectories)
    state = _save_status(
        status_path,
        state,
        stage="sealed_baseline_ready",
        sealed_baseline_path=str(baseline_path.resolve()),
    )
    return baseline, state


def _atomic_write_trajectory_file(
    hooks: RuntimeHooks, path: Path, trajectories: list[Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        hooks.write_trajectories(temporary, trajectories)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _paired_record(paired: Any, *, drop_blank: bool) -> dict[str, Any]:
    repair = int(getattr(paired, "repair", 0))
    regression = int(getattr(paired, "regression", 0))
    baseline_acc = float(getattr(paired, "baseline_acc", 0.0))
    skill_acc = float(getattr(paired, "skill_acc", 0.0))
    return {
        "n_instances": int(getattr(paired, "n_instances", 0)),
        "baseline_acc": baseline_acc,
        "skill_acc": skill_acc,
        "delta_acc": skill_acc - baseline_acc,
        "repair": repair,
        "regression": regression,
        "repair_rate": float(getattr(paired, "repair_rate", 0.0)),
        "regression_rate": float(getattr(paired, "regression_rate", 0.0)),
        "net_gain": int(getattr(paired, "net_gain", repair - regression)),
        "blank_filter": {
            "drop_blank": drop_blank,
            "n_paired_raw": int(getattr(paired, "n_paired_raw", 0)),
            "n_blank_baseline": int(getattr(paired, "n_blank_baseline", 0)),
            "n_blank_skill": int(getattr(paired, "n_blank_skill", 0)),
            "n_blank_either": int(getattr(paired, "n_blank_either", 0)),
            "n_blank_both": int(getattr(paired, "n_blank_both", 0)),
        },
    }


def _verifier_primary_metrics(
    runtime: RuntimeHooks, baseline: Any, skill_result: Any
) -> dict[str, Any]:
    """Score once-run trajectories under both sealed-summary policies.

    SkillsBench's official verifier is the primary authority, so valid
    tool-only trajectories remain in the primary paired table.  The released
    ``eval_skill.py`` default blank-DROP heuristic is retained as a zero-cost
    sensitivity analysis over the exact same trajectories.
    """

    keep = runtime.paired_analysis(baseline, skill_result, drop_blank=False)
    drop = runtime.paired_analysis(baseline, skill_result, drop_blank=True)
    metrics = _paired_record(keep, drop_blank=False)
    metrics["upstream_default_blank_drop_sensitivity"] = _paired_record(
        drop, drop_blank=True
    )
    return metrics


def _write_result(
    spec: TaskSpec,
    state: dict[str, Any],
    status_path: Path,
    result: dict[str, Any],
    *,
    final_stage: str = "complete",
) -> tuple[dict[str, Any], dict[str, Any]]:
    result_path = spec.task_root / "result.json"
    _atomic_write_json(result_path, result)
    state = _save_status(
        status_path,
        state,
        stage=final_stage,
        result_path=str(result_path.resolve()),
    )
    return result, state


def _pipeline_error_result(
    spec: TaskSpec,
    protocol_hash: str,
    state: dict[str, Any],
    status_path: Path,
    reason: str,
    counts: dict[str, Any] | None,
) -> dict[str, Any]:
    result = {
        "task_id": spec.task_id,
        "protocol_hash": protocol_hash,
        "method_status": "pipeline_error",
        "reason": reason,
        "skill_generated": False,
        "sealed_test_executed": False,
        "baseline_counts": counts,
    }
    state = _save_status(
        status_path,
        state,
        stage="failed",
        method_status="pipeline_error",
        baseline_counts=counts,
        failure={"kind": "pipeline_error", "reason": reason},
    )
    _write_result(spec, state, status_path, result, final_stage="failed")
    return result


def _runtime_failure_kind(exc: Exception) -> str:
    try:
        from benchmarks.skillsbench_adapter import SkillsBenchInfrastructureError

        if isinstance(exc, SkillsBenchInfrastructureError):
            return "infrastructure_error"
    except ImportError:
        pass
    return "runtime_exception"


def run_task(
    spec: TaskSpec,
    *,
    execute: bool = False,
    hooks: RuntimeHooks | None = None,
) -> dict[str, Any]:
    """Plan or execute one SkillsBench task under the original SkillGen method."""

    manifest, protocol_hash = build_protocol(spec)
    status_path = spec.task_root / "status.json"

    if not execute:
        existing_stage = None
        recoverable_persisted_condition = False
        if status_path.exists():
            existing = _load_json(status_path)
            existing_stage = existing.get("stage")
            if existing.get("protocol_hash") != protocol_hash:
                raise ValueError("existing status has a different protocol hash")
            recoverable_persisted_condition = (
                existing_stage in {"evaluating_baseline", "sealed_baseline_failed"}
                and (spec.task_root / "sealed" / "baseline.jsonl").is_file()
            ) or (
                existing_stage in {"evaluating", "sealed_skill_failed"}
                and (spec.task_root / "sealed" / "with_skill.jsonl").is_file()
            )
        return {
            "mode": "dry_run",
            "paid_actions_executed": False,
            "task_id": spec.task_id,
            "protocol_hash": protocol_hash,
            "protocol": manifest,
            "existing_stage": existing_stage,
            "paid_retry_required": existing_stage
            in {
                "constructing",
                "evaluating",
                "evaluating_baseline",
                "sealed_baseline_failed",
                "sealed_skill_failed",
                "failed",
            }
            and not recoverable_persisted_condition,
            "planned_stages": [
                "released run_pipeline on construction replicas",
                "inspect persisted baseline checkpoint if pipeline returns None",
                "one sealed no-skill baseline rollout set",
                "existing paired eval or empty-intervention reuse",
            ],
            "warning": (
                "splits are rollout-disjoint replicas of one task package, "
                "not task- or semantic-instance-disjoint"
            ),
        }

    status_path, state = _load_or_initialize_status(spec, manifest, protocol_hash)

    if state.get("stage") == "complete":
        result_path = state.get("result_path")
        if not result_path or not Path(result_path).is_file():
            raise FileNotFoundError("status is complete but result.json is missing")
        return _load_json(Path(result_path))

    # The released pipeline returns early on an all-success construction pool
    # before writing checkpoint.json. If the process dies in the tiny window
    # before this orchestrator records that return, recover only when the
    # persisted trajectories exactly cover every pre-declared construction ID
    # and every outcome is successful. Anything less remains fail-closed.
    if state.get("stage") == "constructing":
        latest_run = _latest_pipeline_run(spec.task_root, spec.task_id, protocol_hash)
        counts = _inspect_baseline_checkpoint(latest_run)
        expected_ids = _prepared_instance_ids(spec.construction_dataset)
        observed_ids = counts.get("instance_ids") or []
        exact_all_success = (
            counts.get("valid") is True
            and counts.get("n_failures") == 0
            and len(observed_ids) == len(expected_ids)
            and set(observed_ids) == set(expected_ids)
        )
        if exact_all_success:
            state = _save_status(
                status_path,
                state,
                stage="constructed_none",
                method_status="not_applicable_no_failure",
                pipeline_run_dir=str(latest_run.resolve()) if latest_run else None,
                baseline_counts=counts,
                skill_id=None,
                skill_repo=None,
                skill_status=None,
                failure=None,
            )

    uncertain_paid_stages = {
        "constructing",
        "evaluating",
        "evaluating_baseline",
        "sealed_baseline_failed",
        "sealed_skill_failed",
    }
    if state.get("stage") == "failed" and state.get("result_path"):
        failed_result = Path(str(state["result_path"]))
        if failed_result.is_file() and not spec.allow_paid_retry:
            return _load_json(failed_result)
    recoverable_baseline_checkpoint = (
        state.get("stage") in {"evaluating_baseline", "sealed_baseline_failed"}
        and (spec.task_root / "sealed" / "baseline.jsonl").is_file()
    )
    recoverable_skill_checkpoint = (
        state.get("stage") in {"evaluating", "sealed_skill_failed"}
        and (spec.task_root / "sealed" / "with_skill.jsonl").is_file()
    )
    if (
        state.get("stage") in uncertain_paid_stages | {"failed"}
        and not recoverable_baseline_checkpoint
        and not recoverable_skill_checkpoint
        and not spec.allow_paid_retry
    ):
        raise RuntimeError(
            "A previous paid stage failed or was interrupted after its request "
            "state became uncertain. Inspect status/artifacts and pass "
            "--retry-paid explicitly if repeating requests is intended."
        )

    runtime = hooks or _load_runtime_hooks()
    runtime_config_path, runtime_config = _write_runtime_config(spec)
    construction = runtime.load_dataset(str(spec.construction_dataset))
    sealed = runtime.load_dataset(str(spec.sealed_test_dataset))
    _validate_datasets(construction, sealed)

    construction_resolved = state.get("method_status") in _FINAL_CONSTRUCTION_STATUSES
    if not construction_resolved:
        latest_run = _latest_pipeline_run(spec.task_root, spec.task_id, protocol_hash)
        recorded_run = Path(state["pipeline_run_dir"]) if state.get("pipeline_run_dir") else None
        resume_dir = recorded_run if _is_resumable_checkpoint(recorded_run) else latest_run
        if not _is_resumable_checkpoint(resume_dir):
            resume_dir = None

        state = _save_status(
            status_path,
            state,
            stage="constructing",
            pipeline_run_dir=str(resume_dir.resolve()) if resume_dir else None,
            failure=None,
        )
        try:
            skill = runtime.run_pipeline(
                construction.instances,
                construction.task_type,
                config_path=str(runtime_config_path),
                dataset_id=spec.task_id,
                task_name=getattr(construction, "task_name", spec.task_id),
                dataset_metadata={
                    **(getattr(construction, "metadata", {}) or {}),
                    "protocol_hash": protocol_hash,
                    "split_kind": "rollout_disjoint",
                },
                generate_scripts=spec.generate_scripts,
                resume_dir=str(resume_dir) if resume_dir else None,
            )
        except Exception as exc:
            latest_run = _latest_pipeline_run(spec.task_root, spec.task_id, protocol_hash)
            failure_kind = _runtime_failure_kind(exc)
            _save_status(
                status_path,
                state,
                stage="failed",
                method_status=(
                    "infra_error"
                    if failure_kind == "infrastructure_error"
                    else "pipeline_error"
                ),
                pipeline_run_dir=str(latest_run.resolve()) if latest_run else None,
                failure={
                    "kind": failure_kind,
                    "reason": str(exc),
                },
            )
            raise

        latest_run = _latest_pipeline_run(spec.task_root, spec.task_id, protocol_hash)
        if skill is None:
            counts = _inspect_baseline_checkpoint(latest_run)
            if counts["valid"] and counts["n_failures"] == 0:
                state = _save_status(
                    status_path,
                    state,
                    stage="constructed_none",
                    method_status="not_applicable_no_failure",
                    pipeline_run_dir=str(latest_run.resolve()) if latest_run else None,
                    baseline_counts=counts,
                    skill_id=None,
                    skill_repo=None,
                    skill_status=None,
                )
            else:
                if counts["valid"]:
                    reason = (
                        "run_pipeline returned None although its baseline checkpoint "
                        f"contains {counts['n_failures']} failure signal(s)"
                    )
                else:
                    reason = (
                        "run_pipeline returned None but no valid non-empty baseline "
                        f"checkpoint proves an all-success induction pool: {counts['reason']}"
                    )
                return _pipeline_error_result(
                    spec, protocol_hash, state, status_path, reason, counts
                )
        else:
            skill_id = str(getattr(skill, "skill_id", ""))
            if not skill_id:
                return _pipeline_error_result(
                    spec,
                    protocol_hash,
                    state,
                    status_path,
                    "run_pipeline returned an object without skill_id",
                    _inspect_baseline_checkpoint(latest_run),
                )
            skill_repo = _find_skill_repo(spec.task_root, skill_id)
            if skill_repo is None:
                return _pipeline_error_result(
                    spec,
                    protocol_hash,
                    state,
                    status_path,
                    f"persisted skill JSON was not found for skill_id={skill_id}",
                    _inspect_baseline_checkpoint(latest_run),
                )
            skill_status = _status_text(getattr(skill, "status", "active"))
            if skill_status not in {"active", "deprecated"}:
                return _pipeline_error_result(
                    spec,
                    protocol_hash,
                    state,
                    status_path,
                    f"unexpected finalized skill status: {skill_status}",
                    _inspect_baseline_checkpoint(latest_run),
                )
            state = _save_status(
                status_path,
                state,
                stage="skill_ready",
                method_status=skill_status,
                pipeline_run_dir=str(latest_run.resolve()) if latest_run else None,
                skill_repo=str(skill_repo.resolve()),
                skill_id=skill_id,
                skill_status=skill_status,
                baseline_counts=_inspect_baseline_checkpoint(latest_run),
            )

    base_model, judge_model, max_workers = _config_models(runtime_config)
    try:
        baseline, state = _ensure_sealed_baseline(
            spec=spec,
            state=state,
            status_path=status_path,
            sealed=sealed,
            hooks=runtime,
            base_model=base_model,
            judge_model=judge_model,
            max_workers=max_workers,
        )
    except Exception as exc:
        _save_status(
            status_path,
            state,
            stage="sealed_baseline_failed",
            failure={"kind": _runtime_failure_kind(exc), "reason": str(exc)},
        )
        raise

    method_status = state.get("method_status")
    if method_status == "not_applicable_no_failure":
        # Empty-intervention deployment semantics: execute sealed baseline once,
        # then reuse the same trajectories as the skill condition.  A second
        # stochastic rollout would not represent an absent intervention.
        metrics = _verifier_primary_metrics(runtime, baseline, baseline)
        metrics.update(
            {
                "task_id": spec.task_id,
                "protocol_hash": protocol_hash,
                "method_status": "not_applicable_no_failure",
                "reason": "no_induction_failure",
                "skill_generated": False,
                "skill_id": None,
                "skill_status": None,
                "deployed_intervention": "empty",
                "skill_condition_executed": False,
                "skill_condition_reused_baseline": True,
                "split_kind": "rollout_disjoint",
                "sealed_test_executed": True,
            }
        )
        # Make the contract explicit even if a custom paired analyser is used.
        metrics["skill_acc"] = metrics["baseline_acc"]
        metrics["delta_acc"] = 0.0
        metrics["repair"] = 0
        metrics["regression"] = 0
        metrics["repair_rate"] = 0.0
        metrics["regression_rate"] = 0.0
        metrics["net_gain"] = 0
        result, _ = _write_result(spec, state, status_path, metrics)
        return result

    skill_repo = state.get("skill_repo")
    skill_id = state.get("skill_id")
    if not skill_repo or not skill_id:
        return _pipeline_error_result(
            spec,
            protocol_hash,
            state,
            status_path,
            "construction status expects a skill but its repository/id is missing",
            state.get("baseline_counts"),
        )
    skill = runtime.load_skill(skill_repo, skill_id=skill_id)
    actual_status = _status_text(getattr(skill, "status", method_status))

    if actual_status == "deprecated":
        skill_result = baseline
        skill_condition_executed = False
        deployed_intervention = "empty"
    elif actual_status == "active":
        skill_path = spec.task_root / "sealed" / "with_skill.jsonl"
        recorded_skill_path = state.get("sealed_skill_path")
        reusable_skill_path = (
            Path(recorded_skill_path)
            if recorded_skill_path and Path(recorded_skill_path).is_file()
            else skill_path
        )
        expected_ids = {str(instance.instance_id) for instance in sealed.instances}
        if reusable_skill_path.is_file():
            skill_result = runtime.load_baseline_condition(
                reusable_skill_path,
                base_model,
                expected_instance_ids=expected_ids,
            )
        else:
            state = _save_status(status_path, state, stage="evaluating")
            try:
                skill_result = runtime.run_condition(
                    sealed.instances,
                    sealed.task_type,
                    base_model,
                    judge_model,
                    skill=skill,
                    max_workers=max_workers,
                    enable_web_search=False,
                    execute_scripts=bool(getattr(skill, "scripts", None)),
                )
            except Exception as exc:
                _save_status(
                    status_path,
                    state,
                    stage="sealed_skill_failed",
                    failure={
                        "kind": _runtime_failure_kind(exc),
                        "reason": str(exc),
                    },
                )
                raise
            _atomic_write_trajectory_file(
                runtime, skill_path, skill_result.trajectories
            )
            state = _save_status(
                status_path,
                state,
                stage="sealed_skill_ready",
                sealed_skill_path=str(skill_path.resolve()),
            )
        skill_condition_executed = True
        deployed_intervention = "generated_skill"
    else:
        return _pipeline_error_result(
            spec,
            protocol_hash,
            state,
            status_path,
            f"loaded skill has unsupported status: {actual_status}",
            state.get("baseline_counts"),
        )

    metrics = _verifier_primary_metrics(runtime, baseline, skill_result)
    metrics.update(
        {
            "task_id": spec.task_id,
            "protocol_hash": protocol_hash,
            "method_status": actual_status,
            "reason": None if actual_status == "active" else "verification_gate_rejected",
            "skill_generated": True,
            "skill_id": skill_id,
            "skill_status": actual_status,
            "deployed_intervention": deployed_intervention,
            "skill_condition_executed": skill_condition_executed,
            "skill_condition_reused_baseline": not skill_condition_executed,
            "split_kind": "rollout_disjoint",
            "sealed_test_executed": True,
        }
    )
    if actual_status == "deprecated":
        metrics["skill_acc"] = metrics["baseline_acc"]
        metrics["delta_acc"] = 0.0
        metrics["repair"] = 0
        metrics["regression"] = 0
        metrics["repair_rate"] = 0.0
        metrics["regression_rate"] = 0.0
        metrics["net_gain"] = 0
    result, _ = _write_result(spec, state, status_path, metrics)
    return result


def _parse_generate_scripts(args: argparse.Namespace) -> bool | None:
    if args.generate_scripts:
        return True
    if args.no_generate_scripts:
        return False
    return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run original SkillGen independently on one SkillsBench task"
    )
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--construction-dataset", required=True, type=Path)
    parser.add_argument("--sealed-test-dataset", required=True, type=Path)
    parser.add_argument("--config", dest="config_path", default=Path("config.yaml"), type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--task-package", type=Path, default=None)
    scripts_group = parser.add_mutually_exclusive_group()
    scripts_group.add_argument("--generate-scripts", action="store_true")
    scripts_group.add_argument("--no-generate-scripts", action="store_true")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="refuse to use an existing status instead of resuming it",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "authorize construction and sealed-test model calls; without this "
            "flag only a dry plan is printed and runtime/API modules are not loaded"
        ),
    )
    parser.add_argument(
        "--retry-paid",
        action="store_true",
        help=(
            "after inspecting artifacts, explicitly allow a failed/interrupted "
            "paid stage to repeat; never implied by --execute or --resume"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    spec = TaskSpec(
        task_id=args.task_id,
        construction_dataset=args.construction_dataset,
        sealed_test_dataset=args.sealed_test_dataset,
        config_path=args.config_path,
        run_root=args.run_root,
        task_package=args.task_package,
        generate_scripts=_parse_generate_scripts(args),
        resume=not args.no_resume,
        allow_paid_retry=args.retry_paid,
    )
    result = run_task(spec, execute=args.execute)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 2 if result.get("method_status") == "pipeline_error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
