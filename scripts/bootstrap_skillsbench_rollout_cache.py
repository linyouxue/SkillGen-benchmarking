"""Import explicitly mapped legacy BenchFlow artifacts into the slot cache.

This command is offline: it neither resolves a BenchFlow executable nor calls
an API.  The mapping file binds each frozen slot to exact result/trajectory
digests so stochastic rollouts cannot be silently reordered or substituted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.skillsbench_adapter import (  # noqa: E402
    _find_rollout_artifacts,
    bootstrap_skillsbench_rollout_cache,
)
from models import TaskInstance  # noqa: E402


MAPPING_SCHEMA = "skillsbench-rollout-bootstrap-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _parse_time(value: Any, *, label: str) -> datetime:
    text = str(value or "").strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"invalid {label}: {value!r}") from exc


def run(*, dataset_path: Path, mapping_path: Path, model: str) -> dict[str, Any]:
    dataset_path = dataset_path.resolve()
    mapping_path = mapping_path.resolve()
    dataset = _read_object(dataset_path)
    mapping = _read_object(mapping_path)
    if mapping.get("schema") != MAPPING_SCHEMA:
        raise ValueError("unsupported bootstrap mapping schema")
    if mapping.get("dataset_sha256") != _sha256(dataset_path):
        raise ValueError("bootstrap mapping does not bind this exact dataset")

    rows = dataset.get("instances") or []
    instances = {
        str(row["instance_id"]): TaskInstance(
            instance_id=str(row["instance_id"]),
            input=row.get("input"),
            ground_truth=row.get("ground_truth"),
            metadata=dict(row.get("metadata") or {}),
        )
        for row in rows
    }
    assignments = mapping.get("assignments") or []
    if not assignments:
        raise ValueError("bootstrap mapping has no assignments")
    assigned_ids = [str(item.get("instance_id") or "") for item in assignments]
    if len(assigned_ids) != len(set(assigned_ids)):
        raise ValueError("bootstrap mapping repeats an instance_id")
    expected_prefix = [str(row["instance_id"]) for row in rows[: len(assignments)]]
    if assigned_ids != expected_prefix:
        raise ValueError(
            "legacy bootstrap assignments must be the exact frozen dataset prefix"
        )
    if any(instance_id not in instances for instance_id in assigned_ids):
        raise ValueError("bootstrap mapping contains an unknown instance_id")

    not_before = _parse_time(mapping.get("not_before"), label="not_before")
    seen_roots: set[Path] = set()
    prior_started: datetime | None = None
    imported: list[dict[str, Any]] = []
    for item in assignments:
        instance_id = str(item["instance_id"])
        run_root = Path(str(item["run_root"])).expanduser().resolve()
        if run_root in seen_roots:
            raise ValueError("bootstrap mapping repeats a run_root")
        seen_roots.add(run_root)
        result_path, trajectory_path = _find_rollout_artifacts(run_root / "jobs")
        if item.get("result_sha256") != _sha256(result_path):
            raise ValueError(f"result digest mismatch for {instance_id}")
        if item.get("trajectory_sha256") != _sha256(trajectory_path):
            raise ValueError(f"trajectory digest mismatch for {instance_id}")
        result = _read_object(result_path)
        started = _parse_time(result.get("started_at"), label="result.started_at")
        if started < not_before:
            raise ValueError(f"artifact predates this formal pipeline run: {run_root}")
        if prior_started is not None and started <= prior_started:
            raise ValueError("mapped artifact start times are not strictly increasing")
        prior_started = started

        trajectory = bootstrap_skillsbench_rollout_cache(
            instance=instances[instance_id],
            skill_bundle=None,
            config=SimpleNamespace(model=model),
            run_root=run_root,
        )
        imported.append(
            {
                "instance_id": instance_id,
                "trajectory_id": trajectory.trajectory_id,
                "score": trajectory.score,
                "success": trajectory.success,
                "run_root": str(run_root),
                "result_sha256": item["result_sha256"],
                "trajectory_sha256": item["trajectory_sha256"],
            }
        )

    return {
        "schema": MAPPING_SCHEMA,
        "dataset": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "model": model,
        "imported": imported,
        "api_calls_made": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(
        dataset_path=args.dataset,
        mapping_path=args.mapping,
        model=args.model,
    )
    serialized = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(args.output)
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
