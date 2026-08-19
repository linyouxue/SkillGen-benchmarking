"""Prepare fixed same-task rollout slots for the SkillGen SkillsBench study."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.skillsbench_adapter import (  # noqa: E402
    ADAPTER_SCHEMA_VERSION,
    parse_task_markdown,
    task_package_digest,
    validate_task_package,
)


def _git_metadata(task_dir: Path) -> dict[str, Any]:
    repo = next(
        (candidate for candidate in (task_dir, *task_dir.parents) if (candidate / ".git").exists()),
        None,
    )
    if repo is None:
        return {"repository_root": None, "commit": None, "task_dirty": None}

    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
            shell=False,
        )
        return completed.stdout.strip()

    try:
        rel = task_dir.relative_to(repo).as_posix()
        return {
            "repository_root": str(repo),
            "commit": run("rev-parse", "HEAD"),
            "task_dirty": bool(run("status", "--porcelain", "--", rel)),
        }
    except (OSError, subprocess.SubprocessError, ValueError):
        return {"repository_root": str(repo), "commit": None, "task_dirty": None}


def _task_relpath(task_dir: Path, repo_root: str | None) -> str | None:
    if not repo_root:
        return None
    try:
        return task_dir.relative_to(Path(repo_root)).as_posix()
    except ValueError:
        return None


def _instances(
    *,
    task_id: str,
    split: str,
    count: int,
    prompt: str,
    common_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "instance_id": f"{task_id}::{split}::r{idx:03d}",
            "input": prompt,
            "ground_truth": None,
            "metadata": {
                **common_metadata,
                "skillsbench_split": split,
                "skillsbench_rollout_index": idx,
            },
        }
        for idx in range(count)
    ]


def build_payloads(
    *,
    task_dir: Path,
    construction_rollouts: int,
    test_rollouts: int,
    agent: str,
    sandbox: str,
    jobs_root: Path,
    bench_executable: str,
    subprocess_timeout_sec: float,
    source_version: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if construction_rollouts <= 0 or test_rollouts <= 0:
        raise ValueError("construction-rollouts and test-rollouts must both be positive")
    if subprocess_timeout_sec <= 0:
        raise ValueError("subprocess-timeout-sec must be positive")
    task_dir = validate_task_package(task_dir)
    frontmatter, prompt = parse_task_markdown(task_dir)
    task_id = task_dir.name
    digest = task_package_digest(task_dir)
    git_meta = _git_metadata(task_dir)
    task_relpath = _task_relpath(task_dir, git_meta.get("repository_root"))
    category_metadata = frontmatter.get("metadata") or {}

    common_metadata: dict[str, Any] = {
        "benchmark": "skillsbench",
        "skillsbench_adapter_schema": ADAPTER_SCHEMA_VERSION,
        "skillsbench_task_id": task_id,
        "skillsbench_task_dir": str(task_dir),
        "skillsbench_task_relpath": task_relpath,
        "skillsbench_task_digest": digest,
        "skillsbench_source_version": source_version,
        "skillsbench_source_commit": git_meta.get("commit"),
        "skillsbench_agent": agent,
        "skillsbench_sandbox": sandbox,
        "skillsbench_jobs_root": str(jobs_root.resolve()),
        "skillsbench_bench_executable": bench_executable,
        "skillsbench_subprocess_timeout_sec": subprocess_timeout_sec,
        "rollout_replica_kind": "stochastic_same_task",
        "seed_control": "unsupported_by_benchflow_cli",
        "official_task_skills_visible": False,
    }
    dataset_metadata = {
        "benchmark": "skillsbench",
        "protocol": "released_skillgen_per_task_sparse_rollouts",
        "adapter_schema": ADAPTER_SCHEMA_VERSION,
        "task_id": task_id,
        "task_digest": digest,
        "source_version": source_version,
        "source_commit": git_meta.get("commit"),
        "source_task_dirty": git_meta.get("task_dirty"),
        "category_metadata": category_metadata,
        "rollout_replica_kind": "stochastic_same_task",
        "seed_control": "unsupported_by_benchflow_cli",
        "cross_task_pooling": False,
        "manual_examples": False,
        "refinement_example_extension": False,
        "official_task_skills_used_for_construction": False,
    }
    construction = {
        "dataset_id": f"skillsbench-{source_version}::{task_id}::construction",
        "task_name": task_id,
        "task_type": "scored",
        "metadata": {**dataset_metadata, "split": "construction"},
        "instances": _instances(
            task_id=task_id,
            split="construction",
            count=construction_rollouts,
            prompt=prompt,
            common_metadata=common_metadata,
        ),
    }
    sealed_test = {
        "dataset_id": f"skillsbench-{source_version}::{task_id}::sealed-test",
        "task_name": task_id,
        "task_type": "scored",
        "metadata": {**dataset_metadata, "split": "sealed-test"},
        "instances": _instances(
            task_id=task_id,
            split="sealed-test",
            count=test_rollouts,
            prompt=prompt,
            common_metadata=common_metadata,
        ),
    }
    manifest = {
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "task_id": task_id,
        "task_dir": str(task_dir),
        "task_digest": digest,
        "source_version": source_version,
        "source_commit": git_meta.get("commit"),
        "source_task_dirty": git_meta.get("task_dirty"),
        "construction_rollouts": construction_rollouts,
        "sealed_test_rollouts": test_rollouts,
        "construction_instance_ids": [x["instance_id"] for x in construction["instances"]],
        "sealed_test_instance_ids": [x["instance_id"] for x in sealed_test["instances"]],
        "agent": agent,
        "sandbox": sandbox,
        "bench_executable": bench_executable,
        "jobs_root": str(jobs_root.resolve()),
        "method_fidelity": {
            "released_skillgen_core_unchanged": True,
            "task_processed_independently": True,
            "cross_task_pooling": False,
            "adaptive_sampling_until_both_classes": False,
            "refinement_positive_example_extension": False,
            "official_curated_skill_is_construction_input": False,
        },
        "interpretation": (
            "Rollout slots are stochastic replicas of one fixed task package, not "
            "independent task instances or evidence of cross-task generalization."
        ),
    }
    return construction, sealed_test, manifest


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare one SkillsBench task as fixed SkillGen rollout slots."
    )
    parser.add_argument("task_dir", type=Path, help="Path to tasks/<task-id>")
    parser.add_argument("--construction-rollouts", type=int, required=True)
    parser.add_argument("--test-rollouts", type=int, required=True)
    parser.add_argument("--agent", required=True, help="BenchFlow agent id")
    parser.add_argument("--sandbox", default="docker")
    parser.add_argument("--bench-executable", default="bench")
    parser.add_argument("--source-version", default="v1.1")
    parser.add_argument(
        "--jobs-root",
        type=Path,
        default=Path("artifacts/skillsbench/benchflow_jobs"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: data/skillsbench/<task-id>",
    )
    parser.add_argument("--subprocess-timeout-sec", type=float, default=7200)
    parser.add_argument("--dry-run", action="store_true", help="Validate and print only")
    parser.add_argument("--force", action="store_true", help="Replace prepared JSON files")
    args = parser.parse_args()

    construction, sealed_test, manifest = build_payloads(
        task_dir=args.task_dir,
        construction_rollouts=args.construction_rollouts,
        test_rollouts=args.test_rollouts,
        agent=args.agent,
        sandbox=args.sandbox,
        jobs_root=args.jobs_root,
        bench_executable=args.bench_executable,
        subprocess_timeout_sec=args.subprocess_timeout_sec,
        source_version=args.source_version,
    )
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    output_dir = args.output_dir or Path("data/skillsbench") / manifest["task_id"]
    paths = [
        output_dir / "construction.json",
        output_dir / "sealed_test.json",
        output_dir / "protocol_manifest.json",
    ]
    existing = [str(path) for path in paths if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "Refusing to overwrite prepared protocol files; pass --force: "
            + ", ".join(existing)
        )
    _atomic_json(paths[0], construction)
    _atomic_json(paths[1], sealed_test)
    _atomic_json(paths[2], manifest)
    print(f"Prepared task: {manifest['task_id']}")
    print(f"Construction: {paths[0]} ({manifest['construction_rollouts']} rollouts)")
    print(f"Sealed test : {paths[1]} ({manifest['sealed_test_rollouts']} rollouts)")
    print(f"Manifest    : {paths[2]}")


if __name__ == "__main__":
    main()
