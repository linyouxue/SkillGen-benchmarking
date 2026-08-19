"""Prepare every SkillsBench task as an independent SkillGen dataset pair."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_skillsbench import build_payloads  # noqa: E402


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def discover_task_dirs(tasks_root: Path, include: list[str] | None = None) -> list[Path]:
    root = tasks_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"SkillsBench tasks root not found: {root}")
    available = {
        child.name: child
        for child in root.iterdir()
        if child.is_dir() and (child / "task.md").is_file()
    }
    if include:
        missing = sorted(set(include) - set(available))
        if missing:
            raise ValueError("Unknown SkillsBench task ids: " + ", ".join(missing))
        names = sorted(set(include))
    else:
        names = sorted(available)
    if not names:
        raise ValueError(f"No native task.md packages found under {root}")
    return [available[name] for name in names]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze independent SkillGen rollout slots for a SkillsBench suite."
    )
    parser.add_argument("tasks_root", type=Path, help="Pinned SkillsBench v1.1 tasks/")
    parser.add_argument("--construction-rollouts", type=int, required=True)
    parser.add_argument("--test-rollouts", type=int, required=True)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--sandbox", default="docker")
    parser.add_argument("--bench-executable", default="bench")
    parser.add_argument("--source-version", default="v1.1")
    parser.add_argument("--jobs-root", type=Path, default=Path("artifacts/skillsbench/benchflow_jobs"))
    parser.add_argument("--output-root", type=Path, default=Path("data/skillsbench"))
    parser.add_argument("--subprocess-timeout-sec", type=float, default=7200)
    parser.add_argument(
        "--include",
        action="append",
        default=None,
        help="Prepare only this task id; repeatable. Omit to freeze every task.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    task_dirs = discover_task_dirs(args.tasks_root, args.include)
    prepared: list[tuple[Path, dict, dict, dict]] = []
    for task_dir in task_dirs:
        construction, sealed_test, manifest = build_payloads(
            task_dir=task_dir,
            construction_rollouts=args.construction_rollouts,
            test_rollouts=args.test_rollouts,
            agent=args.agent,
            sandbox=args.sandbox,
            jobs_root=args.jobs_root,
            bench_executable=args.bench_executable,
            subprocess_timeout_sec=args.subprocess_timeout_sec,
            source_version=args.source_version,
        )
        prepared.append((task_dir, construction, sealed_test, manifest))

    suite_manifest = {
        "schema_version": "skillsbench-skillgen-suite-v1",
        "source_version": args.source_version,
        "tasks_root": str(args.tasks_root.expanduser().resolve()),
        "num_tasks": len(prepared),
        "task_ids": [item[3]["task_id"] for item in prepared],
        "construction_rollouts_per_task": args.construction_rollouts,
        "sealed_test_rollouts_per_task": args.test_rollouts,
        "agent": args.agent,
        "sandbox": args.sandbox,
        "task_digests": {item[3]["task_id"]: item[3]["task_digest"] for item in prepared},
        "task_source_commits": {
            item[3]["task_id"]: item[3]["source_commit"] for item in prepared
        },
        "task_processing": "independent",
        "cross_task_pooling": False,
        "adaptive_sampling_until_both_classes": False,
    }
    if args.dry_run:
        print(json.dumps(suite_manifest, ensure_ascii=False, indent=2))
        return

    paths: list[Path] = [args.output_root / "suite_manifest.json"]
    for _, _, _, manifest in prepared:
        task_output = args.output_root / manifest["task_id"]
        paths.extend(
            [
                task_output / "construction.json",
                task_output / "sealed_test.json",
                task_output / "protocol_manifest.json",
            ]
        )
    existing = [str(path) for path in paths if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "Refusing to overwrite frozen suite files; pass --force: "
            + ", ".join(existing[:10])
            + (f" ... and {len(existing) - 10} more" if len(existing) > 10 else "")
        )

    for _, construction, sealed_test, manifest in prepared:
        task_output = args.output_root / manifest["task_id"]
        _atomic_json(task_output / "construction.json", construction)
        _atomic_json(task_output / "sealed_test.json", sealed_test)
        _atomic_json(task_output / "protocol_manifest.json", manifest)
    _atomic_json(args.output_root / "suite_manifest.json", suite_manifest)
    print(
        f"Prepared {len(prepared)} independent tasks under {args.output_root}; "
        f"suite manifest: {args.output_root / 'suite_manifest.json'}"
    )


if __name__ == "__main__":
    main()
