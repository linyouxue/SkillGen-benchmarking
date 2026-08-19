"""Prepare one task-disjoint SkillsBench family fold for SkillGen.

This is a data/protocol adapter only.  It does not inspect official task
skills, execute an agent, or call a model API.
"""

from __future__ import annotations

import argparse
import hashlib
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


PROTOCOL_NAME = "skillgen_skillsbench_task_disjoint_family_fold_v1"


def _parse_allocations(values: list[str], *, label: str) -> dict[str, int]:
    allocations: dict[str, int] = {}
    for raw in values:
        task_id, separator, count_text = raw.rpartition(":")
        task_id = task_id.strip()
        if not separator or not task_id:
            raise ValueError(f"{label} allocation must be TASK_ID:COUNT: {raw!r}")
        try:
            count = int(count_text)
        except ValueError as exc:
            raise ValueError(f"invalid {label} count in {raw!r}") from exc
        if count <= 0:
            raise ValueError(f"{label} count must be positive: {raw!r}")
        if task_id in allocations:
            raise ValueError(f"duplicate {label} task: {task_id}")
        allocations[task_id] = count
    if not allocations:
        raise ValueError(f"at least one {label} allocation is required")
    return allocations


def _git_metadata(tasks_root: Path) -> dict[str, Any]:
    repo = next(
        (
            candidate
            for candidate in (tasks_root, *tasks_root.parents)
            if (candidate / ".git").exists()
        ),
        None,
    )
    if repo is None:
        return {"repository_root": None, "commit": None, "dirty": None}

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
        return {
            "repository_root": str(repo.resolve()),
            "commit": run("rev-parse", "HEAD"),
            "dirty": bool(run("status", "--porcelain", "--", str(tasks_root))),
        }
    except (OSError, subprocess.SubprocessError):
        return {"repository_root": str(repo.resolve()), "commit": None, "dirty": None}


def _task_relpath(task_dir: Path, repository_root: str | None) -> str | None:
    if not repository_root:
        return None
    try:
        return task_dir.relative_to(Path(repository_root)).as_posix()
    except ValueError:
        return None


def _selection_hash(*, seed: str, family_number: int, task_id: str) -> str:
    raw = f"{seed}|{family_number}|{task_id}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _task_info(
    *,
    task_id: str,
    tasks_root: Path,
    git_meta: dict[str, Any],
) -> dict[str, Any]:
    task_dir = validate_task_package(tasks_root / task_id)
    frontmatter, prompt = parse_task_markdown(task_dir)
    return {
        "task_id": task_id,
        "task_dir": task_dir,
        "task_relpath": _task_relpath(task_dir, git_meta.get("repository_root")),
        "task_digest": task_package_digest(task_dir),
        "prompt": prompt,
        "category_metadata": frontmatter.get("metadata") or {},
    }


def _instances(
    *,
    split: str,
    allocations: dict[str, int],
    task_infos: dict[str, dict[str, Any]],
    family_id: str,
    heldout_task_id: str,
    source_version: str,
    source_commit: str | None,
    agent: str,
    sandbox: str,
    jobs_root: Path,
    bench_executable: str,
    subprocess_timeout_sec: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_id, count in allocations.items():
        info = task_infos[task_id]
        for index in range(count):
            rows.append(
                {
                    "instance_id": f"{family_id}::{task_id}::{split}::r{index:03d}",
                    "input": info["prompt"],
                    "ground_truth": None,
                    "metadata": {
                        "benchmark": "skillsbench",
                        "skillsbench_adapter_schema": ADAPTER_SCHEMA_VERSION,
                        "skillsbench_family_id": family_id,
                        "skillsbench_family_split": split,
                        "skillsbench_heldout_task_id": heldout_task_id,
                        "skillsbench_task_id": task_id,
                        "skillsbench_task_dir": str(info["task_dir"]),
                        "skillsbench_task_relpath": info["task_relpath"],
                        "skillsbench_task_digest": info["task_digest"],
                        "skillsbench_source_version": source_version,
                        "skillsbench_source_commit": source_commit,
                        "skillsbench_agent": agent,
                        "skillsbench_sandbox": sandbox,
                        "skillsbench_jobs_root": str(jobs_root.resolve()),
                        "skillsbench_bench_executable": bench_executable,
                        "skillsbench_subprocess_timeout_sec": subprocess_timeout_sec,
                        "skillsbench_rollout_index": index,
                        "rollout_replica_kind": "stochastic_same_task",
                        "seed_control": "unsupported_by_benchflow_cli",
                        "official_task_skills_visible": False,
                    },
                }
            )
    return rows


def build_payloads(
    *,
    tasks_root: Path,
    family_id: str,
    family_number: int,
    family_label: str,
    coherence: str,
    induction_allocations: dict[str, int],
    verification_allocations: dict[str, int],
    heldout_task_id: str,
    heldout_rollouts: int,
    selection_seed: str,
    agent: str,
    sandbox: str,
    jobs_root: Path,
    bench_executable: str,
    subprocess_timeout_sec: float,
    source_version: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if set(induction_allocations) != set(verification_allocations):
        raise ValueError("induction and verification must contain the same source tasks")
    if heldout_task_id in induction_allocations:
        raise ValueError("heldout task must not appear in either source allocation")
    if sum(induction_allocations.values()) != 10:
        raise ValueError("induction allocations must sum to exactly 10")
    if sum(verification_allocations.values()) != 10:
        raise ValueError("verification allocations must sum to exactly 10")
    if heldout_rollouts != 10:
        raise ValueError("this frozen pilot requires exactly 10 heldout rollout slots")
    if subprocess_timeout_sec <= 0:
        raise ValueError("subprocess timeout must be positive")

    tasks_root = tasks_root.resolve()
    git_meta = _git_metadata(tasks_root)
    all_task_ids = [*induction_allocations.keys(), heldout_task_id]
    task_infos = {
        task_id: _task_info(task_id=task_id, tasks_root=tasks_root, git_meta=git_meta)
        for task_id in all_task_ids
    }
    selection = {
        task_id: _selection_hash(
            seed=selection_seed,
            family_number=family_number,
            task_id=task_id,
        )
        for task_id in all_task_ids
    }
    selected_by_hash = min(selection, key=selection.get)
    if selected_by_hash != heldout_task_id:
        raise ValueError(
            "heldout task does not match the frozen hash rule: "
            f"expected {selected_by_hash}, got {heldout_task_id}"
        )

    common_dataset_metadata = {
        "benchmark": "skillsbench",
        "protocol": PROTOCOL_NAME,
        "adapter_schema": ADAPTER_SCHEMA_VERSION,
        "family_id": family_id,
        "family_number": family_number,
        "family_label": family_label,
        "coherence": coherence,
        "source_task_ids": list(induction_allocations),
        "heldout_task_id": heldout_task_id,
        "source_version": source_version,
        "source_commit": git_meta.get("commit"),
        "cross_task_pooling": True,
        "task_disjoint_heldout": True,
        "manual_examples": False,
        "refinement_example_extension": False,
        "official_task_skills_used_for_construction": False,
        "seed_control": "unsupported_by_benchflow_cli",
    }
    induction_instances = _instances(
        split="induction",
        allocations=induction_allocations,
        task_infos=task_infos,
        family_id=family_id,
        heldout_task_id=heldout_task_id,
        source_version=source_version,
        source_commit=git_meta.get("commit"),
        agent=agent,
        sandbox=sandbox,
        jobs_root=jobs_root,
        bench_executable=bench_executable,
        subprocess_timeout_sec=subprocess_timeout_sec,
    )
    verification_instances = _instances(
        split="verification",
        allocations=verification_allocations,
        task_infos=task_infos,
        family_id=family_id,
        heldout_task_id=heldout_task_id,
        source_version=source_version,
        source_commit=git_meta.get("commit"),
        agent=agent,
        sandbox=sandbox,
        jobs_root=jobs_root,
        bench_executable=bench_executable,
        subprocess_timeout_sec=subprocess_timeout_sec,
    )
    heldout_instances = _instances(
        split="heldout",
        allocations={heldout_task_id: heldout_rollouts},
        task_infos=task_infos,
        family_id=family_id,
        heldout_task_id=heldout_task_id,
        source_version=source_version,
        source_commit=git_meta.get("commit"),
        agent=agent,
        sandbox=sandbox,
        jobs_root=jobs_root,
        bench_executable=bench_executable,
        subprocess_timeout_sec=subprocess_timeout_sec,
    )

    def dataset(split: str, instances: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "dataset_id": f"skillsbench-{source_version}::{family_id}::{split}",
            "task_name": family_label,
            "task_type": "scored",
            "metadata": {**common_dataset_metadata, "split": split},
            "instances": instances,
        }

    induction = dataset("induction", induction_instances)
    verification = dataset("verification", verification_instances)
    heldout = dataset("heldout", heldout_instances)
    manifest = {
        "schema_version": 1,
        "protocol": PROTOCOL_NAME,
        "family_id": family_id,
        "family_number": family_number,
        "family_label": family_label,
        "coherence": coherence,
        "selection_seed": selection_seed,
        "selection_hash_rule": "sha256(seed|family_number|task_id), minimum wins",
        "selection_hashes": selection,
        "heldout_task_id": heldout_task_id,
        "source_task_ids": list(induction_allocations),
        "induction_allocations": induction_allocations,
        "verification_allocations": verification_allocations,
        "heldout_rollouts": heldout_rollouts,
        "task_package_digests": {
            task_id: info["task_digest"] for task_id, info in task_infos.items()
        },
        "source_version": source_version,
        "source_commit": git_meta.get("commit"),
        "source_dirty": git_meta.get("dirty"),
        "agent": agent,
        "sandbox": sandbox,
        "bench_executable": bench_executable,
        "jobs_root": str(jobs_root.resolve()),
        "counts": {"induction": 10, "verification": 10, "heldout": 10},
        "candidate_rounds": 8,
        "gate": {
            "selection": "maximum verification net_gain; earliest round breaks ties",
            "repair_minus_regression_min": 2,
            "heldout_skill_is_conditional_on_gate": True,
        },
        "isolation": {
            "heldout_absent_from_induction": True,
            "heldout_absent_from_verification": True,
            "official_skills_hidden": True,
            "adaptive_sampling": False,
        },
        "api_calls_made": False,
    }
    return induction, verification, heldout, manifest


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_path, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare one task-disjoint SkillsBench family fold (no API calls)."
    )
    parser.add_argument("tasks_root", type=Path)
    parser.add_argument("--family-id", required=True)
    parser.add_argument("--family-number", required=True, type=int)
    parser.add_argument("--family-label", required=True)
    parser.add_argument("--coherence", choices=("A", "B", "C"), required=True)
    parser.add_argument("--induction", action="append", default=[], metavar="TASK:COUNT")
    parser.add_argument("--verification", action="append", default=[], metavar="TASK:COUNT")
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--heldout-rollouts", type=int, default=10)
    parser.add_argument("--selection-seed", default="42")
    parser.add_argument("--agent", default="openhands")
    parser.add_argument("--sandbox", default="docker")
    parser.add_argument("--bench-executable", default="bench")
    parser.add_argument("--source-version", default="v1.1")
    parser.add_argument(
        "--jobs-root",
        type=Path,
        default=Path("artifacts/skillsbench-family/benchflow-jobs"),
    )
    parser.add_argument("--subprocess-timeout-sec", type=float, default=7200)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payloads = build_payloads(
        tasks_root=args.tasks_root,
        family_id=args.family_id,
        family_number=args.family_number,
        family_label=args.family_label,
        coherence=args.coherence,
        induction_allocations=_parse_allocations(args.induction, label="induction"),
        verification_allocations=_parse_allocations(
            args.verification, label="verification"
        ),
        heldout_task_id=args.heldout,
        heldout_rollouts=args.heldout_rollouts,
        selection_seed=args.selection_seed,
        agent=args.agent,
        sandbox=args.sandbox,
        jobs_root=args.jobs_root,
        bench_executable=args.bench_executable,
        subprocess_timeout_sec=args.subprocess_timeout_sec,
        source_version=args.source_version,
    )
    induction, verification, heldout, manifest = payloads
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    paths = [
        args.output_dir / "induction.json",
        args.output_dir / "verification.json",
        args.output_dir / "heldout.json",
        args.output_dir / "protocol_manifest.json",
    ]
    existing = [str(path) for path in paths if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "Refusing to overwrite frozen protocol files; pass --force: "
            + ", ".join(existing)
        )
    for path, payload in zip(paths, payloads, strict=True):
        _atomic_json(path, payload)
    print(json.dumps({"prepared": [str(path) for path in paths], **manifest}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
