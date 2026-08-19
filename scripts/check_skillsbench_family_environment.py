"""Offline preflight for a prepared SkillsBench family fold."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.skillsbench_adapter import (  # noqa: E402
    EXPECTED_BENCHFLOW_VERSION,
    resolve_jobs_root,
    resolve_task_dir,
    task_package_digest,
)
from scripts.run_skillsbench_family import FamilySpec, validate_protocol  # noqa: E402


def _command(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=45,
            check=False,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "command": command, "error": str(exc)}
    return {
        "ok": completed.returncode == 0,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip()[-1000:],
        "stderr": completed.stderr.strip()[-1000:],
    }


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check a family fold without model/API calls."
    )
    parser.add_argument("--family-id", required=True)
    parser.add_argument("--induction-dataset", required=True, type=Path)
    parser.add_argument("--verification-dataset", required=True, type=Path)
    parser.add_argument("--heldout-dataset", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--budget-cny", default="120")
    parser.add_argument("--require-keys", action="store_true")
    args = parser.parse_args(argv)

    spec = FamilySpec(
        family_id=args.family_id,
        induction_dataset=args.induction_dataset,
        verification_dataset=args.verification_dataset,
        heldout_dataset=args.heldout_dataset,
        manifest_path=args.manifest,
        config_path=args.config,
        run_root=args.run_root,
        budget_cny=args.budget_cny,
    )
    protocol, manifest = validate_protocol(spec)
    task_records: dict[str, dict[str, Any]] = {}
    bench_names: set[str] = set()
    jobs_roots: set[str] = set()
    for dataset_path in (
        spec.induction_dataset,
        spec.verification_dataset,
        spec.heldout_dataset,
    ):
        payload = _read(dataset_path)
        for row in payload["instances"]:
            metadata = row.get("metadata") or {}
            task_id = str(metadata["skillsbench_task_id"])
            task_dir = resolve_task_dir(metadata)
            expected = str(metadata["skillsbench_task_digest"])
            actual = task_package_digest(task_dir)
            if expected != actual:
                raise ValueError(f"task package digest drifted for {task_id}")
            task_records[task_id] = {
                "task_dir": str(task_dir),
                "digest": actual,
            }
            bench_names.add(str(metadata.get("skillsbench_bench_executable") or "bench"))
            jobs_roots.add(str(resolve_jobs_root(metadata)))

    if len(bench_names) != 1 or len(jobs_roots) != 1:
        raise ValueError("prepared instances disagree on bench executable or jobs root")
    bench_name = next(iter(bench_names))
    bench_path = str(Path(bench_name).resolve()) if Path(bench_name).is_file() else shutil.which(bench_name)
    docker_path = shutil.which("docker")
    bench_check = _command([bench_path, "--version"]) if bench_path else {"ok": False}
    version_text = f"{bench_check.get('stdout', '')}\n{bench_check.get('stderr', '')}"
    bench_check["ok"] = bool(
        bench_check.get("ok") and EXPECTED_BENCHFLOW_VERSION in version_text
    )
    docker_check = _command([docker_path, "info"]) if docker_path else {"ok": False}
    keys = {
        "DEEPSEEK_API_KEY": bool(os.environ.get("DEEPSEEK_API_KEY", "").strip()),
        "OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY", "").strip()),
    }
    ok = bool(
        protocol
        and len(task_records) == 5
        and bench_check.get("ok")
        and docker_check.get("ok")
        and (all(keys.values()) if args.require_keys else True)
    )
    report = {
        "ok": ok,
        "protocol_hash": protocol["hash"],
        "family_id": args.family_id,
        "heldout_task_id": manifest["heldout_task_id"],
        "tasks": task_records,
        "resolved_jobs_roots": sorted(jobs_roots),
        "benchflow": bench_check,
        "docker": docker_check,
        "required_key_names_present": keys,
        "keys_required_for_this_check": args.require_keys,
        "api_calls_made": False,
    }
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
