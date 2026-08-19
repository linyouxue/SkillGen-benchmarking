"""Offline preflight for a prepared SkillGen x SkillsBench task."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.skillsbench_adapter import (  # noqa: E402
    EXPECTED_BENCHFLOW_VERSION,
    resolve_jobs_root,
    resolve_task_dir,
    task_package_digest,
)


def _command_check(command: list[str]) -> dict:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Check SkillsBench runtime without API calls")
    parser.add_argument("dataset", type=Path, help="Prepared construction/test JSON")
    args = parser.parse_args()
    payload = json.loads(args.dataset.read_text(encoding="utf-8"))
    instances = payload.get("instances") or []
    if not instances:
        raise ValueError("Prepared dataset contains no instances")
    metadata = instances[0].get("metadata") or {}
    invariant_keys = (
        "skillsbench_task_id",
        "skillsbench_task_digest",
        "skillsbench_source_version",
        "skillsbench_agent",
        "skillsbench_sandbox",
        "skillsbench_bench_executable",
    )
    inconsistent = [
        key
        for key in invariant_keys
        if any((item.get("metadata") or {}).get(key) != metadata.get(key) for item in instances)
    ]
    if inconsistent:
        raise ValueError(
            "Prepared dataset mixes incompatible runtime metadata: "
            + ", ".join(inconsistent)
        )
    task_dir = resolve_task_dir(metadata)
    jobs_root = resolve_jobs_root(metadata)
    expected_digest = metadata.get("skillsbench_task_digest")
    actual_digest = task_package_digest(task_dir)
    bench_name = str(metadata.get("skillsbench_bench_executable") or "bench")
    bench_path = (
        str(Path(bench_name).resolve()) if Path(bench_name).is_file() else shutil.which(bench_name)
    )
    docker_path = shutil.which("docker")

    checks = {
        "dataset": str(args.dataset.resolve()),
        "n_instances": len(instances),
        "unique_instance_ids": len({item["instance_id"] for item in instances}),
        "resolved_task_dir": str(task_dir),
        "resolved_jobs_root": str(jobs_root),
        "task_digest": {
            "ok": expected_digest == actual_digest,
            "expected": expected_digest,
            "actual": actual_digest,
        },
        "benchflow": {
            "ok": False,
            "expected_version": EXPECTED_BENCHFLOW_VERSION,
            "executable": bench_path,
        },
        "docker": {"ok": False, "executable": docker_path},
        "api_calls_made": False,
    }
    if bench_path:
        result = _command_check([bench_path, "--version"])
        version_text = result.get("stdout", "") + "\n" + result.get("stderr", "")
        result["ok"] = result["ok"] and EXPECTED_BENCHFLOW_VERSION in version_text
        checks["benchflow"].update(result)
    if docker_path:
        checks["docker"].update(_command_check([docker_path, "info"]))

    checks["ok"] = all(
        (
            checks["task_digest"]["ok"],
            checks["benchflow"]["ok"],
            checks["docker"]["ok"],
            checks["n_instances"] == checks["unique_instance_ids"],
        )
    )
    # ASCII escaping keeps diagnostic output valid on Windows consoles whose
    # active code page cannot encode replacement characters from subprocess
    # stderr. Paths remain losslessly represented in JSON.
    print(json.dumps(checks, ensure_ascii=True, indent=2))
    raise SystemExit(0 if checks["ok"] else 1)


if __name__ == "__main__":
    main()
