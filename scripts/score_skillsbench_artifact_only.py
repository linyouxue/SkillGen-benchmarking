#!/usr/bin/env python3
"""Run an unchanged SkillsBench verifier against an immutable saved artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_evidence(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def tree_evidence(root: Path) -> dict[str, Any]:
    files = {
        path.relative_to(root).as_posix(): {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    canonical = json.dumps(
        files,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return {
        "path": str(root.resolve()),
        "file_count": len(files),
        "files": files,
        "snapshot_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def run_logged(command: list[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def inspect_image(image: str) -> dict[str, Any]:
    raw = subprocess.check_output(
        ["docker", "image", "inspect", image],
        text=True,
        encoding="utf-8",
    )
    value = json.loads(raw)[0]
    return {
        "requested_reference": image,
        "id": value.get("Id"),
        "repo_digests": value.get("RepoDigests") or [],
        "created": value.get("Created"),
    }


def parse_pytest_log(text: str) -> dict[str, Any]:
    collected_match = re.search(r"collected (\d+) items", text)
    passed = re.findall(r"^PASSED (\S+)", text, flags=re.MULTILINE)
    failed = re.findall(r"^FAILED (\S+)(?:\s+-.*)?$", text, flags=re.MULTILINE)
    messages = [
        match.strip()
        for match in re.findall(
            r"^E\s+AssertionError:\s*(.+)$", text, flags=re.MULTILINE
        )
    ]
    failure_details = []
    for index, node_id in enumerate(failed):
        failure_details.append(
            {
                "node_id": node_id,
                "assertion": messages[index] if index < len(messages) else None,
            }
        )
    return {
        "collected": int(collected_match.group(1)) if collected_match else None,
        "passed_node_ids": passed,
        "failed_node_ids": failed,
        "failure_details": failure_details,
    }


def snapshot_files(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    return {str(path.resolve()): file_evidence(path) for path in paths}


def finalize_existing_evidence(args: argparse.Namespace) -> int:
    artifact = args.artifact.resolve()
    task_dir = args.task_dir.resolve()
    output_dir = args.output_dir.resolve()
    verifier_dir = task_dir / "verifier"
    protected_files = [path.resolve() for path in args.protected_file]
    protected_trees = [path.resolve() for path in args.protected_tree]
    baseline_path = args.integrity_baseline_manifest.resolve()
    manifest_path = output_dir / "evidence_manifest.json"
    failure_path = output_dir / "evidence_failure.json"
    log_path = output_dir / "official_verifier.log"
    copied_artifact = output_dir / "input" / "test_demand.xlsx"
    reward_path = output_dir / "verifier" / "reward.txt"

    if not output_dir.is_dir():
        raise FileNotFoundError(f"existing evidence directory is missing: {output_dir}")
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {manifest_path}")
    required = [
        artifact,
        copied_artifact,
        failure_path,
        log_path,
        reward_path,
        output_dir / "verifier" / "ctrf.json",
        verifier_dir / "test.sh",
        verifier_dir / "test_outputs.py",
        task_dir / "environment" / "Dockerfile",
        task_dir / "task.md",
        baseline_path,
        *protected_files,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    missing.extend(str(path) for path in protected_trees if not path.is_dir())
    if missing:
        raise FileNotFoundError(f"required finalization evidence is missing: {missing}")

    artifact_evidence = file_evidence(artifact)
    expected_hash = args.expected_sha256.lower().removeprefix("sha256:")
    if (
        artifact_evidence["sha256"] != expected_hash
        or artifact_evidence["size_bytes"] != args.expected_size
    ):
        raise ValueError("source artifact no longer matches frozen identity")
    copied_evidence = file_evidence(copied_artifact)
    if (
        copied_evidence["sha256"] != expected_hash
        or copied_evidence["size_bytes"] != args.expected_size
    ):
        raise ValueError("scored input copy does not match the frozen source artifact")

    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    if (
        failure.get("error_type") != "RuntimeError"
        or not str(failure.get("error", "")).startswith(
            "official verifier infrastructure did not complete pytest:"
        )
        or failure.get("source_artifact_modified") is not False
    ):
        raise ValueError("existing attempt did not fail solely at the known parser check")

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    pytest_summary = parse_pytest_log(log_text)
    completed_tests = len(pytest_summary["passed_node_ids"]) + len(
        pytest_summary["failed_node_ids"]
    )
    if (
        pytest_summary["collected"] is None
        or completed_tests != pytest_summary["collected"]
    ):
        raise RuntimeError(
            f"pytest log is still incomplete: {pytest_summary}, completed={completed_tests}"
        )
    reward_text = reward_path.read_text(encoding="utf-8").strip()
    if reward_text not in {"0", "1"}:
        raise ValueError(f"invalid official reward: {reward_text!r}")

    current_files = snapshot_files(protected_files)
    current_trees = {str(root): tree_evidence(root) for root in protected_trees}
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_files = baseline["protected_files_before_and_after"]["after"]
    baseline_trees = baseline["protected_cache_trees_before_and_after"]["after"]
    if baseline_files != current_files or baseline_trees != current_trees:
        raise RuntimeError("protected v4 state differs from the prior integrity snapshot")

    verifier_artifacts = {
        path.name: file_evidence(path)
        for path in sorted((output_dir / "verifier").iterdir())
        if path.is_file()
    }
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": "shock-analysis-demand",
        "classification": "timeout_diagnostic_verifier_only",
        "diagnostic_only": True,
        "eligible_for_formal_scored_slot": False,
        "eligible_for_formal_rollout_cache": False,
        "no_llm_or_model_api_calls": True,
        "source_artifact_modified": False,
        "v4_result_or_status_modified": False,
        "rollout_cache_modified": False,
        "formal_use_limitation": (
            "The agent timed out before the normal scored-rollout boundary. "
            "This post-hoc verifier result is diagnostic and cannot directly "
            "replace the formal slot."
        ),
        "offline_finalization": {
            "reason": (
                "The unchanged official verifier completed, but the first log "
                "parser required every FAILED summary line to contain a dash."
            ),
            "original_failure": file_evidence(failure_path),
            "integrity_baseline_manifest": file_evidence(baseline_path),
            "no_verifier_rerun_during_finalization": True,
        },
        "source_artifact": artifact_evidence,
        "copied_input": copied_evidence,
        "official_task_sources": {
            "task_markdown": file_evidence(task_dir / "task.md"),
            "task_dockerfile": file_evidence(task_dir / "environment" / "Dockerfile"),
            "verifier_test_sh": file_evidence(verifier_dir / "test.sh"),
            "verifier_test_outputs": file_evidence(verifier_dir / "test_outputs.py"),
        },
        "official_verifier_unmodified": True,
        "official_verifier_mount_read_only": True,
        "runtime_image": inspect_image(args.image),
        "protected_files_before_and_after": {
            "before": baseline_files,
            "after": current_files,
            "identical": True,
        },
        "protected_cache_trees_before_and_after": {
            "before": baseline_trees,
            "after": current_trees,
            "identical": True,
        },
        "official_verifier": {
            "reward": int(reward_text),
            "pytest": pytest_summary,
            "artifacts": verifier_artifacts,
        },
        "verifier_log": file_evidence(log_path),
    }
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "reward": int(reward_text),
                "pytest": pytest_summary,
                "manifest": str(manifest_path),
                "diagnostic_only": True,
                "offline_finalization": True,
            },
            ensure_ascii=False,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-size", type=int, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--protected-file", type=Path, action="append", default=[])
    parser.add_argument("--protected-tree", type=Path, action="append", default=[])
    parser.add_argument("--finalize-existing", action="store_true")
    parser.add_argument("--integrity-baseline-manifest", type=Path)
    args = parser.parse_args()

    if args.finalize_existing:
        if args.integrity_baseline_manifest is None:
            parser.error("--finalize-existing requires --integrity-baseline-manifest")
        return finalize_existing_evidence(args)

    artifact = args.artifact.resolve()
    task_dir = args.task_dir.resolve()
    output_dir = args.output_dir.resolve()
    verifier_dir = task_dir / "verifier"
    verifier_test = verifier_dir / "test.sh"
    verifier_code = verifier_dir / "test_outputs.py"
    dockerfile = task_dir / "environment" / "Dockerfile"
    task_md = task_dir / "task.md"
    protected_files = [path.resolve() for path in args.protected_file]
    protected_trees = [path.resolve() for path in args.protected_tree]

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite evidence directory: {output_dir}")
    required_files = [
        artifact,
        verifier_test,
        verifier_code,
        dockerfile,
        task_md,
        *protected_files,
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    missing.extend(str(path) for path in protected_trees if not path.is_dir())
    if missing:
        raise FileNotFoundError(f"required evidence is missing: {missing}")

    artifact_evidence = file_evidence(artifact)
    expected_hash = args.expected_sha256.lower().removeprefix("sha256:")
    if artifact_evidence["sha256"] != expected_hash:
        raise ValueError(
            "artifact SHA-256 mismatch: "
            f"expected {expected_hash}, got {artifact_evidence['sha256']}"
        )
    if artifact_evidence["size_bytes"] != args.expected_size:
        raise ValueError(
            "artifact size mismatch: "
            f"expected {args.expected_size}, got {artifact_evidence['size_bytes']}"
        )

    before_files = snapshot_files(protected_files)
    before_trees = {str(root): tree_evidence(root) for root in protected_trees}
    output_dir.mkdir(parents=True)
    (output_dir / "input").mkdir()
    (output_dir / "verifier").mkdir()
    copied_artifact = output_dir / "input" / "test_demand.xlsx"
    shutil.copy2(artifact, copied_artifact)

    try:
        image_evidence = inspect_image(args.image)
        container_script = "\n".join(
            [
                "set -euo pipefail",
                "cp /evidence/input/test_demand.xlsx /root/test_demand.xlsx",
                "bash /verifier/test.sh",
            ]
        )
        command = [
            "docker",
            "run",
            "--rm",
            "--name",
            "skillsbench-shock-demand-timeout-verifier-only",
            "--mount",
            f"type=bind,src={output_dir},dst=/evidence",
            "--mount",
            f"type=bind,src={verifier_dir},dst=/verifier,readonly",
            "--mount",
            f"type=bind,src={output_dir / 'verifier'},dst=/logs/verifier",
            "--env",
            "HOME=/root",
            args.image,
            "bash",
            "-lc",
            container_script,
        ]
        log_path = output_dir / "official_verifier.log"
        run_logged(command, log_path)

        reward_path = output_dir / "verifier" / "reward.txt"
        if not reward_path.is_file():
            raise RuntimeError("official verifier did not emit reward.txt")
        reward_text = reward_path.read_text(encoding="utf-8").strip()
        if reward_text not in {"0", "1"}:
            raise ValueError(f"invalid official reward: {reward_text!r}")

        after_files = snapshot_files(protected_files)
        after_trees = {str(root): tree_evidence(root) for root in protected_trees}
        original_after = file_evidence(artifact)
        if artifact_evidence != original_after:
            raise RuntimeError("source artifact changed during verifier-only scoring")
        if before_files != after_files:
            raise RuntimeError("a protected result/status file changed")
        if before_trees != after_trees:
            raise RuntimeError("the protected rollout cache tree changed")

        verifier_artifacts = {
            path.name: file_evidence(path)
            for path in sorted((output_dir / "verifier").iterdir())
            if path.is_file()
        }
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        pytest_summary = parse_pytest_log(log_text)
        completed_tests = len(pytest_summary["passed_node_ids"]) + len(
            pytest_summary["failed_node_ids"]
        )
        if (
            "ctrf.json" not in verifier_artifacts
            or pytest_summary["collected"] is None
            or completed_tests != pytest_summary["collected"]
        ):
            raise RuntimeError(
                "official verifier infrastructure did not complete pytest: "
                f"collected={pytest_summary['collected']}, completed={completed_tests}, "
                f"ctrf_present={'ctrf.json' in verifier_artifacts}"
            )
        if int(reward_text) == 1 and pytest_summary["failed_node_ids"]:
            raise RuntimeError("reward=1 conflicts with failed official pytest nodes")
        manifest = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "task_id": "shock-analysis-demand",
            "classification": "timeout_diagnostic_verifier_only",
            "diagnostic_only": True,
            "eligible_for_formal_scored_slot": False,
            "eligible_for_formal_rollout_cache": False,
            "no_llm_or_model_api_calls": True,
            "source_artifact_modified": False,
            "v4_result_or_status_modified": False,
            "rollout_cache_modified": False,
            "formal_use_limitation": (
                "The agent timed out before the normal scored-rollout boundary. "
                "This post-hoc verifier result is diagnostic and cannot directly "
                "replace the formal slot."
            ),
            "source_artifact": artifact_evidence,
            "copied_input": file_evidence(copied_artifact),
            "official_task_sources": {
                "task_markdown": file_evidence(task_md),
                "task_dockerfile": file_evidence(dockerfile),
                "verifier_test_sh": file_evidence(verifier_test),
                "verifier_test_outputs": file_evidence(verifier_code),
            },
            "official_verifier_unmodified": True,
            "official_verifier_mount_read_only": True,
            "runtime_image": image_evidence,
            "protected_files_before_and_after": {
                "before": before_files,
                "after": after_files,
                "identical": True,
            },
            "protected_cache_trees_before_and_after": {
                "before": before_trees,
                "after": after_trees,
                "identical": True,
            },
            "official_verifier": {
                "reward": int(reward_text),
                "pytest": pytest_summary,
                "artifacts": verifier_artifacts,
            },
            "verifier_log": file_evidence(log_path),
        }
        write_json(output_dir / "evidence_manifest.json", manifest)
        print(
            json.dumps(
                {
                    "reward": int(reward_text),
                    "pytest": pytest_summary,
                    "manifest": str(output_dir / "evidence_manifest.json"),
                    "diagnostic_only": True,
                },
                ensure_ascii=False,
            )
        )
        return 0
    except Exception as exc:
        write_json(
            output_dir / "evidence_failure.json",
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "classification": "timeout_diagnostic_verifier_only_failure",
                "no_llm_or_model_api_calls": True,
                "source_artifact_modified": file_evidence(artifact)
                != artifact_evidence,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
