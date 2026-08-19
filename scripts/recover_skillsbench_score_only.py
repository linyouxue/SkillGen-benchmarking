#!/usr/bin/env python3
"""Reconstruct and score a completed SkillsBench rollout without an LLM call.

This is an audit/diagnostic utility. It replays only recorded file-editor changes
for a small, explicit script allowlist and then runs a fixed workbook build chain.
It never changes the source job, its result, or the rollout cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


SCRIPT_ALLOWLIST = {
    "copy_sut_sheets.py",
    "build_workbook.py",
    "add_calconload.py",
}
SUT_URL = "https://www.geostat.ge/media/79759/SUT-2024_eng.xlsx"
SUT_REFERER = (
    "https://www.geostat.ge/en/modules/categories/632/"
    "supply-and-use-tables-new"
)


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


def load_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            event = json.loads(line)
            event["_line_number"] = line_number
            events.append(event)
    return events


def parse_file_editor_call(event: dict[str, Any]) -> dict[str, Any] | None:
    title = event.get("title")
    prefix = "file_editor: "
    if event.get("type") != "tool_call" or not isinstance(title, str):
        return None
    if not title.startswith(prefix):
        return None
    value, suffix_offset = json.JSONDecoder().raw_decode(title[len(prefix) :])
    suffix = title[len(prefix) + suffix_offset :].strip()
    if suffix and not suffix.startswith(":"):
        raise ValueError(f"unexpected file_editor title suffix: {suffix[:80]!r}")
    if not isinstance(value, dict):
        raise ValueError("file_editor payload is not an object")
    return value


def extract_recorded_scripts(
    events: list[dict[str, Any]], output_dir: Path
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    contents: dict[str, str] = {}
    operations: list[dict[str, Any]] = []

    for event_index, event in enumerate(events):
        payload = parse_file_editor_call(event)
        if payload is None:
            continue
        raw_path = payload.get("path")
        if not isinstance(raw_path, str):
            continue
        basename = PurePosixPath(raw_path).name
        if basename not in SCRIPT_ALLOWLIST:
            continue

        command = payload.get("command")
        if command == "create":
            text = payload.get("file_text")
            if not isinstance(text, str):
                raise ValueError(f"missing file_text for {raw_path}")
            contents[basename] = text
        elif command == "str_replace":
            if basename not in contents:
                raise ValueError(f"str_replace precedes create for {raw_path}")
            old = payload.get("old_str")
            new = payload.get("new_str")
            if not isinstance(old, str) or not isinstance(new, str):
                raise ValueError(f"invalid str_replace payload for {raw_path}")
            count = contents[basename].count(old)
            if count != 1:
                raise ValueError(
                    f"expected one replacement target in {basename}, found {count}"
                )
            contents[basename] = contents[basename].replace(old, new, 1)
        elif command == "view":
            continue
        else:
            raise ValueError(f"unsupported recorded editor command: {command!r}")

        operations.append(
            {
                "event_index": event_index,
                "trajectory_line_number": event["_line_number"],
                "tool_call_id": event.get("tool_call_id"),
                "command": command,
                "recorded_path": raw_path,
            }
        )

    missing = SCRIPT_ALLOWLIST.difference(contents)
    if missing:
        raise ValueError(f"trajectory is missing scripts: {sorted(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    evidence: dict[str, dict[str, Any]] = {}
    for basename in sorted(contents):
        text = contents[basename]
        if any(marker in text.lower() for marker in ("deepseek", "openrouter", "api_key")):
            raise ValueError(f"model/API marker found in extracted {basename}")
        path = output_dir / basename
        path.write_text(text, encoding="utf-8")
        evidence[basename] = file_evidence(path)
    return evidence, operations


def download_sut(destination: Path) -> dict[str, Any]:
    request = urllib.request.Request(
        SUT_URL,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; SkillsBenchRecovery/1.0)",
            "Referer": SUT_REFERER,
        },
    )
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(request, timeout=120) as response:
        with temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        metadata = {
            "requested_url": SUT_URL,
            "resolved_url": response.geturl(),
            "http_status": response.status,
            "content_type": response.headers.get("Content-Type"),
            "content_length_header": response.headers.get("Content-Length"),
            "last_modified": response.headers.get("Last-Modified"),
            "etag": response.headers.get("ETag"),
        }
    if temporary.stat().st_size < 10_000 or temporary.read_bytes()[:2] != b"PK":
        raise ValueError("downloaded SUT is not a plausible XLSX file")
    temporary.replace(destination)
    metadata.update(file_evidence(destination))
    metadata["provenance_limitation"] = (
        "The original paid container did not persist the downloaded SUT hash; "
        "this newly downloaded version cannot be proven byte-identical."
    )
    return metadata


def run_logged(command: list[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
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
    output = subprocess.check_output(
        ["docker", "image", "inspect", image], text=True, encoding="utf-8"
    )
    value = json.loads(output)[0]
    return {
        "requested_reference": image,
        "id": value.get("Id"),
        "repo_digests": value.get("RepoDigests") or [],
        "created": value.get("Created"),
    }


def run_reconstruction_and_verifier(
    recovery_dir: Path,
    verifier_dir: Path,
    image: str,
) -> None:
    container_script = r"""
set -euo pipefail
cd /root
cp /recovery/inputs/test_demand.original.xlsx /root/test_demand.xlsx
cp /recovery/inputs/SUT-2024_eng.xlsx /root/SUT-2024_eng.xlsx
cp /recovery/extracted_scripts/*.py /root/
python3 /root/copy_sut_sheets.py
python3 /root/build_workbook.py
rm -rf /root/recalc
mkdir -p /root/recalc
timeout 180 soffice --headless --convert-to xlsx --outdir /root/recalc /root/test_demand.xlsx
test -s /root/recalc/test_demand.xlsx
cp /root/recalc/test_demand.xlsx /root/test_demand.xlsx
python3 /root/add_calconload.py
test -s /root/test_demand.xlsx
cp /root/test_demand.xlsx /recovery/workspace/test_demand.xlsx
bash /verifier/test.sh
""".strip()
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        "skillsbench-shock-demand-score-only-recovery",
        "--mount",
        f"type=bind,src={recovery_dir.resolve()},dst=/recovery",
        "--mount",
        f"type=bind,src={verifier_dir.resolve()},dst=/verifier,readonly",
        "--mount",
        (
            "type=bind,src="
            f"{(recovery_dir / 'verifier').resolve()},dst=/logs/verifier"
        ),
        "--env",
        "HOME=/root",
        image,
        "bash",
        "-lc",
        container_script,
    ]
    run_logged(command, recovery_dir / "docker_score_only.log")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--original-result", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    trajectory = args.trajectory.resolve()
    original_result = args.original_result.resolve()
    task_dir = args.task_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite recovery directory: {output_dir}")
    if output_dir == original_result.parent or output_dir in original_result.parents:
        raise ValueError("recovery output must be a sibling, not the original job")

    template = task_dir / "environment" / "test_demand.xlsx"
    dockerfile = task_dir / "environment" / "Dockerfile"
    task_md = task_dir / "task.md"
    verifier_dir = task_dir / "verifier"
    verifier_test = verifier_dir / "test.sh"
    verifier_code = verifier_dir / "test_outputs.py"
    required = [
        trajectory,
        original_result,
        template,
        dockerfile,
        task_md,
        verifier_test,
        verifier_code,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required files missing: {missing}")

    (output_dir / "inputs").mkdir(parents=True)
    (output_dir / "extracted_scripts").mkdir()
    (output_dir / "workspace").mkdir()
    (output_dir / "verifier").mkdir()

    try:
        shutil.copy2(template, output_dir / "inputs" / "test_demand.original.xlsx")
        events = load_events(trajectory)
        script_evidence, editor_operations = extract_recorded_scripts(
            events, output_dir / "extracted_scripts"
        )
        sut_evidence = download_sut(output_dir / "inputs" / "SUT-2024_eng.xlsx")
        image_evidence = inspect_image(args.image)
        run_reconstruction_and_verifier(output_dir, verifier_dir, args.image)

        output_workbook = output_dir / "workspace" / "test_demand.xlsx"
        reward_path = output_dir / "verifier" / "reward.txt"
        if not output_workbook.is_file() or not reward_path.is_file():
            raise RuntimeError("reconstruction/verifier did not emit required artifacts")
        reward_text = reward_path.read_text(encoding="utf-8").strip()
        if reward_text not in {"0", "1"}:
            raise ValueError(f"invalid official reward: {reward_text!r}")

        verifier_artifacts = {
            path.name: file_evidence(path)
            for path in sorted((output_dir / "verifier").iterdir())
            if path.is_file()
        }
        manifest = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "task_id": "shock-analysis-demand",
            "classification": "diagnostic_score_only_recovery",
            "diagnostic_only": True,
            "eligible_for_formal_rollout_cache": False,
            "no_llm_or_model_api_calls": True,
            "original_job_or_result_modified": False,
            "rollout_cache_modified": False,
            "formal_use_limitation": (
                "The original rollout did not preserve the downloaded SUT hash. "
                "The score is diagnostic and must not be imported as the formal slot."
            ),
            "sources": {
                "trajectory": file_evidence(trajectory),
                "original_result": file_evidence(original_result),
                "task_markdown": file_evidence(task_md),
                "task_dockerfile": file_evidence(dockerfile),
                "original_template": file_evidence(template),
                "official_verifier_test_sh": file_evidence(verifier_test),
                "official_verifier_test_outputs": file_evidence(verifier_code),
                "redownloaded_sut": sut_evidence,
            },
            "trajectory": {
                "event_count": len(events),
                "tool_call_count": sum(
                    event.get("type") == "tool_call" for event in events
                ),
                "editor_operations_replayed": editor_operations,
            },
            "extracted_scripts": script_evidence,
            "runtime_image": image_evidence,
            "fixed_replay_chain": [
                "copy_sut_sheets.py",
                "build_workbook.py",
                "LibreOffice headless XLSX recalculation",
                "add_calconload.py",
                "official verifier/test.sh",
            ],
            "output_workbook": file_evidence(output_workbook),
            "official_verifier": {
                "reward": int(reward_text),
                "artifacts": verifier_artifacts,
            },
            "docker_log": file_evidence(output_dir / "docker_score_only.log"),
        }
        write_json(output_dir / "recovery_manifest.json", manifest)
        print(
            json.dumps(
                {
                    "reward": int(reward_text),
                    "output": str(output_workbook),
                    "output_sha256": manifest["output_workbook"]["sha256"],
                    "manifest": str(output_dir / "recovery_manifest.json"),
                    "diagnostic_only": True,
                },
                ensure_ascii=False,
            )
        )
        return 0
    except Exception as exc:
        failure = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "classification": "diagnostic_score_only_recovery_failure",
            "no_llm_or_model_api_calls": True,
            "original_job_or_result_modified": False,
            "rollout_cache_modified": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        write_json(output_dir / "recovery_failure.json", failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
