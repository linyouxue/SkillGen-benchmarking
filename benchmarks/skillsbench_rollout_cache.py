"""Fail-closed, content-addressed cache for paid SkillsBench rollout slots.

The cache is deliberately below SkillGen's algorithmic boundary.  It only
memoizes a completed official-verifier result for the *same pre-declared
slot* and execution treatment.  A corrupt or mismatched entry raises instead
of silently causing another paid request.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from models import SkillItem, TaskInstance, Trajectory


CACHE_REQUEST_SCHEMA = "skillsbench-rollout-request-v1"
CACHE_ENTRY_SCHEMA = "skillsbench-rollout-cache-entry-v1"
CACHE_DIRECTORY = ".skillgen-rollout-cache"
ATTEMPT_MANIFEST = "skillgen_cache_request.json"
ATTEMPT_RECEIPT = "skillgen_cache_process.json"
ATTEMPT_KEY_CHARS = 16


class SkillsBenchRolloutCacheError(RuntimeError):
    """A cache entry exists but cannot be trusted."""


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    raise TypeError(f"value is not canonically JSON serializable: {type(value).__name__}")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def skill_content_digest(skill: SkillItem | None) -> str:
    """Hash every persisted field of the exact SkillItem intervention."""

    if skill is None:
        return "none"
    return _sha256(asdict(skill))


@dataclass(frozen=True)
class RolloutCacheRequest:
    """Canonical identity of one pre-declared paid rollout slot."""

    payload: dict[str, Any]
    key: str

    @classmethod
    def build(
        cls,
        *,
        instance: TaskInstance,
        task_digest: str,
        model: str,
        agent: str,
        sandbox: str,
        skill: SkillItem | None,
        adapter_schema_version: str,
        benchflow_version: str,
    ) -> "RolloutCacheRequest":
        condition = "with-skill" if skill is not None else "no-skill"
        payload = {
            "schema": CACHE_REQUEST_SCHEMA,
            "instance_id": str(instance.instance_id),
            "task_digest": str(task_digest),
            "model": str(model),
            "agent": str(agent),
            "sandbox": str(sandbox),
            "condition": condition,
            "skill_id": str(skill.skill_id) if skill is not None else None,
            "skill_content_digest": skill_content_digest(skill),
            "adapter_schema_version": str(adapter_schema_version),
            "benchflow_version": str(benchflow_version),
        }
        return cls(payload=payload, key=_sha256(payload))


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
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
            temporary = handle.name
            json.dump(
                _jsonable(payload),
                handle,
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                Path(temporary).unlink()
            except FileNotFoundError:
                pass


def cache_entry_path(jobs_root: Path, request: RolloutCacheRequest) -> Path:
    hex_key = request.key.removeprefix("sha256:")
    return (
        jobs_root
        / CACHE_DIRECTORY
        / CACHE_ENTRY_SCHEMA
        / hex_key[:2]
        / f"{hex_key}.json"
    )


def attempt_directory_name(
    run_name: str, request: RolloutCacheRequest, attempt_id: str
) -> str:
    """Name an attempt so an orphaned manifest can be attributed fail-closed."""

    hex_key = request.key.removeprefix("sha256:")
    return (
        f"{run_name}--{request.payload['condition']}--"
        f"cache-{hex_key[:ATTEMPT_KEY_CHARS]}--{attempt_id}"
    )


@contextmanager
def slot_lock(jobs_root: Path, request: RolloutCacheRequest):
    """Prevent two processes from paying for the same slot concurrently.

    A process crash deliberately leaves a stale lock.  Recovery may still
    return a valid cache/artifact before entering this context, but an
    ambiguous interrupted attempt requires explicit inspection rather than an
    automatic duplicate charge.
    """

    hex_key = request.key.removeprefix("sha256:")
    lock_path = jobs_root / CACHE_DIRECTORY / "locks" / f"{hex_key}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as exc:
        raise SkillsBenchRolloutCacheError(
            "rollout slot is locked by another or interrupted process; "
            f"inspect before retrying: {lock_path}"
        ) from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                {"cache_key": request.key, "request": request.payload},
                handle,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        yield lock_path
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def write_attempt_manifest(run_root: Path, request: RolloutCacheRequest) -> Path:
    """Persist request identity before launching the paid subprocess."""

    path = run_root / ATTEMPT_MANIFEST
    _atomic_json(
        path,
        {
            "schema": CACHE_REQUEST_SCHEMA,
            "cache_key": request.key,
            "request": request.payload,
        },
    )
    return path


def write_attempt_receipt(
    run_root: Path, *, process_returncode: int, elapsed: float
) -> Path:
    """Persist subprocess completion before parsing verifier artifacts."""

    path = run_root / ATTEMPT_RECEIPT
    _atomic_json(
        path,
        {
            "process_returncode": int(process_returncode),
            "elapsed": float(elapsed),
        },
    )
    return path


def read_attempt_receipt(run_root: Path) -> tuple[int, float] | None:
    path = run_root / ATTEMPT_RECEIPT
    if not path.is_file():
        return None
    payload = _read_object(path)
    return int(payload["process_returncode"]), float(payload["elapsed"])


def manifested_attempts(
    jobs_root: Path, request: RolloutCacheRequest
) -> list[Path]:
    """Return newest-first prior attempts carrying this exact request."""

    matches: list[Path] = []
    if not jobs_root.is_dir():
        return matches
    hex_key = request.key.removeprefix("sha256:")
    condition = request.payload["condition"]
    pattern = (
        f"*--{condition}--cache-{hex_key[:ATTEMPT_KEY_CHARS]}--*"
    )
    for run_root in jobs_root.glob(pattern):
        manifest_path = run_root / ATTEMPT_MANIFEST
        if not manifest_path.is_file():
            raise SkillsBenchRolloutCacheError(
                "an attempt directory claims this cache key but has no request "
                f"manifest: {run_root}"
            )
        payload = _read_object(manifest_path)
        if (
            payload.get("cache_key") != request.key
            or payload.get("request") != request.payload
        ):
            raise SkillsBenchRolloutCacheError(
                "attempt directory key prefix collides with a different request: "
                f"{run_root}"
            )
        matches.append(run_root)
    return sorted(matches, key=lambda path: path.stat().st_mtime_ns, reverse=True)


def _trajectory_record(trajectory: Trajectory) -> dict[str, Any]:
    return _jsonable(asdict(trajectory))


def _trajectory_from_record(record: Mapping[str, Any]) -> Trajectory:
    required = {
        "trajectory_id",
        "instance_id",
        "agent_config",
        "messages",
        "final_output",
    }
    missing = sorted(required - set(record))
    if missing:
        raise SkillsBenchRolloutCacheError(
            "cached trajectory is missing fields: " + ", ".join(missing)
        )
    return Trajectory(
        trajectory_id=str(record["trajectory_id"]),
        instance_id=str(record["instance_id"]),
        agent_config=dict(record.get("agent_config") or {}),
        messages=list(record.get("messages") or []),
        final_output=record.get("final_output"),
        success=record.get("success"),
        score=record.get("score"),
        error_summary=record.get("error_summary"),
        token_usage=(
            dict(record["token_usage"])
            if isinstance(record.get("token_usage"), Mapping)
            else None
        ),
        latency=record.get("latency"),
        timestamp=record.get("timestamp"),
        metadata=dict(record.get("metadata") or {}),
    )


def _validate_trajectory(
    request: RolloutCacheRequest, trajectory: Trajectory
) -> None:
    payload = request.payload
    if str(trajectory.instance_id) != payload["instance_id"]:
        raise SkillsBenchRolloutCacheError("cached trajectory instance_id mismatch")

    score = trajectory.score
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise SkillsBenchRolloutCacheError(
            "cached trajectory lacks an official numeric reward"
        )
    score = float(score)
    if not 0.0 <= score <= 1.0:
        raise SkillsBenchRolloutCacheError("cached reward is outside [0, 1]")
    if trajectory.success is not (score == 1.0):
        raise SkillsBenchRolloutCacheError(
            "cached success flag disagrees with the official reward"
        )

    metadata = trajectory.metadata or {}
    if metadata.get("benchmark") != "skillsbench":
        raise SkillsBenchRolloutCacheError("cached trajectory is not SkillsBench")
    if metadata.get("adapter_schema_version") != payload["adapter_schema_version"]:
        raise SkillsBenchRolloutCacheError("cached adapter schema mismatch")
    if metadata.get("task_digest") != payload["task_digest"]:
        raise SkillsBenchRolloutCacheError("cached task digest mismatch")
    if metadata.get("skill_mode") != payload["condition"]:
        raise SkillsBenchRolloutCacheError("cached treatment mismatch")
    if metadata.get("benchflow_returncode") != 0:
        raise SkillsBenchRolloutCacheError("cached subprocess was not successful")
    checks = metadata.get("real_run_checks") or {}
    if checks.get("has_verifier_reward") is not True:
        raise SkillsBenchRolloutCacheError("cached official reward evidence is absent")
    if metadata.get("agent_exception") or metadata.get("verifier_error"):
        raise SkillsBenchRolloutCacheError("cached trajectory contains an error")

    config = trajectory.agent_config or {}
    observed_model = config.get("inference_model") or config.get("model")
    if observed_model != payload["model"]:
        raise SkillsBenchRolloutCacheError("cached model mismatch")
    if config.get("agent") != payload["agent"]:
        raise SkillsBenchRolloutCacheError("cached agent mismatch")
    if config.get("skill_mode") != payload["condition"]:
        raise SkillsBenchRolloutCacheError("cached agent treatment mismatch")
    expected_skill_id = payload["skill_id"]
    if config.get("skill_id") != expected_skill_id:
        raise SkillsBenchRolloutCacheError("cached skill_id mismatch")


def write_cached_trajectory(
    jobs_root: Path,
    request: RolloutCacheRequest,
    trajectory: Trajectory,
    *,
    source: Mapping[str, Any],
) -> Path:
    """Atomically publish one fully validated official rollout result."""

    path = cache_entry_path(jobs_root, request)
    existing = load_cached_trajectory(jobs_root, request)
    if existing is not None:
        if existing != trajectory:
            raise SkillsBenchRolloutCacheError(
                "a different valid trajectory is already cached for this slot"
            )
        return path
    _validate_trajectory(request, trajectory)
    record = _trajectory_record(trajectory)
    _atomic_json(
        path,
        {
            "schema": CACHE_ENTRY_SCHEMA,
            "cache_key": request.key,
            "request": request.payload,
            "trajectory_sha256": _sha256(record),
            "trajectory": record,
            "source": dict(source),
        },
    )
    return path


def _read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SkillsBenchRolloutCacheError(
            f"cannot parse rollout cache metadata {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise SkillsBenchRolloutCacheError(
            f"rollout cache metadata is not an object: {path}"
        )
    return payload


def load_cached_trajectory(
    jobs_root: Path, request: RolloutCacheRequest
) -> Trajectory | None:
    """Load the exact cached Trajectory, or return ``None`` for a clean miss."""

    path = cache_entry_path(jobs_root, request)
    if not path.is_file():
        return None
    payload = _read_object(path)
    if payload.get("schema") != CACHE_ENTRY_SCHEMA:
        raise SkillsBenchRolloutCacheError("rollout cache entry schema mismatch")
    if payload.get("cache_key") != request.key:
        raise SkillsBenchRolloutCacheError("rollout cache key mismatch")
    if payload.get("request") != request.payload:
        raise SkillsBenchRolloutCacheError("rollout cache request mismatch")
    record = payload.get("trajectory")
    if not isinstance(record, Mapping):
        raise SkillsBenchRolloutCacheError("rollout cache trajectory is malformed")
    if payload.get("trajectory_sha256") != _sha256(record):
        raise SkillsBenchRolloutCacheError("rollout cache trajectory digest mismatch")
    trajectory = _trajectory_from_record(record)
    _validate_trajectory(request, trajectory)
    return trajectory
