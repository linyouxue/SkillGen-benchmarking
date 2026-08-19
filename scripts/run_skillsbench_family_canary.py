"""Run exactly one frozen no-skill D_ind SkillsBench rollout.

This is an infrastructure canary, not a shortened SkillGen experiment.  It
only populates (or reads) the content-addressed rollout cache used by the full
family runner.  Dry-run is the default; ``--execute`` is the sole path that
may start BenchFlow or contact a provider.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterator, Mapping
from zoneinfo import ZoneInfo

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.skillsbench_adapter import (  # noqa: E402
    ADAPTER_SCHEMA_VERSION,
    EXPECTED_BENCHFLOW_VERSION,
    _rollout_cache_request,
    resolve_jobs_root,
    resolve_task_dir,
    run_skillsbench_agent,
    task_package_digest,
)
from benchmarks.skillsbench_rollout_cache import (  # noqa: E402
    RolloutCacheRequest,
    cache_entry_path,
    load_cached_trajectory,
)
from models import TaskInstance, Trajectory  # noqa: E402
from trajectory import AgentConfig  # noqa: E402


CANARY_PROTOCOL_VERSION = "skillsbench-family-single-slot-canary-v1"
RECEIPT_SCHEMA_VERSION = 1
EXPECTED_MODEL = "deepseek-v4-flash"
EXPECTED_AGENT_MODEL = "deepseek/deepseek-v4-flash"
EXPECTED_AGENT = "openhands"
EXPECTED_SANDBOX = "docker"
EXPECTED_PROVIDER = "deepseek_official"
EXPECTED_BASE_URL = "https://api.deepseek.com"
_SHANGHAI = ZoneInfo("Asia/Shanghai")
_SECRET_PATTERN = re.compile(r"(?i)(?:bearer\s+)?sk-[A-Za-z0-9_-]{8,}")


@dataclass(frozen=True)
class CanarySpec:
    induction_dataset: Path
    manifest_path: Path
    config_path: Path
    instance_id: str
    run_root: Path
    budget_cny: str = "120"


@dataclass(frozen=True)
class PreparedCanary:
    spec: CanarySpec
    instance: TaskInstance
    family_id: str
    task_id: str
    task_digest: str
    task_dir: Path
    jobs_root: Path
    model: str
    judge_model: str
    agent: str
    sandbox: str
    request: RolloutCacheRequest
    protocol: dict[str, Any]
    protocol_hash: str

    @property
    def fold_root(self) -> Path:
        return self.spec.run_root.resolve() / self.family_id

    @property
    def receipt_path(self) -> Path:
        return self.fold_root / "canary" / "receipt.json"

    @property
    def budget_ledger_path(self) -> Path:
        return self.fold_root / "budget_ledger.json"


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is not a file: {resolved}")
    with resolved.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {resolved}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
                dict(payload),
                handle,
                indent=2,
                sort_keys=True,
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


def _normalise_path(value: object) -> Path:
    return Path(str(value)).expanduser().resolve()


def _validate_config(config: Mapping[str, Any]) -> tuple[str, str]:
    models = config.get("models") or {}
    model = str(models.get("baseline_agent") or "")
    judge_model = str(models.get("baseline_judge") or "")
    if model != EXPECTED_AGENT_MODEL:
        raise ValueError(
            f"config models.baseline_agent must be {EXPECTED_AGENT_MODEL!r}"
        )
    if judge_model != EXPECTED_MODEL:
        raise ValueError(
            f"config models.baseline_judge must be {EXPECTED_MODEL!r}"
        )
    experiment = config.get("experiment") or {}
    if experiment.get("provider") != EXPECTED_PROVIDER:
        raise ValueError(
            f"config experiment.provider must be {EXPECTED_PROVIDER!r}"
        )
    if experiment.get("base_url") != EXPECTED_BASE_URL:
        raise ValueError(
            f"config experiment.base_url must be {EXPECTED_BASE_URL!r}"
        )
    if float((config.get("llm") or {}).get("temperature", 0.0)) != 0.0:
        raise ValueError("config llm.temperature must be 0.0 for the frozen canary")
    return model, judge_model


def _validate_budget(value: object) -> str:
    try:
        amount = Decimal(str(value).strip())
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("budget_cny must be a finite positive amount") from exc
    if not amount.is_finite() or amount <= 0:
        raise ValueError("budget_cny must be a finite positive amount")
    return str(amount)


def _validate_dataset_and_select(
    dataset: Mapping[str, Any],
    manifest: Mapping[str, Any],
    instance_id: str,
) -> tuple[dict[str, Any], str, str]:
    family_id = str(manifest.get("family_id") or "")
    if not family_id:
        raise ValueError("manifest is missing family_id")
    metadata = dataset.get("metadata") or {}
    if metadata.get("benchmark") != "skillsbench":
        raise ValueError("induction dataset is not SkillsBench")
    if metadata.get("family_id") != family_id:
        raise ValueError("induction dataset family_id differs from manifest")
    if metadata.get("split") != "induction":
        raise ValueError("canary dataset must be the frozen induction split")
    if metadata.get("protocol") != manifest.get("protocol"):
        raise ValueError("induction dataset protocol differs from manifest")

    rows = dataset.get("instances") or []
    if not isinstance(rows, list) or not rows:
        raise ValueError("induction dataset has no instances")
    ids = [str(row.get("instance_id") or "") for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("induction dataset has blank or duplicate instance IDs")

    expected_count = int((manifest.get("counts") or {}).get("induction", -1))
    if len(rows) != expected_count:
        raise ValueError("induction dataset count differs from manifest")

    source_tasks = [str(item) for item in manifest.get("source_task_ids") or []]
    if not source_tasks:
        raise ValueError("manifest has no D_ind source tasks")
    heldout_task = str(manifest.get("heldout_task_id") or "")
    allocations = {
        str(key): int(value)
        for key, value in (manifest.get("induction_allocations") or {}).items()
    }
    observed: dict[str, int] = {}
    selected: dict[str, Any] | None = None
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("induction dataset contains a non-object instance")
        row_metadata = row.get("metadata") or {}
        task_id = str(row_metadata.get("skillsbench_task_id") or "")
        observed[task_id] = observed.get(task_id, 0) + 1
        if str(row.get("instance_id")) == instance_id:
            selected = row
    if observed != allocations:
        raise ValueError("D_ind task allocation differs from manifest")
    if selected is None:
        raise ValueError("requested instance_id is not in the frozen D_ind dataset")

    selected_metadata = selected.get("metadata") or {}
    task_id = str(selected_metadata.get("skillsbench_task_id") or "")
    if task_id not in source_tasks or task_id == heldout_task:
        raise ValueError("requested instance is not a D_ind source-task slot")
    expected_prefix = f"{family_id}::{task_id}::induction::"
    if not instance_id.startswith(expected_prefix):
        raise ValueError("requested instance ID is not a canonical D_ind slot ID")
    if selected_metadata.get("benchmark") != "skillsbench":
        raise ValueError("requested instance is not SkillsBench")
    if selected_metadata.get("skillsbench_family_id") != family_id:
        raise ValueError("requested instance family differs from manifest")
    if selected_metadata.get("skillsbench_family_split") != "induction":
        raise ValueError("requested instance is not marked as induction")
    if selected_metadata.get("official_task_skills_visible") is not False:
        raise ValueError("official task skills are not frozen hidden")
    if selected_metadata.get("skillsbench_adapter_schema") != ADAPTER_SCHEMA_VERSION:
        raise ValueError("requested instance adapter schema differs from runtime")

    expected_digests = manifest.get("task_package_digests") or {}
    task_digest = str(selected_metadata.get("skillsbench_task_digest") or "")
    if not task_digest or task_digest != expected_digests.get(task_id):
        raise ValueError("requested instance task digest differs from manifest")
    return selected, family_id, task_id


def prepare_canary(spec: CanarySpec) -> PreparedCanary:
    budget_cny = _validate_budget(spec.budget_cny)
    dataset = _read_object(spec.induction_dataset, label="induction dataset")
    manifest = _read_object(spec.manifest_path, label="family manifest")
    with spec.config_path.resolve().open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("config must contain a YAML mapping")
    model, judge_model = _validate_config(config)
    row, family_id, task_id = _validate_dataset_and_select(
        dataset, manifest, spec.instance_id
    )
    metadata = dict(row.get("metadata") or {})
    task_dir = resolve_task_dir(metadata)
    task_digest = str(metadata["skillsbench_task_digest"])
    actual_digest = task_package_digest(task_dir)
    if actual_digest != task_digest:
        raise ValueError("requested task package digest has drifted")

    agent = str(metadata.get("skillsbench_agent") or "")
    sandbox = str(metadata.get("skillsbench_sandbox") or "")
    if agent != EXPECTED_AGENT or agent != manifest.get("agent"):
        raise ValueError("requested slot does not use the frozen OpenHands agent")
    if sandbox != EXPECTED_SANDBOX or sandbox != manifest.get("sandbox"):
        raise ValueError("requested slot does not use the frozen Docker sandbox")
    if str(metadata.get("skillsbench_bench_executable") or "") != str(
        manifest.get("bench_executable") or ""
    ):
        raise ValueError("requested slot BenchFlow executable differs from manifest")
    jobs_root = resolve_jobs_root(metadata)
    if jobs_root != _normalise_path(manifest.get("jobs_root")):
        raise ValueError("requested slot jobs root differs from manifest")

    instance = TaskInstance(
        instance_id=str(row["instance_id"]),
        input=row.get("input"),
        ground_truth=row.get("ground_truth"),
        metadata=metadata,
    )
    request = _rollout_cache_request(
        instance=instance,
        task_dir=task_dir,
        model=model,
        agent=agent,
        sandbox=sandbox,
        skill_bundle=None,
    )
    if request.payload.get("condition") != "no-skill":
        raise RuntimeError("canary request unexpectedly contains a skill")
    if request.payload.get("skill_id") is not None:
        raise RuntimeError("canary request unexpectedly has a skill_id")

    protocol = {
        "protocol_version": CANARY_PROTOCOL_VERSION,
        "family_id": family_id,
        "instance_id": instance.instance_id,
        "task_id": task_id,
        "task_digest": task_digest,
        "induction_sha256": _sha256_file(spec.induction_dataset),
        "manifest_sha256": _sha256_file(spec.manifest_path),
        "config_sha256": _sha256_file(spec.config_path),
        "model": model,
        "judge_model": judge_model,
        "agent": agent,
        "sandbox": sandbox,
        "condition": "no-skill",
        "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
        "benchflow_version": EXPECTED_BENCHFLOW_VERSION,
        "cache_key": request.key,
        "budget_cny": budget_cny,
        "agent_rollout_reserve_cny": "30",
        "code_sha256": {
            "canary_runner": _sha256_file(Path(__file__)),
            "skillsbench_adapter": _sha256_file(
                REPO_ROOT / "benchmarks" / "skillsbench_adapter.py"
            ),
            "rollout_cache": _sha256_file(
                REPO_ROOT / "benchmarks" / "skillsbench_rollout_cache.py"
            ),
            "budget_guard": _sha256_file(REPO_ROOT / "pilot_budget_guard.py"),
        },
    }
    return PreparedCanary(
        spec=spec,
        instance=instance,
        family_id=family_id,
        task_id=task_id,
        task_digest=task_digest,
        task_dir=task_dir,
        jobs_root=jobs_root,
        model=model,
        judge_model=judge_model,
        agent=agent,
        sandbox=sandbox,
        request=request,
        protocol=protocol,
        protocol_hash=_sha256_json(protocol),
    )


def _validate_official_trajectory(
    prepared: PreparedCanary, trajectory: Trajectory
) -> float:
    score = trajectory.score
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise RuntimeError("canary result lacks an official numeric reward")
    reward = float(score)
    if not math.isfinite(reward) or reward not in {0.0, 1.0}:
        raise RuntimeError("canary official reward must be numeric 0 or 1")
    if trajectory.success is not (reward == 1.0):
        raise RuntimeError("canary success flag disagrees with official reward")
    if str(trajectory.instance_id) != prepared.instance.instance_id:
        raise RuntimeError("canary result instance_id differs from frozen slot")

    metadata = trajectory.metadata or {}
    checks = metadata.get("real_run_checks") or {}
    if metadata.get("benchmark") != "skillsbench":
        raise RuntimeError("canary result is not SkillsBench")
    if metadata.get("adapter_schema_version") != ADAPTER_SCHEMA_VERSION:
        raise RuntimeError("canary result adapter schema differs from runtime")
    if metadata.get("task_digest") != prepared.task_digest:
        raise RuntimeError("canary result task digest differs from frozen task")
    if metadata.get("skill_mode") != "no-skill":
        raise RuntimeError("canary result is not no-skill")
    if metadata.get("benchflow_returncode") != 0:
        raise RuntimeError("canary BenchFlow subprocess did not return zero")
    if checks.get("has_verifier_reward") is not True:
        raise RuntimeError("canary result lacks official verifier evidence")
    if metadata.get("agent_exception") or metadata.get("verifier_error"):
        raise RuntimeError("canary result contains an agent or verifier error")

    agent_config = trajectory.agent_config or {}
    observed_model = agent_config.get("inference_model") or agent_config.get("model")
    if observed_model != prepared.model:
        raise RuntimeError("canary result model differs from frozen model")
    if agent_config.get("agent") != prepared.agent:
        raise RuntimeError("canary result agent differs from frozen agent")
    if agent_config.get("skill_mode") != "no-skill":
        raise RuntimeError("canary result agent config is not no-skill")
    if agent_config.get("skill_id") is not None:
        raise RuntimeError("canary result unexpectedly contains a skill_id")
    return reward


def _cache_source(prepared: PreparedCanary) -> str:
    path = cache_entry_path(prepared.jobs_root, prepared.request)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("canary result was not durably published to cache") from exc
    source = payload.get("source") or {}
    kind = str(source.get("kind") or "")
    if not kind:
        raise RuntimeError("canary cache entry has no auditable source kind")
    return kind


def _redact_error(exc: BaseException) -> str:
    text = str(exc)
    for name in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
        secret = os.environ.get(name, "")
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return _SECRET_PATTERN.sub("[REDACTED]", text)[:2000]


@contextmanager
def _canary_environment(prepared: PreparedCanary) -> Iterator[None]:
    updates = {
        "SKILLGEN_CHAT_PROVIDER": "deepseek",
        "DEEPSEEK_BASE_URL": EXPECTED_BASE_URL,
        "SKILLGEN_DEEPSEEK_BUDGET_CNY": str(prepared.protocol["budget_cny"]),
        "SKILLGEN_META_REQUEST_RESERVE_CNY": "5",
        "SKILLGEN_AGENT_ROLLOUT_RESERVE_CNY": "30",
        "SKILLGEN_BUDGET_LEDGER": str(prepared.budget_ledger_path),
    }
    old = {key: os.environ.get(key) for key in updates}
    conflict = os.environ.get("SKILLGEN_CHAT_PROVIDER")
    if conflict not in (None, "", "deepseek"):
        raise RuntimeError("SKILLGEN_CHAT_PROVIDER conflicts with frozen DeepSeek route")
    try:
        os.environ.update(updates)
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _assert_announced_offpeak(now: datetime | None = None) -> None:
    local = now.astimezone(_SHANGHAI) if now is not None else datetime.now(_SHANGHAI)
    minutes = local.hour * 60 + local.minute
    if 9 * 60 <= minutes < 12 * 60 or 14 * 60 <= minutes < 18 * 60:
        raise RuntimeError(
            "DeepSeek peak window is active; canary is allowed only "
            "12:00-14:00 or 18:00-09:00 Asia/Shanghai"
        )


def _base_record(
    prepared: PreparedCanary, *, mode: str, cache_hit: bool
) -> dict[str, Any]:
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "mode": mode,
        "protocol_hash": prepared.protocol_hash,
        "protocol": prepared.protocol,
        "family_id": prepared.family_id,
        "instance_id": prepared.instance.instance_id,
        "task_id": prepared.task_id,
        "condition": "no-skill",
        "cache_key": prepared.request.key,
        "cache_hit_before_execution": cache_hit,
        "receipt_path": str(prepared.receipt_path),
        "budget_ledger": str(prepared.budget_ledger_path),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }


def _load_existing_receipt(prepared: PreparedCanary) -> dict[str, Any] | None:
    path = prepared.receipt_path
    if not path.is_file():
        return None
    payload = _read_object(path, label="canary receipt")
    if payload.get("protocol_hash") != prepared.protocol_hash:
        raise RuntimeError("existing canary receipt has a different protocol hash")
    return payload


def run_canary(
    spec: CanarySpec,
    *,
    execute: bool = False,
    allow_paid_retry: bool = False,
) -> dict[str, Any]:
    prepared = prepare_canary(spec)
    cached = load_cached_trajectory(prepared.jobs_root, prepared.request)
    cache_hit = cached is not None
    plan = {
        **_base_record(prepared, mode="execute" if execute else "dry-run", cache_hit=cache_hit),
        "infrastructure_success": None,
        "official_reward": None,
        "task_success": None,
        "agent_runner_invocations": 0,
        "new_benchflow_attempts": 0,
        "api_calls_made": False,
    }
    if not execute:
        return plan

    existing = _load_existing_receipt(prepared)
    if existing is not None:
        if existing.get("infrastructure_success") is True:
            if not cache_hit:
                raise RuntimeError(
                    "successful canary receipt exists but its central cache entry is missing"
                )
            return existing
        if not cache_hit and not allow_paid_retry:
            raise RuntimeError(
                "a failed canary receipt already exists; inspect it and pass "
                "--allow-paid-retry only after explicit review"
            )

    runner_invoked = False
    try:
        if cache_hit:
            trajectory = cached
        else:
            if not os.environ.get("DEEPSEEK_API_KEY", "").strip():
                raise RuntimeError("DEEPSEEK_API_KEY is required for a cache miss")
            _assert_announced_offpeak()
            config = AgentConfig(
                model=prepared.model,
                judge_model=prepared.judge_model,
                temperature=0.0,
            )
            with _canary_environment(prepared):
                runner_invoked = True
                trajectory = run_skillsbench_agent(prepared.instance, None, config)
        if trajectory is None:
            raise RuntimeError("canary did not return a trajectory")
        reward = _validate_official_trajectory(prepared, trajectory)
        source_kind = _cache_source(prepared)
        new_benchflow_attempts = int(not cache_hit and source_kind == "benchflow")
        if new_benchflow_attempts not in {0, 1}:
            raise RuntimeError("canary attempted more than one new BenchFlow rollout")
        receipt = {
            **plan,
            "infrastructure_success": True,
            "official_reward": reward,
            "task_success": reward == 1.0,
            "trajectory_id": str(trajectory.trajectory_id),
            "cache_source_kind": source_kind,
            "agent_runner_invocations": int(runner_invoked),
            "new_benchflow_attempts": new_benchflow_attempts,
            "api_calls_made": bool(new_benchflow_attempts),
        }
        _atomic_json(prepared.receipt_path, receipt)
        return receipt
    except Exception as exc:
        failure = {
            **plan,
            "infrastructure_success": False,
            "agent_runner_invocations": int(runner_invoked),
            "new_benchflow_attempts": None if runner_invoked else 0,
            "api_calls_made": None if runner_invoked else False,
            "error": {
                "kind": type(exc).__name__,
                "reason": _redact_error(exc),
            },
        }
        if runner_invoked:
            _atomic_json(prepared.receipt_path, failure)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dry-plan or run exactly one frozen no-skill D_ind canary slot."
    )
    parser.add_argument("--induction-dataset", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--budget-cny", default="120")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--allow-paid-retry",
        action="store_true",
        help="Permit retry only after a failed receipt was explicitly reviewed.",
    )
    args = parser.parse_args(argv)
    if args.allow_paid_retry and not args.execute:
        parser.error("--allow-paid-retry requires --execute")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    spec = CanarySpec(
        induction_dataset=args.induction_dataset,
        manifest_path=args.manifest,
        config_path=args.config,
        instance_id=args.instance_id,
        run_root=args.run_root,
        budget_cny=args.budget_cny,
    )
    result = run_canary(
        spec,
        execute=args.execute,
        allow_paid_retry=args.allow_paid_retry,
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
