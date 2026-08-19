"""Run one frozen task-disjoint SkillsBench family fold through SkillGen.

The command is dry-run by default.  ``--execute`` is the only paid path.
The held-out dataset is not loaded until SkillGen construction and gate
selection are complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pilot_budget_guard  # noqa: E402


PROTOCOL_VERSION = "skillsbench-family-heldout-pilot-v2"
EXPECTED_MODEL = "deepseek-v4-flash"
EXPECTED_AGENT_MODEL = "deepseek/deepseek-v4-flash"
FROZEN_MAX_WORKERS = 3
FROZEN_BALANCE_STOP_CNY = "2"
V12_PRE_BALANCE_STOP_PROTOCOL_HASH = (
    "b1f37cc09f8b48cd54be332441902d1f00f4ff3796c64ee11f359682db2ad86b"
)
V12_PRE_BALANCE_STOP_CONFIG_SHA256 = (
    "c5a651901016edbe1d6fdc3d568d0b0ba39199740a3dfb0738a4a23fd729fa60"
)
_OLD_BUDGET_GUARD_SEMANTICS = (
    "official-balance persistent active reservations with cross-process "
    "locking; provider hard limit requires account available balance <= "
    "approved cap and auto-recharge disabled"
)
_BUDGET_GUARD_SEMANTICS = (
    _OLD_BUDGET_GUARD_SEMANTICS
    + "; deny new paid units at or below the frozen 2 CNY balance floor"
)
_BUDGET_POLICY_AMENDMENT_FLAG = "--authorize-budget-policy-amendment"


@dataclass(frozen=True)
class FamilySpec:
    family_id: str
    induction_dataset: Path
    verification_dataset: Path
    heldout_dataset: Path
    manifest_path: Path
    config_path: Path
    run_root: Path
    budget_cny: str = "120"
    resume: bool = False
    authorize_budget_policy_amendment: bool = False
    expected_new_protocol_hash: str | None = None

    @property
    def fold_root(self) -> Path:
        return self.run_root / self.family_id


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_runtime_source_tree() -> str:
    """Hash every Python module that can affect this frozen pilot."""

    files = list(REPO_ROOT.glob("*.py"))
    for directory in ("agents", "benchmarks", "prompts"):
        files.extend((REPO_ROOT / directory).rglob("*.py"))
    files.append(Path(__file__).resolve())
    digest = hashlib.sha256()
    for path in sorted(set(files), key=lambda item: item.relative_to(REPO_ROOT).as_posix()):
        relative = path.relative_to(REPO_ROOT).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
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
            temp_name = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        temp_name = None
    finally:
        if temp_name is not None:
            try:
                Path(temp_name).unlink()
            except FileNotFoundError:
                pass


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        temp_name = None
    finally:
        if temp_name is not None:
            try:
                Path(temp_name).unlink()
            except FileNotFoundError:
                pass


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is not a file: {resolved}")
    with resolved.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {resolved}")
    return payload


def _dataset_ids(payload: dict[str, Any]) -> list[str]:
    return [str(row["instance_id"]) for row in (payload.get("instances") or [])]


def _dataset_task_ids(payload: dict[str, Any]) -> set[str]:
    return {
        str((row.get("metadata") or {}).get("skillsbench_task_id") or "")
        for row in (payload.get("instances") or [])
    } - {""}


def _validate_config(config: dict[str, Any]) -> None:
    models = config.get("models") or {}
    agent_keys = ("baseline_agent", "verification_agent")
    for key in agent_keys:
        if models.get(key) != EXPECTED_AGENT_MODEL:
            raise ValueError(
                f"config models.{key} must be {EXPECTED_AGENT_MODEL!r}, "
                f"got {models.get(key)!r}"
            )
    meta_keys = (
        "default",
        "baseline_judge",
        "induction",
        "induction_contextual",
        "induction_summary",
        "induction_pattern",
        "induction_contrastive",
        "generation_plan",
        "generation_execute",
        "refinement",
        "verification_judge",
        "verification_case_analyst",
        "verification_revision_synthesiser",
    )
    for key in meta_keys:
        if models.get(key) != EXPECTED_MODEL:
            raise ValueError(
                f"config models.{key} must be {EXPECTED_MODEL!r}, "
                f"got {models.get(key)!r}"
            )
    verification = config.get("verification") or {}
    if int(verification.get("min_net_gain_abs", -1)) != 2:
        raise ValueError("pilot gate requires verification.min_net_gain_abs=2")
    if float(verification.get("min_net_gain_rel", -1)) != 0.0:
        raise ValueError("pilot gate requires verification.min_net_gain_rel=0.0")
    pipeline = config.get("pipeline") or {}
    if int(pipeline.get("max_refine_rounds", -1)) != 8:
        raise ValueError("pilot requires pipeline.max_refine_rounds=8")
    if int(pipeline.get("max_workers", -1)) != FROZEN_MAX_WORKERS:
        raise ValueError(
            f"SkillsBench pilot requires max_workers={FROZEN_MAX_WORKERS}"
        )
    verification_analysis = config.get("verification_analysis") or {}
    if (
        int(verification_analysis.get("case_analyst_workers", -1))
        != FROZEN_MAX_WORKERS
    ):
        raise ValueError(
            "SkillsBench pilot requires "
            f"verification_analysis.case_analyst_workers={FROZEN_MAX_WORKERS}"
        )
    generation = config.get("generation") or {}
    if generation.get("use_web_search") is not False:
        raise ValueError("pilot requires generation.use_web_search=false")
    if generation.get("generate_scripts") is not False:
        raise ValueError("pilot requires generation.generate_scripts=false")
    if (config.get("router") or {}).get("enabled") is not False:
        raise ValueError("pilot requires router.enabled=false")
    balance_stop_cny = (config.get("experiment") or {}).get("balance_stop_cny")
    if type(balance_stop_cny) is not int or balance_stop_cny != 2:
        raise ValueError("pilot requires experiment.balance_stop_cny=2")


def validate_protocol(spec: FamilySpec) -> tuple[dict[str, Any], dict[str, Any]]:
    induction = _read_object(spec.induction_dataset, label="induction dataset")
    verification = _read_object(spec.verification_dataset, label="verification dataset")
    heldout = _read_object(spec.heldout_dataset, label="heldout dataset")
    manifest = _read_object(spec.manifest_path, label="family manifest")
    with spec.config_path.resolve().open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("config must be a YAML mapping")
    _validate_config(config)

    datasets = {
        "induction": induction,
        "verification": verification,
        "heldout": heldout,
    }
    expected_counts = {"induction": 10, "verification": 10, "heldout": 10}
    all_ids: dict[str, set[str]] = {}
    for split, payload in datasets.items():
        rows = payload.get("instances") or []
        if len(rows) != expected_counts[split]:
            raise ValueError(
                f"{split} must contain exactly {expected_counts[split]} slots"
            )
        ids = _dataset_ids(payload)
        if len(ids) != len(set(ids)):
            raise ValueError(f"{split} instance IDs are not unique")
        all_ids[split] = set(ids)
        metadata = payload.get("metadata") or {}
        if metadata.get("family_id") != spec.family_id:
            raise ValueError(f"{split} family_id differs from CLI family_id")
        if metadata.get("split") != split:
            raise ValueError(f"{split} dataset metadata has the wrong split")

    for left, right in (
        ("induction", "verification"),
        ("induction", "heldout"),
        ("verification", "heldout"),
    ):
        overlap = all_ids[left] & all_ids[right]
        if overlap:
            raise ValueError(f"{left}/{right} instance IDs overlap")

    source_ids = set(str(item) for item in manifest.get("source_task_ids") or [])
    heldout_id = str(manifest.get("heldout_task_id") or "")
    if not source_ids or not heldout_id:
        raise ValueError("manifest is missing source or heldout task IDs")
    if _dataset_task_ids(induction) != source_ids:
        raise ValueError("induction task membership differs from frozen manifest")
    if _dataset_task_ids(verification) != source_ids:
        raise ValueError("verification task membership differs from frozen manifest")
    if _dataset_task_ids(heldout) != {heldout_id}:
        raise ValueError("heldout dataset does not contain exactly the frozen target")
    if heldout_id in source_ids:
        raise ValueError("heldout target leaked into source tasks")

    expected_allocations = {
        "induction": manifest.get("induction_allocations") or {},
        "verification": manifest.get("verification_allocations") or {},
    }
    for split in ("induction", "verification"):
        observed: dict[str, int] = {}
        for row in datasets[split]["instances"]:
            task_id = str((row.get("metadata") or {})["skillsbench_task_id"])
            observed[task_id] = observed.get(task_id, 0) + 1
        if observed != expected_allocations[split]:
            raise ValueError(
                f"{split} allocation differs from manifest: {observed!r}"
            )

    expected_digests = manifest.get("task_package_digests") or {}
    for split, payload in datasets.items():
        for row in payload["instances"]:
            metadata = row.get("metadata") or {}
            task_id = str(metadata.get("skillsbench_task_id"))
            if metadata.get("skillsbench_task_digest") != expected_digests.get(task_id):
                raise ValueError(f"{split} task digest mismatch for {task_id}")

    if manifest.get("candidate_rounds") != 8:
        raise ValueError("manifest candidate_rounds must be 8")
    gate = manifest.get("gate") or {}
    if gate.get("repair_minus_regression_min") != 2:
        raise ValueError("manifest gate must require repair-regression >= 2")
    if gate.get("heldout_skill_is_conditional_on_gate") is not True:
        raise ValueError("heldout skill execution must be conditional on gate activation")

    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "family_id": spec.family_id,
        "induction_sha256": _hash_file(spec.induction_dataset.resolve()),
        "verification_sha256": _hash_file(spec.verification_dataset.resolve()),
        "heldout_sha256": _hash_file(spec.heldout_dataset.resolve()),
        "manifest_sha256": _hash_file(spec.manifest_path.resolve()),
        "config_sha256": _hash_file(spec.config_path.resolve()),
        "code_sha256": {
            name: _hash_file(REPO_ROOT / name)
            for name in (
                "pipeline.py",
                "llm.py",
                "pilot_budget_guard.py",
                "benchmarks/skillsbench_adapter.py",
                "scripts/run_skillsbench_family.py",
            )
        },
        "runtime_source_tree_sha256": _hash_runtime_source_tree(),
        "provider": "deepseek_official",
        "chat_model": EXPECTED_MODEL,
        "agent_model": EXPECTED_AGENT_MODEL,
        "budget_cny": str(spec.budget_cny),
        "budget_guard_semantics": _BUDGET_GUARD_SEMANTICS,
        "balance_stop_cny": FROZEN_BALANCE_STOP_CNY,
        "soft_reserve_cny": {"meta_request": "5", "agent_rollout": "10"},
        "stage_max_workers": FROZEN_MAX_WORKERS,
        "rolling_fail_stop": True,
        "candidate_rounds": 8,
        "gate_net_gain": 2,
        "heldout_skill_conditional": True,
    }
    serialized = json.dumps(
        protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    protocol_hash = hashlib.sha256(serialized).hexdigest()
    return {"hash": protocol_hash, **protocol}, manifest


def _write_runtime_config(spec: FamilySpec) -> Path:
    with spec.config_path.resolve().open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    runtime = dict(config)
    runtime["pipeline"] = {
        **(runtime.get("pipeline") or {}),
        "artifact_root": str((spec.fold_root / "pipeline-runs").resolve()),
    }
    runtime["skill_output"] = {
        **(runtime.get("skill_output") or {}),
        "path": str((spec.fold_root / "skill-output").resolve()),
    }
    path = spec.fold_root / "runtime_config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".yaml.tmp")
    temporary.write_text(
        yaml.safe_dump(runtime, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def _latest_pipeline_run(root: Path, protocol_hash: str) -> Path | None:
    candidates: list[Path] = []
    if not root.is_dir():
        return None
    for child in root.iterdir():
        metadata_path = child / "run_metadata.json"
        if not metadata_path.is_file():
            continue
        try:
            metadata = _read_object(metadata_path, label="pipeline metadata")
        except Exception:
            continue
        dataset_metadata = metadata.get("dataset_metadata") or {}
        if dataset_metadata.get("protocol_hash") == protocol_hash:
            candidates.append(child)
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _trajectory_counts(path: Path | None) -> dict[str, int] | None:
    if path is None or not path.is_file():
        return None
    total = successes = failures = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            if bool(row.get("success")):
                successes += 1
            else:
                failures += 1
    return {"total": total, "successes": successes, "failures": failures}


def _baseline_counts(run_dir: Path | None) -> dict[str, int] | None:
    return _trajectory_counts(
        run_dir / "baseline_trajectories.jsonl" if run_dir is not None else None
    )


def _verification_baseline_counts(run_dir: Path | None) -> dict[str, int] | None:
    return _trajectory_counts(
        run_dir / "verification_baseline_trajectories.jsonl"
        if run_dir is not None
        else None
    )


def _find_skill_dir(root: Path, skill_id: str) -> Path | None:
    matches = list(root.rglob(f"{skill_id}.json")) if root.is_dir() else []
    return matches[0].parent if len(matches) == 1 else None


def _paired_record(paired: Any, *, drop_blank: bool) -> dict[str, Any]:
    return {
        "n_instances": int(paired.n_instances),
        "baseline_acc": float(paired.baseline_acc),
        "skill_acc": float(paired.skill_acc),
        "delta_acc": float(paired.skill_acc - paired.baseline_acc),
        "repair": int(paired.repair),
        "regression": int(paired.regression),
        "net_gain": int(paired.net_gain),
        "repair_rate": float(paired.repair_rate),
        "regression_rate": float(paired.regression_rate),
        "blank_filter": {
            "drop_blank": drop_blank,
            "n_paired_raw": int(paired.n_paired_raw),
            "n_blank_either": int(paired.n_blank_either),
        },
    }


def _condition_from_trajectories(*, model: str, skill: Any, trajectories: list[Any]) -> Any:
    from eval_skill import ConditionResult

    successes = sum(1 for trajectory in trajectories if trajectory.success)
    latencies = [
        trajectory.latency
        for trajectory in trajectories
        if trajectory.latency is not None
    ]
    return ConditionResult(
        model=model,
        with_skill=skill is not None,
        skill_id=skill.skill_id if skill else None,
        n_instances=len(trajectories),
        n_success=successes,
        success_rate=successes / len(trajectories) if trajectories else 0.0,
        latency_mean=sum(latencies) / len(latencies) if latencies else 0.0,
        trajectories=trajectories,
    )


def _write_trajectories_atomic(path: Path, trajectories: list[Any]) -> None:
    from artifacts import write_trajectories

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_trajectories(temporary, trajectories)
    os.replace(temporary, path)


def _load_trajectory_condition(
    path: Path,
    *,
    model: str,
    skill: Any,
    expected_instance_ids: set[str] | None = None,
) -> Any:
    from artifacts import read_trajectories

    trajectories = read_trajectories(path)
    observed = [str(item.instance_id) for item in trajectories]
    if len(observed) != len(set(observed)):
        raise RuntimeError(f"trajectory condition contains duplicate IDs: {path}")
    if expected_instance_ids is not None and set(observed) != expected_instance_ids:
        raise RuntimeError(
            f"trajectory condition IDs differ from the frozen dataset: {path}"
        )
    return _condition_from_trajectories(
        model=model,
        skill=skill,
        trajectories=trajectories,
    )


def _status_path(spec: FamilySpec) -> Path:
    return spec.fold_root / "status.json"


def _save_status(spec: FamilySpec, status: dict[str, Any], **updates: Any) -> dict[str, Any]:
    status = {**status, **updates}
    _atomic_json(_status_path(spec), status)
    return status


def _budget_policy_amendment_id(new_protocol_hash: str) -> str:
    lineage = (
        f"{V12_PRE_BALANCE_STOP_PROTOCOL_HASH}->{new_protocol_hash}:"
        f"balance_stop_cny={FROZEN_BALANCE_STOP_CNY}"
    )
    return "balance-stop-policy-" + hashlib.sha256(lineage.encode("ascii")).hexdigest()


def _budget_policy_amendment_path(spec: FamilySpec) -> Path:
    return spec.fold_root / "budget_policy_amendments.json"


@contextmanager
def _budget_policy_amendment_lock(spec: FamilySpec):
    """Acquire one non-blocking cross-platform lock for same-fold migration."""

    path = spec.fold_root / "budget_policy_amendment.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    locked = False
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            raise RuntimeError(
                "another process holds the fold budget-policy amendment lock"
            ) from exc
        locked = True
        yield
    finally:
        if locked:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _validate_exact_config_policy_diff(
    spec: FamilySpec,
    old_protocol: dict[str, Any],
    new_protocol: dict[str, Any],
) -> None:
    if old_protocol.get("config_sha256") != V12_PRE_BALANCE_STOP_CONFIG_SHA256:
        raise RuntimeError("allowlisted v12 config hash is not exact")
    config_bytes = spec.config_path.resolve().read_bytes()
    marker = b"  balance_stop_cny: 2\n"
    if config_bytes.count(marker) != 1:
        raise RuntimeError("config amendment is not the exact frozen one-line policy diff")
    reconstructed_old = config_bytes.replace(marker, b"", 1)
    if hashlib.sha256(reconstructed_old).hexdigest() != V12_PRE_BALANCE_STOP_CONFIG_SHA256:
        raise RuntimeError("config contains changes outside the balance-stop policy")
    if hashlib.sha256(config_bytes).hexdigest() != new_protocol.get("config_sha256"):
        raise RuntimeError("new protocol does not bind the amended config bytes")


def _canonical_sha256(payload: Any) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _read_budget_policy_amendment_events(spec: FamilySpec) -> list[dict[str, Any]]:
    path = _budget_policy_amendment_path(spec)
    if not path.is_file():
        return []
    payload = _read_object(path, label="budget policy amendment ledger")
    if set(payload) != {"schema_version", "events"} or payload.get("schema_version") != 1:
        raise RuntimeError("invalid budget policy amendment ledger")
    events = payload.get("events")
    if not isinstance(events, list) or not all(isinstance(item, dict) for item in events):
        raise RuntimeError("invalid budget policy amendment ledger events")
    return list(events)


def _append_budget_policy_amendment_event(
    spec: FamilySpec,
    event: dict[str, Any],
) -> None:
    events = _read_budget_policy_amendment_events(spec)
    amendment_id = event.get("amendment_id")
    phase = event.get("event")
    phases = [existing.get("event") for existing in events]
    if phases not in ([], ["authorized"], ["authorized", "applied"]):
        raise RuntimeError("budget policy amendment events are not an exact prefix")
    if any(existing.get("amendment_id") != amendment_id for existing in events):
        raise RuntimeError("unexpected amendment lineage in the frozen fold")
    if phase == "authorized":
        if phases:
            if events[0] != event:
                raise RuntimeError(
                    "budget policy amendment event conflicts with frozen record"
                )
            return
    elif phase == "applied":
        if phases == []:
            raise RuntimeError("budget policy amendment is missing its authorization event")
        if phases == ["authorized", "applied"]:
            if events[1] != event:
                raise RuntimeError(
                    "budget policy amendment event conflicts with frozen record"
                )
            return
    else:
        raise RuntimeError("unknown budget policy amendment event phase")
    _atomic_json(
        _budget_policy_amendment_path(spec),
        {"schema_version": 1, "events": [*events, event]},
    )


def _validate_budget_policy_protocol_change(
    old_protocol: dict[str, Any],
    new_protocol: dict[str, Any],
) -> None:
    if old_protocol.get("hash") != V12_PRE_BALANCE_STOP_PROTOCOL_HASH:
        raise RuntimeError("budget policy amendment old protocol is not allowlisted")
    old_protocol_body = {
        key: value for key, value in old_protocol.items() if key != "hash"
    }
    if _canonical_sha256(old_protocol_body) != V12_PRE_BALANCE_STOP_PROTOCOL_HASH:
        raise RuntimeError("allowlisted old protocol content does not match its hash")
    new_protocol_hash = new_protocol.get("hash")
    new_protocol_body = {
        key: value for key, value in new_protocol.items() if key != "hash"
    }
    if (
        not isinstance(new_protocol_hash, str)
        or _canonical_sha256(new_protocol_body) != new_protocol_hash
    ):
        raise RuntimeError("new protocol content does not match its hash")
    if set(new_protocol) != set(old_protocol) | {"balance_stop_cny"}:
        raise RuntimeError("budget policy amendment changes protocol fields outside policy")
    if "balance_stop_cny" in old_protocol:
        raise RuntimeError("allowlisted old protocol already records a balance floor")
    if new_protocol.get("balance_stop_cny") != FROZEN_BALANCE_STOP_CNY:
        raise RuntimeError("new protocol does not record the frozen 2 CNY balance floor")
    if old_protocol.get("budget_guard_semantics") != _OLD_BUDGET_GUARD_SEMANTICS:
        raise RuntimeError("allowlisted old budget semantics are not exact")
    if new_protocol.get("budget_guard_semantics") != _BUDGET_GUARD_SEMANTICS:
        raise RuntimeError("new budget semantics are not exact")

    allowed_top_level_changes = {
        "hash",
        "config_sha256",
        "code_sha256",
        "runtime_source_tree_sha256",
        "budget_guard_semantics",
        "balance_stop_cny",
    }
    for key in set(old_protocol) - allowed_top_level_changes:
        if old_protocol.get(key) != new_protocol.get(key):
            raise RuntimeError(
                f"budget policy amendment changes non-policy protocol field {key!r}"
            )
    for key in ("hash", "config_sha256", "runtime_source_tree_sha256"):
        if old_protocol.get(key) == new_protocol.get(key):
            raise RuntimeError(f"budget policy amendment did not change derived field {key!r}")

    old_code = old_protocol.get("code_sha256")
    new_code = new_protocol.get("code_sha256")
    if not isinstance(old_code, dict) or not isinstance(new_code, dict):
        raise RuntimeError("budget policy amendment protocol is missing code hashes")
    if set(old_code) != set(new_code):
        raise RuntimeError("budget policy amendment changes the frozen code hash inventory")
    changed_code = {
        name for name in old_code if old_code.get(name) != new_code.get(name)
    }
    if changed_code != {
        "pilot_budget_guard.py",
        "scripts/run_skillsbench_family.py",
    }:
        raise RuntimeError(
            "budget policy amendment code diff is not limited to runner and guard"
        )


def _read_checkpoint_instance_ids(path: Path) -> list[str]:
    instance_ids: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict) or "instance_id" not in payload:
                raise RuntimeError("invalid baseline trajectory checkpoint")
            instance_ids.append(str(payload["instance_id"]))
    return instance_ids


def _validate_budget_policy_checkpoint(
    spec: FamilySpec,
    status: dict[str, Any],
    *,
    new_protocol_hash: str,
) -> dict[str, Any]:
    raw_run_dir = status.get("pipeline_run_dir")
    if not isinstance(raw_run_dir, str) or not raw_run_dir.strip():
        raise RuntimeError("budget policy amendment requires a recorded pipeline run")
    pipeline_root = (spec.fold_root / "pipeline-runs").resolve()
    run_dir = Path(raw_run_dir).expanduser().resolve()
    if run_dir.parent != pipeline_root or not run_dir.is_dir():
        raise RuntimeError("recorded pipeline run is outside the frozen fold")

    induction = _read_object(spec.induction_dataset, label="induction dataset")
    verification = _read_object(spec.verification_dataset, label="verification dataset")
    expected_ids = _dataset_ids(induction)
    expected_verification_ids = _dataset_ids(verification)
    if len(expected_ids) != 10 or len(expected_ids) != len(set(expected_ids)):
        raise RuntimeError("induction dataset is inconsistent with the frozen checkpoint")

    checkpoint_relative_paths = (
        "checkpoint.json",
        "checkpoint_trajectories.jsonl",
        "baseline_trajectories.jsonl",
        "analysis/skill_analysis.json",
        "analysis/skill_analysis_summary.json",
        "verification_baseline_trajectories.jsonl",
        "refinement_checkpoint.json",
    )
    checkpoint_paths = {
        relative: run_dir / relative for relative in checkpoint_relative_paths
    }
    for relative, path in checkpoint_paths.items():
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(run_dir):
            raise RuntimeError(f"pipeline checkpoint artifact is unsafe or missing: {relative}")
    checkpoint_bindings = {
        relative: _hash_file(path) for relative, path in checkpoint_paths.items()
    }

    metadata_path = run_dir / "run_metadata.json"
    if (
        metadata_path.is_symlink()
        or not metadata_path.is_file()
        or metadata_path.resolve().parent != run_dir
    ):
        raise RuntimeError("pipeline metadata is unsafe or missing")
    metadata_bytes = metadata_path.read_bytes()
    metadata = json.loads(metadata_bytes)
    if not isinstance(metadata, dict):
        raise RuntimeError("pipeline metadata must be a JSON object")
    dataset_metadata = metadata.get("dataset_metadata")
    if not isinstance(dataset_metadata, dict):
        raise RuntimeError("pipeline metadata is missing dataset metadata")
    metadata_protocol_hash = dataset_metadata.get("protocol_hash")
    if metadata_protocol_hash not in {
        V12_PRE_BALANCE_STOP_PROTOCOL_HASH,
        new_protocol_hash,
    }:
        raise RuntimeError("pipeline metadata has an unrelated protocol hash")
    if (
        metadata.get("dataset_id") != induction.get("dataset_id")
        or metadata.get("task_name") != induction.get("task_name")
        or metadata.get("n_instances") != len(expected_ids)
        or metadata.get("n_verification_instances")
        != len(verification.get("instances") or [])
        or metadata.get("separate_verification_pool") is not True
        or metadata.get("exhaustive_refinement") is not True
    ):
        raise RuntimeError("pipeline metadata is inconsistent with the frozen datasets")

    baseline_checkpoint = _read_object(
        checkpoint_paths["checkpoint.json"], label="checkpoint"
    )
    baseline_slot_checkpoint = (
        baseline_checkpoint.get("schema_version") == 1
        and baseline_checkpoint.get("stage") == "baseline_done"
        and baseline_checkpoint.get("expected_instance_ids") == expected_ids
        and baseline_checkpoint.get("completed_instance_ids") == expected_ids
    )
    analysis_checkpoint = baseline_checkpoint == {
        "stage": "analysis_done",
        "total_stages": 1,
        "completed_stages": ["analysis"],
    }
    if not baseline_slot_checkpoint and not analysis_checkpoint:
        raise RuntimeError("pipeline progress checkpoint is not a reviewed frozen state")
    trajectory_ids = _read_checkpoint_instance_ids(
        checkpoint_paths["checkpoint_trajectories.jsonl"]
    )
    if trajectory_ids != expected_ids or len(trajectory_ids) != len(set(trajectory_ids)):
        raise RuntimeError("baseline trajectory checkpoint differs from frozen D_ind")
    if (
        _read_checkpoint_instance_ids(checkpoint_paths["baseline_trajectories.jsonl"])
        != expected_ids
    ):
        raise RuntimeError("published baseline trajectories differ from frozen D_ind")
    if (
        _read_checkpoint_instance_ids(
            checkpoint_paths["verification_baseline_trajectories.jsonl"]
        )
        != expected_verification_ids
    ):
        raise RuntimeError("verification baseline checkpoint differs from frozen D_ver")
    for relative in ("analysis/skill_analysis.json", "analysis/skill_analysis_summary.json"):
        _read_object(checkpoint_paths[relative], label=relative)

    refinement = _read_object(
        checkpoint_paths["refinement_checkpoint.json"], label="refinement checkpoint"
    )
    completed_rounds = refinement.get("completed_rounds")
    in_progress_round = refinement.get("in_progress_round")
    round_history = refinement.get("round_history")
    if (
        refinement.get("schema_version") != 2
        or type(completed_rounds) is not int
        or not 0 <= completed_rounds <= 8
        or refinement.get("max_rounds") != 8
        or refinement.get("exhaustive_refinement") is not True
        or not isinstance(round_history, list)
        or len(round_history) != completed_rounds
        or (
            in_progress_round is not None
            and (
                type(in_progress_round) is not int
                or in_progress_round != completed_rounds
                or not isinstance(refinement.get("candidate"), dict)
            )
        )
    ):
        raise RuntimeError("refinement checkpoint is internally inconsistent")
    candidate_sha256 = (
        _canonical_sha256(refinement.get("candidate"))
        if isinstance(refinement.get("candidate"), dict)
        else None
    )

    relative_metadata_path = metadata_path.relative_to(spec.fold_root.resolve()).as_posix()
    metadata_sha256 = hashlib.sha256(metadata_bytes).hexdigest()
    return {
        "run_metadata_path": metadata_path,
        "run_metadata_relative_path": relative_metadata_path,
        "run_metadata": metadata,
        "run_metadata_bytes": metadata_bytes,
        "run_metadata_sha256": metadata_sha256,
        "run_metadata_protocol_hash": metadata_protocol_hash,
        "checkpoint_bindings": checkpoint_bindings,
        "refinement_checkpoint_sha256": checkpoint_bindings[
            "refinement_checkpoint.json"
        ],
        "candidate_sha256": candidate_sha256,
        "completed_rounds": completed_rounds,
        "in_progress_round": in_progress_round,
    }


def _assert_no_inflight_budget_policy_work(spec: FamilySpec) -> None:
    jobs_roots: set[str] = set()
    for path, label in (
        (spec.induction_dataset, "induction dataset"),
        (spec.verification_dataset, "verification dataset"),
    ):
        dataset = _read_object(path, label=label)
        for row in dataset.get("instances") or []:
            metadata = row.get("metadata") or {}
            jobs_root = metadata.get("skillsbench_jobs_root")
            if not isinstance(jobs_root, str) or not jobs_root.strip():
                raise RuntimeError("frozen dataset is missing its SkillsBench jobs root")
            jobs_roots.add(str(Path(jobs_root).expanduser().resolve()))
    if len(jobs_roots) != 1:
        raise RuntimeError("frozen datasets do not share one SkillsBench jobs root")
    jobs_root = Path(next(iter(jobs_roots)))
    cache_locks = jobs_root / ".skillgen-rollout-cache" / "locks"
    if cache_locks.exists():
        if cache_locks.is_symlink() or not cache_locks.is_dir():
            raise RuntimeError("rollout cache lock directory is unsafe")
        if any(cache_locks.iterdir()):
            raise RuntimeError("rollout cache still contains active or stale slot locks")

    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return
    own_pid = os.getpid()
    for process_dir in proc_root.iterdir():
        if not process_dir.name.isdigit() or int(process_dir.name) == own_pid:
            continue
        try:
            raw_command = (process_dir / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        arguments = [
            item.decode("utf-8", errors="replace")
            for item in raw_command.split(b"\0")
            if item
        ]
        is_family_runner = any(
            Path(argument).name == "run_skillsbench_family.py"
            for argument in arguments
        )
        if is_family_runner and (
            spec.family_id in arguments or str(spec.run_root.resolve()) in arguments
        ):
            raise RuntimeError("another SkillsBench family runner targets this fold")


def _budget_policy_authorization_event(
    *,
    amendment_id: str,
    new_protocol_hash: str,
    relative_metadata_path: str,
    metadata_sha256_before: str,
    checkpoint_bindings: dict[str, str],
    candidate_sha256: str | None,
    completed_rounds: int,
    in_progress_round: int | None,
) -> dict[str, Any]:
    return {
        "event": "authorized",
        "amendment_id": amendment_id,
        "old_protocol_hash": V12_PRE_BALANCE_STOP_PROTOCOL_HASH,
        "new_protocol_hash": new_protocol_hash,
        "old_balance_stop_cny": None,
        "new_balance_stop_cny": FROZEN_BALANCE_STOP_CNY,
        "authorization": {
            "authorized": True,
            "mechanism": "explicit_cli_flag",
            "flag": _BUDGET_POLICY_AMENDMENT_FLAG,
            "expected_new_protocol_hash": new_protocol_hash,
        },
        "run_metadata_path": relative_metadata_path,
        "run_metadata_sha256_before": metadata_sha256_before,
        "checkpoint_bindings": checkpoint_bindings,
        "refinement_checkpoint_sha256": checkpoint_bindings[
            "refinement_checkpoint.json"
        ],
        "candidate_sha256": candidate_sha256,
        "completed_rounds": completed_rounds,
        "in_progress_round": in_progress_round,
    }


def _find_amendment_event(
    events: list[dict[str, Any]],
    *,
    amendment_id: str,
    phase: str,
) -> dict[str, Any] | None:
    matches = [
        event
        for event in events
        if event.get("amendment_id") == amendment_id and event.get("event") == phase
    ]
    if len(matches) > 1:
        raise RuntimeError("duplicate budget policy amendment event")
    return matches[0] if matches else None


def _validate_expected_new_protocol_hash(
    spec: FamilySpec,
    new_protocol: dict[str, Any],
) -> None:
    expected_new_hash = spec.expected_new_protocol_hash
    if (
        not isinstance(expected_new_hash, str)
        or len(expected_new_hash) != 64
        or any(character not in "0123456789abcdef" for character in expected_new_hash)
        or expected_new_hash != new_protocol.get("hash")
    ):
        raise RuntimeError(
            "budget policy amendment requires an exact 64-hex "
            "--expected-new-protocol-hash matching the computed protocol"
        )


def _prepare_budget_policy_amendment(
    spec: FamilySpec,
    status: dict[str, Any],
    new_protocol: dict[str, Any],
) -> dict[str, Any]:
    if not spec.resume or not spec.authorize_budget_policy_amendment:
        raise RuntimeError(
            "protocol migration requires --resume and explicit "
            f"{_BUDGET_POLICY_AMENDMENT_FLAG}"
        )
    _validate_expected_new_protocol_hash(spec, new_protocol)
    if status.get("stage") != "budget_stopped":
        raise RuntimeError("budget policy amendment is limited to budget_stopped folds")
    failure = status.get("failure")
    if not isinstance(failure, dict) or failure.get("kind") != "PilotBudgetStop":
        raise RuntimeError("budget_stopped fold does not record PilotBudgetStop")
    if status.get("protocol_hash") != V12_PRE_BALANCE_STOP_PROTOCOL_HASH:
        raise RuntimeError("existing fold protocol is not the allowlisted v12 protocol")
    old_protocol = status.get("protocol")
    if not isinstance(old_protocol, dict):
        raise RuntimeError("existing fold is missing its frozen protocol record")
    if status.get("family_id") != spec.family_id:
        raise RuntimeError("existing fold family differs from the requested family")
    if (spec.fold_root / "result.json").exists():
        raise RuntimeError("budget policy amendment cannot modify a completed fold")
    _validate_budget_policy_protocol_change(old_protocol, new_protocol)
    _validate_exact_config_policy_diff(spec, old_protocol, new_protocol)

    context = _validate_budget_policy_checkpoint(
        spec,
        status,
        new_protocol_hash=str(new_protocol["hash"]),
    )
    amendment_id = _budget_policy_amendment_id(str(new_protocol["hash"]))
    events = _read_budget_policy_amendment_events(spec)
    if any(event.get("amendment_id") != amendment_id for event in events):
        raise RuntimeError("unexpected amendment lineage in the frozen fold")
    event_phases = [event.get("event") for event in events]
    if event_phases not in ([], ["authorized"], ["authorized", "applied"]):
        raise RuntimeError("budget policy amendment events are not an exact prefix")
    authorization = _find_amendment_event(
        events, amendment_id=amendment_id, phase="authorized"
    )
    applied = _find_amendment_event(events, amendment_id=amendment_id, phase="applied")

    if context["run_metadata_protocol_hash"] == V12_PRE_BALANCE_STOP_PROTOCOL_HASH:
        if event_phases not in ([], ["authorized"]):
            raise RuntimeError("old run metadata conflicts with amendment event state")
        metadata_sha256_before = context["run_metadata_sha256"]
        expected_authorization = _budget_policy_authorization_event(
            amendment_id=amendment_id,
            new_protocol_hash=str(new_protocol["hash"]),
            relative_metadata_path=context["run_metadata_relative_path"],
            metadata_sha256_before=metadata_sha256_before,
            checkpoint_bindings=context["checkpoint_bindings"],
            candidate_sha256=context["candidate_sha256"],
            completed_rounds=context["completed_rounds"],
            in_progress_round=context["in_progress_round"],
        )
        if authorization is not None and authorization != expected_authorization:
            raise RuntimeError("budget policy authorization record conflicts with metadata")
        if applied is not None:
            raise RuntimeError("applied amendment record conflicts with old run metadata")
    else:
        if event_phases not in (["authorized"], ["authorized", "applied"]):
            raise RuntimeError("new run metadata lacks an exact amendment event prefix")
        if authorization is None:
            raise RuntimeError("amended run metadata lacks an append-only authorization")
        metadata_sha256_before = authorization.get("run_metadata_sha256_before")
        expected_authorization = _budget_policy_authorization_event(
            amendment_id=amendment_id,
            new_protocol_hash=str(new_protocol["hash"]),
            relative_metadata_path=context["run_metadata_relative_path"],
            metadata_sha256_before=str(metadata_sha256_before),
            checkpoint_bindings=context["checkpoint_bindings"],
            candidate_sha256=context["candidate_sha256"],
            completed_rounds=context["completed_rounds"],
            in_progress_round=context["in_progress_round"],
        )
        if authorization != expected_authorization:
            raise RuntimeError("budget policy authorization record is invalid")
        if not isinstance(metadata_sha256_before, str) or len(metadata_sha256_before) != 64:
            raise RuntimeError("budget policy authorization lacks the pre-amendment hash")
        if applied is not None and applied.get("run_metadata_sha256_after") != context[
            "run_metadata_sha256"
        ]:
            raise RuntimeError("applied amendment record conflicts with run metadata")

    return {
        **context,
        "amendment_id": amendment_id,
        "metadata_sha256_before": metadata_sha256_before,
        "authorization_event": expected_authorization,
    }


def _budget_policy_applied_event(
    *,
    context: dict[str, Any],
    new_protocol_hash: str,
    metadata_sha256_after: str,
) -> dict[str, Any]:
    return {
        "event": "applied",
        "amendment_id": context["amendment_id"],
        "old_protocol_hash": V12_PRE_BALANCE_STOP_PROTOCOL_HASH,
        "new_protocol_hash": new_protocol_hash,
        "old_balance_stop_cny": None,
        "new_balance_stop_cny": FROZEN_BALANCE_STOP_CNY,
        "run_metadata_path": context["run_metadata_relative_path"],
        "run_metadata_sha256_before": context["metadata_sha256_before"],
        "run_metadata_sha256_after": metadata_sha256_after,
        "guard_ledger_schema_version": 3,
        "checkpoint_bindings": context["checkpoint_bindings"],
        "candidate_sha256": context["candidate_sha256"],
        "completed_rounds": context["completed_rounds"],
        "in_progress_round": context["in_progress_round"],
    }


def _prepare_applied_budget_policy_lineage(
    spec: FamilySpec,
    status: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    _validate_expected_new_protocol_hash(spec, protocol)
    lineage = status.get("protocol_lineage")
    if not isinstance(lineage, dict):
        raise RuntimeError("amended fold is missing immutable protocol lineage")
    old_protocol = lineage.get("old_protocol")
    if not isinstance(old_protocol, dict):
        raise RuntimeError("amended fold did not retain the complete old protocol")
    _validate_budget_policy_protocol_change(old_protocol, protocol)
    _validate_exact_config_policy_diff(spec, old_protocol, protocol)
    amendment_id = _budget_policy_amendment_id(str(protocol["hash"]))
    if (
        lineage.get("kind") != "balance_stop_policy_amendment"
        or lineage.get("amendment_id") != amendment_id
        or lineage.get("old_protocol_hash") != V12_PRE_BALANCE_STOP_PROTOCOL_HASH
        or lineage.get("new_protocol_hash") != protocol.get("hash")
        or status.get("protocol") != protocol
        or status.get("protocol_hash") != protocol.get("hash")
    ):
        raise RuntimeError("amended fold protocol lineage is inconsistent")

    context = _validate_budget_policy_checkpoint(
        spec,
        status,
        new_protocol_hash=str(protocol["hash"]),
    )
    if context["run_metadata_protocol_hash"] != protocol["hash"]:
        raise RuntimeError("amended fold run metadata did not retain the new protocol")
    events = _read_budget_policy_amendment_events(spec)
    if any(event.get("amendment_id") != amendment_id for event in events):
        raise RuntimeError("amended fold contains an unrelated amendment lineage")
    if [event.get("event") for event in events] != ["authorized", "applied"]:
        raise RuntimeError("amended fold must contain exactly authorized then applied")
    authorization = _find_amendment_event(
        events, amendment_id=amendment_id, phase="authorized"
    )
    applied = _find_amendment_event(events, amendment_id=amendment_id, phase="applied")
    if authorization is None or applied is None:
        raise RuntimeError("amended fold is missing append-only amendment events")
    metadata_sha256_before = authorization.get("run_metadata_sha256_before")
    recorded_bindings = authorization.get("checkpoint_bindings")
    expected_binding_names = {
        "checkpoint.json",
        "checkpoint_trajectories.jsonl",
        "baseline_trajectories.jsonl",
        "analysis/skill_analysis.json",
        "analysis/skill_analysis_summary.json",
        "verification_baseline_trajectories.jsonl",
        "refinement_checkpoint.json",
    }
    if (
        not isinstance(recorded_bindings, dict)
        or set(recorded_bindings) != expected_binding_names
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in recorded_bindings.values()
        )
    ):
        raise RuntimeError("amended fold authorization has invalid checkpoint bindings")
    recorded_candidate_sha256 = authorization.get("candidate_sha256")
    if recorded_candidate_sha256 is not None and (
        not isinstance(recorded_candidate_sha256, str)
        or len(recorded_candidate_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in recorded_candidate_sha256
        )
    ):
        raise RuntimeError("amended fold authorization has an invalid candidate digest")
    recorded_completed_rounds = authorization.get("completed_rounds")
    recorded_in_progress_round = authorization.get("in_progress_round")
    if (
        type(recorded_completed_rounds) is not int
        or not 0 <= recorded_completed_rounds <= 8
        or (
            recorded_in_progress_round is not None
            and (
                type(recorded_in_progress_round) is not int
                or recorded_in_progress_round != recorded_completed_rounds
            )
        )
    ):
        raise RuntimeError("amended fold authorization has invalid refinement progress")
    expected_authorization = _budget_policy_authorization_event(
        amendment_id=amendment_id,
        new_protocol_hash=str(protocol["hash"]),
        relative_metadata_path=context["run_metadata_relative_path"],
        metadata_sha256_before=str(metadata_sha256_before),
        checkpoint_bindings=recorded_bindings,
        candidate_sha256=recorded_candidate_sha256,
        completed_rounds=recorded_completed_rounds,
        in_progress_round=recorded_in_progress_round,
    )
    if authorization != expected_authorization:
        raise RuntimeError("amended fold authorization event no longer matches state")
    full_context = {
        **context,
        "amendment_id": amendment_id,
        "metadata_sha256_before": metadata_sha256_before,
        "authorization_event": authorization,
        "checkpoint_bindings": recorded_bindings,
        "candidate_sha256": recorded_candidate_sha256,
        "completed_rounds": recorded_completed_rounds,
        "in_progress_round": recorded_in_progress_round,
    }
    expected_applied = _budget_policy_applied_event(
        context=full_context,
        new_protocol_hash=str(protocol["hash"]),
        metadata_sha256_after=context["run_metadata_sha256"],
    )
    if applied != expected_applied:
        raise RuntimeError("amended fold applied event no longer matches state")
    if (
        lineage.get("authorization_event_sha256") != _canonical_sha256(authorization)
        or lineage.get("applied_event_sha256") != _canonical_sha256(applied)
    ):
        raise RuntimeError("amended fold status does not bind its append-only events")
    return full_context


def _validate_migrated_guard_ledger(
    ledger: object,
    *,
    amendment_id: str,
    new_protocol_hash: str,
) -> None:
    try:
        valid_summary = (
            isinstance(ledger, dict)
            and ledger.get("schema_version") == 3
            and Decimal(str(ledger.get("balance_stop_cny")))
            == Decimal(FROZEN_BALANCE_STOP_CNY)
            and ledger.get("active_reservations") == {}
            and Decimal(str(ledger.get("active_reserved_cny"))) == Decimal("0")
        )
    except Exception as exc:
        raise RuntimeError("budget guard returned an invalid migrated ledger") from exc
    if not valid_summary:
        raise RuntimeError("budget guard returned an invalid migrated ledger")
    amendments = ledger.get("balance_stop_policy_amendments")
    if not isinstance(amendments, list) or len(amendments) != 1:
        raise RuntimeError("budget guard ledger has non-exact amendment lineage")
    amendment = amendments[0]
    expected_identity = {
        "amendment_id": amendment_id,
        "old_balance_stop_cny": None,
        "new_balance_stop_cny": FROZEN_BALANCE_STOP_CNY,
        "old_protocol_hash": V12_PRE_BALANCE_STOP_PROTOCOL_HASH,
        "new_protocol_hash": new_protocol_hash,
    }
    if (
        not isinstance(amendment, dict)
        or set(amendment) != set(expected_identity) | {"migrated_at"}
        or {key: amendment.get(key) for key in expected_identity} != expected_identity
        or not isinstance(amendment.get("migrated_at"), str)
        or not amendment["migrated_at"].strip()
        or amendment["migrated_at"] != amendment["migrated_at"].strip()
    ):
        raise RuntimeError("budget guard ledger amendment identity is not exact")


def _verify_applied_budget_policy_lineage(
    spec: FamilySpec,
    protocol: dict[str, Any],
    context: dict[str, Any],
) -> None:
    _assert_no_inflight_budget_policy_work(spec)
    ledger = pilot_budget_guard.migrate_balance_stop_policy(
        expected_old_balance_stop_cny=None,
        new_balance_stop_cny=Decimal(FROZEN_BALANCE_STOP_CNY),
        amendment_id=context["amendment_id"],
        old_protocol_hash=V12_PRE_BALANCE_STOP_PROTOCOL_HASH,
        new_protocol_hash=str(protocol["hash"]),
    )
    _validate_migrated_guard_ledger(
        ledger,
        amendment_id=context["amendment_id"],
        new_protocol_hash=str(protocol["hash"]),
    )


def _apply_budget_policy_amendment(
    spec: FamilySpec,
    status: dict[str, Any],
    new_protocol: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    _assert_no_inflight_budget_policy_work(spec)
    current = _validate_budget_policy_checkpoint(
        spec,
        status,
        new_protocol_hash=str(new_protocol["hash"]),
    )
    for key in (
        "run_metadata_sha256",
        "run_metadata_protocol_hash",
        "checkpoint_bindings",
        "refinement_checkpoint_sha256",
        "candidate_sha256",
        "completed_rounds",
        "in_progress_round",
    ):
        if current.get(key) != context.get(key):
            raise RuntimeError(
                f"budget policy checkpoint changed after authorization: {key}"
            )
    ledger = pilot_budget_guard.migrate_balance_stop_policy(
        expected_old_balance_stop_cny=None,
        new_balance_stop_cny=Decimal(FROZEN_BALANCE_STOP_CNY),
        amendment_id=context["amendment_id"],
        old_protocol_hash=V12_PRE_BALANCE_STOP_PROTOCOL_HASH,
        new_protocol_hash=str(new_protocol["hash"]),
    )
    _validate_migrated_guard_ledger(
        ledger,
        amendment_id=context["amendment_id"],
        new_protocol_hash=str(new_protocol["hash"]),
    )

    _append_budget_policy_amendment_event(spec, context["authorization_event"])
    metadata_path = context["run_metadata_path"]
    if context["run_metadata_protocol_hash"] == V12_PRE_BALANCE_STOP_PROTOCOL_HASH:
        old_hash_bytes = V12_PRE_BALANCE_STOP_PROTOCOL_HASH.encode("ascii")
        metadata_bytes = context["run_metadata_bytes"]
        if metadata_bytes.count(old_hash_bytes) != 1:
            raise RuntimeError("old protocol hash is not unique in run metadata")
        amended_bytes = metadata_bytes.replace(
            old_hash_bytes, str(new_protocol["hash"]).encode("ascii"), 1
        )
        amended_metadata = json.loads(amended_bytes)
        expected_metadata = json.loads(json.dumps(context["run_metadata"]))
        expected_metadata["dataset_metadata"]["protocol_hash"] = new_protocol["hash"]
        if amended_metadata != expected_metadata:
            raise RuntimeError("run metadata amendment would change non-policy fields")
        _atomic_bytes(metadata_path, amended_bytes)
    metadata_sha256_after = _hash_file(metadata_path)
    applied_event = _budget_policy_applied_event(
        context=context,
        new_protocol_hash=str(new_protocol["hash"]),
        metadata_sha256_after=metadata_sha256_after,
    )
    _append_budget_policy_amendment_event(spec, applied_event)
    protocol_lineage = {
        "kind": "balance_stop_policy_amendment",
        "amendment_id": context["amendment_id"],
        "old_protocol_hash": V12_PRE_BALANCE_STOP_PROTOCOL_HASH,
        "new_protocol_hash": str(new_protocol["hash"]),
        "old_protocol": status["protocol"],
        "authorization_event_sha256": _canonical_sha256(
            context["authorization_event"]
        ),
        "applied_event_sha256": _canonical_sha256(applied_event),
    }
    return _save_status(
        spec,
        status,
        protocol_hash=new_protocol["hash"],
        protocol=new_protocol,
        protocol_lineage=protocol_lineage,
    )


def run(spec: FamilySpec, *, execute: bool) -> dict[str, Any]:
    protocol, manifest = validate_protocol(spec)
    plan = {
        "mode": "execute" if execute else "dry-run",
        "protocol": protocol,
        "manifest": manifest,
        "agent_trajectory_counts": {
            "no_induction_failure": 20,
            "gate_rejected_after_8_rounds": 110,
            "gate_active_after_8_rounds": 120,
        },
        "heldout_not_passed_to_construction_or_gate": True,
        "api_calls_made": False,
    }
    if not execute:
        return plan

    existing_status = (
        _read_object(_status_path(spec), label="status")
        if _status_path(spec).is_file()
        else None
    )
    new_status = existing_status is None
    needs_budget_policy_amendment = False
    needs_budget_policy_lineage_check = False
    completed_result_after_lineage: Path | None = None
    if existing_status:
        if existing_status.get("protocol_hash") != protocol["hash"]:
            needs_budget_policy_amendment = True
        elif existing_status.get("stage") == "complete":
            result_path = spec.fold_root / "result.json"
            if "protocol_lineage" not in existing_status:
                return _read_object(result_path, label="completed result")
            needs_budget_policy_lineage_check = True
            completed_result_after_lineage = result_path
        elif not spec.resume:
            raise RuntimeError(
                "an incomplete paid fold already exists; inspect it and pass --resume "
                "only if its checkpoint is safe"
            )
        elif "protocol_lineage" in existing_status:
            needs_budget_policy_lineage_check = True
        status = existing_status
    else:
        status = {
            "schema_version": 1,
            "family_id": spec.family_id,
            "protocol_hash": protocol["hash"],
            "protocol": protocol,
            "stage": "prepared",
            "pipeline_run_dir": None,
            "skill_id": None,
            "skill_dir": None,
            "failure": None,
        }

    if os.environ.get("SKILLGEN_CHAT_PROVIDER") not in (None, "", "deepseek"):
        raise RuntimeError("SKILLGEN_CHAT_PROVIDER conflicts with the frozen DeepSeek route")
    os.environ["SKILLGEN_CHAT_PROVIDER"] = "deepseek"
    os.environ.setdefault("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    os.environ["SKILLGEN_DEEPSEEK_BUDGET_CNY"] = str(spec.budget_cny)
    os.environ["SKILLGEN_DEEPSEEK_BALANCE_STOP_CNY"] = FROZEN_BALANCE_STOP_CNY
    os.environ["SKILLGEN_META_REQUEST_RESERVE_CNY"] = "5"
    os.environ["SKILLGEN_AGENT_ROLLOUT_RESERVE_CNY"] = "10"
    expected_ledger_path = (spec.fold_root / "budget_ledger.json").resolve()
    configured_ledger_path = os.environ.get("SKILLGEN_BUDGET_LEDGER", "").strip()
    if (
        configured_ledger_path
        and Path(configured_ledger_path).expanduser().resolve() != expected_ledger_path
    ):
        raise RuntimeError("SKILLGEN_BUDGET_LEDGER conflicts with the frozen fold ledger")
    os.environ["SKILLGEN_BUDGET_LEDGER"] = str(expected_ledger_path)
    if not os.environ.get("DEEPSEEK_API_KEY", "").strip():
        raise RuntimeError("DEEPSEEK_API_KEY is required")
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise RuntimeError(
            "OPENAI_API_KEY is required only for the original text-embedding-3-small step"
        )
    if needs_budget_policy_amendment or needs_budget_policy_lineage_check:
        with _budget_policy_amendment_lock(spec):
            locked_status = _read_object(_status_path(spec), label="locked status")
            if locked_status != status:
                raise RuntimeError("fold status changed while entering amendment lock")
            if needs_budget_policy_amendment:
                amendment_context = _prepare_budget_policy_amendment(
                    spec, locked_status, protocol
                )
                status = _apply_budget_policy_amendment(
                    spec,
                    locked_status,
                    protocol,
                    amendment_context,
                )
            else:
                lineage_context = _prepare_applied_budget_policy_lineage(
                    spec, locked_status, protocol
                )
                _verify_applied_budget_policy_lineage(
                    spec, protocol, lineage_context
                )
                status = locked_status
    if completed_result_after_lineage is not None:
        return _read_object(completed_result_after_lineage, label="completed result")
    try:
        budget_snapshot = pilot_budget_guard.initialize()
    except pilot_budget_guard.PilotBudgetStop as exc:
        if not new_status:
            _save_status(
                spec,
                status,
                stage="budget_stopped",
                failure={"kind": type(exc).__name__, "reason": str(exc)},
            )
        raise
    if new_status:
        status = _save_status(spec, status)

    runtime_config = _write_runtime_config(spec)

    from eval_skill import paired_analysis, run_condition
    from main import load_dataset
    from models import SkillStatus
    from pipeline import run_pipeline
    from skill_store import load_skill

    induction = load_dataset(str(spec.induction_dataset))
    verification = load_dataset(str(spec.verification_dataset))
    if induction.task_type != verification.task_type:
        raise ValueError("induction and verification task types differ")

    resume_dir: str | None = None
    recorded_run = status.get("pipeline_run_dir")
    if spec.resume and not recorded_run:
        discovered = _latest_pipeline_run(
            spec.fold_root / "pipeline-runs", protocol["hash"]
        )
        recorded_run = str(discovered) if discovered is not None else None
    if spec.resume and recorded_run:
        run_path = Path(str(recorded_run))
        if (run_path / "checkpoint_trajectories.jsonl").is_file() and (
            run_path / "checkpoint.json"
        ).is_file():
            resume_dir = str(run_path)

    skill = None
    construction_complete = status.get("method_status") in {
        "active",
        "deprecated",
        "not_applicable_no_failure",
    }
    # The released pipeline returns immediately on an all-success induction
    # pool, leaving no checkpoint.json.  If the process was killed in the tiny
    # gap before the runner recorded that outcome, the complete frozen
    # baseline artifact is sufficient to recover without paying for D_ind
    # again.
    if spec.resume and not construction_complete and recorded_run:
        recovered_counts = _baseline_counts(Path(str(recorded_run)))
        if recovered_counts == {"total": 10, "successes": 10, "failures": 0}:
            status = _save_status(
                spec,
                status,
                stage="skill_ready",
                method_status="not_applicable_no_failure",
                pipeline_run_dir=str(Path(str(recorded_run)).resolve()),
                baseline_counts=recovered_counts,
                failure=None,
            )
            construction_complete = True
    if construction_complete:
        skill_dir = status.get("skill_dir")
        skill_id = status.get("skill_id")
        if skill_dir and skill_id:
            skill = load_skill(skill_dir, skill_id=skill_id)
    else:
        status = _save_status(spec, status, stage="constructing", failure=None)
        try:
            skill = run_pipeline(
                induction.instances,
                induction.task_type,
                config_path=str(runtime_config),
                dataset_id=induction.dataset_id,
                task_name=induction.task_name,
                dataset_metadata={
                    **(induction.metadata or {}),
                    "protocol_hash": protocol["hash"],
                },
                generate_scripts=False,
                resume_dir=resume_dir,
                verification_instances=verification.instances,
                exhaustive_refinement=True,
            )
        except pilot_budget_guard.PilotBudgetStop as exc:
            latest = _latest_pipeline_run(
                spec.fold_root / "pipeline-runs", protocol["hash"]
            )
            status = _save_status(
                spec,
                status,
                stage="budget_stopped",
                pipeline_run_dir=str(latest) if latest else resume_dir,
                failure={"kind": type(exc).__name__, "reason": str(exc)},
            )
            raise
        except Exception as exc:
            latest = _latest_pipeline_run(
                spec.fold_root / "pipeline-runs", protocol["hash"]
            )
            status = _save_status(
                spec,
                status,
                stage="failed",
                pipeline_run_dir=str(latest) if latest else resume_dir,
                failure={"kind": type(exc).__name__, "reason": str(exc)},
            )
            raise

        latest = _latest_pipeline_run(
            spec.fold_root / "pipeline-runs", protocol["hash"]
        )
        counts = _baseline_counts(latest)
        verification_counts = _verification_baseline_counts(latest)
        ceiling_limited = bool(
            verification_counts is not None
            and verification_counts.get("failures", 0) < 2
        )
        if skill is None:
            if not counts or counts.get("failures") != 0:
                raise RuntimeError("pipeline returned no skill without an all-success baseline")
            status = _save_status(
                spec,
                status,
                stage="skill_ready",
                method_status="not_applicable_no_failure",
                pipeline_run_dir=str(latest) if latest else None,
                baseline_counts=counts,
                verification_baseline_counts=verification_counts,
                ceiling_limited=False,
            )
        else:
            skill_dir = _find_skill_dir(spec.fold_root / "skill-output", skill.skill_id)
            if skill_dir is None:
                raise RuntimeError("could not locate the generated skill artifact")
            status = _save_status(
                spec,
                status,
                stage="skill_ready",
                method_status=(
                    "active" if skill.status == SkillStatus.ACTIVE else "deprecated"
                ),
                pipeline_run_dir=str(latest) if latest else None,
                baseline_counts=counts,
                verification_baseline_counts=verification_counts,
                ceiling_limited=ceiling_limited,
                skill_id=skill.skill_id,
                skill_dir=str(skill_dir.resolve()),
            )

    # The heldout target is deliberately loaded only after construction,
    # best-of-8 selection, and activation gating have frozen the skill status.
    heldout = load_dataset(str(spec.heldout_dataset))
    model = EXPECTED_AGENT_MODEL
    judge_model = EXPECTED_MODEL
    heldout_dir = spec.fold_root / "heldout"
    baseline_path = heldout_dir / "baseline.jsonl"
    skill_path = heldout_dir / "with_skill.jsonl"
    expected_heldout_ids = {str(item.instance_id) for item in heldout.instances}

    if baseline_path.is_file():
        baseline = _load_trajectory_condition(
            baseline_path,
            model=model,
            skill=None,
            expected_instance_ids=expected_heldout_ids,
        )
    else:
        status = _save_status(spec, status, stage="evaluating_heldout_baseline")
        try:
            baseline = run_condition(
                heldout.instances,
                heldout.task_type,
                model,
                judge_model,
                skill=None,
                max_workers=FROZEN_MAX_WORKERS,
                enable_web_search=False,
                execute_scripts=False,
            )
        except pilot_budget_guard.PilotBudgetStop as exc:
            _save_status(
                spec,
                status,
                stage="budget_stopped",
                failure={"kind": type(exc).__name__, "reason": str(exc)},
            )
            raise
        except Exception as exc:
            _save_status(
                spec,
                status,
                stage="failed",
                failure={"kind": type(exc).__name__, "reason": str(exc)},
            )
            raise
        _write_trajectories_atomic(baseline_path, baseline.trajectories)
        status = _save_status(spec, status, stage="heldout_baseline_ready")

    active = skill is not None and skill.status == SkillStatus.ACTIVE
    if active:
        if skill_path.is_file():
            skill_condition = _load_trajectory_condition(
                skill_path,
                model=model,
                skill=skill,
                expected_instance_ids=expected_heldout_ids,
            )
        else:
            status = _save_status(spec, status, stage="evaluating_skill")
            try:
                skill_condition = run_condition(
                    heldout.instances,
                    heldout.task_type,
                    model,
                    judge_model,
                    skill=skill,
                    max_workers=FROZEN_MAX_WORKERS,
                    enable_web_search=False,
                    execute_scripts=False,
                )
            except pilot_budget_guard.PilotBudgetStop as exc:
                _save_status(
                    spec,
                    status,
                    stage="budget_stopped",
                    failure={"kind": type(exc).__name__, "reason": str(exc)},
                )
                raise
            except Exception as exc:
                _save_status(
                    spec,
                    status,
                    stage="failed",
                    failure={"kind": type(exc).__name__, "reason": str(exc)},
                )
                raise
            _write_trajectories_atomic(skill_path, skill_condition.trajectories)
    else:
        # Released deployment semantics: a rejected/absent skill is the empty
        # intervention, so no second paid target rollout is generated.
        skill_condition = _condition_from_trajectories(
            model=model,
            skill=None,
            trajectories=list(baseline.trajectories),
        )

    keep = paired_analysis(baseline, skill_condition, drop_blank=False)
    drop = paired_analysis(baseline, skill_condition, drop_blank=True)
    budget_completion_warning = None
    try:
        pilot_budget_guard.record_balance("pilot_complete")
    except pilot_budget_guard.PilotBudgetStop as exc:
        # No paid work remains.  Preserve the scientifically complete result
        # even if the final balance audit endpoint is temporarily unavailable.
        budget_completion_warning = str(exc)
    result = {
        "schema_version": 1,
        "family_id": spec.family_id,
        "protocol_hash": protocol["hash"],
        "method_status": status.get("method_status"),
        "skill_id": skill.skill_id if skill else None,
        "skill_status": str(skill.status.value) if skill else None,
        "heldout_task_id": manifest["heldout_task_id"],
        "heldout_skill_executed": active,
        "primary": _paired_record(keep, drop_blank=False),
        "upstream_blank_drop_sensitivity": _paired_record(drop, drop_blank=True),
        "baseline_counts": status.get("baseline_counts"),
        "verification_baseline_counts": status.get(
            "verification_baseline_counts"
        ),
        "ceiling_limited": bool(status.get("ceiling_limited", False)),
        "budget": {
            "cap_cny": str(spec.budget_cny),
            "balance_stop_cny": FROZEN_BALANCE_STOP_CNY,
            "ledger": os.environ["SKILLGEN_BUDGET_LEDGER"],
            "initial_snapshot": {
                key: value
                for key, value in budget_snapshot.items()
                if key != "events"
            },
            "hard_limit_note": (
                "The between-unit guard observes official account balance. "
                "A provider-side hard cap additionally requires keeping account "
                "available balance at or below the approved pilot amount."
            ),
            "completion_balance_warning": budget_completion_warning,
        },
    }
    _atomic_json(spec.fold_root / "result.json", result)
    _save_status(spec, status, stage="complete", result_path=str((spec.fold_root / "result.json").resolve()))
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one frozen SkillGen SkillsBench family fold (dry-run by default)."
    )
    parser.add_argument("--family-id", required=True)
    parser.add_argument("--induction-dataset", required=True, type=Path)
    parser.add_argument("--verification-dataset", required=True, type=Path)
    parser.add_argument("--heldout-dataset", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--budget-cny", default="120")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a reviewed checkpoint; never retries an incomplete fold silently.",
    )
    parser.add_argument(
        _BUDGET_POLICY_AMENDMENT_FLAG,
        action="store_true",
        help=(
            "Explicitly authorize the one allowlisted v12 same-root amendment that "
            "adds the frozen 2 CNY balance stop policy."
        ),
    )
    parser.add_argument(
        "--expected-new-protocol-hash",
        help=(
            "Exact 64-hex dry-run protocol hash authorized for the v12 "
            "budget-policy amendment and all later resumes of that lineage."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    spec = FamilySpec(
        family_id=args.family_id,
        induction_dataset=args.induction_dataset,
        verification_dataset=args.verification_dataset,
        heldout_dataset=args.heldout_dataset,
        manifest_path=args.manifest,
        config_path=args.config,
        run_root=args.run_root,
        budget_cny=args.budget_cny,
        resume=args.resume,
        authorize_budget_policy_amendment=args.authorize_budget_policy_amendment,
        expected_new_protocol_hash=args.expected_new_protocol_hash,
    )
    result = run(spec, execute=args.execute)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
