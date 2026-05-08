"""Utilities for persisting trajectories, clusters, and run artifacts."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from models import Trajectory


def make_run_dir(root: str = "./artifacts/runs") -> Path:
    """Create a timestamped run directory."""
    run_id = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    run_dir = Path(root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(path: str | Path, payload: Any) -> Path:
    """Write a JSON artifact with stable formatting."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, indent=2, ensure_ascii=True)
        f.write("\n")
    return output_path


def _to_jsonable(value: Any) -> Any:
    """Recursively coerce numpy/scalar-like values into plain JSON types."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]

    # Handle numpy / pandas scalar-likes without importing heavy deps here.
    if hasattr(value, "item") and callable(value.item):
        try:
            return _to_jsonable(value.item())
        except Exception:
            pass

    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return _to_jsonable(value.tolist())
        except Exception:
            pass

    return str(value)


def trajectory_to_record(traj: Trajectory) -> dict[str, Any]:
    """Serialize a trajectory and make the inference model explicit."""
    record = _to_jsonable(asdict(traj))
    agent_cfg = dict(record.get("agent_config") or {})
    inference_model = agent_cfg.get("inference_model") or agent_cfg.get("model")
    judge_model = agent_cfg.get("judge_model")
    record["agent_config"] = agent_cfg
    record["inference_model"] = inference_model
    record["judge_model"] = judge_model
    return record


def write_trajectories(path: str | Path, trajectories: list[Trajectory]) -> Path:
    """Write trajectories to JSONL for easy inspection and streaming."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for traj in trajectories:
            f.write(json.dumps(_to_jsonable(trajectory_to_record(traj)), ensure_ascii=True))
            f.write("\n")
    return output_path


# Checkpoint helpers

_TRAJECTORIES_FILE = "checkpoint_trajectories.jsonl"
_PROGRESS_FILE = "checkpoint.json"


def _trajectory_from_dict(d: dict) -> Trajectory:
    """Reconstruct a Trajectory from a serialized dict."""
    return Trajectory(
        trajectory_id=d["trajectory_id"],
        instance_id=d["instance_id"],
        agent_config=d.get("agent_config", {}),
        messages=d.get("messages", []),
        final_output=d.get("final_output"),
        success=d.get("success"),
        score=d.get("score"),
        error_summary=d.get("error_summary"),
        token_usage=d.get("token_usage"),
        latency=d.get("latency"),
        timestamp=d.get("timestamp"),
        metadata=d.get("metadata", {}),
    )


def save_trajectories(run_dir: Path, trajectories: list[Trajectory]) -> None:
    """Persist all baseline trajectories to disk for checkpoint/resume."""
    path = run_dir / _TRAJECTORIES_FILE
    with path.open("w", encoding="utf-8") as f:
        for traj in trajectories:
            f.write(json.dumps(_to_jsonable(asdict(traj)), ensure_ascii=True))
            f.write("\n")


def load_trajectories(run_dir: Path) -> list[Trajectory]:
    """Load trajectories saved by save_trajectories."""
    path = run_dir / _TRAJECTORIES_FILE
    trajs: list[Trajectory] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                trajs.append(_trajectory_from_dict(json.loads(line)))
    return trajs


def save_progress(run_dir: Path, completed_stages: list[str],
                  total_stages: int, stage: str = "pipeline") -> None:
    """Write/update the checkpoint file with the latest completed stages."""
    path = run_dir / _PROGRESS_FILE
    payload = {
        "stage": stage,
        "total_stages": total_stages,
        "completed_stages": completed_stages,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_progress(run_dir: Path) -> dict | None:
    """Return the checkpoint dict, or None if not present."""
    path = run_dir / _PROGRESS_FILE
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def checkpoint_exists(run_dir: Path) -> bool:
    """Return True if the baseline trajectory checkpoint is present in run_dir."""
    return (
        (run_dir / _TRAJECTORIES_FILE).exists()
        and (run_dir / _PROGRESS_FILE).exists()
    )
