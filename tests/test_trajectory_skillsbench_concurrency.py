from __future__ import annotations

import threading
import time

import pytest

import trajectory as trajectory_module
from models import TaskInstance, TaskType, Trajectory


def _instances(count: int) -> list[TaskInstance]:
    return [
        TaskInstance(
            instance_id=f"slot-{index}",
            input={},
            metadata={"benchmark": "skillsbench"},
        )
        for index in range(count)
    ]


def _trajectory(instance: TaskInstance) -> Trajectory:
    return Trajectory(
        trajectory_id=f"trajectory-{instance.instance_id}",
        instance_id=instance.instance_id,
        agent_config={},
        messages=[],
        final_output="official result",
        success=True,
        score=1.0,
    )


def test_skillsbench_bounded_collector_returns_frozen_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances = _instances(8)
    lock = threading.Lock()
    active = 0
    peak = 0
    callback_ids: list[str] = []

    def fake_run(instance, *_args):
        nonlocal active, peak
        index = int(instance.instance_id.rsplit("-", 1)[1])
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            # Reverse completion order inside each rolling window.
            time.sleep((3 - (index % 3)) * 0.01)
            return _trajectory(instance)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(trajectory_module, "_run_and_eval", fake_run)
    results = trajectory_module.collect_trajectories(
        instances,
        TaskType.SCORED,
        max_workers=3,
        on_trajectory=lambda item: callback_ids.append(str(item.instance_id)),
    )

    assert peak == 3
    assert [item.instance_id for item in results] == [
        instance.instance_id for instance in instances
    ]
    assert set(callback_ids) == {instance.instance_id for instance in instances}
    assert len(callback_ids) == len(set(callback_ids))
    assert callback_ids != [instance.instance_id for instance in instances]


def test_skillsbench_first_error_stops_submission_but_drains_inflight_successes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances = _instances(6)
    all_initial_started = threading.Event()
    failure_released = threading.Event()
    lock = threading.Lock()
    started: list[str] = []
    callback_ids: list[str] = []

    def fake_run(instance, *_args):
        index = int(instance.instance_id.rsplit("-", 1)[1])
        with lock:
            started.append(instance.instance_id)
            if len(started) == 3:
                all_initial_started.set()
        assert all_initial_started.wait(timeout=2)
        if index == 0:
            failure_released.set()
            raise RuntimeError("simulated infrastructure failure")
        assert failure_released.wait(timeout=2)
        time.sleep(0.05)
        return _trajectory(instance)

    monkeypatch.setattr(trajectory_module, "_run_and_eval", fake_run)
    with pytest.raises(RuntimeError, match="infrastructure failure"):
        trajectory_module.collect_trajectories(
            instances,
            TaskType.SCORED,
            max_workers=3,
            on_trajectory=lambda item: callback_ids.append(str(item.instance_id)),
        )

    assert set(started) == {"slot-0", "slot-1", "slot-2"}
    assert set(callback_ids) == {"slot-1", "slot-2"}
    assert not ({"slot-3", "slot-4", "slot-5"} & set(started))

