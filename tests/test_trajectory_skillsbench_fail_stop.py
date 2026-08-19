from __future__ import annotations

import unittest
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import trajectory
from benchmarks.skillsbench_adapter import SkillsBenchInfrastructureError
from models import TaskInstance, TaskType, Trajectory


class SkillsBenchCollectionFailStopTests(unittest.TestCase):
    def _instance(self, suffix: str, *, benchmark: str = "skillsbench") -> TaskInstance:
        return TaskInstance(
            instance_id=f"demo::{suffix}",
            input="task",
            metadata={"benchmark": benchmark},
        )

    def test_infrastructure_error_stops_refill_and_drains_started_slots(self) -> None:
        calls: list[str] = []
        callbacks: list[str] = []
        lock = threading.Lock()
        initial_started = threading.Event()
        failure_released = threading.Event()

        def fail_first(instance, *_args):
            with lock:
                calls.append(instance.instance_id)
                if len(calls) == 3:
                    initial_started.set()
            self.assertTrue(initial_started.wait(timeout=2))
            if instance.instance_id == "demo::0":
                failure_released.set()
                raise SkillsBenchInfrastructureError("proxy unavailable")
            self.assertTrue(failure_released.wait(timeout=2))
            time.sleep(0.05)
            return Trajectory(
                trajectory_id=f"traj-{instance.instance_id}",
                instance_id=instance.instance_id,
                agent_config={},
                messages=[],
                final_output="official result",
                success=True,
                score=1.0,
            )

        with patch("trajectory._run_and_eval", side_effect=fail_first):
            with self.assertRaises(SkillsBenchInfrastructureError):
                trajectory.collect_trajectories(
                    [
                        self._instance("0"),
                        self._instance("1"),
                        self._instance("2"),
                        self._instance("3"),
                    ],
                    TaskType.SCORED,
                    config=SimpleNamespace(),
                    max_workers=3,
                    on_trajectory=lambda item: callbacks.append(item.instance_id),
                )
        self.assertEqual({"demo::0", "demo::1", "demo::2"}, set(calls))
        self.assertNotIn("demo::3", calls)
        self.assertEqual({"demo::1", "demo::2"}, set(callbacks))

    def test_valid_verifier_failure_does_not_stop_later_slots(self) -> None:
        calls: list[str] = []

        def valid_failure(instance, *_args):
            calls.append(instance.instance_id)
            return Trajectory(
                trajectory_id=f"traj-{instance.instance_id}",
                instance_id=instance.instance_id,
                agent_config={},
                messages=[],
                final_output="",
                success=False,
                score=0.0,
            )

        with patch("trajectory._run_and_eval", side_effect=valid_failure):
            results = trajectory.collect_trajectories(
                [self._instance("0"), self._instance("1")],
                TaskType.SCORED,
                config=SimpleNamespace(),
                max_workers=16,
            )
        self.assertEqual(["demo::0", "demo::1"], calls)
        self.assertEqual([False, False], [result.success for result in results])


if __name__ == "__main__":
    unittest.main()
