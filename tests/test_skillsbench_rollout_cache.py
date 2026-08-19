from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from benchmarks.skillsbench_rollout_cache import (
    ATTEMPT_MANIFEST,
    RolloutCacheRequest,
    SkillsBenchRolloutCacheError,
    attempt_directory_name,
    cache_entry_path,
    load_cached_trajectory,
    manifested_attempts,
    read_attempt_receipt,
    skill_content_digest,
    slot_lock,
    write_attempt_manifest,
    write_attempt_receipt,
    write_cached_trajectory,
)
from models import SkillItem, TaskInstance, Trajectory


ADAPTER_SCHEMA = "skillsbench-skillgen-v1"
BENCHFLOW_VERSION = "0.6.7"
TASK_DIGEST = "sha256:" + "a" * 64


def make_instance(instance_id: str = "family::task::induction::r000") -> TaskInstance:
    return TaskInstance(instance_id=instance_id, input="task", metadata={})


def make_skill(body: str = "# Procedure\nDo it.") -> SkillItem:
    return SkillItem(
        skill_id="skill-1",
        body=body,
        contextual_abstract="Reusable method",
        scripts=["solve.py"],
        requirements=["numpy"],
        references=["reference.md"],
        reference_docs=[
            {"name": "reference.md", "summary": "facts", "content": "full text"}
        ],
        task_name="family",
        verification_history=[{"round": 1, "net_gain": 2}],
    )


def make_request(
    *,
    instance: TaskInstance | None = None,
    model: str = "deepseek/deepseek-v4-flash",
    agent: str = "openhands",
    skill: SkillItem | None = None,
) -> RolloutCacheRequest:
    return RolloutCacheRequest.build(
        instance=instance or make_instance(),
        task_digest=TASK_DIGEST,
        model=model,
        agent=agent,
        sandbox="docker",
        skill=skill,
        adapter_schema_version=ADAPTER_SCHEMA,
        benchflow_version=BENCHFLOW_VERSION,
    )


def make_trajectory(
    request: RolloutCacheRequest, *, score: float = 1.0
) -> Trajectory:
    payload = request.payload
    return Trajectory(
        trajectory_id="trajectory-original",
        instance_id=payload["instance_id"],
        agent_config={
            "model": payload["model"],
            "inference_model": payload["model"],
            "agent": payload["agent"],
            "skill_id": payload["skill_id"],
            "skill_mode": payload["condition"],
        },
        messages=[{"role": "assistant", "content": "done"}],
        final_output="done",
        success=score == 1.0,
        score=score,
        token_usage={"total_tokens": 123},
        latency=2.5,
        timestamp="2026-08-15T00:00:00Z",
        metadata={
            "benchmark": "skillsbench",
            "adapter_schema_version": ADAPTER_SCHEMA,
            "task_digest": TASK_DIGEST,
            "skill_mode": payload["condition"],
            "benchflow_returncode": 0,
            "real_run_checks": {"has_verifier_reward": True},
        },
    )


class SkillsBenchRolloutCacheTests(unittest.TestCase):
    def test_key_covers_slot_task_model_agent_condition_and_full_skill(self) -> None:
        base = make_request()
        variants = [
            make_request(instance=make_instance("family::task::induction::r001")),
            RolloutCacheRequest.build(
                instance=make_instance(),
                task_digest="sha256:" + "b" * 64,
                model=base.payload["model"],
                agent=base.payload["agent"],
                sandbox="docker",
                skill=None,
                adapter_schema_version=ADAPTER_SCHEMA,
                benchflow_version=BENCHFLOW_VERSION,
            ),
            make_request(model="deepseek/deepseek-v4"),
            make_request(agent="codex"),
            make_request(skill=make_skill()),
        ]
        self.assertEqual(len(variants), len({item.key for item in variants}))
        self.assertTrue(all(item.key != base.key for item in variants))

        skill = make_skill()
        changed_reference = replace(
            skill,
            reference_docs=[
                {"name": "reference.md", "summary": "facts", "content": "changed"}
            ],
        )
        self.assertNotEqual(
            skill_content_digest(skill), skill_content_digest(changed_reference)
        )

    def test_valid_official_failure_is_cached_and_original_trajectory_returns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = make_request()
            original = make_trajectory(request, score=0.4)
            path = write_cached_trajectory(
                root,
                request,
                original,
                source={"kind": "benchflow", "run_root": "/tmp/run"},
            )
            self.assertEqual(cache_entry_path(root, request), path)
            loaded = load_cached_trajectory(root, request)
            self.assertIsNotNone(loaded)
            self.assertEqual(original, loaded)
            self.assertEqual("trajectory-original", loaded.trajectory_id)

    def test_invalid_or_infrastructure_result_is_never_cached(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = make_request()

            missing_reward = make_trajectory(request)
            missing_reward.score = None
            with self.assertRaisesRegex(
                SkillsBenchRolloutCacheError, "official numeric reward"
            ):
                write_cached_trajectory(
                    root, request, missing_reward, source={"kind": "test"}
                )

            verifier_error = make_trajectory(request)
            verifier_error.metadata["verifier_error"] = "crashed"
            with self.assertRaisesRegex(SkillsBenchRolloutCacheError, "error"):
                write_cached_trajectory(
                    root, request, verifier_error, source={"kind": "test"}
                )

            nonzero = make_trajectory(request)
            nonzero.metadata["benchflow_returncode"] = 1
            with self.assertRaisesRegex(
                SkillsBenchRolloutCacheError, "subprocess was not successful"
            ):
                write_cached_trajectory(root, request, nonzero, source={"kind": "test"})
            self.assertFalse(cache_entry_path(root, request).exists())

    def test_corrupt_entry_fails_closed_instead_of_becoming_a_paid_miss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = make_request()
            write_cached_trajectory(
                root, request, make_trajectory(request), source={"kind": "test"}
            )
            path = cache_entry_path(root, request)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["trajectory"]["score"] = 0.0
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                SkillsBenchRolloutCacheError, "trajectory digest mismatch"
            ):
                load_cached_trajectory(root, request)

    def test_attempt_manifest_and_receipt_support_crash_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = make_request()
            run_root = root / attempt_directory_name("slot", request, "attempt")
            run_root.mkdir()
            manifest = write_attempt_manifest(run_root, request)
            self.assertEqual(run_root / ATTEMPT_MANIFEST, manifest)
            self.assertEqual([run_root], manifested_attempts(root, request))
            self.assertIsNone(read_attempt_receipt(run_root))
            write_attempt_receipt(run_root, process_returncode=0, elapsed=3.25)
            self.assertEqual((0, 3.25), read_attempt_receipt(run_root))

    def test_slot_lock_prevents_a_concurrent_duplicate_charge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = make_request()
            with slot_lock(root, request) as lock_path:
                self.assertTrue(lock_path.is_file())
                with self.assertRaisesRegex(
                    SkillsBenchRolloutCacheError, "locked"
                ):
                    with slot_lock(root, request):
                        self.fail("a duplicate process acquired the same slot")
            self.assertFalse(lock_path.exists())

    def test_slot_lock_is_left_stale_if_process_dies_before_finally(self) -> None:
        """Document the fail-closed contract without actually killing a process."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = make_request()
            context = slot_lock(root, request)
            lock_path = context.__enter__()
            self.assertTrue(lock_path.exists())
            # A real process death skips __exit__.  Suppress cleanup here to
            # emulate the next process observing that persistent file.
            with patch.object(Path, "unlink", side_effect=PermissionError):
                with self.assertRaises(PermissionError):
                    context.__exit__(None, None, None)
            with self.assertRaisesRegex(SkillsBenchRolloutCacheError, "locked"):
                with slot_lock(root, request):
                    self.fail("stale lock was ignored")
            lock_path.unlink()


if __name__ == "__main__":
    unittest.main()
