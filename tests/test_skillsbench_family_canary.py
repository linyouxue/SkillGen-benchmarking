from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from benchmarks.skillsbench_adapter import _cached_task_digest, task_package_digest
from benchmarks.skillsbench_rollout_cache import write_cached_trajectory
from models import Trajectory
from scripts import run_skillsbench_family_canary as canary


def _fixture(tmp_path: Path, *, model: str = canary.EXPECTED_AGENT_MODEL):
    family_id = "family-test"
    task_id = "shock-analysis-demand"
    instance_id = f"{family_id}::{task_id}::induction::r000"
    task_dir = tmp_path / "tasks" / task_id
    task_dir.mkdir(parents=True)
    (task_dir / "environment").mkdir()
    (task_dir / "verifier").mkdir()
    (task_dir / "task.md").write_text(
        "---\nname: shock-analysis-demand\n---\nDo the spreadsheet task.\n",
        encoding="utf-8",
    )
    (task_dir / "environment" / "Dockerfile").write_text(
        "FROM scratch\n", encoding="utf-8"
    )
    (task_dir / "verifier" / "test.sh").write_text(
        "#!/bin/sh\nexit 0\n", encoding="utf-8"
    )
    _cached_task_digest.cache_clear()
    digest = task_package_digest(task_dir)
    jobs_root = tmp_path / "jobs"
    bench = tmp_path / "bench"
    bench.write_text("offline test executable", encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "protocol": "skillgen_skillsbench_task_disjoint_family_fold_v1",
        "family_id": family_id,
        "heldout_task_id": "heldout-task",
        "source_task_ids": [task_id],
        "induction_allocations": {task_id: 1},
        "task_package_digests": {task_id: digest, "heldout-task": "sha256:held"},
        "agent": canary.EXPECTED_AGENT,
        "sandbox": canary.EXPECTED_SANDBOX,
        "bench_executable": str(bench.resolve()),
        "jobs_root": str(jobs_root.resolve()),
        "counts": {"induction": 1, "verification": 1, "heldout": 1},
    }
    dataset = {
        "metadata": {
            "benchmark": "skillsbench",
            "protocol": manifest["protocol"],
            "family_id": family_id,
            "split": "induction",
        },
        "instances": [
            {
                "instance_id": instance_id,
                "input": "Do the spreadsheet task.",
                "ground_truth": None,
                "metadata": {
                    "benchmark": "skillsbench",
                    "skillsbench_adapter_schema": canary.ADAPTER_SCHEMA_VERSION,
                    "skillsbench_family_id": family_id,
                    "skillsbench_family_split": "induction",
                    "skillsbench_heldout_task_id": "heldout-task",
                    "skillsbench_task_id": task_id,
                    "skillsbench_task_dir": str(task_dir.resolve()),
                    "skillsbench_task_digest": digest,
                    "skillsbench_agent": canary.EXPECTED_AGENT,
                    "skillsbench_sandbox": canary.EXPECTED_SANDBOX,
                    "skillsbench_jobs_root": str(jobs_root.resolve()),
                    "skillsbench_bench_executable": str(bench.resolve()),
                    "official_task_skills_visible": False,
                },
            }
        ],
    }
    config = {
        "models": {
            "baseline_agent": model,
            "baseline_judge": canary.EXPECTED_MODEL,
        },
        "experiment": {
            "provider": canary.EXPECTED_PROVIDER,
            "base_url": canary.EXPECTED_BASE_URL,
        },
    }
    dataset_path = tmp_path / "induction.json"
    manifest_path = tmp_path / "manifest.json"
    config_path = tmp_path / "config.yaml"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    spec = canary.CanarySpec(
        induction_dataset=dataset_path,
        manifest_path=manifest_path,
        config_path=config_path,
        instance_id=instance_id,
        run_root=tmp_path / "run",
    )
    return spec, task_dir


def _trajectory(prepared: canary.PreparedCanary, reward: int) -> Trajectory:
    return Trajectory(
        trajectory_id=f"canary-{reward}",
        instance_id=prepared.instance.instance_id,
        agent_config={
            "inference_model": prepared.model,
            "agent": prepared.agent,
            "skill_mode": "no-skill",
            "skill_id": None,
        },
        messages=[],
        final_output="",
        success=reward == 1,
        score=float(reward),
        metadata={
            "benchmark": "skillsbench",
            "adapter_schema_version": canary.ADAPTER_SCHEMA_VERSION,
            "task_digest": prepared.task_digest,
            "skill_mode": "no-skill",
            "benchflow_returncode": 0,
            "real_run_checks": {"has_verifier_reward": True},
            "agent_exception": None,
            "verifier_error": None,
        },
    )


def test_dry_run_is_read_only_and_never_invokes_agent(tmp_path, monkeypatch) -> None:
    spec, _ = _fixture(tmp_path)
    monkeypatch.setattr(canary, "load_cached_trajectory", lambda *_args: None)
    monkeypatch.setattr(
        canary,
        "run_skillsbench_agent",
        lambda *_args: pytest.fail("dry-run invoked the agent"),
    )

    plan = canary.run_canary(spec)

    assert plan["mode"] == "dry-run"
    assert plan["cache_hit_before_execution"] is False
    assert plan["agent_runner_invocations"] == 0
    assert plan["api_calls_made"] is False
    assert not (spec.run_root / "family-test").exists()


def test_prepare_rejects_non_dind_id_and_wrong_model(tmp_path) -> None:
    spec, _ = _fixture(tmp_path)
    wrong_id = canary.CanarySpec(
        **{**spec.__dict__, "instance_id": "family-test::heldout-task::heldout::r000"}
    )
    with pytest.raises(ValueError, match="not in the frozen D_ind"):
        canary.prepare_canary(wrong_id)

    wrong_model_root = tmp_path / "wrong-model"
    wrong_model_root.mkdir()
    wrong_spec, _ = _fixture(
        wrong_model_root, model="deepseek/deepseek-chat"
    )
    with pytest.raises(ValueError, match="baseline_agent"):
        canary.prepare_canary(wrong_spec)


def test_prepare_detects_task_package_digest_drift(tmp_path) -> None:
    spec, task_dir = _fixture(tmp_path)
    (task_dir / "task.md").write_text(
        "---\nname: shock-analysis-demand\n---\nChanged after freeze.\n",
        encoding="utf-8",
    )
    _cached_task_digest.cache_clear()

    with pytest.raises(Exception, match="changed after dataset preparation"):
        canary.prepare_canary(spec)


def test_cache_hit_reward_zero_has_no_key_budget_or_agent_call(
    tmp_path, monkeypatch
) -> None:
    spec, _ = _fixture(tmp_path)
    prepared = canary.prepare_canary(spec)
    trajectory = _trajectory(prepared, 0)
    write_cached_trajectory(
        prepared.jobs_root,
        prepared.request,
        trajectory,
        source={"kind": "offline_test_cache"},
    )
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("SKILLGEN_DEEPSEEK_BUDGET_CNY", raising=False)
    monkeypatch.setattr(
        canary,
        "run_skillsbench_agent",
        lambda *_args: pytest.fail("cache hit invoked the agent"),
    )

    receipt = canary.run_canary(spec, execute=True)

    assert receipt["infrastructure_success"] is True
    assert receipt["official_reward"] == 0.0
    assert receipt["task_success"] is False
    assert receipt["agent_runner_invocations"] == 0
    assert receipt["new_benchflow_attempts"] == 0
    assert receipt["api_calls_made"] is False
    assert "SKILLGEN_DEEPSEEK_BUDGET_CNY" not in __import__("os").environ


@pytest.mark.parametrize("reward", [0, 1])
def test_cache_miss_runs_exactly_one_no_skill_slot_and_accepts_both_rewards(
    tmp_path, monkeypatch, reward
) -> None:
    spec, _ = _fixture(tmp_path)
    prepared = canary.prepare_canary(spec)
    calls = []

    def fake_runner(instance, skill, config):
        import os

        calls.append(instance.instance_id)
        assert skill is None
        assert config.model == canary.EXPECTED_AGENT_MODEL
        assert os.environ["SKILLGEN_DEEPSEEK_BUDGET_CNY"] == "120"
        assert os.environ["SKILLGEN_AGENT_ROLLOUT_RESERVE_CNY"] == "30"
        assert Path(os.environ["SKILLGEN_BUDGET_LEDGER"]) == (
            spec.run_root / "family-test" / "budget_ledger.json"
        ).resolve()
        trajectory = _trajectory(prepared, reward)
        write_cached_trajectory(
            prepared.jobs_root,
            prepared.request,
            trajectory,
            source={"kind": "benchflow"},
        )
        return trajectory

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-unit-test-placeholder")
    monkeypatch.delenv("SKILLGEN_DEEPSEEK_BUDGET_CNY", raising=False)
    monkeypatch.setattr(canary, "_assert_announced_offpeak", lambda: None)
    monkeypatch.setattr(canary, "run_skillsbench_agent", fake_runner)

    receipt = canary.run_canary(spec, execute=True)

    assert calls == [spec.instance_id]
    assert receipt["infrastructure_success"] is True
    assert receipt["official_reward"] == float(reward)
    assert receipt["task_success"] is (reward == 1)
    assert receipt["agent_runner_invocations"] == 1
    assert receipt["new_benchflow_attempts"] == 1
    assert receipt["api_calls_made"] is True
    assert "sk-unit-test-placeholder" not in prepared.receipt_path.read_text()
    assert "SKILLGEN_DEEPSEEK_BUDGET_CNY" not in __import__("os").environ


def test_failed_runner_writes_redacted_receipt_and_requires_review(
    tmp_path, monkeypatch
) -> None:
    spec, _ = _fixture(tmp_path)
    secret = "sk-unit-test-secret-value"
    calls = 0

    def fail_runner(*_args):
        nonlocal calls
        calls += 1
        raise RuntimeError(f"provider echoed {secret}")

    monkeypatch.setenv("DEEPSEEK_API_KEY", secret)
    monkeypatch.setattr(canary, "_assert_announced_offpeak", lambda: None)
    monkeypatch.setattr(canary, "run_skillsbench_agent", fail_runner)

    with pytest.raises(RuntimeError, match="provider echoed"):
        canary.run_canary(spec, execute=True)

    prepared = canary.prepare_canary(spec)
    receipt_text = prepared.receipt_path.read_text(encoding="utf-8")
    receipt = json.loads(receipt_text)
    assert secret not in receipt_text
    assert "[REDACTED]" in receipt["error"]["reason"]
    assert receipt["infrastructure_success"] is False
    assert receipt["agent_runner_invocations"] == 1

    with pytest.raises(RuntimeError, match="failed canary receipt"):
        canary.run_canary(spec, execute=True)
    assert calls == 1
