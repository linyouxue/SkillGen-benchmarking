from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "scripts" / "run_skillsbench_task.py"
MODULE_SPEC = importlib.util.spec_from_file_location("run_skillsbench_task", RUNNER_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
runner = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = runner
MODULE_SPEC.loader.exec_module(runner)


def _write_dataset(path: Path, prefix: str, outcomes: list[bool]) -> None:
    payload = {
        "dataset_id": "demo",
        "task_name": "Demo task",
        "task_type": "binary",
        "instances": [
            {
                "instance_id": f"{prefix}-{index}",
                "input": "the same fixed SkillsBench task package",
                "metadata": {"fake_success": outcome},
            }
            for index, outcome in enumerate(outcomes)
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_spec(tmp_path: Path, task_id: str = "demo-task"):
    construction = tmp_path / "construction.json"
    sealed = tmp_path / "sealed.json"
    config = tmp_path / "config.yaml"
    _write_dataset(construction, "construction", [True, True])
    _write_dataset(sealed, "sealed", [True, False])
    config.write_text(
        """
models:
  default: fake/default
  baseline_agent: fake/base
  baseline_judge: fake/judge
pipeline:
  max_workers: 2
generation: {}
skill_output: {}
""".lstrip(),
        encoding="utf-8",
    )
    return runner.TaskSpec(
        task_id=task_id,
        construction_dataset=construction,
        sealed_test_dataset=sealed,
        config_path=config,
        run_root=tmp_path / "runs",
    )


class FakeRuntime:
    def __init__(self, pipeline_outcomes: list[bool], skill_status: str | None = None):
        self.pipeline_outcomes = pipeline_outcomes
        self.skill_status = skill_status
        self.pipeline_calls = 0
        self.condition_calls = 0
        self.loaded_skill = None
        self.paired_drop_flags: list[bool] = []

    def hooks(self):
        return runner.RuntimeHooks(
            load_dataset=self.load_dataset,
            run_pipeline=self.run_pipeline,
            run_condition=self.run_condition,
            load_baseline_condition=self.load_baseline_condition,
            paired_analysis=self.paired_analysis,
            write_trajectories=self.write_trajectories,
            load_skill=self.load_skill,
        )

    @staticmethod
    def load_dataset(path: str):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        instances = [
            SimpleNamespace(
                instance_id=item["instance_id"],
                input=item["input"],
                metadata=item.get("metadata", {}),
            )
            for item in raw["instances"]
        ]
        return SimpleNamespace(
            dataset_id=raw["dataset_id"],
            task_name=raw["task_name"],
            task_type=raw["task_type"],
            instances=instances,
            metadata={},
        )

    def run_pipeline(self, instances, task_type, **kwargs):
        del instances, task_type
        self.pipeline_calls += 1
        task_root = Path(kwargs["config_path"]).parent
        run_dir = task_root / "pipeline_runs" / "fake-run"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run_metadata.json").write_text(
            json.dumps(
                {
                    "dataset_id": kwargs["dataset_id"],
                    "dataset_metadata": kwargs["dataset_metadata"],
                }
            ),
            encoding="utf-8",
        )
        with (run_dir / "checkpoint_trajectories.jsonl").open("w", encoding="utf-8") as handle:
            for index, outcome in enumerate(self.pipeline_outcomes):
                handle.write(
                    json.dumps(
                        {
                            "trajectory_id": f"construction-{index}",
                            "instance_id": f"construction-{index}",
                            "success": outcome,
                        }
                    )
                    + "\n"
                )
        (run_dir / "checkpoint.json").write_text("{}", encoding="utf-8")

        if self.skill_status is None:
            return None

        self.loaded_skill = SimpleNamespace(
            skill_id="fake-skill",
            status=self.skill_status,
            body="generated body",
            scripts=[],
        )
        skill_repo = task_root / "skill_output" / "fake-skill-run"
        skill_repo.mkdir(parents=True, exist_ok=True)
        (skill_repo / "fake-skill.json").write_text(
            json.dumps({"skill_id": "fake-skill", "status": self.skill_status}),
            encoding="utf-8",
        )
        return self.loaded_skill

    def run_condition(
        self,
        instances,
        task_type,
        model,
        judge_model,
        skill,
        max_workers,
        **kwargs,
    ):
        del task_type, judge_model, max_workers, kwargs
        self.condition_calls += 1
        trajectories = [
            SimpleNamespace(
                trajectory_id=f"traj-{instance.instance_id}",
                instance_id=instance.instance_id,
                success=bool(instance.metadata["fake_success"]),
                final_output="nonblank",
                agent_config={"model": model},
                messages=[],
                score=None,
                error_summary=None,
                token_usage=None,
                latency=0.0,
                timestamp=None,
                metadata={},
            )
            for instance in instances
        ]
        return SimpleNamespace(
            model=model,
            with_skill=skill is not None,
            skill_id=getattr(skill, "skill_id", None),
            trajectories=trajectories,
        )

    @staticmethod
    def write_trajectories(path, trajectories):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for trajectory in trajectories:
                handle.write(
                    json.dumps(
                        {
                            "trajectory_id": trajectory.trajectory_id,
                            "instance_id": trajectory.instance_id,
                            "success": trajectory.success,
                            "final_output": trajectory.final_output,
                            "agent_config": trajectory.agent_config,
                        }
                    )
                    + "\n"
                )

    @staticmethod
    def load_baseline_condition(path, model, expected_instance_ids=None):
        trajectories = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            raw = json.loads(line)
            trajectories.append(SimpleNamespace(**raw))
        if expected_instance_ids is not None:
            assert {trajectory.instance_id for trajectory in trajectories} == expected_instance_ids
        return SimpleNamespace(
            model=model,
            with_skill=False,
            skill_id=None,
            trajectories=trajectories,
        )

    def paired_analysis(self, baseline, skill, drop_blank=True):
        self.paired_drop_flags.append(drop_blank)
        baseline_map = {item.instance_id: item for item in baseline.trajectories}
        skill_map = {item.instance_id: item for item in skill.trajectories}
        both_pass = both_fail = repair = regression = 0
        for instance_id, before in baseline_map.items():
            after = skill_map[instance_id]
            if before.success and after.success:
                both_pass += 1
            elif not before.success and not after.success:
                both_fail += 1
            elif not before.success and after.success:
                repair += 1
            else:
                regression += 1
        total = both_pass + both_fail + repair + regression
        baseline_pass = both_pass + regression
        skill_pass = both_pass + repair
        return SimpleNamespace(
            n_instances=total,
            baseline_acc=baseline_pass / total,
            skill_acc=skill_pass / total,
            repair=repair,
            regression=regression,
            repair_rate=repair / (repair + both_fail) if repair + both_fail else 0.0,
            regression_rate=(
                regression / (regression + both_pass) if regression + both_pass else 0.0
            ),
            net_gain=repair - regression,
            n_paired_raw=total,
            n_blank_baseline=0,
            n_blank_skill=0,
            n_blank_either=0,
            n_blank_both=0,
        )

    def load_skill(self, repo, skill_id=None):
        del repo
        assert skill_id == "fake-skill"
        return self.loaded_skill


def test_default_is_dry_plan_and_does_not_load_runtime(tmp_path, monkeypatch):
    spec = _make_spec(tmp_path)

    def fail_if_loaded():
        raise AssertionError("dry plan must not import/load paid runtime")

    monkeypatch.setattr(runner, "_load_runtime_hooks", fail_if_loaded)
    result = runner.run_task(spec)

    assert result["mode"] == "dry_run"
    assert result["paid_actions_executed"] is False
    assert not (spec.task_root / "status.json").exists()


def test_none_with_all_success_is_not_applicable_and_complete_resumes(tmp_path):
    spec = _make_spec(tmp_path)
    fake = FakeRuntime([True, True], skill_status=None)

    first = runner.run_task(spec, execute=True, hooks=fake.hooks())

    assert first["method_status"] == "not_applicable_no_failure"
    assert first["reason"] == "no_induction_failure"
    assert first["net_gain"] == 0
    assert first["skill_condition_executed"] is False
    assert first["skill_condition_reused_baseline"] is True
    assert first["blank_filter"]["drop_blank"] is False
    assert first["upstream_default_blank_drop_sensitivity"]["blank_filter"][
        "drop_blank"
    ] is True
    assert fake.paired_drop_flags == [False, True]
    assert fake.pipeline_calls == 1
    assert fake.condition_calls == 1  # sealed baseline exactly once
    status = json.loads((spec.task_root / "status.json").read_text(encoding="utf-8"))
    assert status["stage"] == "complete"
    assert status["method_status"] == "not_applicable_no_failure"
    assert status["protocol_hash"] == first["protocol_hash"]

    second = runner.run_task(spec, execute=True, hooks=fake.hooks())
    assert second == first
    assert fake.pipeline_calls == 1
    assert fake.condition_calls == 1


def test_no_failure_state_resumes_after_sealed_baseline_interruption(tmp_path):
    spec = _make_spec(tmp_path)
    interrupted = FakeRuntime([True, True], skill_status=None)

    def fail_sealed_once(*args, **kwargs):
        del args, kwargs
        interrupted.condition_calls += 1
        raise RuntimeError("simulated interruption before sealed baseline persisted")

    interrupted.run_condition = fail_sealed_once
    with pytest.raises(RuntimeError, match="simulated interruption"):
        runner.run_task(spec, execute=True, hooks=interrupted.hooks())

    state = json.loads((spec.task_root / "status.json").read_text(encoding="utf-8"))
    assert state["stage"] == "sealed_baseline_failed"
    assert state["method_status"] == "not_applicable_no_failure"
    assert state["failure"]["kind"] == "runtime_exception"

    resumed = FakeRuntime([True, True], skill_status=None)
    with pytest.raises(RuntimeError, match="--retry-paid"):
        runner.run_task(spec, execute=True, hooks=resumed.hooks())
    result = runner.run_task(
        replace(spec, allow_paid_retry=True),
        execute=True,
        hooks=resumed.hooks(),
    )

    assert result["method_status"] == "not_applicable_no_failure"
    assert result["net_gain"] == 0
    assert resumed.pipeline_calls == 0  # construction was not repeated
    assert resumed.condition_calls == 1


def test_none_with_failure_checkpoint_is_pipeline_error_not_no_failure(tmp_path):
    spec = _make_spec(tmp_path)
    fake = FakeRuntime([True, False], skill_status=None)

    result = runner.run_task(spec, execute=True, hooks=fake.hooks())

    assert result["method_status"] == "pipeline_error"
    assert "1 failure signal" in result["reason"]
    assert result["sealed_test_executed"] is False
    assert fake.condition_calls == 0
    status = json.loads((spec.task_root / "status.json").read_text(encoding="utf-8"))
    assert status["stage"] == "failed"
    assert status["baseline_counts"]["n_failures"] == 1


def test_persisted_baseline_recovers_without_paid_retry_or_rerun(tmp_path):
    spec = _make_spec(tmp_path)
    manifest, protocol_hash = runner.build_protocol(spec)
    status_path, state = runner._load_or_initialize_status(
        spec, manifest, protocol_hash
    )
    runner._save_status(
        status_path,
        state,
        stage="evaluating_baseline",
        method_status="not_applicable_no_failure",
        baseline_counts={"valid": True, "n_successes": 2, "n_failures": 0},
    )
    fake = FakeRuntime([True, True], skill_status=None)
    sealed = fake.load_dataset(str(spec.sealed_test_dataset))
    baseline = fake.run_condition(
        sealed.instances,
        sealed.task_type,
        "fake/base",
        "fake/judge",
        skill=None,
        max_workers=1,
    )
    persisted = spec.task_root / "sealed" / "baseline.jsonl"
    fake.write_trajectories(persisted, baseline.trajectories)
    fake.condition_calls = 0

    result = runner.run_task(spec, execute=True, hooks=fake.hooks())

    assert result["method_status"] == "not_applicable_no_failure"
    assert fake.pipeline_calls == 0
    assert fake.condition_calls == 0


def test_complete_all_success_construction_checkpoint_recovers_after_hard_kill(
    tmp_path,
):
    spec = _make_spec(tmp_path)
    manifest, protocol_hash = runner.build_protocol(spec)
    status_path, state = runner._load_or_initialize_status(
        spec, manifest, protocol_hash
    )
    runner._save_status(status_path, state, stage="constructing")
    run_dir = spec.task_root / "pipeline_runs" / "interrupted-all-success"
    run_dir.mkdir(parents=True)
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "dataset_id": spec.task_id,
                "dataset_metadata": {"protocol_hash": protocol_hash},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "checkpoint_trajectories.jsonl").write_text(
        "\n".join(
            json.dumps({"instance_id": f"construction-{index}", "success": True})
            for index in range(2)
        )
        + "\n",
        encoding="utf-8",
    )
    fake = FakeRuntime([True, True], skill_status=None)

    result = runner.run_task(spec, execute=True, hooks=fake.hooks())

    assert result["method_status"] == "not_applicable_no_failure"
    assert fake.pipeline_calls == 0
    assert fake.condition_calls == 1


def test_active_skill_executes_one_sealed_treatment(tmp_path):
    spec = _make_spec(tmp_path)
    fake = FakeRuntime([False, True], skill_status="active")

    result = runner.run_task(spec, execute=True, hooks=fake.hooks())

    assert result["method_status"] == "active"
    assert result["deployed_intervention"] == "generated_skill"
    assert result["skill_condition_executed"] is True
    assert fake.pipeline_calls == 1
    assert fake.condition_calls == 2
    assert (spec.task_root / "sealed" / "with_skill.jsonl").is_file()

    status_path = spec.task_root / "status.json"
    state = json.loads(status_path.read_text(encoding="utf-8"))
    runner._save_status(
        status_path,
        state,
        stage="evaluating",
        result_path=None,
        sealed_skill_path=None,
    )
    fake.condition_calls = 0

    resumed = runner.run_task(spec, execute=True, hooks=fake.hooks())

    assert resumed["method_status"] == "active"
    assert fake.pipeline_calls == 1
    assert fake.condition_calls == 0


def test_deprecated_skill_uses_empty_intervention_without_second_rollout(tmp_path):
    spec = _make_spec(tmp_path)
    fake = FakeRuntime([False, True], skill_status="deprecated")

    result = runner.run_task(spec, execute=True, hooks=fake.hooks())

    assert result["method_status"] == "deprecated"
    assert result["deployed_intervention"] == "empty"
    assert result["skill_generated"] is True
    assert result["skill_condition_executed"] is False
    assert result["net_gain"] == 0
    assert fake.pipeline_calls == 1
    assert fake.condition_calls == 1  # baseline only
