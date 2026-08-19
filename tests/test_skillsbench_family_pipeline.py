from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Callable

import pytest

import pipeline
from models import (
    CandidateSkill,
    EffectivenessResult,
    SkillAnalysis,
    SkillItem,
    SkillStatus,
    TaskInstance,
    TaskType,
    Trajectory,
    VerificationFeedback,
)


def _config(root: Path) -> dict:
    return {
        "llm": {"temperature": 0.0, "max_tokens_generation": 64},
        "models": {"default": "mock-model"},
        "clustering": {},
        "induction": {},
        "generation": {
            "generate_scripts": False,
            "generate_references": False,
            "use_web_search": False,
            "candidate_output_dir": str(root / "candidates"),
        },
        "verification": {
            "min_net_gain_abs": 2,
            "min_net_gain_rel": 0.0,
        },
        "verification_analysis": {},
        "router": {"enabled": False},
        "embedding": {"model": "mock-embedding"},
        "pipeline": {
            "artifact_root": str(root / "runs"),
            "max_refine_rounds": 8,
            "max_workers": 3,
            "baseline_runs_per_instance": 1,
        },
        "skill_output": {"path": str(root / "skills")},
    }


def _instances(prefix: str) -> list[TaskInstance]:
    return [TaskInstance(f"{prefix}-{index}", {}) for index in range(10)]


def _trajectory(instance: TaskInstance) -> Trajectory:
    index = int(instance.instance_id.rsplit("-", 1)[1])
    return Trajectory(
        trajectory_id=f"trajectory-{instance.instance_id}",
        instance_id=instance.instance_id,
        agent_config={},
        messages=[],
        final_output="mock output",
        success=index % 2 == 1,
    )


def _analysis() -> SkillAnalysis:
    return SkillAnalysis(
        analysis_id="analysis-1",
        dataset_id="dataset-1",
        task_name="family-source",
        task_type=TaskType.SCORED.value,
        contextual_abstract="mock analysis",
        n_total=10,
        n_failures=5,
        n_successes=5,
    )


def _feedback(net_gain: int, round_idx: int) -> VerificationFeedback:
    effectiveness = EffectivenessResult(
        passed=net_gain >= 2,
        n_target=5,
        n_boundary=5,
        paired_n=10,
        baseline_acc=0.5,
        skill_acc=(5 + net_gain) / 10,
        repair_count=max(net_gain, 0),
        regression_count=max(-net_gain, 0),
        repair_rate=0.0,
        regression_rate=0.0,
        net_gain=net_gain,
        target_repair_count=max(net_gain, 0),
        target_fail_count=5,
        success_guard_regression_count=max(-net_gain, 0),
        success_guard_pass_count=5,
        repaired_ids=[],
        regression_ids=[],
        failed_ids_after_skill=[],
        diagnostic_summary=f"net_gain={net_gain}",
        cases=[],
    )
    return VerificationFeedback(
        effectiveness=effectiveness,
        round_idx=round_idx,
        revision_guidance=f"guidance-{round_idx}",
    )


def _finalize(candidate: CandidateSkill, *_args, **_kwargs) -> SkillItem:
    return SkillItem(
        skill_id="skill-1",
        body=candidate.body,
        contextual_abstract=candidate.contextual_abstract,
        analysis_id=candidate.analysis_id,
    )


def _install_common_mocks(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    *,
    collect: Callable,
    induce: Callable,
    generate: Callable,
    refine: Callable,
    verify: Callable,
) -> None:
    monkeypatch.setattr(pipeline, "load_config", lambda _path: _config(root))
    monkeypatch.setattr(pipeline, "collect_trajectories", collect)
    monkeypatch.setattr(pipeline, "run_induction", induce)
    monkeypatch.setattr(pipeline, "generate_skill", generate)
    monkeypatch.setattr(pipeline, "refine_skill", refine)
    monkeypatch.setattr(pipeline, "run_verification", verify)
    monkeypatch.setattr(pipeline, "finalize_skill", _finalize)
    monkeypatch.setattr(pipeline, "save_skill", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, "_progress_iter", lambda items, **_kwargs: items)
    monkeypatch.setattr(pipeline.llm, "reset_token_stats", lambda: None)
    monkeypatch.setattr(
        pipeline.llm,
        "stage_scope",
        lambda *_args, **_kwargs: contextlib.nullcontext(),
    )
    monkeypatch.setattr(pipeline.llm, "dump_token_stats", lambda path: path)
    monkeypatch.setattr(pipeline.llm, "get_token_stats", lambda: [])


def _unexpected(label: str) -> Callable:
    def fail(*_args, **_kwargs):
        raise AssertionError(f"resume unexpectedly repeated {label}")

    return fail


def test_disjoint_dind_dver_runs_all_k8_and_selects_earliest_tied_best(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    induction = _instances("ind")
    verification = _instances("ver")
    heldout_ids = {instance.instance_id for instance in _instances("heldout")}
    induction_ids = {instance.instance_id for instance in induction}
    verification_ids = {instance.instance_id for instance in verification}
    gains = [3, 3, -1, 2, 0, 1, -2, 2]

    collected_ids: list[set[str]] = []
    verified_rounds: list[tuple[int, str]] = []

    def collect(instances, *_args, **_kwargs):
        ids = {instance.instance_id for instance in instances}
        assert ids.isdisjoint(heldout_ids)
        collected_ids.append(ids)
        return [_trajectory(instance) for instance in instances]

    def induce(_failures, _successes, instances, *_args, **kwargs):
        assert {instance.instance_id for instance in instances} == induction_ids
        result = _analysis()
        pipeline.save_analysis(result, kwargs["artifact_dir"])
        return result

    def generate(*_args, **_kwargs):
        return CandidateSkill("candidate-1", "analysis-1", "r1", "mock")

    def refine(candidate, *_args, **_kwargs):
        # The production refiner mutates the candidate in place. This makes the
        # final r1 assertion exercise the pipeline's deepcopy snapshot as well
        # as its strict-greater-than (earliest tie) selection rule.
        candidate.body = f"r{int(candidate.body[1:]) + 1}"
        return candidate

    def verify(candidate, target, guard, *_args, **kwargs):
        round_idx = kwargs["round_idx"]
        verified_rounds.append((round_idx, candidate.body))
        assert {instance.instance_id for instance in target + guard} == verification_ids
        assert set(kwargs["baseline_cache"]) == verification_ids
        return _feedback(gains[round_idx], round_idx), kwargs["baseline_cache"]

    _install_common_mocks(
        monkeypatch,
        tmp_path,
        collect=collect,
        induce=induce,
        generate=generate,
        refine=refine,
        verify=verify,
    )

    skill = pipeline.run_pipeline(
        induction,
        TaskType.SCORED,
        config_path="mock.yaml",
        verification_instances=verification,
        exhaustive_refinement=True,
    )

    assert skill is not None
    assert skill.status is SkillStatus.ACTIVE
    assert verified_rounds == [(index, f"r{index + 1}") for index in range(8)]
    assert collected_ids == [induction_ids, verification_ids]
    assert skill.body == "r1"
    assert len(skill.verification_history) == 8
    assert [
        row["round_idx"]
        for row in skill.verification_history
        if row["selected_best"]
    ] == [0]


def test_round4_interruption_resumes_only_r4_through_r8(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    induction = _instances("ind")
    verification = _instances("ver")
    gains = [1, 4, 0, 2, 4, 3, -1, 2]

    collected_ids: list[list[str]] = []
    attempted_rounds: list[int] = []

    def collect(instances, *_args, **_kwargs):
        collected_ids.append([instance.instance_id for instance in instances])
        return [_trajectory(instance) for instance in instances]

    def induce(_failures, _successes, _instances_arg, *_args, **kwargs):
        result = _analysis()
        # A real analysis artifact is required by the production resume path.
        pipeline.save_analysis(result, kwargs["artifact_dir"])
        return result

    def generate(*_args, **_kwargs):
        return CandidateSkill("candidate-1", "analysis-1", "r1", "mock")

    def refine(candidate, *_args, **_kwargs):
        candidate.body = f"r{int(candidate.body[1:]) + 1}"
        return candidate

    def verify_before_interruption(candidate, _target, _guard, *_args, **kwargs):
        round_idx = kwargs["round_idx"]
        attempted_rounds.append(round_idx)
        if round_idx == 3:
            raise RuntimeError("simulated interruption during round 4")
        return _feedback(gains[round_idx], round_idx), kwargs["baseline_cache"]

    _install_common_mocks(
        monkeypatch,
        tmp_path,
        collect=collect,
        induce=induce,
        generate=generate,
        refine=refine,
        verify=verify_before_interruption,
    )

    with pytest.raises(RuntimeError, match="round 4"):
        pipeline.run_pipeline(
            induction,
            TaskType.SCORED,
            config_path="mock.yaml",
            verification_instances=verification,
            exhaustive_refinement=True,
        )

    assert attempted_rounds == [0, 1, 2, 3]
    run_dirs = list((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    checkpoint = json.loads(
        (run_dir / "refinement_checkpoint.json").read_text(encoding="utf-8")
    )
    assert checkpoint["completed_rounds"] == 3
    assert checkpoint["in_progress_round"] == 3
    assert checkpoint["candidate"]["body"] == "r4"
    assert checkpoint["best_candidate"]["body"] == "r2"

    resumed_rounds: list[tuple[int, str]] = []

    def refine_after_resume(candidate, _analysis_arg, feedback, **_kwargs):
        # These isinstance checks specifically exercise round-checkpoint
        # deserialization, not merely the loop's start index.
        assert isinstance(candidate, CandidateSkill)
        assert isinstance(feedback, VerificationFeedback)
        assert isinstance(feedback.effectiveness, EffectivenessResult)
        candidate.body = f"r{int(candidate.body[1:]) + 1}"
        return candidate

    def verify_after_resume(candidate, _target, _guard, *_args, **kwargs):
        round_idx = kwargs["round_idx"]
        resumed_rounds.append((round_idx, candidate.body))
        return _feedback(gains[round_idx], round_idx), kwargs["baseline_cache"]

    _install_common_mocks(
        monkeypatch,
        tmp_path,
        collect=_unexpected("Dind/Dver baseline collection"),
        induce=_unexpected("induction"),
        generate=_unexpected("round-1 generation"),
        refine=refine_after_resume,
        verify=verify_after_resume,
    )

    skill = pipeline.run_pipeline(
        induction,
        TaskType.SCORED,
        config_path="mock.yaml",
        verification_instances=verification,
        exhaustive_refinement=True,
        resume_dir=str(run_dir),
    )

    assert skill is not None
    assert resumed_rounds == [
        (3, "r4"),
        (4, "r5"),
        (5, "r6"),
        (6, "r7"),
        (7, "r8"),
    ]
    assert len(collected_ids) == 2
    assert skill.body == "r2"
    assert len(skill.verification_history) == 8
    assert [
        row["round_idx"]
        for row in skill.verification_history
        if row["selected_best"]
    ] == [1]


def test_dind_slot_checkpoint_resumes_after_third_scored_slot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    induction = _instances("ind")
    verification = _instances("ver")
    attempted_before: list[str] = []

    def successful_trajectory(instance: TaskInstance) -> Trajectory:
        return Trajectory(
            trajectory_id=f"trajectory-{instance.instance_id}",
            instance_id=instance.instance_id,
            agent_config={},
            messages=[],
            final_output="mock output",
            success=True,
            score=1.0,
        )

    def collect_before_interruption(instances, *_args, **kwargs):
        callback = kwargs["on_trajectory"]
        trajectories: list[Trajectory] = []
        for instance in instances:
            attempted_before.append(instance.instance_id)
            if len(attempted_before) == 4:
                raise RuntimeError("simulated infrastructure failure in D_ind slot 4")
            trajectory = successful_trajectory(instance)
            trajectories.append(trajectory)
            callback(trajectory)
        return trajectories

    _install_common_mocks(
        monkeypatch,
        tmp_path,
        collect=collect_before_interruption,
        induce=_unexpected("induction"),
        generate=_unexpected("generation"),
        refine=_unexpected("refinement"),
        verify=_unexpected("verification"),
    )

    with pytest.raises(RuntimeError, match="D_ind slot 4"):
        pipeline.run_pipeline(
            induction,
            TaskType.SCORED,
            config_path="mock.yaml",
            verification_instances=verification,
            exhaustive_refinement=True,
        )

    run_dirs = list((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    partial = pipeline.load_trajectories(run_dir)
    assert [trajectory.instance_id for trajectory in partial] == [
        "ind-0",
        "ind-1",
        "ind-2",
    ]
    progress = json.loads((run_dir / "checkpoint.json").read_text())
    assert progress["stage"] == "baseline_collecting"
    assert progress["completed_instance_ids"] == ["ind-0", "ind-1", "ind-2"]

    attempted_after: list[str] = []

    def collect_after_resume(instances, *_args, **kwargs):
        callback = kwargs["on_trajectory"]
        trajectories = []
        for instance in instances:
            attempted_after.append(instance.instance_id)
            trajectory = successful_trajectory(instance)
            trajectories.append(trajectory)
            callback(trajectory)
        return trajectories

    _install_common_mocks(
        monkeypatch,
        tmp_path,
        collect=collect_after_resume,
        induce=_unexpected("induction"),
        generate=_unexpected("generation"),
        refine=_unexpected("refinement"),
        verify=_unexpected("verification"),
    )

    skill = pipeline.run_pipeline(
        induction,
        TaskType.SCORED,
        config_path="mock.yaml",
        verification_instances=verification,
        exhaustive_refinement=True,
        resume_dir=str(run_dir),
    )

    assert skill is None
    assert attempted_before == ["ind-0", "ind-1", "ind-2", "ind-3"]
    assert attempted_after == [
        "ind-3",
        "ind-4",
        "ind-5",
        "ind-6",
        "ind-7",
        "ind-8",
        "ind-9",
    ]
    complete = pipeline.load_trajectories(run_dir)
    assert [trajectory.instance_id for trajectory in complete] == [
        f"ind-{index}" for index in range(10)
    ]
    progress = json.loads((run_dir / "checkpoint.json").read_text())
    assert progress["stage"] == "baseline_done"


def test_dind_sparse_out_of_order_checkpoint_resumes_only_missing_slots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    induction = _instances("ind")
    verification = _instances("ver")

    def successful(instance: TaskInstance) -> Trajectory:
        return Trajectory(
            trajectory_id=f"trajectory-{instance.instance_id}",
            instance_id=instance.instance_id,
            agent_config={},
            messages=[],
            final_output="official result",
            success=True,
            score=1.0,
        )

    def collect_then_fail(instances, *_args, **kwargs):
        assert [item.instance_id for item in instances] == [
            f"ind-{index}" for index in range(10)
        ]
        callback = kwargs["on_trajectory"]
        # Parallel workers may finish a non-prefix subset. Repeating the same
        # callback must remain idempotent.
        callback(successful(instances[2]))
        callback(successful(instances[0]))
        callback(successful(instances[2]))
        raise RuntimeError("simulated concurrent infrastructure failure")

    _install_common_mocks(
        monkeypatch,
        tmp_path,
        collect=collect_then_fail,
        induce=_unexpected("induction"),
        generate=_unexpected("generation"),
        refine=_unexpected("refinement"),
        verify=_unexpected("verification"),
    )

    with pytest.raises(RuntimeError, match="concurrent infrastructure failure"):
        pipeline.run_pipeline(
            induction,
            TaskType.SCORED,
            config_path="mock.yaml",
            verification_instances=verification,
            exhaustive_refinement=True,
        )

    run_dir = next((tmp_path / "runs").iterdir())
    partial = pipeline.load_trajectories(run_dir)
    assert [item.instance_id for item in partial] == ["ind-0", "ind-2"]
    progress = json.loads((run_dir / "checkpoint.json").read_text())
    assert progress["stage"] == "baseline_collecting"
    assert progress["completed_instance_ids"] == ["ind-0", "ind-2"]

    attempted_after: list[str] = []

    def collect_missing(instances, *_args, **kwargs):
        attempted_after.extend(item.instance_id for item in instances)
        callback = kwargs["on_trajectory"]
        results = [successful(item) for item in instances]
        for result in reversed(results):
            callback(result)
        return results

    _install_common_mocks(
        monkeypatch,
        tmp_path,
        collect=collect_missing,
        induce=_unexpected("induction"),
        generate=_unexpected("generation"),
        refine=_unexpected("refinement"),
        verify=_unexpected("verification"),
    )
    skill = pipeline.run_pipeline(
        induction,
        TaskType.SCORED,
        config_path="mock.yaml",
        verification_instances=verification,
        exhaustive_refinement=True,
        resume_dir=str(run_dir),
    )

    assert skill is None
    assert attempted_after == [
        "ind-1",
        "ind-3",
        "ind-4",
        "ind-5",
        "ind-6",
        "ind-7",
        "ind-8",
        "ind-9",
    ]
    complete = pipeline.load_trajectories(run_dir)
    assert [item.instance_id for item in complete] == [
        f"ind-{index}" for index in range(10)
    ]
