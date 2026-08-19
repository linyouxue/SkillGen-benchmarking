from __future__ import annotations

import threading

import pytest

import agents.verification as verification
import pilot_budget_guard
from models import CandidateSkill, CaseAnalysis, EffectivenessResult, TaskType


def _effectiveness(cases: list[dict]) -> EffectivenessResult:
    return EffectivenessResult(
        passed=False,
        n_target=0,
        n_boundary=0,
        paired_n=0,
        baseline_acc=0.0,
        skill_acc=0.0,
        repair_count=0,
        regression_count=0,
        repair_rate=0.0,
        regression_rate=0.0,
        net_gain=0,
        target_repair_count=0,
        target_fail_count=0,
        success_guard_regression_count=0,
        success_guard_pass_count=0,
        repaired_ids=[],
        regression_ids=[],
        failed_ids_after_skill=[],
        diagnostic_summary="mock",
        cases=cases,
    )


def test_case_analysis_budget_stop_drains_inflight_without_submitting_more(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = [
        {"instance_id": str(index), "outcome": "both_fail"}
        for index in range(6)
    ]
    monkeypatch.setattr(
        verification,
        "verify_effectiveness",
        lambda *_args, **_kwargs: (_effectiveness(cases), {}),
    )
    monkeypatch.setattr(
        verification,
        "_synthesise_revision_guidance",
        lambda **_kwargs: pytest.fail("synthesis must not start after budget stop"),
    )

    barrier = threading.Barrier(3)
    release_inflight = threading.Event()
    started: list[int] = []
    finished: list[int] = []
    lock = threading.Lock()

    def analyse(case, **_kwargs):
        index = int(case["instance_id"])
        with lock:
            started.append(index)
        barrier.wait(timeout=2)
        if index == 0:
            threading.Timer(0.1, release_inflight.set).start()
            raise pilot_budget_guard.PilotBudgetStop("cap reached")
        assert release_inflight.wait(timeout=2)
        with lock:
            finished.append(index)
        return CaseAnalysis(
            instance_id=str(index),
            bucket="still_failing",
            analysis="done",
            skill_influence="none",
            micro_recommendation="none",
        )

    monkeypatch.setattr(verification, "_analyse_one_case", analyse)
    candidate = CandidateSkill("candidate", "analysis", "body", "abstract")

    with pytest.raises(pilot_budget_guard.PilotBudgetStop, match="cap reached"):
        verification.run_verification(
            candidate,
            [],
            [],
            TaskType.SCORED,
            case_analyst_max_workers=3,
        )

    assert sorted(started) == [0, 1, 2]
    assert sorted(finished) == [1, 2]
