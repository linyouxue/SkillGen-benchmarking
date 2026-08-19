from __future__ import annotations

import threading
import time

import pytest

import llm


def test_budget_concurrency_uses_bounded_workers_and_preserves_input_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm.pilot_budget_guard, "enabled", lambda: True)
    barrier = threading.Barrier(3)

    def work(index: int) -> str:
        if index < 3:
            barrier.wait(timeout=2)
            time.sleep((2 - index) * 0.01)
        return f"result-{index}"

    results = llm.run_concurrent(
        work,
        [(index,) for index in range(6)],
        max_workers=3,
    )
    assert results == [f"result-{index}" for index in range(6)]


def test_budget_concurrency_stops_submission_and_drains_inflight_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm.pilot_budget_guard, "enabled", lambda: True)
    barrier = threading.Barrier(3)
    release_inflight = threading.Event()
    started: list[int] = []
    finished: list[int] = []
    lock = threading.Lock()

    def work(index: int) -> int:
        with lock:
            started.append(index)
        barrier.wait(timeout=2)
        if index == 0:
            threading.Timer(0.1, release_inflight.set).start()
            raise RuntimeError("first failure")
        assert release_inflight.wait(timeout=2)
        with lock:
            finished.append(index)
        return index

    with pytest.raises(RuntimeError, match="first failure"):
        llm.run_concurrent(
            work,
            [(index,) for index in range(6)],
            max_workers=3,
        )

    assert sorted(started) == [0, 1, 2]
    assert sorted(finished) == [1, 2]
