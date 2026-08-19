from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from types import SimpleNamespace

import pytest

import pilot_budget_guard as guard
import llm


_GUARD_ENV = (
    "SKILLGEN_DEEPSEEK_BUDGET_CNY",
    "SKILLGEN_DEEPSEEK_BALANCE_STOP_CNY",
    "SKILLGEN_BUDGET_LEDGER",
    "SKILLGEN_META_REQUEST_RESERVE_CNY",
    "SKILLGEN_AGENT_ROLLOUT_RESERVE_CNY",
    "SKILLGEN_ALLOW_PEAK_LAUNCH",
    "DEEPSEEK_API_KEY",
)


@pytest.fixture(autouse=True)
def _isolate_guard_and_forbid_network(monkeypatch: pytest.MonkeyPatch):
    for name in _GUARD_ENV:
        monkeypatch.delenv(name, raising=False)

    def forbidden_urlopen(*args, **kwargs):
        raise AssertionError("budget guard tests must never perform network I/O")

    monkeypatch.setattr(guard.urllib.request, "urlopen", forbidden_urlopen)


def _enable_guard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    *,
    cap: str = "120",
) -> None:
    monkeypatch.setenv("SKILLGEN_DEEPSEEK_BUDGET_CNY", cap)
    monkeypatch.setenv("SKILLGEN_BUDGET_LEDGER", str(tmp_path / "ledger.json"))
    monkeypatch.setattr(guard, "assert_offpeak", lambda now=None: None)


def test_explicit_peak_launch_override_bypasses_only_the_time_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("SKILLGEN_DEEPSEEK_BUDGET_CNY", "120")
    monkeypatch.setenv("SKILLGEN_BUDGET_LEDGER", str(tmp_path / "ledger.json"))
    peak = guard._PEAK_PRICING_EFFECTIVE.replace(hour=9)

    with pytest.raises(guard.PilotBudgetStop, match="peak-price window"):
        guard.assert_offpeak(peak)

    monkeypatch.setenv("SKILLGEN_ALLOW_PEAK_LAUNCH", "1")
    guard.assert_offpeak(peak)

    monkeypatch.setenv("SKILLGEN_ALLOW_PEAK_LAUNCH", "true")
    with pytest.raises(guard.PilotBudgetStop, match="peak-price window"):
        guard.assert_offpeak(peak)


@pytest.mark.parametrize(
    ("env_name", "entrypoint"),
    (
        ("SKILLGEN_META_REQUEST_RESERVE_CNY", guard.before_meta_request),
        ("SKILLGEN_AGENT_ROLLOUT_RESERVE_CNY", guard.before_agent_rollout),
    ),
)
@pytest.mark.parametrize(
    "raw",
    ("", "not-a-number", "NaN", "sNaN", "Infinity", "-Infinity", "-0.01"),
)
def test_invalid_reserve_env_fails_before_balance_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    env_name: str,
    entrypoint,
    raw: str,
) -> None:
    _enable_guard(monkeypatch, tmp_path)
    monkeypatch.setenv(env_name, raw)

    def forbidden_balance_read():
        raise AssertionError("invalid reserve must fail before reading balance")

    monkeypatch.setattr(guard, "_fetch_cny_balance", forbidden_balance_read)

    with pytest.raises(guard.PilotBudgetStop, match=env_name):
        entrypoint()


@pytest.mark.parametrize("raw", ("0", "-0", "0.000001", "30", "1E+2"))
def test_nonnegative_reserve_env_is_accepted_offline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    raw: str,
) -> None:
    _enable_guard(monkeypatch, tmp_path)
    monkeypatch.setenv("SKILLGEN_AGENT_ROLLOUT_RESERVE_CNY", raw)
    monkeypatch.setattr(guard, "_fetch_cny_balance", lambda: Decimal("1000"))

    guard.before_agent_rollout()

    ledger = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    assert Decimal(ledger["events"][-1]["reserve_cny"]) == Decimal(raw)


def test_default_reserves_are_frozen_without_network(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _enable_guard(monkeypatch, tmp_path)
    observed: list[tuple[str, Decimal, bool, str | None]] = []

    def capture(
        kind: str,
        *,
        reserve_cny: Decimal,
        initialize_only: bool,
        create_reservation: str | None = None,
        **_kwargs,
    ):
        observed.append((kind, reserve_cny, initialize_only, create_reservation))
        return {}

    monkeypatch.setattr(guard, "_snapshot", capture)

    guard.before_meta_request()
    guard.before_agent_rollout()

    assert [(kind, reserve, initialize) for kind, reserve, initialize, _ in observed] == [
        ("before_meta_request", Decimal("5"), False),
        ("before_agent_rollout", Decimal("30"), False),
    ]
    assert all(token for *_, token in observed)
    assert observed[0][3] != observed[1][3]


def test_three_concurrent_reservations_are_atomic_and_fourth_is_denied(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _enable_guard(monkeypatch, tmp_path, cap="90")
    monkeypatch.setattr(guard, "_fetch_cny_balance", lambda: Decimal("1000"))

    def reserve_one(_index: int):
        try:
            return guard.before_agent_rollout()
        except guard.PilotBudgetStop:
            return None

    with ThreadPoolExecutor(max_workers=4) as executor:
        tokens = list(executor.map(reserve_one, range(4)))

    accepted = [token for token in tokens if token is not None]
    assert len(accepted) == 3
    assert len(set(accepted)) == 3
    ledger = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    assert ledger["schema_version"] == 3
    assert Decimal(ledger["balance_stop_cny"]) == Decimal("0")
    assert set(ledger["active_reservations"]) == set(accepted)
    assert Decimal(ledger["active_reserved_cny"]) == Decimal("90")

    guard.record_balance("after_agent_rollout", reservation_token=accepted[0])
    replacement = guard.before_agent_rollout()
    assert replacement is not None


def test_initialize_fails_closed_on_stale_active_reservation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _enable_guard(monkeypatch, tmp_path)
    monkeypatch.setattr(guard, "_fetch_cny_balance", lambda: Decimal("1000"))
    token = guard.before_agent_rollout()
    assert token is not None

    with pytest.raises(guard.PilotBudgetStop, match="unfinished run"):
        guard.initialize()

    with pytest.raises(guard.PilotBudgetStop, match="unfinished run"):
        guard.record_balance("pilot_complete")


def test_failed_settlement_keeps_reservation_for_fail_closed_resume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _enable_guard(monkeypatch, tmp_path)
    monkeypatch.setattr(guard, "_fetch_cny_balance", lambda: Decimal("1000"))
    token = guard.before_agent_rollout()
    assert token is not None

    def failed_balance():
        raise guard.PilotBudgetStop("balance unavailable")

    monkeypatch.setattr(guard, "_fetch_cny_balance", failed_balance)
    with pytest.raises(guard.PilotBudgetStop, match="balance unavailable"):
        guard.record_balance("after_agent_rollout", reservation_token=token)

    ledger = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    assert token in ledger["active_reservations"]


@pytest.mark.parametrize("raises", (False, True))
def test_deepseek_meta_attempt_settles_the_exact_reservation_token(
    monkeypatch: pytest.MonkeyPatch,
    raises: bool,
) -> None:
    monkeypatch.setenv("SKILLGEN_CHAT_PROVIDER", "deepseek")
    calls = []
    monkeypatch.setattr(
        llm.pilot_budget_guard,
        "before_meta_request",
        lambda: "meta-reservation",
    )
    monkeypatch.setattr(
        llm.pilot_budget_guard,
        "record_balance",
        lambda kind, *, reservation_token=None: calls.append(
            (kind, reservation_token)
        ),
    )

    def create(**_kwargs):
        if raises:
            raise ValueError("non-retryable provider failure")
        return object()

    monkeypatch.setattr(
        llm,
        "_get_router_client",
        lambda: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
    )
    if raises:
        with pytest.raises(ValueError, match="non-retryable"):
            llm._router_chat_create(model="test")
        expected_kind = "after_meta_request_error"
    else:
        llm._router_chat_create(model="test")
        expected_kind = "after_meta_request"
    assert calls == [(expected_kind, "meta-reservation")]


@pytest.mark.parametrize("raw", ("", "garbage", "NaN", "Infinity", "0", "-1"))
def test_cap_must_be_finite_and_positive(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    monkeypatch.setenv("SKILLGEN_DEEPSEEK_BUDGET_CNY", raw)

    with pytest.raises(
        guard.PilotBudgetStop,
        match="SKILLGEN_DEEPSEEK_BUDGET_CNY",
    ):
        guard._cap()


@pytest.mark.parametrize("raw", (Decimal("NaN"), Decimal("Infinity"), Decimal("-1")))
def test_direct_snapshot_rejects_invalid_reserve_before_balance_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    raw: Decimal,
) -> None:
    _enable_guard(monkeypatch, tmp_path)

    def forbidden_balance_read():
        raise AssertionError("invalid reserve must fail before reading balance")

    monkeypatch.setattr(guard, "_fetch_cny_balance", forbidden_balance_read)

    with pytest.raises(guard.PilotBudgetStop, match="reserve_cny"):
        guard._snapshot("test", reserve_cny=raw, initialize_only=False)


def test_snapshot_rejects_nonfinite_mocked_balance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _enable_guard(monkeypatch, tmp_path)
    monkeypatch.setattr(guard, "_fetch_cny_balance", lambda: Decimal("NaN"))

    with pytest.raises(guard.PilotBudgetStop, match="DeepSeek CNY balance"):
        guard._snapshot("test", reserve_cny=Decimal("0"), initialize_only=True)


@pytest.mark.parametrize(
    "bad_field",
    (
        {"cap_cny": "NaN", "starting_balance_cny": "100", "events": []},
        {"cap_cny": "120", "starting_balance_cny": "Infinity", "events": []},
        {"cap_cny": "120", "starting_balance_cny": "100", "events": {}},
    ),
)
def test_snapshot_rejects_invalid_ledger_numbers_and_events(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    bad_field: dict,
) -> None:
    _enable_guard(monkeypatch, tmp_path)
    ledger = {
        "schema_version": 1,
        "provider": "deepseek_official",
        "currency": "CNY",
        **bad_field,
    }
    (tmp_path / "ledger.json").write_text(json.dumps(ledger), encoding="utf-8")
    monkeypatch.setattr(guard, "_fetch_cny_balance", lambda: Decimal("100"))

    with pytest.raises(guard.PilotBudgetStop, match="invalid budget ledger"):
        guard._snapshot("test", reserve_cny=Decimal("0"), initialize_only=True)


def test_snapshot_rejects_malformed_ledger_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _enable_guard(monkeypatch, tmp_path)
    (tmp_path / "ledger.json").write_text("{not-json", encoding="utf-8")
    monkeypatch.setattr(guard, "_fetch_cny_balance", lambda: Decimal("100"))

    with pytest.raises(guard.PilotBudgetStop, match="invalid budget ledger"):
        guard._snapshot("test", reserve_cny=Decimal("0"), initialize_only=True)


def test_disabled_guard_ignores_reserve_env_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SKILLGEN_META_REQUEST_RESERVE_CNY", "-1")
    monkeypatch.setenv("SKILLGEN_AGENT_ROLLOUT_RESERVE_CNY", "NaN")

    guard.before_meta_request()
    guard.before_agent_rollout()
    assert guard.initialize() == {"enabled": False}


@pytest.mark.parametrize(
    "raw",
    ("", "not-a-number", "NaN", "sNaN", "Infinity", "-Infinity", "-0.01"),
)
def test_invalid_balance_stop_env_fails_before_balance_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    raw: str,
) -> None:
    _enable_guard(monkeypatch, tmp_path)
    monkeypatch.setenv("SKILLGEN_DEEPSEEK_BALANCE_STOP_CNY", raw)

    def forbidden_balance_read():
        raise AssertionError("invalid stop floor must fail before reading balance")

    monkeypatch.setattr(guard, "_fetch_cny_balance", forbidden_balance_read)

    with pytest.raises(
        guard.PilotBudgetStop,
        match="SKILLGEN_DEEPSEEK_BALANCE_STOP_CNY",
    ):
        guard.before_agent_rollout()


def test_new_ledger_freezes_default_zero_balance_stop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _enable_guard(monkeypatch, tmp_path)
    monkeypatch.setattr(guard, "_fetch_cny_balance", lambda: Decimal("10"))

    guard.initialize()

    ledger = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    assert ledger["schema_version"] == 3
    assert ledger["balance_stop_cny"] == "0"
    assert ledger["balance_stop_policy_amendments"] == []


@pytest.mark.parametrize("current", ("2", "1.99", "0"))
def test_paid_preflight_rejects_balance_at_or_below_frozen_floor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    current: str,
) -> None:
    _enable_guard(monkeypatch, tmp_path)
    monkeypatch.setenv("SKILLGEN_DEEPSEEK_BALANCE_STOP_CNY", "2")
    monkeypatch.setattr(guard, "_fetch_cny_balance", lambda: Decimal(current))

    with pytest.raises(guard.PilotBudgetStop, match="at or below frozen stop floor 2"):
        guard.before_agent_rollout()

    assert not (tmp_path / "ledger.json").exists()


def test_active_reserve_affects_cap_but_not_balance_floor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _enable_guard(monkeypatch, tmp_path, cap="90")
    monkeypatch.setenv("SKILLGEN_DEEPSEEK_BALANCE_STOP_CNY", "2")
    monkeypatch.setattr(guard, "_fetch_cny_balance", lambda: Decimal("2.01"))

    tokens = [guard.before_agent_rollout() for _ in range(3)]
    assert all(tokens)
    with pytest.raises(guard.PilotBudgetStop, match="exceeds cap 90"):
        guard.before_agent_rollout()

    ledger = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    assert Decimal(ledger["active_reserved_cny"]) == Decimal("90")


def test_low_balance_settlement_releases_reservation_without_floor_stop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _enable_guard(monkeypatch, tmp_path)
    monkeypatch.setenv("SKILLGEN_DEEPSEEK_BALANCE_STOP_CNY", "2")
    balances = iter((Decimal("2.01"), Decimal("1.50")))
    monkeypatch.setattr(guard, "_fetch_cny_balance", lambda: next(balances))

    token = guard.before_agent_rollout()
    guard.record_balance("after_agent_rollout", reservation_token=token)

    ledger = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    assert ledger["latest_balance_cny"] == "1.50"
    assert ledger["active_reservations"] == {}
    assert ledger["active_reserved_cny"] == "0"


def _write_legacy_ledger(path) -> dict:
    ledger = {
        "schema_version": 2,
        "provider": "deepseek_official",
        "currency": "CNY",
        "cap_cny": "120",
        "starting_balance_cny": "80",
        "latest_balance_cny": "60",
        "observed_spend_cny": "20",
        "active_reserved_cny": "0",
        "active_reservations": {},
        "events": [{"at": "2026-08-18T00:00:00+08:00", "kind": "initialize"}],
    }
    path.write_text(json.dumps(ledger), encoding="utf-8")
    return ledger


def test_legacy_ledger_requires_explicit_balance_stop_migration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _enable_guard(monkeypatch, tmp_path)
    monkeypatch.setenv("SKILLGEN_DEEPSEEK_BALANCE_STOP_CNY", "2")
    _write_legacy_ledger(tmp_path / "ledger.json")
    monkeypatch.setattr(guard, "_fetch_cny_balance", lambda: Decimal("60"))

    with pytest.raises(
        guard.PilotBudgetStop,
        match=r"explicit migrate_balance_stop_policy\(\)",
    ):
        guard.initialize()


def test_explicit_balance_stop_migration_is_atomic_cas_and_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _enable_guard(monkeypatch, tmp_path)
    monkeypatch.setenv("SKILLGEN_DEEPSEEK_BALANCE_STOP_CNY", "2")
    original = _write_legacy_ledger(tmp_path / "ledger.json")

    def forbidden_balance_read():
        raise AssertionError("ledger migration must not read provider balance")

    monkeypatch.setattr(guard, "_fetch_cny_balance", forbidden_balance_read)
    kwargs = {
        "expected_old_balance_stop_cny": None,
        "new_balance_stop_cny": Decimal("2"),
        "amendment_id": "balance-floor-v1",
        "old_protocol_hash": "old-protocol",
        "new_protocol_hash": "new-protocol",
    }

    migrated = guard.migrate_balance_stop_policy(**kwargs)
    assert migrated["schema_version"] == 3
    assert migrated["balance_stop_cny"] == "2"
    assert migrated["latest_balance_cny"] == original["latest_balance_cny"]
    assert migrated["observed_spend_cny"] == original["observed_spend_cny"]
    assert migrated["events"][:-1] == original["events"]
    assert migrated["events"][-1]["kind"] == "migrate_balance_stop_policy"
    assert len(migrated["balance_stop_policy_amendments"]) == 1

    first_bytes = (tmp_path / "ledger.json").read_bytes()
    repeated = guard.migrate_balance_stop_policy(
        **{**kwargs, "new_balance_stop_cny": "2.00"}
    )
    assert repeated == migrated
    assert (tmp_path / "ledger.json").read_bytes() == first_bytes


def test_schema_three_requires_explicit_policy_amendment_list(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _enable_guard(monkeypatch, tmp_path)
    monkeypatch.setenv("SKILLGEN_DEEPSEEK_BALANCE_STOP_CNY", "2")
    monkeypatch.setattr(guard, "_fetch_cny_balance", lambda: Decimal("60"))
    guard.initialize()

    path = tmp_path / "ledger.json"
    ledger = json.loads(path.read_text(encoding="utf-8"))
    del ledger["balance_stop_policy_amendments"]
    path.write_text(json.dumps(ledger), encoding="utf-8")

    with pytest.raises(guard.PilotBudgetStop, match="invalid budget ledger"):
        guard.initialize()


@pytest.mark.parametrize(
    "tamper",
    (
        "missing_field",
        "extra_field",
        "noncanonical_id",
        "bad_old_floor_type",
        "noncanonical_new_floor",
        "empty_old_hash",
        "bad_new_hash_type",
        "bad_timestamp_type",
        "bad_timestamp_value",
        "duplicate_id",
    ),
)
def test_policy_amendment_tampering_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    tamper: str,
) -> None:
    _enable_guard(monkeypatch, tmp_path)
    monkeypatch.setenv("SKILLGEN_DEEPSEEK_BALANCE_STOP_CNY", "2")
    path = tmp_path / "ledger.json"
    _write_legacy_ledger(path)
    guard.migrate_balance_stop_policy(
        expected_old_balance_stop_cny=None,
        new_balance_stop_cny="2",
        amendment_id="balance-floor-v1",
        old_protocol_hash="old-protocol",
        new_protocol_hash="new-protocol",
    )
    ledger = json.loads(path.read_text(encoding="utf-8"))
    amendment = ledger["balance_stop_policy_amendments"][0]
    if tamper == "missing_field":
        del amendment["new_protocol_hash"]
    elif tamper == "extra_field":
        amendment["unexpected"] = True
    elif tamper == "noncanonical_id":
        amendment["amendment_id"] = " balance-floor-v1 "
    elif tamper == "bad_old_floor_type":
        amendment["old_balance_stop_cny"] = 0
    elif tamper == "noncanonical_new_floor":
        amendment["new_balance_stop_cny"] = "2.00"
    elif tamper == "empty_old_hash":
        amendment["old_protocol_hash"] = ""
    elif tamper == "bad_new_hash_type":
        amendment["new_protocol_hash"] = 123
    elif tamper == "bad_timestamp_type":
        amendment["migrated_at"] = 123
    elif tamper == "bad_timestamp_value":
        amendment["migrated_at"] = "not-an-iso-timestamp"
    elif tamper == "duplicate_id":
        ledger["balance_stop_policy_amendments"].append(dict(amendment))
    else:  # pragma: no cover - the parametrization is closed above
        raise AssertionError(f"unknown tamper case: {tamper}")
    path.write_text(json.dumps(ledger), encoding="utf-8")
    monkeypatch.setattr(guard, "_fetch_cny_balance", lambda: Decimal("60"))

    with pytest.raises(guard.PilotBudgetStop, match="invalid budget ledger"):
        guard.initialize()


def test_boolean_ledger_schema_version_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _enable_guard(monkeypatch, tmp_path)
    path = tmp_path / "ledger.json"
    ledger = _write_legacy_ledger(path)
    ledger["schema_version"] = True
    path.write_text(json.dumps(ledger), encoding="utf-8")
    monkeypatch.setattr(guard, "_fetch_cny_balance", lambda: Decimal("60"))

    with pytest.raises(guard.PilotBudgetStop, match="invalid budget ledger"):
        guard.initialize()


def test_balance_stop_migration_rejects_cas_and_lineage_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _enable_guard(monkeypatch, tmp_path)
    monkeypatch.setenv("SKILLGEN_DEEPSEEK_BALANCE_STOP_CNY", "2")
    _write_legacy_ledger(tmp_path / "ledger.json")
    guard.migrate_balance_stop_policy(
        expected_old_balance_stop_cny=None,
        new_balance_stop_cny="2",
        amendment_id="balance-floor-v1",
        old_protocol_hash="old-protocol",
        new_protocol_hash="new-protocol",
    )

    with pytest.raises(guard.PilotBudgetStop, match="already used"):
        guard.migrate_balance_stop_policy(
            expected_old_balance_stop_cny=None,
            new_balance_stop_cny="2",
            amendment_id="balance-floor-v1",
            old_protocol_hash="different-old",
            new_protocol_hash="new-protocol",
        )

    monkeypatch.setenv("SKILLGEN_DEEPSEEK_BALANCE_STOP_CNY", "1")
    with pytest.raises(guard.PilotBudgetStop, match="compare-and-swap failed"):
        guard.migrate_balance_stop_policy(
            expected_old_balance_stop_cny="0",
            new_balance_stop_cny="1",
            amendment_id="balance-floor-v2",
            old_protocol_hash="new-protocol",
            new_protocol_hash="next-protocol",
        )


def test_balance_stop_migration_rejects_active_reservation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _enable_guard(monkeypatch, tmp_path)
    monkeypatch.setenv("SKILLGEN_DEEPSEEK_BALANCE_STOP_CNY", "2")
    ledger = _write_legacy_ledger(tmp_path / "ledger.json")
    ledger["active_reservations"] = {
        "token": {
            "kind": "before_agent_rollout",
            "reserve_cny": "10",
            "created_at": "2026-08-18T00:00:00+08:00",
            "pid": 1,
        }
    }
    ledger["active_reserved_cny"] = "10"
    (tmp_path / "ledger.json").write_text(json.dumps(ledger), encoding="utf-8")

    with pytest.raises(guard.PilotBudgetStop, match="active reservations"):
        guard.migrate_balance_stop_policy(
            expected_old_balance_stop_cny=None,
            new_balance_stop_cny="2",
            amendment_id="balance-floor-v1",
            old_protocol_hash="old-protocol",
            new_protocol_hash="new-protocol",
        )


def test_frozen_balance_stop_rejects_env_drift_for_new_paid_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _enable_guard(monkeypatch, tmp_path)
    monkeypatch.setenv("SKILLGEN_DEEPSEEK_BALANCE_STOP_CNY", "2")
    monkeypatch.setattr(guard, "_fetch_cny_balance", lambda: Decimal("10"))
    guard.initialize()

    monkeypatch.setenv("SKILLGEN_DEEPSEEK_BALANCE_STOP_CNY", "1")
    with pytest.raises(guard.PilotBudgetStop, match="differs from the frozen ledger"):
        guard.before_agent_rollout()
