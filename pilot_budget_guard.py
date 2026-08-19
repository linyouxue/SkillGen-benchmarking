"""Time-window and between-unit account-balance guard for DeepSeek pilots.

The guard is opt-in through environment variables and never reads a key or
performs network I/O when disabled.  It uses DeepSeek's official balance
endpoint, so it does not trust BenchFlow/LiteLLM's placeholder price table.
It is deliberately described as a *soft* stop: a provider-side hard cap also
requires the account's available balance to be no greater than the approved
amount and automatic recharge to be disabled.

The frozen balance-stop floor is a paid-preflight policy: it denies creation
of a new paid reservation when the current balance is at or below the floor.
It never blocks settlement of an existing reservation.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import urllib.request
import uuid
from contextlib import contextmanager
from datetime import datetime, time, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


_LOCK = threading.Lock()
_SHANGHAI = ZoneInfo("Asia/Shanghai")
_PEAK_PRICING_EFFECTIVE = datetime(2026, 8, 17, 0, 0, tzinfo=_SHANGHAI)
_LEDGER_SCHEMA_VERSION = 3
_BALANCE_STOP_ENV = "SKILLGEN_DEEPSEEK_BALANCE_STOP_CNY"
_POLICY_AMENDMENTS_FIELD = "balance_stop_policy_amendments"


class PilotBudgetStop(RuntimeError):
    """Raised before another paid unit starts when a pilot guard denies it."""


def _cny_amount(
    value: object,
    *,
    label: str,
    allow_zero: bool,
) -> Decimal:
    """Parse one finite CNY amount and fail closed on unsafe values."""

    try:
        amount = value if isinstance(value, Decimal) else Decimal(str(value).strip())
    except (InvalidOperation, TypeError, ValueError) as exc:
        qualifier = "non-negative" if allow_zero else "positive"
        raise PilotBudgetStop(
            f"{label} must be a finite {qualifier} CNY amount"
        ) from exc

    if not amount.is_finite():
        qualifier = "non-negative" if allow_zero else "positive"
        raise PilotBudgetStop(
            f"{label} must be a finite {qualifier} CNY amount"
        )
    invalid_sign = amount < 0 if allow_zero else amount <= 0
    if invalid_sign:
        qualifier = "non-negative" if allow_zero else "positive"
        raise PilotBudgetStop(
            f"{label} must be a finite {qualifier} CNY amount"
        )
    return amount


def _reserve_from_env(name: str, default: str) -> Decimal:
    return _cny_amount(
        os.environ.get(name, default),
        label=name,
        allow_zero=True,
    )


def _balance_stop_from_env() -> Decimal:
    """Return the global balance floor for admitting a new paid request."""

    return _cny_amount(
        os.environ.get(_BALANCE_STOP_ENV, "0"),
        label=_BALANCE_STOP_ENV,
        allow_zero=True,
    )


def enabled() -> bool:
    return bool(os.environ.get("SKILLGEN_DEEPSEEK_BUDGET_CNY", "").strip())


def _ledger_path() -> Path:
    raw = os.environ.get("SKILLGEN_BUDGET_LEDGER", "").strip()
    if not raw:
        raise PilotBudgetStop(
            "SKILLGEN_BUDGET_LEDGER is required when the DeepSeek budget guard is enabled"
        )
    return Path(raw).expanduser().resolve()


def _cap() -> Decimal:
    return _cny_amount(
        os.environ.get("SKILLGEN_DEEPSEEK_BUDGET_CNY"),
        label="SKILLGEN_DEEPSEEK_BUDGET_CNY",
        allow_zero=False,
    )


def _now_shanghai() -> datetime:
    return datetime.now(timezone.utc).astimezone(_SHANGHAI)


def _is_peak(now: datetime) -> bool:
    local = now.astimezone(_SHANGHAI)
    if local < _PEAK_PRICING_EFFECTIVE:
        return False
    clock = local.time()
    return time(9, 0) <= clock < time(12, 0) or time(14, 0) <= clock < time(18, 0)


def assert_offpeak(now: datetime | None = None) -> None:
    if not enabled():
        return
    if os.environ.get("SKILLGEN_ALLOW_PEAK_LAUNCH", "").strip() == "1":
        return
    local = (now or _now_shanghai()).astimezone(_SHANGHAI)
    if _is_peak(local):
        raise PilotBudgetStop(
            "DeepSeek peak-price window is active in Asia/Shanghai; "
            "new paid work is allowed only 12:00-14:00 or 18:00-09:00"
        )


def _fetch_cny_balance() -> Decimal:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise PilotBudgetStop("DEEPSEEK_API_KEY is not set")
    request = urllib.request.Request(
        "https://api.deepseek.com/user/balance",
        method="GET",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {key}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise PilotBudgetStop(
            "could not verify the official DeepSeek account balance; "
            "refusing to start paid work"
        ) from exc
    if not payload.get("is_available"):
        raise PilotBudgetStop("DeepSeek reports that the account balance is unavailable")
    for item in payload.get("balance_infos") or []:
        if str(item.get("currency") or "").upper() == "CNY":
            try:
                raw_balance = item["total_balance"]
            except KeyError as exc:
                raise PilotBudgetStop("DeepSeek returned an invalid CNY balance") from exc
            try:
                return _cny_amount(
                    raw_balance,
                    label="DeepSeek CNY total_balance",
                    allow_zero=True,
                )
            except PilotBudgetStop as exc:
                raise PilotBudgetStop("DeepSeek returned an invalid CNY balance") from exc
    raise PilotBudgetStop(
        "DeepSeek did not return a CNY balance; this CNY-denominated pilot guard "
        "will not guess an exchange rate"
    )


def _read_ledger(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotBudgetStop(f"invalid budget ledger: {path}") from exc
    if not isinstance(payload, dict):
        raise PilotBudgetStop(f"invalid budget ledger: {path}")
    return payload


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        temp_name = None
    finally:
        if temp_name is not None:
            try:
                Path(temp_name).unlink()
            except FileNotFoundError:
                pass


@contextmanager
def _ledger_file_lock(path: Path):
    """Serialize balance checks and reservations across threads/processes."""

    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        with lock_path.open("a+b") as handle:
            if os.name == "nt":
                import msvcrt

                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _active_reservations(ledger: dict[str, Any], path: Path) -> dict[str, Any]:
    active = ledger.get("active_reservations")
    if active is None and ledger.get("schema_version") == 1:
        active = {}
        ledger["active_reservations"] = active
        ledger["schema_version"] = 2
    if not isinstance(active, dict):
        raise PilotBudgetStop(f"invalid budget ledger: {path}")
    for token, record in active.items():
        if not isinstance(token, str) or not token or not isinstance(record, dict):
            raise PilotBudgetStop(f"invalid budget ledger: {path}")
        try:
            _cny_amount(
                record.get("reserve_cny"),
                label="budget ledger active reserve_cny",
                allow_zero=True,
            )
        except PilotBudgetStop as exc:
            raise PilotBudgetStop(f"invalid budget ledger: {path}") from exc
    return active


def _active_total(active: dict[str, Any]) -> Decimal:
    return sum(
        (
            _cny_amount(
                record.get("reserve_cny"),
                label="budget ledger active reserve_cny",
                allow_zero=True,
            )
            for record in active.values()
        ),
        Decimal("0"),
    )


def _policy_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PilotBudgetStop(f"{label} must be a non-empty string")
    return value.strip()


def _decimal_text(value: Decimal) -> str:
    """Canonical fixed-point text for policy identity and idempotence."""

    if value == 0:
        return "0"
    return format(value.normalize(), "f")


def _validate_policy_amendments(
    ledger: dict[str, Any],
    path: Path,
) -> list[dict[str, Any]]:
    amendments = ledger.get(_POLICY_AMENDMENTS_FIELD, [])
    if not isinstance(amendments, list):
        raise PilotBudgetStop(f"invalid budget ledger: {path}")
    seen_ids: set[str] = set()
    required_fields = {
        "amendment_id",
        "old_balance_stop_cny",
        "new_balance_stop_cny",
        "old_protocol_hash",
        "new_protocol_hash",
        "migrated_at",
    }
    for amendment in amendments:
        if not isinstance(amendment, dict) or set(amendment) != required_fields:
            raise PilotBudgetStop(f"invalid budget ledger: {path}")
        try:
            amendment_id = _policy_text(
                amendment.get("amendment_id"),
                label="budget ledger amendment_id",
            )
            old_protocol_hash = _policy_text(
                amendment.get("old_protocol_hash"),
                label="budget ledger old_protocol_hash",
            )
            new_protocol_hash = _policy_text(
                amendment.get("new_protocol_hash"),
                label="budget ledger new_protocol_hash",
            )
            migrated_at = _policy_text(
                amendment.get("migrated_at"),
                label="budget ledger migrated_at",
            )
            if (
                amendment["amendment_id"] != amendment_id
                or amendment["old_protocol_hash"] != old_protocol_hash
                or amendment["new_protocol_hash"] != new_protocol_hash
                or amendment["migrated_at"] != migrated_at
            ):
                raise PilotBudgetStop("budget ledger amendment text is not canonical")
            try:
                parsed_migrated_at = datetime.fromisoformat(migrated_at)
            except ValueError as exc:
                raise PilotBudgetStop(
                    "budget ledger migrated_at must be an ISO-8601 timestamp"
                ) from exc
            if parsed_migrated_at.tzinfo is None:
                raise PilotBudgetStop(
                    "budget ledger migrated_at must include a timezone"
                )
            old_stop = amendment.get("old_balance_stop_cny")
            if old_stop is not None:
                if not isinstance(old_stop, str):
                    raise PilotBudgetStop(
                        "budget ledger old_balance_stop_cny must be text or null"
                    )
                parsed_old = _cny_amount(
                    old_stop,
                    label="budget ledger old_balance_stop_cny",
                    allow_zero=True,
                )
                if old_stop != _decimal_text(parsed_old):
                    raise PilotBudgetStop(
                        "budget ledger old_balance_stop_cny is not canonical"
                    )
            new_stop = amendment.get("new_balance_stop_cny")
            if not isinstance(new_stop, str):
                raise PilotBudgetStop(
                    "budget ledger new_balance_stop_cny must be text"
                )
            parsed_new = _cny_amount(
                new_stop,
                label="budget ledger new_balance_stop_cny",
                allow_zero=True,
            )
            if new_stop != _decimal_text(parsed_new):
                raise PilotBudgetStop(
                    "budget ledger new_balance_stop_cny is not canonical"
                )
        except PilotBudgetStop as exc:
            raise PilotBudgetStop(f"invalid budget ledger: {path}") from exc
        if amendment_id in seen_ids:
            raise PilotBudgetStop(f"invalid budget ledger: {path}")
        seen_ids.add(amendment_id)
    return amendments


def _validate_ledger_common(
    ledger: dict[str, Any],
    path: Path,
    cap: Decimal,
) -> tuple[int, Decimal, list[Any], dict[str, Any]]:
    """Validate fields shared by normal snapshots and explicit migrations."""

    schema_version = ledger.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or schema_version not in (1, 2, _LEDGER_SCHEMA_VERSION)
        or ledger.get("provider") != "deepseek_official"
        or ledger.get("currency") != "CNY"
    ):
        raise PilotBudgetStop(f"invalid budget ledger: {path}")
    try:
        ledger_cap = _cny_amount(
            ledger.get("cap_cny"),
            label="budget ledger cap_cny",
            allow_zero=False,
        )
        starting = _cny_amount(
            ledger.get("starting_balance_cny"),
            label="budget ledger starting_balance_cny",
            allow_zero=True,
        )
        for field in ("latest_balance_cny", "observed_spend_cny"):
            if field in ledger:
                _cny_amount(
                    ledger[field],
                    label=f"budget ledger {field}",
                    allow_zero=True,
                )
    except PilotBudgetStop as exc:
        raise PilotBudgetStop(f"invalid budget ledger: {path}") from exc
    if ledger_cap != cap:
        raise PilotBudgetStop(
            "budget cap differs from the frozen ledger; use a new ledger path"
        )
    events = ledger.get("events")
    if not isinstance(events, list) or any(not isinstance(event, dict) for event in events):
        raise PilotBudgetStop(f"invalid budget ledger: {path}")
    active = _active_reservations(ledger, path)
    active_total = _active_total(active)
    if "active_reserved_cny" in ledger:
        try:
            recorded_active = _cny_amount(
                ledger["active_reserved_cny"],
                label="budget ledger active_reserved_cny",
                allow_zero=True,
            )
        except PilotBudgetStop as exc:
            raise PilotBudgetStop(f"invalid budget ledger: {path}") from exc
        if recorded_active != active_total:
            raise PilotBudgetStop(f"invalid budget ledger: {path}")
    if (
        schema_version == _LEDGER_SCHEMA_VERSION
        and _POLICY_AMENDMENTS_FIELD not in ledger
    ):
        raise PilotBudgetStop(f"invalid budget ledger: {path}")
    _validate_policy_amendments(ledger, path)
    return int(schema_version), starting, events, active


def _frozen_balance_stop(ledger: dict[str, Any], path: Path) -> Decimal:
    if (
        ledger.get("schema_version") != _LEDGER_SCHEMA_VERSION
        or "balance_stop_cny" not in ledger
    ):
        raise PilotBudgetStop(
            "budget ledger requires explicit migrate_balance_stop_policy() "
            "before new paid work"
        )
    try:
        return _cny_amount(
            ledger["balance_stop_cny"],
            label="budget ledger balance_stop_cny",
            allow_zero=True,
        )
    except PilotBudgetStop as exc:
        raise PilotBudgetStop(f"invalid budget ledger: {path}") from exc


def migrate_balance_stop_policy(
    *,
    expected_old_balance_stop_cny: object | None,
    new_balance_stop_cny: object,
    amendment_id: str,
    old_protocol_hash: str,
    new_protocol_hash: str,
) -> dict[str, Any]:
    """Explicitly CAS-migrate the frozen balance floor without network I/O.

    ``expected_old_balance_stop_cny=None`` means that the old ledger must be a
    legacy schema with no frozen floor.  Repeating the exact same amendment is
    idempotent; reusing its ID with different content fails closed.
    """

    configured_stop = _balance_stop_from_env()
    new_stop = _cny_amount(
        new_balance_stop_cny,
        label="new_balance_stop_cny",
        allow_zero=True,
    )
    if configured_stop != new_stop:
        raise PilotBudgetStop(
            f"{_BALANCE_STOP_ENV} must equal new_balance_stop_cny for migration"
        )
    expected_old = (
        None
        if expected_old_balance_stop_cny is None
        else _cny_amount(
            expected_old_balance_stop_cny,
            label="expected_old_balance_stop_cny",
            allow_zero=True,
        )
    )
    amendment_id = _policy_text(amendment_id, label="amendment_id")
    old_protocol_hash = _policy_text(
        old_protocol_hash,
        label="old_protocol_hash",
    )
    new_protocol_hash = _policy_text(
        new_protocol_hash,
        label="new_protocol_hash",
    )
    identity = {
        "amendment_id": amendment_id,
        "old_balance_stop_cny": (
            None if expected_old is None else _decimal_text(expected_old)
        ),
        "new_balance_stop_cny": _decimal_text(new_stop),
        "old_protocol_hash": old_protocol_hash,
        "new_protocol_hash": new_protocol_hash,
    }
    cap = _cap()
    path = _ledger_path()
    with _ledger_file_lock(path):
        ledger = _read_ledger(path)
        if ledger is None:
            raise PilotBudgetStop(
                "budget ledger does not exist; initialize a new ledger instead of migrating"
            )
        schema_version, _starting, events, active = _validate_ledger_common(
            ledger,
            path,
            cap,
        )
        if active:
            raise PilotBudgetStop(
                "budget ledger contains active reservations; settle them before migration"
            )
        amendments = _validate_policy_amendments(ledger, path)
        for amendment in amendments:
            if amendment.get("amendment_id") != amendment_id:
                continue
            recorded_identity = {
                key: amendment.get(key)
                for key in identity
            }
            if (
                recorded_identity == identity
                and schema_version == _LEDGER_SCHEMA_VERSION
                and _frozen_balance_stop(ledger, path) == new_stop
            ):
                return ledger
            raise PilotBudgetStop(
                "balance-stop amendment_id was already used with different policy data"
            )

        old_missing = "balance_stop_cny" not in ledger
        if old_missing:
            if schema_version == _LEDGER_SCHEMA_VERSION:
                raise PilotBudgetStop(f"invalid budget ledger: {path}")
            old_stop = None
        else:
            try:
                old_stop = _cny_amount(
                    ledger["balance_stop_cny"],
                    label="budget ledger balance_stop_cny",
                    allow_zero=True,
                )
            except PilotBudgetStop as exc:
                raise PilotBudgetStop(f"invalid budget ledger: {path}") from exc
        if expected_old is None:
            expected_matches = old_missing
        else:
            expected_matches = not old_missing and old_stop == expected_old
        if not expected_matches:
            raise PilotBudgetStop(
                "balance-stop policy compare-and-swap failed: expected old floor "
                "does not match the ledger"
            )
        if old_stop == new_stop:
            raise PilotBudgetStop(
                "balance-stop policy already has the requested floor under a "
                "different amendment lineage"
            )

        migrated_at = _now_shanghai().isoformat()
        amendment = {**identity, "migrated_at": migrated_at}
        amendments.append(amendment)
        migration_event = {
            "at": migrated_at,
            "kind": "migrate_balance_stop_policy",
            "reserve_cny": "0",
            "active_reserve_before_cny": "0",
            "active_reserve_after_cny": "0",
            **identity,
        }
        for field in ("latest_balance_cny", "observed_spend_cny"):
            if field in ledger:
                event_field = (
                    "balance_cny" if field == "latest_balance_cny" else field
                )
                migration_event[event_field] = ledger[field]
        events.append(migration_event)
        ledger["schema_version"] = _LEDGER_SCHEMA_VERSION
        ledger["balance_stop_cny"] = _decimal_text(new_stop)
        ledger[_POLICY_AMENDMENTS_FIELD] = amendments
        ledger["active_reserved_cny"] = str(_active_total(active))
        ledger["updated_at"] = migrated_at
        _atomic_write(path, ledger)
        return ledger


def _snapshot(
    kind: str,
    *,
    reserve_cny: Decimal,
    initialize_only: bool,
    create_reservation: str | None = None,
    release_reservation: str | None = None,
    reject_active: bool = False,
) -> dict[str, Any]:
    reserve_cny = _cny_amount(
        reserve_cny,
        label="reserve_cny",
        allow_zero=True,
    )
    configured_stop = _balance_stop_from_env()
    cap = _cap()
    path = _ledger_path()
    if release_reservation is None:
        assert_offpeak()
    if create_reservation is not None and release_reservation is not None:
        raise PilotBudgetStop("cannot create and release a budget reservation together")
    with _ledger_file_lock(path):
        current = _cny_amount(
            _fetch_cny_balance(),
            label="DeepSeek CNY balance",
            allow_zero=True,
        )
        ledger = _read_ledger(path)
        if ledger is None:
            ledger = {
                "schema_version": _LEDGER_SCHEMA_VERSION,
                "provider": "deepseek_official",
                "currency": "CNY",
                "cap_cny": str(cap),
                "balance_stop_cny": _decimal_text(configured_stop),
                _POLICY_AMENDMENTS_FIELD: [],
                "starting_balance_cny": str(current),
                "created_at": _now_shanghai().isoformat(),
                "events": [],
                "active_reservations": {},
            }
        _schema_version, starting, events, active = _validate_ledger_common(
            ledger,
            path,
            cap,
        )
        frozen_stop: Decimal | None = None
        if release_reservation is None:
            frozen_stop = _frozen_balance_stop(ledger, path)
            if frozen_stop != configured_stop:
                raise PilotBudgetStop(
                    "balance stop threshold differs from the frozen ledger; "
                    "call migrate_balance_stop_policy() explicitly or use a new ledger path"
                )
        if reject_active and active:
            raise PilotBudgetStop(
                "budget ledger contains active reservations from an unfinished "
                "run; inspect the paid attempts before resuming"
            )
        spent = max(Decimal("0"), starting - current)
        active_before = _active_total(active)

        if release_reservation is not None:
            if release_reservation not in active:
                raise PilotBudgetStop(
                    "budget reservation is missing or already settled; "
                    "refusing ambiguous accounting"
                )
            del active[release_reservation]
        elif create_reservation is not None:
            if not create_reservation or create_reservation in active:
                raise PilotBudgetStop("budget reservation token is invalid or duplicated")
            projected = spent + active_before + reserve_cny
            if not initialize_only and projected > cap:
                raise PilotBudgetStop(
                    f"pilot budget guard denied {kind}: observed spend {spent} CNY + "
                    f"active reserve {active_before} CNY + new reserve {reserve_cny} "
                    f"CNY exceeds cap {cap} CNY"
                )
            assert frozen_stop is not None
            if current <= frozen_stop:
                raise PilotBudgetStop(
                    f"DeepSeek balance {current} CNY is at or below frozen stop "
                    f"floor {frozen_stop} CNY for {kind}"
                )
            active[create_reservation] = {
                "kind": kind,
                "reserve_cny": str(reserve_cny),
                "created_at": _now_shanghai().isoformat(),
                "pid": os.getpid(),
            }

        active_after = _active_total(active)
        event = {
            "at": _now_shanghai().isoformat(),
            "kind": kind,
            "balance_cny": str(current),
            "observed_spend_cny": str(spent),
            "reserve_cny": str(reserve_cny),
            "active_reserve_before_cny": str(active_before),
            "active_reserve_after_cny": str(active_after),
        }
        if create_reservation is not None:
            event["reservation_action"] = "create"
            event["reservation_token"] = create_reservation
        elif release_reservation is not None:
            event["reservation_action"] = "release"
            event["reservation_token"] = release_reservation
        events.append(event)
        ledger["latest_balance_cny"] = str(current)
        ledger["observed_spend_cny"] = str(spent)
        ledger["active_reserved_cny"] = str(active_after)
        ledger["updated_at"] = event["at"]
        _atomic_write(path, ledger)
        if (
            create_reservation is None
            and release_reservation is None
            and not initialize_only
            and spent + active_after + reserve_cny > cap
        ):
            raise PilotBudgetStop(
                f"pilot budget guard denied {kind}: observed spend {spent} CNY + "
                f"active reserve {active_after} CNY + reserve {reserve_cny} CNY "
                f"exceeds cap {cap} CNY"
            )
        return ledger


def initialize() -> dict[str, Any]:
    """Freeze the starting account balance without authorizing a model call."""

    if not enabled():
        return {"enabled": False}
    return _snapshot(
        "initialize",
        reserve_cny=Decimal("0"),
        initialize_only=True,
        reject_active=True,
    )


def before_meta_request() -> str | None:
    if enabled():
        reserve = _reserve_from_env("SKILLGEN_META_REQUEST_RESERVE_CNY", "5")
        token = uuid.uuid4().hex
        _snapshot(
            "before_meta_request",
            reserve_cny=reserve,
            initialize_only=False,
            create_reservation=token,
        )
        return token
    return None


def before_agent_rollout() -> str | None:
    if enabled():
        reserve = _reserve_from_env("SKILLGEN_AGENT_ROLLOUT_RESERVE_CNY", "30")
        token = uuid.uuid4().hex
        _snapshot(
            "before_agent_rollout",
            reserve_cny=reserve,
            initialize_only=False,
            create_reservation=token,
        )
        return token
    return None


def record_balance(kind: str, *, reservation_token: str | None = None) -> None:
    if enabled():
        _snapshot(
            kind,
            reserve_cny=Decimal("0"),
            initialize_only=True,
            release_reservation=reservation_token,
            reject_active=(kind == "pilot_complete" and reservation_token is None),
        )
