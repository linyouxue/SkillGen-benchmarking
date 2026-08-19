from __future__ import annotations

import json
import os
import tempfile
import unittest
from copy import deepcopy
from decimal import Decimal
from datetime import datetime
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

import yaml

import pilot_budget_guard
from scripts import run_skillsbench_family as family_runner
from benchmarks.skillsbench_adapter import build_benchflow_command
from scripts.prepare_skillsbench_family import build_payloads
from scripts.run_skillsbench_family import FamilySpec, validate_protocol


def make_task(root: Path, task_id: str) -> Path:
    task = root / task_id
    (task / "environment").mkdir(parents=True)
    (task / "verifier").mkdir()
    (task / "task.md").write_text(
        "---\nschema_version: '1.3'\n---\n\nComplete the quantitative task.\n",
        encoding="utf-8",
    )
    (task / "environment" / "Dockerfile").write_text(
        "FROM python:3.12\n", encoding="utf-8"
    )
    (task / "verifier" / "test.sh").write_text(
        "#!/bin/sh\nexit 0\n", encoding="utf-8"
    )
    return task


def config_payload() -> dict:
    meta = "deepseek-v4-flash"
    agent = "deepseek/deepseek-v4-flash"
    return {
        "models": {
            "default": meta,
            "baseline_agent": agent,
            "baseline_judge": meta,
            "induction": meta,
            "induction_contextual": meta,
            "induction_summary": meta,
            "induction_pattern": meta,
            "induction_contrastive": meta,
            "generation_plan": meta,
            "generation_execute": meta,
            "refinement": meta,
            "verification_agent": agent,
            "verification_judge": meta,
            "verification_case_analyst": meta,
            "verification_revision_synthesiser": meta,
        },
        "generation": {"use_web_search": False, "generate_scripts": False},
        "verification": {"min_net_gain_abs": 2, "min_net_gain_rel": 0.0},
        "verification_analysis": {"case_analyst_workers": 3},
        "router": {"enabled": False},
        "pipeline": {"max_refine_rounds": 8, "max_workers": 3},
        "experiment": {"balance_stop_cny": 2},
    }


class FamilyProtocolTests(unittest.TestCase):
    def _prepared(self, root: Path):
        task_ids = ["source-a", "source-b", "source-c", "source-d", "target"]
        for task_id in task_ids:
            make_task(root, task_id)
        # This test uses a direct heldout id rather than the production hash
        # rule, so discover a seed for which target is the minimum.
        seed = next(
            str(value)
            for value in range(10000)
            if min(
                task_ids,
                key=lambda task_id: __import__("hashlib").sha256(
                    f"{value}|13|{task_id}".encode()
                ).hexdigest(),
            )
            == "target"
        )
        return build_payloads(
            tasks_root=root,
            family_id="family-13",
            family_number=13,
            family_label="Economic family",
            coherence="A",
            induction_allocations={
                "source-a": 3,
                "source-b": 3,
                "source-c": 2,
                "source-d": 2,
            },
            verification_allocations={
                "source-a": 2,
                "source-b": 2,
                "source-c": 3,
                "source-d": 3,
            },
            heldout_task_id="target",
            heldout_rollouts=10,
            selection_seed=seed,
            agent="openhands",
            sandbox="docker",
            jobs_root=root / "jobs",
            bench_executable="bench",
            subprocess_timeout_sec=7200,
            source_version="v1.1",
        )

    def _migration_fixture(self, root: Path):
        instance_ids = [f"slot-{index:02d}" for index in range(10)]
        jobs_root = root / "jobs"
        (jobs_root / ".skillgen-rollout-cache" / "locks").mkdir(parents=True)
        induction = {
            "dataset_id": "frozen-induction",
            "task_name": "Economic family",
            "instances": [
                {
                    "instance_id": item,
                    "metadata": {"skillsbench_jobs_root": str(jobs_root)},
                }
                for item in instance_ids
            ],
        }
        verification_ids = [f"verification-{index:02d}" for index in range(10)]
        verification = {
            "dataset_id": "frozen-verification",
            "task_name": "Economic family",
            "instances": [
                {
                    "instance_id": item,
                    "metadata": {"skillsbench_jobs_root": str(jobs_root)},
                }
                for item in verification_ids
            ],
        }
        paths = {
            "induction": root / "induction.json",
            "verification": root / "verification.json",
            "heldout": root / "heldout.json",
            "manifest": root / "manifest.json",
            "config": root / "config.yaml",
        }
        paths["induction"].write_text(json.dumps(induction), encoding="utf-8")
        paths["verification"].write_text(json.dumps(verification), encoding="utf-8")
        paths["heldout"].write_text("{}", encoding="utf-8")
        paths["manifest"].write_text("{}", encoding="utf-8")
        amended_config_bytes = (
            family_runner.REPO_ROOT
            / "config.skillsbench.deepseek-v4-flash-family-pilot.yaml"
        ).read_bytes()
        paths["config"].write_bytes(amended_config_bytes)
        spec = FamilySpec(
            family_id="family-13-economic-financial-quant",
            induction_dataset=paths["induction"],
            verification_dataset=paths["verification"],
            heldout_dataset=paths["heldout"],
            manifest_path=paths["manifest"],
            config_path=paths["config"],
            run_root=root / "runs",
            resume=True,
            authorize_budget_policy_amendment=True,
        )
        run_dir = spec.fold_root / "pipeline-runs" / "20260817-141302"
        run_dir.mkdir(parents=True)
        metadata = {
            "dataset_id": induction["dataset_id"],
            "task_name": induction["task_name"],
            "dataset_metadata": {
                "protocol_hash": family_runner.V12_PRE_BALANCE_STOP_PROTOCOL_HASH,
                "family_id": spec.family_id,
            },
            "n_instances": 10,
            "n_verification_instances": 10,
            "separate_verification_pool": True,
            "exhaustive_refinement": True,
            "unrelated_frozen_field": {"must": "survive byte-for-byte"},
        }
        (run_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        (run_dir / "checkpoint.json").write_text(
            json.dumps(
                {
                    "stage": "analysis_done",
                    "total_stages": 1,
                    "completed_stages": ["analysis"],
                }
            ),
            encoding="utf-8",
        )
        baseline_payload = "".join(
            json.dumps({"instance_id": item}) + "\n" for item in instance_ids
        )
        for name in ("checkpoint_trajectories.jsonl", "baseline_trajectories.jsonl"):
            (run_dir / name).write_text(baseline_payload, encoding="utf-8")
        (run_dir / "verification_baseline_trajectories.jsonl").write_text(
            "".join(
                json.dumps({"instance_id": item}) + "\n" for item in verification_ids
            ),
            encoding="utf-8",
        )
        (run_dir / "analysis").mkdir()
        (run_dir / "analysis" / "skill_analysis.json").write_text(
            '{"failure_clusters": []}', encoding="utf-8"
        )
        (run_dir / "analysis" / "skill_analysis_summary.json").write_text(
            '{"summary": "frozen"}', encoding="utf-8"
        )
        (run_dir / "refinement_checkpoint.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "completed_rounds": 3,
                    "in_progress_round": 3,
                    "max_rounds": 8,
                    "exhaustive_refinement": True,
                    "candidate": {"body": "round-4"},
                    "round_history": [{"round": item} for item in range(1, 4)],
                }
            ),
            encoding="utf-8",
        )
        cache_path = run_dir / "verification" / "round_4" / "cache.json"
        cache_path.parent.mkdir(parents=True)
        cache_path.write_text('{"scored_slots": 5}', encoding="utf-8")
        score_path = run_dir / "verification" / "round_4" / "score.json"
        score_path.write_text('{"reward": 1}', encoding="utf-8")

        old_protocol = {
            "agent_model": "deepseek/deepseek-v4-flash",
            "budget_cny": "120",
            "budget_guard_semantics": family_runner._OLD_BUDGET_GUARD_SEMANTICS,
            "candidate_rounds": 8,
            "chat_model": "deepseek-v4-flash",
            "code_sha256": {
                "benchmarks/skillsbench_adapter.py": "dbad396c962ced0ddd6c0f180cf9da79a3f69948be4ab2d13c3b963170b8eccd",
                "llm.py": "3c680713132dac7bb7b62a826bdc7d85d04bcecda9fb0e759d1b790cc7aca134",
                "pilot_budget_guard.py": "295401bc1e1de91947c6ce18938903313a1a49ec2a9108aeb778c27540c99f56",
                "pipeline.py": "ef6a3cbddf04dab0f480ff82ab659e41358eab5739864a11793058bd33ade62c",
                "scripts/run_skillsbench_family.py": "fdecae5bfee2f4477eee8010791cbbeec4a1eddfc9123c78e719957b89677975",
            },
            "config_sha256": family_runner.V12_PRE_BALANCE_STOP_CONFIG_SHA256,
            "family_id": spec.family_id,
            "gate_net_gain": 2,
            "hash": family_runner.V12_PRE_BALANCE_STOP_PROTOCOL_HASH,
            "heldout_sha256": "c83f26a8b404945ef3559fafc32ef0e1f737002a150be43217a03cad34df0620",
            "heldout_skill_conditional": True,
            "induction_sha256": "5cd2529e64e973a5d00a761aa21b8cf84865eec8d5ec7c856df551d7ad400dfc",
            "manifest_sha256": "5784c8d7c51852c7ba867d4a02d9ef81ffbdd870965b2317fba32fe4cb251f28",
            "protocol_version": family_runner.PROTOCOL_VERSION,
            "provider": "deepseek_official",
            "rolling_fail_stop": True,
            "runtime_source_tree_sha256": "060df2a23f2e96ba9d7f449eb808cc2637585090e8194563a010fef7832c6886",
            "soft_reserve_cny": {"meta_request": "5", "agent_rollout": "10"},
            "stage_max_workers": 3,
            "verification_sha256": "f800a9d2b8cdecd2fdf1c802e4a75f430143096a740e7cc0118be5a7ab1cbf53",
        }
        new_protocol = deepcopy(old_protocol)
        new_protocol.update(
            {
                "config_sha256": __import__("hashlib").sha256(
                    amended_config_bytes
                ).hexdigest(),
                "runtime_source_tree_sha256": "new-runtime-tree",
                "budget_guard_semantics": family_runner._BUDGET_GUARD_SEMANTICS,
                "balance_stop_cny": "2",
            }
        )
        new_protocol["code_sha256"] = {
            **old_protocol["code_sha256"],
            "pilot_budget_guard.py": "new-guard",
            "scripts/run_skillsbench_family.py": "new-runner",
        }
        new_protocol["hash"] = family_runner._canonical_sha256(
            {key: value for key, value in new_protocol.items() if key != "hash"}
        )
        spec = FamilySpec(
            **{
                **spec.__dict__,
                "expected_new_protocol_hash": new_protocol["hash"],
            }
        )
        status = {
            "schema_version": 1,
            "family_id": spec.family_id,
            "protocol_hash": family_runner.V12_PRE_BALANCE_STOP_PROTOCOL_HASH,
            "protocol": old_protocol,
            "stage": "budget_stopped",
            "pipeline_run_dir": str(run_dir.resolve()),
            "failure": {
                "kind": "PilotBudgetStop",
                "reason": "balance below the previous reserve",
            },
        }
        family_runner._atomic_json(family_runner._status_path(spec), status)
        return spec, status, old_protocol, new_protocol, run_dir, cache_path, score_path

    def _migrated_ledger(self, context: dict, new_protocol: dict) -> dict:
        return {
            "schema_version": 3,
            "balance_stop_cny": "2",
            "active_reservations": {},
            "active_reserved_cny": "0",
            "balance_stop_policy_amendments": [
                {
                    "amendment_id": context["amendment_id"],
                    "old_balance_stop_cny": None,
                    "new_balance_stop_cny": "2",
                    "old_protocol_hash": family_runner.V12_PRE_BALANCE_STOP_PROTOCOL_HASH,
                    "new_protocol_hash": new_protocol["hash"],
                    "migrated_at": "2026-08-18T12:00:00+08:00",
                }
            ],
        }

    def test_family_payloads_are_task_disjoint_and_exactly_allocated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            induction, verification, heldout, manifest = self._prepared(Path(temporary))
            self.assertEqual(10, len(induction["instances"]))
            self.assertEqual(10, len(verification["instances"]))
            self.assertEqual(10, len(heldout["instances"]))
            ids = [
                {row["instance_id"] for row in payload["instances"]}
                for payload in (induction, verification, heldout)
            ]
            self.assertFalse(ids[0] & ids[1])
            self.assertFalse(ids[0] & ids[2])
            self.assertFalse(ids[1] & ids[2])
            self.assertEqual("target", manifest["heldout_task_id"])
            self.assertTrue(manifest["gate"]["heldout_skill_is_conditional_on_gate"])

    def test_runner_validation_rejects_heldout_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payloads = self._prepared(root)
            paths = [root / name for name in ("ind.json", "ver.json", "held.json", "manifest.json")]
            for path, payload in zip(paths, payloads, strict=True):
                path.write_text(json.dumps(payload), encoding="utf-8")
            config = root / "config.yaml"
            config.write_text(yaml.safe_dump(config_payload()), encoding="utf-8")
            spec = FamilySpec(
                family_id="family-13",
                induction_dataset=paths[0],
                verification_dataset=paths[1],
                heldout_dataset=paths[2],
                manifest_path=paths[3],
                config_path=config,
                run_root=root / "runs",
            )
            protocol, _ = validate_protocol(spec)
            self.assertEqual(
                {"meta_request": "5", "agent_rollout": "10"},
                protocol["soft_reserve_cny"],
            )
            self.assertEqual("2", protocol["balance_stop_cny"])
            self.assertEqual(3, protocol["stage_max_workers"])
            induction = json.loads(paths[0].read_text(encoding="utf-8"))
            induction["instances"][0]["metadata"]["skillsbench_task_id"] = "target"
            paths[0].write_text(json.dumps(induction), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "membership"):
                validate_protocol(spec)

    def test_config_requires_exact_frozen_balance_stop(self) -> None:
        payload = config_payload()
        family_runner._validate_config(payload)
        for invalid in (None, 0, 2.0, "2", True):
            changed = deepcopy(payload)
            if invalid is None:
                changed["experiment"].pop("balance_stop_cny")
            else:
                changed["experiment"]["balance_stop_cny"] = invalid
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "balance_stop_cny=2"):
                    family_runner._validate_config(changed)

    def test_v12_budget_policy_amendment_is_atomic_append_only_and_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            (
                spec,
                status,
                _old_protocol,
                new_protocol,
                run_dir,
                cache_path,
                score_path,
            ) = self._migration_fixture(Path(temporary))
            protected_paths = [
                run_dir / "checkpoint.json",
                run_dir / "checkpoint_trajectories.jsonl",
                run_dir / "baseline_trajectories.jsonl",
                run_dir / "verification_baseline_trajectories.jsonl",
                run_dir / "analysis" / "skill_analysis.json",
                run_dir / "analysis" / "skill_analysis_summary.json",
                run_dir / "refinement_checkpoint.json",
                cache_path,
                score_path,
            ]
            protected_before = {path: path.read_bytes() for path in protected_paths}
            metadata_before = (run_dir / "run_metadata.json").read_bytes()
            context = family_runner._prepare_budget_policy_amendment(
                spec, status, new_protocol
            )
            migrated_ledger = self._migrated_ledger(context, new_protocol)
            with mock.patch.object(
                pilot_budget_guard,
                "migrate_balance_stop_policy",
                return_value=migrated_ledger,
            ) as migrate:
                amended_status = family_runner._apply_budget_policy_amendment(
                    spec, status, new_protocol, context
                )

            migrate.assert_called_once_with(
                expected_old_balance_stop_cny=None,
                new_balance_stop_cny=Decimal("2"),
                amendment_id=context["amendment_id"],
                old_protocol_hash=family_runner.V12_PRE_BALANCE_STOP_PROTOCOL_HASH,
                new_protocol_hash=new_protocol["hash"],
            )
            self.assertEqual(new_protocol["hash"], amended_status["protocol_hash"])
            self.assertEqual(new_protocol, amended_status["protocol"])
            metadata_after = (run_dir / "run_metadata.json").read_bytes()
            self.assertEqual(len(metadata_before), len(metadata_after))
            self.assertEqual(
                metadata_before.replace(
                    family_runner.V12_PRE_BALANCE_STOP_PROTOCOL_HASH.encode("ascii"),
                    new_protocol["hash"].encode("ascii"),
                    1,
                ),
                metadata_after,
            )
            self.assertEqual(
                protected_before,
                {path: path.read_bytes() for path in protected_paths},
            )
            amendment = json.loads(
                family_runner._budget_policy_amendment_path(spec).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(["authorized", "applied"], [e["event"] for e in amendment["events"]])
            authorized, applied = amendment["events"]
            self.assertTrue(authorized["authorization"]["authorized"])
            self.assertEqual(
                "--authorize-budget-policy-amendment",
                authorized["authorization"]["flag"],
            )
            self.assertEqual(
                family_runner.V12_PRE_BALANCE_STOP_PROTOCOL_HASH,
                authorized["old_protocol_hash"],
            )
            self.assertEqual(new_protocol["hash"], authorized["new_protocol_hash"])
            self.assertEqual(
                __import__("hashlib").sha256(metadata_before).hexdigest(),
                applied["run_metadata_sha256_before"],
            )
            self.assertEqual(
                __import__("hashlib").sha256(metadata_after).hexdigest(),
                applied["run_metadata_sha256_after"],
            )

    def test_v12_budget_policy_amendment_recovers_without_duplicate_events(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            spec, status, _, new_protocol, _, _, _ = self._migration_fixture(
                Path(temporary)
            )
            first_context = family_runner._prepare_budget_policy_amendment(
                spec, status, new_protocol
            )
            ledger = self._migrated_ledger(first_context, new_protocol)
            with mock.patch.object(
                pilot_budget_guard, "migrate_balance_stop_policy", return_value=ledger
            ):
                family_runner._apply_budget_policy_amendment(
                    spec, status, new_protocol, first_context
                )
                recovery_context = family_runner._prepare_budget_policy_amendment(
                    spec, status, new_protocol
                )
                family_runner._apply_budget_policy_amendment(
                    spec, status, new_protocol, recovery_context
                )
            amendment = json.loads(
                family_runner._budget_policy_amendment_path(spec).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(2, len(amendment["events"]))

    def test_v12_budget_policy_amendment_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            spec, status, _, new_protocol, run_dir, _, _ = self._migration_fixture(
                Path(temporary)
            )
            metadata_path = run_dir / "run_metadata.json"
            metadata_before = metadata_path.read_bytes()

            unauthorized = FamilySpec(
                **{
                    **spec.__dict__,
                    "authorize_budget_policy_amendment": False,
                }
            )
            with self.assertRaisesRegex(RuntimeError, "explicit"):
                family_runner._prepare_budget_policy_amendment(
                    unauthorized, status, new_protocol
                )

            wrong_hash = FamilySpec(
                **{
                    **spec.__dict__,
                    "expected_new_protocol_hash": "b" * 64,
                }
            )
            with self.assertRaisesRegex(RuntimeError, "expected-new-protocol-hash"):
                family_runner._prepare_budget_policy_amendment(
                    wrong_hash, status, new_protocol
                )

            unrelated_protocol = deepcopy(new_protocol)
            unrelated_protocol["stage_max_workers"] = 2
            unrelated_protocol["hash"] = family_runner._canonical_sha256(
                {
                    key: value
                    for key, value in unrelated_protocol.items()
                    if key != "hash"
                }
            )
            unrelated_spec = FamilySpec(
                **{
                    **spec.__dict__,
                    "expected_new_protocol_hash": unrelated_protocol["hash"],
                }
            )
            with self.assertRaisesRegex(RuntimeError, "non-policy"):
                family_runner._prepare_budget_policy_amendment(
                    unrelated_spec, status, unrelated_protocol
                )

            context = family_runner._prepare_budget_policy_amendment(
                spec, status, new_protocol
            )
            with mock.patch.object(
                pilot_budget_guard,
                "migrate_balance_stop_policy",
                side_effect=pilot_budget_guard.PilotBudgetStop(
                    "budget ledger contains active reservations"
                ),
            ):
                with self.assertRaisesRegex(
                    pilot_budget_guard.PilotBudgetStop, "active reservations"
                ):
                    family_runner._apply_budget_policy_amendment(
                        spec, status, new_protocol, context
                    )
            self.assertEqual(metadata_before, metadata_path.read_bytes())
            self.assertFalse(
                family_runner._budget_policy_amendment_path(spec).exists()
            )
            self.assertEqual(
                family_runner.V12_PRE_BALANCE_STOP_PROTOCOL_HASH,
                json.loads(family_runner._status_path(spec).read_text())["protocol_hash"],
            )

    def test_amendment_rejects_protocol_event_and_guard_lineage_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            spec, status, _, new_protocol, run_dir, _, _ = self._migration_fixture(
                Path(temporary)
            )
            tampered_status = deepcopy(status)
            tampered_status["protocol"]["code_sha256"]["pilot_budget_guard.py"] = (
                "tampered"
            )
            with self.assertRaisesRegex(RuntimeError, "content.*hash"):
                family_runner._prepare_budget_policy_amendment(
                    spec, tampered_status, new_protocol
                )

            context = family_runner._prepare_budget_policy_amendment(
                spec, status, new_protocol
            )
            family_runner._atomic_json(
                family_runner._budget_policy_amendment_path(spec),
                {
                    "schema_version": 1,
                    "events": [
                        context["authorization_event"],
                        {
                            "event": "unexpected",
                            "amendment_id": context["amendment_id"],
                        },
                    ],
                },
            )
            with self.assertRaisesRegex(RuntimeError, "exact prefix"):
                family_runner._prepare_budget_policy_amendment(
                    spec, status, new_protocol
                )
            family_runner._budget_policy_amendment_path(spec).unlink()

            bad_ledger = self._migrated_ledger(context, new_protocol)
            bad_ledger["balance_stop_policy_amendments"].append(
                {
                    **bad_ledger["balance_stop_policy_amendments"][0],
                    "amendment_id": "unrelated",
                }
            )
            metadata_before = (run_dir / "run_metadata.json").read_bytes()
            with mock.patch.object(
                pilot_budget_guard,
                "migrate_balance_stop_policy",
                return_value=bad_ledger,
            ):
                with self.assertRaisesRegex(RuntimeError, "non-exact"):
                    family_runner._apply_budget_policy_amendment(
                        spec, status, new_protocol, context
                    )
            self.assertEqual(
                metadata_before,
                (run_dir / "run_metadata.json").read_bytes(),
            )
            self.assertFalse(
                family_runner._budget_policy_amendment_path(spec).exists()
            )

    def test_applied_lineage_is_revalidated_on_every_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            spec, status, old_protocol, new_protocol, run_dir, _, _ = (
                self._migration_fixture(Path(temporary))
            )
            context = family_runner._prepare_budget_policy_amendment(
                spec, status, new_protocol
            )
            ledger = self._migrated_ledger(context, new_protocol)
            with mock.patch.object(
                pilot_budget_guard, "migrate_balance_stop_policy", return_value=ledger
            ):
                amended_status = family_runner._apply_budget_policy_amendment(
                    spec, status, new_protocol, context
                )
            self.assertEqual(
                old_protocol,
                amended_status["protocol_lineage"]["old_protocol"],
            )
            resume_context = family_runner._prepare_applied_budget_policy_lineage(
                spec, amended_status, new_protocol
            )
            with mock.patch.object(
                pilot_budget_guard, "migrate_balance_stop_policy", return_value=ledger
            ) as migrate:
                family_runner._verify_applied_budget_policy_lineage(
                    spec, new_protocol, resume_context
                )
            migrate.assert_called_once()

            refinement_path = run_dir / "refinement_checkpoint.json"
            refinement = json.loads(refinement_path.read_text())
            refinement.update(
                {
                    "completed_rounds": 4,
                    "in_progress_round": 4,
                    "candidate": {"body": "round-5"},
                    "round_history": [
                        {"round": item} for item in range(1, 5)
                    ],
                }
            )
            refinement_path.write_text(json.dumps(refinement), encoding="utf-8")
            family_runner._prepare_applied_budget_policy_lineage(
                spec, amended_status, new_protocol
            )

            amendment_path = family_runner._budget_policy_amendment_path(spec)
            amendment = json.loads(amendment_path.read_text())
            amendment["events"][0]["candidate_sha256"] = "b" * 64
            amendment_path.write_text(json.dumps(amendment), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "event"):
                family_runner._prepare_applied_budget_policy_lineage(
                    spec, amended_status, new_protocol
                )

    def test_amendment_rejects_cache_locks_and_concurrent_fold_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            spec, status, _, new_protocol, _, _, _ = self._migration_fixture(
                Path(temporary)
            )
            induction = json.loads(spec.induction_dataset.read_text())
            jobs_root = Path(
                induction["instances"][0]["metadata"]["skillsbench_jobs_root"]
            )
            cache_lock = jobs_root / ".skillgen-rollout-cache" / "locks" / "active.lock"
            cache_lock.write_text("active", encoding="utf-8")
            context = family_runner._prepare_budget_policy_amendment(
                spec, status, new_protocol
            )
            with (
                mock.patch.object(
                    pilot_budget_guard, "migrate_balance_stop_policy"
                ) as migrate,
                self.assertRaisesRegex(RuntimeError, "cache.*locks"),
            ):
                family_runner._apply_budget_policy_amendment(
                    spec, status, new_protocol, context
                )
            migrate.assert_not_called()
            cache_lock.unlink()

            with family_runner._budget_policy_amendment_lock(spec):
                with self.assertRaisesRegex(RuntimeError, "another process"):
                    with family_runner._budget_policy_amendment_lock(spec):
                        self.fail("nested fold lock unexpectedly succeeded")

    def test_cli_requires_explicit_budget_policy_authorization_flag(self) -> None:
        common = [
            "--family-id",
            "family-13",
            "--induction-dataset",
            "induction.json",
            "--verification-dataset",
            "verification.json",
            "--heldout-dataset",
            "heldout.json",
            "--manifest",
            "manifest.json",
            "--config",
            "config.yaml",
            "--run-root",
            "runs",
            "--resume",
        ]
        self.assertFalse(
            family_runner.parse_args(common).authorize_budget_policy_amendment
        )
        self.assertTrue(
            family_runner.parse_args(
                [
                    *common,
                    "--authorize-budget-policy-amendment",
                    "--expected-new-protocol-hash",
                    "a" * 64,
                ]
            ).authorize_budget_policy_amendment
        )
        self.assertEqual(
            "a" * 64,
            family_runner.parse_args(
                [*common, "--expected-new-protocol-hash", "a" * 64]
            ).expected_new_protocol_hash,
        )

    def test_runner_sets_frozen_balance_stop_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = FamilySpec(
                family_id="family-13",
                induction_dataset=root / "induction.json",
                verification_dataset=root / "verification.json",
                heldout_dataset=root / "heldout.json",
                manifest_path=root / "manifest.json",
                config_path=root / "config.yaml",
                run_root=root / "runs",
                resume=True,
            )
            protocol = {"hash": "frozen", "balance_stop_cny": "2"}
            status = {
                "schema_version": 1,
                "family_id": spec.family_id,
                "protocol_hash": protocol["hash"],
                "protocol": protocol,
                "stage": "prepared",
            }
            family_runner._atomic_json(family_runner._status_path(spec), status)
            with (
                mock.patch.object(
                    family_runner,
                    "validate_protocol",
                    return_value=(protocol, {}),
                ),
                mock.patch.object(
                    pilot_budget_guard,
                    "initialize",
                    side_effect=pilot_budget_guard.PilotBudgetStop("stop before paid work"),
                ),
                mock.patch.dict(
                    os.environ,
                    {
                        "DEEPSEEK_API_KEY": "test",
                        "OPENAI_API_KEY": "test",
                        "SKILLGEN_BUDGET_LEDGER": str(
                            (spec.fold_root / "budget_ledger.json").resolve()
                        ),
                    },
                    clear=False,
                ),
            ):
                with self.assertRaisesRegex(
                    pilot_budget_guard.PilotBudgetStop, "stop before paid work"
                ):
                    family_runner.run(spec, execute=True)
                self.assertEqual(
                    "2", os.environ["SKILLGEN_DEEPSEEK_BALANCE_STOP_CNY"]
                )

    def test_deepseek_openhands_command_sets_reasoning_effort(self) -> None:
        command = build_benchflow_command(
            bench_executable="bench",
            task_dir=Path("/tasks/demo"),
            jobs_dir=Path("/jobs/demo"),
            agent="openhands",
            model="deepseek/deepseek-v4-flash",
            sandbox="docker",
            skills_dir=None,
        )
        self.assertIn("LLM_REASONING_EFFORT=high", command)

    def test_peak_window_boundaries(self) -> None:
        zone = ZoneInfo("Asia/Shanghai")
        self.assertTrue(pilot_budget_guard._is_peak(datetime(2026, 8, 17, 9, 0, tzinfo=zone)))
        self.assertTrue(pilot_budget_guard._is_peak(datetime(2026, 8, 17, 17, 59, tzinfo=zone)))
        self.assertFalse(pilot_budget_guard._is_peak(datetime(2026, 8, 17, 18, 0, tzinfo=zone)))
        self.assertFalse(pilot_budget_guard._is_peak(datetime(2026, 8, 17, 12, 0, tzinfo=zone)))


if __name__ == "__main__":
    unittest.main()
