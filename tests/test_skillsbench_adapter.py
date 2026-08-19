from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import patch

from benchmarks.skillsbench_adapter import (
    SkillsBenchInfrastructureError,
    _benchflow_subprocess_env,
    _cached_task_digest,
    _trajectory_from_artifacts,
    _validate_benchflow_executable,
    acp_events_to_skillgen_messages,
    bootstrap_skillsbench_rollout_cache,
    build_benchflow_command,
    materialize_generated_skill,
    parse_task_markdown,
    run_skillsbench_agent,
    task_package_digest,
)
from benchmarks.skillsbench_rollout_cache import (
    RolloutCacheRequest,
    attempt_directory_name,
    cache_entry_path,
    read_attempt_receipt,
    write_attempt_manifest,
    write_attempt_receipt,
)
from models import SkillItem, TaskInstance
from scripts.prepare_skillsbench import build_payloads
from scripts.prepare_skillsbench_suite import discover_task_dirs


def make_task(root: Path, name: str = "demo-task") -> Path:
    task = root / name
    (task / "environment").mkdir(parents=True)
    (task / "verifier").mkdir()
    (task / "task.md").write_text(
        "---\nschema_version: '1.3'\nmetadata:\n  category: mathematics-or-formal-reasoning\n---\n\n"
        "Create /root/output/answer.txt with the requested answer.\n",
        encoding="utf-8",
    )
    (task / "environment" / "Dockerfile").write_text(
        "FROM python:3.12\n", encoding="utf-8"
    )
    (task / "verifier" / "test.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    return task


class SkillsBenchAdapterTests(unittest.TestCase):
    def _deepseek_callback_compat_harness(self, root: Path):
        with (
            patch.dict(os.environ, {"WSL_DISTRO_NAME": "Ubuntu"}, clear=False),
            patch(
                "benchmarks.skillsbench_adapter.platform.release",
                return_value="6.18.0-microsoft-standard-WSL2",
            ),
        ):
            subprocess_env = _benchflow_subprocess_env(
                sandbox="docker", run_root=root
            )
        shim_root = Path(subprocess_env["PYTHONPATH"].split(os.pathsep)[0])
        shim = (shim_root / "sitecustomize.py").read_text(encoding="utf-8")

        callback_source = """\
def _jsonable(value):
    return value

def _gate_opencode_skill_catalog(data):
    return None

class BenchFlowLiteLLMLogger:
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        if not isinstance(data, dict):
            return None

        _gate_opencode_skill_catalog(data)

        cleaned = data
        if cleaned is data:
            return None
        return cleaned

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        return None

proxy_handler_instance = BenchFlowLiteLLMLogger()
"""
        fake_runtime = ModuleType("benchflow.providers.litellm_runtime")
        fake_runtime._docker_host_address = lambda: "172.17.0.1"
        fake_runtime.callback_module_source = lambda: callback_source
        fake_logging = ModuleType("benchflow.providers.litellm_logging")
        fake_logging.callback_module_source = lambda: callback_source
        fake_providers = ModuleType("benchflow.providers")
        fake_providers.litellm_runtime = fake_runtime
        fake_providers.litellm_logging = fake_logging
        fake_registry = ModuleType("benchflow.agents.registry")
        fake_registry.AGENTS = {}
        fake_registry.AGENT_LAUNCH = {}
        fake_agents = ModuleType("benchflow.agents")
        fake_agents.registry = fake_registry
        fake_benchflow = ModuleType("benchflow")
        fake_benchflow.providers = fake_providers
        fake_benchflow.agents = fake_agents
        with (
            patch.dict(
                sys.modules,
                {
                    "benchflow": fake_benchflow,
                    "benchflow.agents": fake_agents,
                    "benchflow.agents.registry": fake_registry,
                    "benchflow.providers": fake_providers,
                    "benchflow.providers.litellm_logging": fake_logging,
                    "benchflow.providers.litellm_runtime": fake_runtime,
                },
            ),
            patch.dict(
                os.environ,
                {
                    "SKILLSBENCH_APPLY_WSL_DOCKER_COMPAT": "1",
                    "SKILLSBENCH_BENCHFLOW_HARDENING_COMPAT": "0",
                    "SKILLSBENCH_BENCHFLOW_TRAJECTORY_PUBLISH_COMPAT": "0",
                    "SKILLSBENCH_BENCHFLOW_FAILURE_FORENSICS": "0",
                    "SKILLSBENCH_DEEPSEEK_OPENHANDS_COMPAT": "1",
                },
                clear=False,
            ),
        ):
            exec(compile(shim, str(shim_root / "sitecustomize.py"), "exec"), {})

        patched_source = fake_logging.callback_module_source()
        namespace: dict[str, object] = {}
        exec(compile(patched_source, "benchflow_litellm_callback.py", "exec"), namespace)
        return namespace["proxy_handler_instance"]

    def _publish_compat_harness(
        self,
        root: Path,
        *,
        publish_error: Exception | None,
        target_state: str = "match",
        docker: bool = True,
        raw_target_state: str = "match",
        ps_stdout: str = "a1b2c3d4e5f6\n",
        wrapped_env: bool = True,
    ) -> SimpleNamespace:
        with (
            patch.dict(os.environ, {"WSL_DISTRO_NAME": "Ubuntu"}, clear=False),
            patch(
                "benchmarks.skillsbench_adapter.platform.release",
                return_value="6.18.0-microsoft-standard-WSL2",
            ),
        ):
            subprocess_env = _benchflow_subprocess_env(
                sandbox="docker", run_root=root
            )
        shim_root = Path(subprocess_env["PYTHONPATH"].split(os.pathsep)[0])
        shim = (shim_root / "sitecustomize.py").read_text(encoding="utf-8")

        def redact(trajectory):
            return "\n".join(
                json.dumps(event, sort_keys=True, separators=(",", ":"))
                for event in trajectory
            )

        original_calls = []

        async def original_publish(_env, trajectory, agent_dir):
            original_calls.append((trajectory, agent_dir))
            if publish_error is not None:
                raise publish_error
            return "original-result"

        probes = []
        raw_calls = []

        class DockerSandbox:
            def __init__(self, agent_dir):
                self.session_id = "shock-analysis-demand__publish"
                self._env_vars = SimpleNamespace(
                    host_agent_logs_path=str(
                        agent_dir
                        if target_state != "path_mismatch"
                        else root / "other-agent"
                    ),
                    env_agent_logs_path=(
                        "/logs/agent"
                        if target_state != "env_mismatch"
                        else "/not-the-agent-bind"
                    ),
                )

            async def _run_docker_compose_command(
                self, command, *, check=True, timeout_sec=None
            ):
                probes.append((command, check, timeout_sec))
                raise AssertionError("proactive publish must not use docker compose")

        DockerSandbox.__module__ = "benchflow.sandbox.docker"

        class ProcessSandbox(DockerSandbox):
            pass

        ProcessSandbox.__module__ = "benchflow.sandbox.process"

        agent_dir = root / "agent"
        agent_dir.mkdir(parents=True, exist_ok=True)
        inner = DockerSandbox(agent_dir) if docker else ProcessSandbox(agent_dir)
        fake_env = SimpleNamespace(inner=inner) if wrapped_env else inner
        fake_rollout = ModuleType("benchflow.rollout")
        fake_rollout._publish_trajectory_for_verifier = original_publish
        fake_setup = ModuleType("benchflow.rollout._setup")
        fake_setup._publish_trajectory_for_verifier = original_publish
        fake_rollout._setup = fake_setup
        fake_types = ModuleType("benchflow.trajectories.types")
        fake_types.redact_acp_trajectory_jsonl = redact
        fake_trajectories = ModuleType("benchflow.trajectories")
        fake_trajectories.types = fake_types
        fake_runtime = ModuleType("benchflow.providers.litellm_runtime")
        fake_runtime._docker_host_address = lambda: "172.17.0.1"
        fake_providers = ModuleType("benchflow.providers")
        fake_providers.litellm_runtime = fake_runtime
        fake_docker = ModuleType("benchflow.sandbox.docker")
        fake_docker.DockerSandbox = DockerSandbox
        fake_docker._sanitize_docker_compose_project_name = lambda value: value
        fake_sandbox = ModuleType("benchflow.sandbox")
        fake_sandbox.docker = fake_docker
        fake_benchflow = ModuleType("benchflow")
        fake_benchflow.providers = fake_providers
        fake_benchflow.rollout = fake_rollout
        fake_benchflow.sandbox = fake_sandbox
        fake_benchflow.trajectories = fake_trajectories

        def raw_run(command, **kwargs):
            raw_calls.append((list(command), kwargs))
            if command[:3] == ["docker", "ps", "-q"]:
                return SimpleNamespace(returncode=0, stdout=ps_stdout, stderr="")
            if command[:2] != ["docker", "exec"]:
                raise AssertionError(f"unexpected raw command: {command!r}")
            if raw_target_state == "missing":
                return SimpleNamespace(returncode=1, stdout="", stderr="missing")
            payload = (agent_dir / "acp_trajectory.jsonl").read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            if raw_target_state == "mismatch":
                digest = "0" * 64
            return SimpleNamespace(
                returncode=0,
                stdout=f"{len(payload)}\n{digest}  "
                "/logs/agent/acp_trajectory.jsonl\n",
                stderr="",
            )

        modules = {
            "benchflow": fake_benchflow,
            "benchflow.providers": fake_providers,
            "benchflow.providers.litellm_runtime": fake_runtime,
            "benchflow.rollout": fake_rollout,
            "benchflow.rollout._setup": fake_setup,
            "benchflow.sandbox": fake_sandbox,
            "benchflow.sandbox.docker": fake_docker,
            "benchflow.trajectories": fake_trajectories,
            "benchflow.trajectories.types": fake_types,
        }

        with (
            patch.dict(sys.modules, modules),
            patch.dict(
                os.environ,
                {
                    "SKILLSBENCH_APPLY_WSL_DOCKER_COMPAT": "1",
                    "SKILLSBENCH_BENCHFLOW_HARDENING_COMPAT": "0",
                    "SKILLSBENCH_BENCHFLOW_TRAJECTORY_PUBLISH_COMPAT": "1",
                    "SKILLSBENCH_DEEPSEEK_OPENHANDS_COMPAT": "0",
                },
                clear=False,
            ),
        ):
            exec(compile(shim, str(shim_root / "sitecustomize.py"), "exec"), {})

        return SimpleNamespace(
            agent_dir=agent_dir,
            env=fake_env,
            probes=probes,
            raw_calls=raw_calls,
            raw_run=raw_run,
            modules=modules,
            original_calls=original_calls,
            publish=fake_rollout._publish_trajectory_for_verifier,
            setup_publish=fake_setup._publish_trajectory_for_verifier,
            subprocess_env=subprocess_env,
        )

    def _hardening_compat_harness(
        self,
        root: Path,
        *,
        original_error: BaseException | None,
        command: list[str] | None = None,
        command_sequence: list[tuple[list[str], int | float | None]] | None = None,
        timeout_sec: int | float | None = 10,
        check: bool = False,
        docker: bool = True,
        ps_stdout: str | None = None,
        ps_returncode: int = 0,
        inspect_labels: dict[str, str] | None = None,
        inspect_returncode: int = 0,
        raw_returncode: int = 0,
        raw_error: BaseException | None = None,
        container_user: str = "",
        verifier_user: str | int | None = None,
        default_user: str | int | None = None,
        verifier_exec_sequence: list[dict[str, object]] | None = None,
        verifier_timeout_sec: int | float = 240,
        verifier_env: dict[str, str] | None = None,
        verifier_type: str = "test-script",
        verifier_service: str = "main",
        persistent_env: dict[str, str] | None = None,
        verifier_error: BaseException | None = None,
        long_compose_error: BaseException | None = None,
        block_async_raw: bool = False,
    ) -> SimpleNamespace:
        with (
            patch.dict(os.environ, {"WSL_DISTRO_NAME": "Ubuntu"}, clear=False),
            patch(
                "benchmarks.skillsbench_adapter.platform.release",
                return_value="6.18.0-microsoft-standard-WSL2",
            ),
        ):
            subprocess_env = _benchflow_subprocess_env(
                sandbox="docker", run_root=root
            )
        shim_root = Path(subprocess_env["PYTHONPATH"].split(os.pathsep)[0])
        shim = (shim_root / "sitecustomize.py").read_text(encoding="utf-8")

        clear_command = (
            "if [ -L /logs/verifier ]; then rm -f /logs/verifier; fi && "
            "mkdir -p /logs/verifier"
        )
        hardening_command = command or [
            "exec",
            "-T",
            "-u",
            "root",
            "main",
            "sh",
            "-c",
            (
                "pkill -u agent 2>/dev/null || true; sleep 1; "
                "pkill -9 -u agent 2>/dev/null || true; sleep 1; "
                "! pgrep -u agent > /dev/null 2>&1"
            ),
        ]
        container_id = "a1" * 32
        resolved_ps_stdout = (
            container_id + "\n" if ps_stdout is None else ps_stdout
        )
        compose_calls = []
        raw_calls = []
        async_raw_calls = []
        async_processes = []
        resolved_verifier_env = (
            dict(verifier_env)
            if verifier_env is not None
            else {
                "PATH": "/trusted/verifier/bin",
                "PYTHONPATH": "/trusted/verifier/python",
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            }
        )

        class ExecResult(SimpleNamespace):
            pass

        class DockerSandbox:
            def __init__(self) -> None:
                self.session_id = "shock-analysis-demand__24667780"
                self.default_user = default_user
                self._persistent_env = dict(persistent_env or {})
                self.rollout_paths = SimpleNamespace(
                    verifier_dir=root / "verifier"
                )
                self._env_vars = SimpleNamespace(
                    host_agent_logs_path=str(root / "agent"),
                    host_verifier_logs_path=str(root / "verifier"),
                    host_artifacts_path=str(root / "artifacts"),
                    env_agent_logs_path="/logs/agent",
                    env_verifier_logs_path="/logs/verifier",
                    env_artifacts_path="/logs/artifacts",
                )

            def _docker_compose_env(self):
                return {"DOCKER_HOST": "mock://benchflow"}

            def _resolve_user(self, user):
                return user if user is not None else self.default_user

            def _merge_env(self, env):
                if not self._persistent_env and not env:
                    return None
                return {**self._persistent_env, **(env or {})}

            @classmethod
            def _wrap_command_with_env_file(cls, env, requested):
                encoded = json.dumps(
                    env, sort_keys=True, separators=(",", ":")
                )
                return f"mock-env-file:{encoded}\n{requested}"

            async def exec(
                self,
                requested,
                cwd=None,
                env=None,
                timeout_sec=None,
                user=None,
                service="main",
            ):
                resolved_user = self._resolve_user(user)
                merged_env = self._merge_env(env)
                command = ["exec", "-T"]
                if cwd:
                    command.extend(["-w", cwd])
                if resolved_user is not None:
                    command.extend(["-u", str(resolved_user)])
                command.append(service)
                if merged_env:
                    requested = self._wrap_command_with_env_file(
                        merged_env, requested
                    )
                command.extend(["sh", "-c", requested])
                return await self._run_docker_compose_command(
                    command, check=False, timeout_sec=timeout_sec
                )

            async def _run_docker_compose_command(
                self, requested, check=True, timeout_sec=None
            ):
                compose_calls.append((requested, check, timeout_sec))
                if original_error is not None:
                    raise original_error
                if (
                    long_compose_error is not None
                    and isinstance(timeout_sec, (int, float))
                    and timeout_sec > 15
                ):
                    raise long_compose_error
                return ExecResult(
                    stdout="compose stdout", stderr=None, return_code=0
                )

        DockerSandbox.__module__ = "benchflow.sandbox.docker"

        class ProcessSandbox(DockerSandbox):
            pass

        ProcessSandbox.__module__ = "benchflow.sandbox.process"
        inner = DockerSandbox() if docker else ProcessSandbox()
        fake_env = SimpleNamespace(inner=inner)

        fake_lockdown = ModuleType("benchflow.sandbox.lockdown")
        fake_lockdown._CLEAR_VERIFIER_DIR_CMD = clear_command
        fake_lockdown._ENSURE_APP_DIR_CMD = "mkdir -p /app"

        async def original_harden(env, *_args, **_kwargs):
            result = None
            for requested, requested_timeout in (
                command_sequence or [(hardening_command, timeout_sec)]
            ):
                result = await env.inner._run_docker_compose_command(
                    requested, check=check, timeout_sec=requested_timeout
                )
            return result

        async def original_kill(*_args, **_kwargs):
            return None

        fake_lockdown.harden_before_verify = original_harden
        fake_lockdown._kill_sandbox_user_procs = original_kill
        verifier_test_command = (
            "/verifier/test.sh > /logs/verifier/test-stdout.txt 2>&1"
        )

        class Verifier:
            def __init__(self) -> None:
                self._sandbox = inner
                self._task = SimpleNamespace(
                    config=SimpleNamespace(
                        verifier=SimpleNamespace(
                            type=verifier_type,
                            service=verifier_service,
                            timeout_sec=verifier_timeout_sec,
                            user=verifier_user,
                            env=dict(resolved_verifier_env),
                        )
                    )
                )

            async def _verify_test_script(self, *_args, **_kwargs):
                if verifier_error is not None:
                    raise verifier_error
                result = None
                for requested in (
                    verifier_exec_sequence
                    or [
                        {
                            "command": verifier_test_command,
                            "cwd": None,
                            "env": (
                                dict(resolved_verifier_env)
                                if resolved_verifier_env
                                else None
                            ),
                            "timeout_sec": verifier_timeout_sec,
                            "user": verifier_user,
                            "service": verifier_service,
                        }
                    ]
                ):
                    result = await self._sandbox.exec(**requested)
                return result

        Verifier.__module__ = "benchflow.task.verifier_core"
        original_verify_test_script = Verifier._verify_test_script
        fake_docker = ModuleType("benchflow.sandbox.docker")
        fake_docker.DockerSandbox = DockerSandbox
        fake_docker._sanitize_docker_compose_project_name = lambda value: value
        fake_base = ModuleType("benchflow.sandbox._base")
        fake_base.ExecResult = ExecResult
        fake_sandbox = ModuleType("benchflow.sandbox")
        fake_sandbox.lockdown = fake_lockdown
        fake_sandbox.docker = fake_docker
        fake_verifier_core = ModuleType("benchflow.task.verifier_core")
        fake_verifier_core.Verifier = Verifier
        fake_task = ModuleType("benchflow.task")
        fake_task.verifier_core = fake_verifier_core
        fake_runtime = ModuleType("benchflow.providers.litellm_runtime")
        fake_runtime._docker_host_address = lambda: "172.17.0.1"
        fake_providers = ModuleType("benchflow.providers")
        fake_providers.litellm_runtime = fake_runtime
        fake_benchflow = ModuleType("benchflow")
        fake_benchflow.providers = fake_providers
        fake_benchflow.sandbox = fake_sandbox
        fake_benchflow.task = fake_task

        resolved_inspect_labels = (
            {
            "benchflow.owned": "true",
            "com.docker.compose.project": inner.session_id,
            "com.docker.compose.service": "main",
            "com.docker.compose.container-number": "1",
            "com.docker.compose.oneoff": "False",
            }
            if inspect_labels is None
            else inspect_labels
        )

        def fake_run(args, **kwargs):
            raw_calls.append((list(args), dict(kwargs)))
            if list(args[:3]) == ["docker", "ps", "-q"]:
                return subprocess.CompletedProcess(
                    args,
                    ps_returncode,
                    stdout=resolved_ps_stdout,
                    stderr="ps failed" if ps_returncode else "",
                )
            if list(args[:2]) == ["docker", "inspect"]:
                return subprocess.CompletedProcess(
                    args,
                    inspect_returncode,
                    stdout=(
                        json.dumps(resolved_inspect_labels)
                        + "|"
                        + json.dumps(container_user)
                    ),
                    stderr="inspect failed" if inspect_returncode else "",
                )
            if list(args[:2]) == ["docker", "exec"]:
                if raw_error is not None:
                    raise raw_error
                return subprocess.CompletedProcess(
                    args,
                    raw_returncode,
                    stdout="raw stdout",
                    stderr="raw stderr" if raw_returncode else "",
                )
            raise AssertionError(f"unexpected raw subprocess: {args!r}")

        class FakeAsyncProcess:
            def __init__(self) -> None:
                self.returncode = None
                self.terminated = False
                self.killed = False
                self.communicate_calls = 0
                self._error_emitted = False
                self.communicate_started = asyncio.Event()
                self.release = asyncio.Event()

            async def communicate(self):
                self.communicate_calls += 1
                if block_async_raw and not self.terminated and not self.killed:
                    self.communicate_started.set()
                    await self.release.wait()
                if not self._error_emitted and raw_error is not None:
                    self._error_emitted = True
                    if isinstance(raw_error, subprocess.TimeoutExpired):
                        raise TimeoutError()
                    if isinstance(raw_error, asyncio.CancelledError):
                        raise raw_error
                    if not isinstance(raw_error, OSError):
                        raise raw_error
                if self.killed:
                    self.returncode = -9
                elif self.terminated:
                    self.returncode = -15
                else:
                    self.returncode = raw_returncode
                return b"raw stdout", None

            def terminate(self):
                self.terminated = True

            def kill(self):
                self.killed = True

        async def fake_create_subprocess_exec(*args, **kwargs):
            async_raw_calls.append((list(args), dict(kwargs)))
            if isinstance(raw_error, OSError):
                raise raw_error
            process = FakeAsyncProcess()
            async_processes.append(process)
            return process

        modules = {
            "benchflow": fake_benchflow,
            "benchflow.providers": fake_providers,
            "benchflow.providers.litellm_runtime": fake_runtime,
            "benchflow.sandbox": fake_sandbox,
            "benchflow.sandbox.lockdown": fake_lockdown,
            "benchflow.sandbox.docker": fake_docker,
            "benchflow.sandbox._base": fake_base,
            "benchflow.task": fake_task,
            "benchflow.task.verifier_core": fake_verifier_core,
        }
        with (
            patch.dict(sys.modules, modules),
            patch.dict(
                os.environ,
                {
                    "SKILLSBENCH_APPLY_WSL_DOCKER_COMPAT": "1",
                    "SKILLSBENCH_BENCHFLOW_HARDENING_COMPAT": "1",
                    "SKILLSBENCH_BENCHFLOW_TRAJECTORY_PUBLISH_COMPAT": "0",
                    "SKILLSBENCH_DEEPSEEK_OPENHANDS_COMPAT": "0",
                },
                clear=False,
            ),
        ):
            exec(compile(shim, str(shim_root / "sitecustomize.py"), "exec"), {})

        return SimpleNamespace(
            agent_dir=root / "agent",
            verifier_dir=root / "verifier",
            diagnostic_dir=root / "benchflow-compat-diagnostics",
            container_id=container_id,
            command=hardening_command,
            compose_calls=compose_calls,
            raw_calls=raw_calls,
            async_raw_calls=async_raw_calls,
            async_processes=async_processes,
            raw_run=fake_run,
            raw_create_subprocess_exec=fake_create_subprocess_exec,
            env=fake_env,
            inner=inner,
            verifier=Verifier(),
            verifier_test_command=verifier_test_command,
            verifier_env=resolved_verifier_env,
            verify_test_script=Verifier._verify_test_script,
            original_verify_test_script=original_verify_test_script,
            harden=fake_lockdown.harden_before_verify,
            original_harden=original_harden,
            original_exec=DockerSandbox.exec,
            original_run=DockerSandbox._run_docker_compose_command,
            modules=modules,
        )

    def _failure_forensics_harness(
        self,
        root: Path,
        *,
        rewards=None,
        verifier_error=None,
        agent_error=None,
        phase: str = "verified",
        terminal: bool = True,
        terminal_kind: str = "agent_message",
        docker: bool = True,
        skip_verify: bool = False,
        wrapped_env: bool = True,
    ) -> SimpleNamespace:
        with (
            patch.dict(os.environ, {"WSL_DISTRO_NAME": "Ubuntu"}, clear=False),
            patch(
                "benchmarks.skillsbench_adapter.platform.release",
                return_value="6.18.0-microsoft-standard-WSL2",
            ),
        ):
            subprocess_env = _benchflow_subprocess_env(
                sandbox="docker", run_root=root
            )
        shim_root = Path(subprocess_env["PYTHONPATH"].split(os.pathsep)[0])
        shim = (shim_root / "sitecustomize.py").read_text(encoding="utf-8")
        (root / "agent").mkdir(parents=True, exist_ok=True)
        cleanup_observations = []

        class DockerSandbox:
            def __init__(self) -> None:
                self._keep_containers = False
                self.session_id = "shock-analysis-demand__forensics"
                self._env_vars = SimpleNamespace(
                    host_agent_logs_path=str(root / "agent")
                )

        DockerSandbox.__module__ = "benchflow.sandbox.docker"

        class ProcessSandbox(DockerSandbox):
            pass

        ProcessSandbox.__module__ = "benchflow.sandbox.process"

        class Rollout:
            def __init__(self) -> None:
                inner = DockerSandbox() if docker else ProcessSandbox()
                self._env = SimpleNamespace(inner=inner) if wrapped_env else inner
                if not terminal:
                    self._trajectory = [
                        {"type": "tool_call", "status": "completed"}
                    ]
                elif terminal_kind == "agent_timeout":
                    self._trajectory = [
                        {
                            "type": "agent_timeout",
                            "reason": "wall_clock_timeout",
                            "timeout_sec": 1800.0,
                            "pending_tool_call_ids": [],
                            "terminal_trajectory_complete": True,
                        }
                    ]
                elif terminal_kind == "partial_timeout":
                    self._trajectory = [
                        {
                            "type": "agent_timeout",
                            "reason": "wall_clock_timeout",
                            "timeout_sec": 1800.0,
                            "pending_tool_call_ids": ["pending"],
                            "terminal_trajectory_complete": False,
                        }
                    ]
                else:
                    self._trajectory = [
                        {"type": "agent_message", "text": "deliverable ready"}
                    ]
                self._rewards = rewards
                self._verifier_error = verifier_error
                self._error = agent_error
                self._phase = phase
                self._config = SimpleNamespace(skip_verify=skip_verify)

            async def cleanup(self):
                inner = getattr(self._env, "inner", self._env)
                cleanup_observations.append(
                    inner._keep_containers
                )
                return "cleanup-result"

        original_cleanup = Rollout.cleanup
        fake_rollout = ModuleType("benchflow.rollout")
        fake_rollout.Rollout = Rollout
        fake_runtime = ModuleType("benchflow.providers.litellm_runtime")
        fake_runtime._docker_host_address = lambda: "172.17.0.1"
        fake_providers = ModuleType("benchflow.providers")
        fake_providers.litellm_runtime = fake_runtime
        fake_docker = ModuleType("benchflow.sandbox.docker")
        fake_docker.DockerSandbox = DockerSandbox
        fake_docker._sanitize_docker_compose_project_name = lambda value: value
        fake_sandbox = ModuleType("benchflow.sandbox")
        fake_sandbox.docker = fake_docker
        fake_benchflow = ModuleType("benchflow")
        fake_benchflow.providers = fake_providers
        fake_benchflow.sandbox = fake_sandbox
        fake_benchflow.rollout = fake_rollout
        modules = {
            "benchflow": fake_benchflow,
            "benchflow.providers": fake_providers,
            "benchflow.providers.litellm_runtime": fake_runtime,
            "benchflow.rollout": fake_rollout,
            "benchflow.sandbox": fake_sandbox,
            "benchflow.sandbox.docker": fake_docker,
        }
        with (
            patch.dict(sys.modules, modules),
            patch.dict(
                os.environ,
                {
                    "SKILLSBENCH_APPLY_WSL_DOCKER_COMPAT": "1",
                    "SKILLSBENCH_BENCHFLOW_HARDENING_COMPAT": "0",
                    "SKILLSBENCH_BENCHFLOW_TRAJECTORY_PUBLISH_COMPAT": "0",
                    "SKILLSBENCH_BENCHFLOW_FAILURE_FORENSICS": "1",
                    "SKILLSBENCH_DEEPSEEK_OPENHANDS_COMPAT": "0",
                },
                clear=False,
            ),
        ):
            exec(compile(shim, str(shim_root / "sitecustomize.py"), "exec"), {})

        rollout = Rollout()
        return SimpleNamespace(
            rollout=rollout,
            cleanup=rollout.cleanup,
            original_cleanup=original_cleanup,
            cleanup_observations=cleanup_observations,
            marker=root / "agent" / "retained_container_forensics.json",
            modules=modules,
        )

    def test_parse_and_digest_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task = make_task(Path(tmp))
            frontmatter, prompt = parse_task_markdown(task)
            self.assertEqual("1.3", frontmatter["schema_version"])
            self.assertIn("answer.txt", prompt)
            first = task_package_digest(task)
            (task / "environment" / "Dockerfile").write_text("FROM python:3.13\n")
            _cached_task_digest.cache_clear()
            self.assertNotEqual(first, task_package_digest(task))

    def test_preparation_makes_disjoint_unique_rollout_slots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task = make_task(Path(tmp))
            construction, test, manifest = build_payloads(
                task_dir=task,
                construction_rollouts=4,
                test_rollouts=3,
                agent="codex",
                sandbox="docker",
                jobs_root=Path(tmp) / "jobs",
                bench_executable="bench",
                subprocess_timeout_sec=7200,
                source_version="v1.1",
            )
            construction_ids = {x["instance_id"] for x in construction["instances"]}
            test_ids = {x["instance_id"] for x in test["instances"]}
            self.assertEqual(4, len(construction_ids))
            self.assertEqual(3, len(test_ids))
            self.assertFalse(construction_ids & test_ids)
            self.assertFalse(manifest["method_fidelity"]["cross_task_pooling"])
            self.assertTrue(
                all(x["ground_truth"] is None for x in construction["instances"])
            )

    def test_suite_discovery_is_sorted_and_include_is_strict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_task(root, "task-b")
            make_task(root, "task-a")
            self.assertEqual(
                ["task-a", "task-b"],
                [path.name for path in discover_task_dirs(root)],
            )
            self.assertEqual(
                ["task-b"],
                [path.name for path in discover_task_dirs(root, ["task-b"])],
            )
            with self.assertRaisesRegex(ValueError, "Unknown"):
                discover_task_dirs(root, ["missing"])

    def test_command_isolates_skill_modes(self) -> None:
        base = build_benchflow_command(
            bench_executable="bench",
            task_dir=Path("C:/task"),
            jobs_dir=Path("C:/jobs"),
            agent="codex",
            model="openai/gpt-5.4-nano",
            sandbox="docker",
            skills_dir=None,
        )
        self.assertIn("no-skill", base)
        self.assertNotIn("--skills-dir", base)
        retry_index = base.index("--retry-attempts")
        self.assertEqual("0", base[retry_index + 1])
        treated = build_benchflow_command(
            bench_executable="bench",
            task_dir=Path("C:/task"),
            jobs_dir=Path("C:/jobs"),
            agent="codex",
            model="openai/gpt-5.4-nano",
            sandbox="docker",
            skills_dir=Path("C:/generated"),
        )
        self.assertIn("with-skill", treated)
        self.assertIn("--skills-dir", treated)

    def test_materialized_skill_has_required_layout_and_frontmatter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            skill = SkillItem(
                skill_id="skill-1",
                body="# Procedure\nDo the work carefully.",
                contextual_abstract="A reusable procedure",
                task_name="Demo Task",
            )
            root = materialize_generated_skill(skill, Path(tmp) / "skills")
            files = list(root.glob("*/SKILL.md"))
            self.assertEqual(1, len(files))
            text = files[0].read_text(encoding="utf-8")
            self.assertTrue(text.startswith("---\nname:"))
            self.assertIn("# Procedure", text)

    def test_acp_conversion_preserves_order_and_tool_evidence(self) -> None:
        events = [
            {"type": "user_message", "text": "do it"},
            {"type": "agent_thought", "text": "plan"},
            {
                "type": "tool_call",
                "kind": "execute",
                "title": "python solve.py",
                "status": "completed",
                "content": [{"type": "content", "content": {"text": "ok"}}],
            },
            {"type": "agent_message", "text": "done"},
        ]
        messages = acp_events_to_skillgen_messages(events)
        self.assertEqual(
            ["user", "assistant", "tool", "assistant"],
            [message["role"] for message in messages],
        )
        self.assertIn("python solve.py", messages[2]["content"])
        self.assertEqual("done", messages[-1]["content"])

    def _write_artifacts(
        self,
        root: Path,
        *,
        with_skill: bool,
        reward: float | None = 0.0,
        verifier_error=None,
        trajectory: list[dict] | None = None,
        task_name: str = "demo",
        model: str = "openai/gpt-5.4-nano",
        agent: str = "codex",
        error=None,
        error_category=None,
        agent_timeout_info=None,
        partial_trajectory: bool = False,
    ) -> tuple[Path, Path]:
        rollout = root / "job" / "demo__12345678"
        (rollout / "trajectory").mkdir(parents=True)
        result = {
            "task_name": task_name,
            "rollout_name": "demo__12345678",
            "rewards": {"reward": reward},
            "agent": agent,
            "agent_name": agent,
            "model": model,
            "skill_mode": "with-skill" if with_skill else "no-skill",
            "skill_source": "custom_runtime" if with_skill else "none",
            "include_task_skills": False,
            "n_tool_calls": 1,
            "n_skill_invocations": 1 if with_skill else 0,
            "agent_result": {
                "total_tokens": 100,
                "n_input_tokens": 70,
                "n_output_tokens": 30,
            },
            "error": error,
            "error_category": error_category,
            "verifier_error": verifier_error,
            "agent_timeout_info": agent_timeout_info,
            "trajectory_summary": {
                "partial_trajectory": partial_trajectory,
            },
            "timing": {"total": 2.5},
            "finished_at": "2026-08-13 00:00:00",
        }
        result_path = rollout / "result.json"
        trajectory_path = rollout / "trajectory" / "acp_trajectory.jsonl"
        result_path.write_text(json.dumps(result), encoding="utf-8")
        if trajectory is None:
            trajectory = [
                {"type": "user_message", "text": "task"},
                {"type": "agent_message", "text": "finished"},
            ]
        trajectory_path.write_text(
            "\n".join(json.dumps(item) for item in trajectory) + "\n",
            encoding="utf-8",
        )
        return result_path, trajectory_path

    def test_valid_verifier_failure_is_a_negative_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path, trajectory_path = self._write_artifacts(
                root, with_skill=False, reward=0.4
            )
            traj = _trajectory_from_artifacts(
                instance=TaskInstance("demo::construction::r000", "task", metadata={}),
                skill_bundle=None,
                result_path=result_path,
                trajectory_path=trajectory_path,
                process_returncode=0,
                run_root=root,
                elapsed=3.0,
            )
            self.assertFalse(traj.success)
            self.assertEqual(0.4, traj.score)
            self.assertEqual("finished", traj.final_output)
            self.assertIn("full pass", traj.error_summary)

    def test_official_scored_clean_timeout_is_a_valid_negative_trajectory(
        self,
    ) -> None:
        timeout_event = {
            "type": "agent_timeout",
            "reason": "wall_clock_timeout",
            "timeout_sec": 1800.0,
            "pending_tool_call_ids": [],
            "terminal_trajectory_complete": True,
        }
        timeout_info = {
            "reason": "wall_clock_timeout",
            "timeout_sec": 1800.0,
            "n_tool_calls": 1,
            "pending_tool_call_ids": [],
            "terminal_event_recorded": True,
            "terminal_trajectory_complete": True,
        }
        for official_reward in (0.0, 0.4):
            with self.subTest(official_reward=official_reward):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    result_path, trajectory_path = self._write_artifacts(
                        root,
                        with_skill=False,
                        reward=official_reward,
                        error="Agent prompt exceeded wall-clock budget 1800s",
                        error_category="timeout",
                        agent_timeout_info=timeout_info,
                        trajectory=[
                            {"type": "user_message", "text": "task"},
                            {"type": "tool_call", "status": "completed"},
                            timeout_event,
                        ],
                    )
                    traj = _trajectory_from_artifacts(
                        instance=TaskInstance("id", "task", metadata={}),
                        skill_bundle=None,
                        result_path=result_path,
                        trajectory_path=trajectory_path,
                        process_returncode=0,
                        run_root=root,
                        elapsed=1801.0,
                    )
                    self.assertFalse(traj.success)
                    self.assertEqual(official_reward, traj.score)
                    self.assertTrue(traj.metadata["agent_timed_out"])
                    self.assertEqual(
                        "normal_timeout", traj.metadata["benchflow_outcome"]
                    )
                    self.assertEqual(
                        timeout_info, traj.metadata["agent_timeout_info"]
                    )

    def test_official_scored_partial_timeout_is_a_valid_trajectory(self) -> None:
        timeout_event = {
            "type": "agent_timeout",
            "reason": "wall_clock_timeout",
            "timeout_sec": 1800.0,
            "pending_tool_call_ids": ["pending"],
            "terminal_trajectory_complete": False,
        }
        timeout_info = {
            "reason": "wall_clock_timeout",
            "timeout_sec": 1800.0,
            "n_tool_calls": 1,
            "pending_tool_call_ids": ["pending"],
            "terminal_event_recorded": True,
            "terminal_trajectory_complete": False,
        }
        for official_reward in (0.0, 1.0):
            with self.subTest(official_reward=official_reward):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    result_path, trajectory_path = self._write_artifacts(
                        root,
                        with_skill=False,
                        reward=official_reward,
                        error="Agent prompt exceeded wall-clock budget 1800s",
                        error_category="timeout",
                        agent_timeout_info=timeout_info,
                        partial_trajectory=True,
                        trajectory=[
                            {"type": "user_message", "text": "task"},
                            {"type": "tool_call", "status": "pending"},
                            timeout_event,
                        ],
                    )
                    traj = _trajectory_from_artifacts(
                        instance=TaskInstance("id", "task", metadata={}),
                        skill_bundle=None,
                        result_path=result_path,
                        trajectory_path=trajectory_path,
                        process_returncode=0,
                        run_root=root,
                        elapsed=1801.0,
                    )
                    self.assertEqual(official_reward == 1.0, traj.success)
                    self.assertEqual(official_reward, traj.score)
                    self.assertTrue(traj.metadata["agent_timed_out"])
                    self.assertTrue(traj.metadata["partial_trajectory"])
                    self.assertEqual(
                        "partial_timeout", traj.metadata["benchflow_outcome"]
                    )
                    self.assertEqual(
                        timeout_info, traj.metadata["agent_timeout_info"]
                    )

    def test_unscored_or_malformed_timeout_is_never_admitted(self) -> None:
        base_event = {
            "type": "agent_timeout",
            "reason": "wall_clock_timeout",
            "timeout_sec": 1800.0,
            "pending_tool_call_ids": [],
            "terminal_trajectory_complete": True,
        }
        base_info = {
            "reason": "wall_clock_timeout",
            "timeout_sec": 1800.0,
            "n_tool_calls": 1,
            "pending_tool_call_ids": [],
            "terminal_event_recorded": True,
            "terminal_trajectory_complete": True,
        }
        cases = (
            {"reward": None},
            {
                "reward": 0.0,
                "partial_trajectory": True,
            },
            {
                "reward": 0.0,
                "trajectory": [
                    {
                        **base_event,
                        "pending_tool_call_ids": ["pending"],
                        "terminal_trajectory_complete": False,
                    }
                ],
                "agent_timeout_info": {
                    **base_info,
                    "pending_tool_call_ids": ["pending"],
                    "terminal_trajectory_complete": False,
                },
                "partial_trajectory": False,
            },
        )
        for case in cases:
            with self.subTest(case=case):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    result_path, trajectory_path = self._write_artifacts(
                        root,
                        with_skill=False,
                        reward=case.get("reward"),
                        error="Agent prompt exceeded wall-clock budget 1800s",
                        error_category="timeout",
                        agent_timeout_info=case.get(
                            "agent_timeout_info", base_info
                        ),
                        partial_trajectory=case.get("partial_trajectory", False),
                        trajectory=case.get("trajectory", [base_event]),
                    )
                    with self.assertRaises(SkillsBenchInfrastructureError):
                        _trajectory_from_artifacts(
                            instance=TaskInstance("id", "task", metadata={}),
                            skill_bundle=None,
                            result_path=result_path,
                            trajectory_path=trajectory_path,
                            process_returncode=0,
                            run_root=root,
                            elapsed=1801.0,
                        )

    def test_tool_only_valid_reward_remains_a_scored_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path, trajectory_path = self._write_artifacts(
                root,
                with_skill=False,
                reward=1.0,
                trajectory=[
                    {"type": "user_message", "text": "task"},
                    {
                        "type": "tool_call",
                        "kind": "write_file",
                        "title": "create deliverable",
                        "status": "completed",
                        "content": [{"text": "saved"}],
                    },
                ],
            )
            traj = _trajectory_from_artifacts(
                instance=TaskInstance("id", "task", metadata={}),
                skill_bundle=None,
                result_path=result_path,
                trajectory_path=trajectory_path,
                process_returncode=0,
                run_root=root,
                elapsed=3.0,
            )
            self.assertTrue(traj.success)
            self.assertEqual("", traj.final_output)
            self.assertEqual(1.0, traj.score)

    def test_verifier_crash_is_not_a_negative_example(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path, trajectory_path = self._write_artifacts(
                root,
                with_skill=False,
                reward=0.0,
                verifier_error="container disappeared",
            )
            with self.assertRaises(SkillsBenchInfrastructureError):
                _trajectory_from_artifacts(
                    instance=TaskInstance("id", "task", metadata={}),
                    skill_bundle=None,
                    result_path=result_path,
                    trajectory_path=trajectory_path,
                    process_returncode=0,
                    run_root=root,
                    elapsed=3.0,
                )

    def test_non_timeout_error_is_rejected_even_with_numeric_reward(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path, trajectory_path = self._write_artifacts(
                root,
                with_skill=False,
                reward=0.0,
                error="agent protocol failed",
                error_category="agent_protocol",
            )
            with self.assertRaises(SkillsBenchInfrastructureError):
                _trajectory_from_artifacts(
                    instance=TaskInstance("id", "task", metadata={}),
                    skill_bundle=None,
                    result_path=result_path,
                    trajectory_path=trajectory_path,
                    process_returncode=0,
                    run_root=root,
                    elapsed=3.0,
                )

    def test_official_task_skill_contamination_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path, trajectory_path = self._write_artifacts(
                root, with_skill=False, reward=1.0
            )
            payload = json.loads(result_path.read_text())
            payload["skill_source"] = "task_bundled"
            payload["include_task_skills"] = True
            result_path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(SkillsBenchInfrastructureError, "contaminated"):
                _trajectory_from_artifacts(
                    instance=TaskInstance("id", "task", metadata={}),
                    skill_bundle=None,
                    result_path=result_path,
                    trajectory_path=trajectory_path,
                    process_returncode=0,
                    run_root=root,
                    elapsed=3.0,
                )

    def test_runner_executes_one_fake_benchflow_attempt_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = make_task(root)
            digest = task_package_digest(task)
            instance = TaskInstance(
                "demo-task::construction::r000",
                "task",
                metadata={
                    "benchmark": "skillsbench",
                    "skillsbench_task_id": "demo-task",
                    "skillsbench_task_dir": str(task),
                    "skillsbench_task_digest": digest,
                    "skillsbench_agent": "codex",
                    "skillsbench_sandbox": "docker",
                    "skillsbench_jobs_root": str(root / "jobs-root"),
                    "skillsbench_bench_executable": sys.executable,
                },
            )
            seen_rollout_commands: list[list[str]] = []

            def fake_run(command, **kwargs):
                if command[-1] == "--version":
                    return subprocess.CompletedProcess(
                        command, 0, stdout="benchflow 0.6.7\n", stderr=""
                    )
                seen_rollout_commands.append(list(command))
                jobs_dir = Path(command[command.index("--jobs-dir") + 1])
                rollout = jobs_dir / "job" / "demo-task__12345678"
                (rollout / "trajectory").mkdir(parents=True)
                result = {
                    "task_name": "demo-task",
                    "rollout_name": "demo-task__12345678",
                    "rewards": {"reward": 1.0},
                    "agent": "codex",
                    "agent_name": "codex",
                    "model": "openai/gpt-5.4-nano",
                    "skill_mode": "no-skill",
                    "skill_source": "none",
                    "include_task_skills": False,
                    "n_tool_calls": 1,
                    "n_skill_invocations": 0,
                    "agent_result": {"total_tokens": 10},
                    "error": None,
                    "verifier_error": None,
                    "finished_at": "2026-08-13T00:00:00Z",
                }
                (rollout / "result.json").write_text(
                    json.dumps(result), encoding="utf-8"
                )
                (rollout / "trajectory" / "acp_trajectory.jsonl").write_text(
                    json.dumps(
                        {
                            "type": "tool_call",
                            "kind": "write_file",
                            "title": "save",
                            "status": "completed",
                            "content": [{"text": "done"}],
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            _validate_benchflow_executable.cache_clear()
            with (
                patch(
                    "benchmarks.skillsbench_adapter.subprocess.run",
                    side_effect=fake_run,
                ),
                patch(
                    "benchmarks.skillsbench_adapter.pilot_budget_guard.before_agent_rollout",
                    return_value="reservation-token",
                ) as before_rollout,
                patch(
                    "benchmarks.skillsbench_adapter.pilot_budget_guard.record_balance"
                ) as record_balance,
            ):
                first = run_skillsbench_agent(
                    instance,
                    None,
                    SimpleNamespace(model="openai/gpt-5.4-nano"),
                )
                second = run_skillsbench_agent(
                    instance,
                    None,
                    SimpleNamespace(model="openai/gpt-5.4-nano"),
                )
            self.assertTrue(first.success)
            self.assertEqual("", first.final_output)
            self.assertEqual(first, second)
            self.assertEqual(first.trajectory_id, second.trajectory_id)
            self.assertEqual(1, len(seen_rollout_commands))
            before_rollout.assert_called_once_with()
            record_balance.assert_called_once_with(
                "after_agent_rollout",
                reservation_token="reservation-token",
            )
            command = seen_rollout_commands[0]
            self.assertEqual("0", command[command.index("--retry-attempts") + 1])

    def test_manifested_valid_artifact_recovers_without_subprocess_or_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = make_task(root)
            digest = task_package_digest(task)
            jobs_root = root / "jobs-root"
            jobs_root.mkdir()
            instance = TaskInstance(
                "demo-task::construction::r000",
                "task",
                metadata={
                    "benchmark": "skillsbench",
                    "skillsbench_task_id": "demo-task",
                    "skillsbench_task_dir": str(task),
                    "skillsbench_task_digest": digest,
                    "skillsbench_agent": "codex",
                    "skillsbench_sandbox": "docker",
                    "skillsbench_jobs_root": str(jobs_root),
                    "skillsbench_bench_executable": "missing-is-fine-on-recovery",
                },
            )
            request = RolloutCacheRequest.build(
                instance=instance,
                task_digest=digest,
                model="openai/gpt-5.4-nano",
                agent="codex",
                sandbox="docker",
                skill=None,
                adapter_schema_version="skillsbench-skillgen-v1",
                benchflow_version="0.6.7",
            )
            run_root = jobs_root / attempt_directory_name(
                "demo-slot", request, "prior"
            )
            run_root.mkdir()
            write_attempt_manifest(run_root, request)
            write_attempt_receipt(
                run_root, process_returncode=0, elapsed=4.0
            )
            self._write_artifacts(
                run_root / "jobs",
                with_skill=False,
                reward=1.0,
                task_name="demo-task",
            )
            with (
                patch(
                    "benchmarks.skillsbench_adapter.subprocess.run",
                    side_effect=AssertionError("subprocess must not run"),
                ),
                patch(
                    "benchmarks.skillsbench_adapter.pilot_budget_guard.before_agent_rollout",
                    side_effect=AssertionError("budget guard must not run"),
                ),
            ):
                trajectory = run_skillsbench_agent(
                    instance,
                    None,
                    SimpleNamespace(model="openai/gpt-5.4-nano"),
                )
            self.assertTrue(trajectory.success)
            self.assertEqual(digest, trajectory.metadata["task_digest"])
            self.assertTrue(cache_entry_path(jobs_root, request).is_file())

    def test_explicit_legacy_bootstrap_revalidates_and_then_hits_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = make_task(root)
            digest = task_package_digest(task)
            jobs_root = root / "jobs-root"
            jobs_root.mkdir()
            instance = TaskInstance(
                "demo-task::construction::r000",
                "task",
                metadata={
                    "benchmark": "skillsbench",
                    "skillsbench_task_id": "demo-task",
                    "skillsbench_task_dir": str(task),
                    "skillsbench_task_digest": digest,
                    "skillsbench_agent": "codex",
                    "skillsbench_sandbox": "docker",
                    "skillsbench_jobs_root": str(jobs_root),
                    "skillsbench_bench_executable": "not-needed-after-bootstrap",
                },
            )
            legacy_root = jobs_root / "legacy-paid-artifact"
            legacy_root.mkdir()
            self._write_artifacts(
                legacy_root / "jobs",
                with_skill=False,
                reward=0.4,
                task_name="demo-task",
            )
            config = SimpleNamespace(model="openai/gpt-5.4-nano")
            imported = bootstrap_skillsbench_rollout_cache(
                instance=instance,
                skill_bundle=None,
                config=config,
                run_root=legacy_root,
            )
            with (
                patch(
                    "benchmarks.skillsbench_adapter._resolve_executable",
                    side_effect=AssertionError("cache hit must precede executable lookup"),
                ),
                patch(
                    "benchmarks.skillsbench_adapter.pilot_budget_guard.before_agent_rollout",
                    side_effect=AssertionError("cache hit must not check paid budget"),
                ),
            ):
                loaded = run_skillsbench_agent(instance, None, config)
            self.assertEqual(imported, loaded)
            self.assertFalse(loaded.success)
            self.assertEqual(0.4, loaded.score)

    def test_bootstrap_rejects_wrong_model_even_with_valid_reward(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = make_task(root)
            digest = task_package_digest(task)
            jobs_root = root / "jobs-root"
            jobs_root.mkdir()
            instance = TaskInstance(
                "demo-task::construction::r000",
                "task",
                metadata={
                    "benchmark": "skillsbench",
                    "skillsbench_task_id": "demo-task",
                    "skillsbench_task_dir": str(task),
                    "skillsbench_task_digest": digest,
                    "skillsbench_agent": "codex",
                    "skillsbench_sandbox": "docker",
                    "skillsbench_jobs_root": str(jobs_root),
                },
            )
            legacy_root = jobs_root / "wrong-model"
            legacy_root.mkdir()
            self._write_artifacts(
                legacy_root / "jobs",
                with_skill=False,
                reward=1.0,
                task_name="demo-task",
                model="some-other-model",
            )
            with self.assertRaisesRegex(
                SkillsBenchInfrastructureError, "model identity mismatch"
            ):
                bootstrap_skillsbench_rollout_cache(
                    instance=instance,
                    skill_bundle=None,
                    config=SimpleNamespace(model="openai/gpt-5.4-nano"),
                    run_root=legacy_root,
                )

    def test_budget_stop_before_launch_records_known_unpaid_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = make_task(root)
            digest = task_package_digest(task)
            jobs_root = root / "jobs-root"
            instance = TaskInstance(
                "demo-task::construction::r000",
                "task",
                metadata={
                    "benchmark": "skillsbench",
                    "skillsbench_task_id": "demo-task",
                    "skillsbench_task_dir": str(task),
                    "skillsbench_task_digest": digest,
                    "skillsbench_agent": "codex",
                    "skillsbench_sandbox": "docker",
                    "skillsbench_jobs_root": str(jobs_root),
                    "skillsbench_bench_executable": sys.executable,
                },
            )
            _validate_benchflow_executable.cache_clear()
            with (
                patch(
                    "benchmarks.skillsbench_adapter._validate_benchflow_executable"
                ),
                patch(
                    "benchmarks.skillsbench_adapter.pilot_budget_guard.before_agent_rollout",
                    side_effect=RuntimeError("budget stop"),
                ),
                patch(
                    "benchmarks.skillsbench_adapter.subprocess.run",
                    side_effect=AssertionError("paid subprocess must not launch"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "budget stop"):
                    run_skillsbench_agent(
                        instance,
                        None,
                        SimpleNamespace(model="openai/gpt-5.4-nano"),
                    )
            attempts = [
                path
                for path in jobs_root.iterdir()
                if path.is_dir() and not path.name.startswith(".")
            ]
            self.assertEqual(1, len(attempts))
            receipt = read_attempt_receipt(attempts[0])
            self.assertIsNotNone(receipt)
            self.assertEqual(-3, receipt[0])

    def test_timeout_settlement_keeps_primary_infrastructure_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = make_task(root)
            instance = TaskInstance(
                "demo-task::construction::r000",
                "task",
                metadata={
                    "benchmark": "skillsbench",
                    "skillsbench_task_id": "demo-task",
                    "skillsbench_task_dir": str(task),
                    "skillsbench_task_digest": task_package_digest(task),
                    "skillsbench_agent": "codex",
                    "skillsbench_sandbox": "docker",
                    "skillsbench_jobs_root": str(root / "jobs-root"),
                    "skillsbench_bench_executable": sys.executable,
                    "skillsbench_subprocess_timeout_sec": 10,
                },
            )
            with (
                patch(
                    "benchmarks.skillsbench_adapter._validate_benchflow_executable"
                ),
                patch(
                    "benchmarks.skillsbench_adapter.pilot_budget_guard.before_agent_rollout",
                    return_value="timeout-reservation",
                ),
                patch(
                    "benchmarks.skillsbench_adapter.pilot_budget_guard.record_balance",
                    side_effect=RuntimeError("balance audit unavailable"),
                ) as record_balance,
                patch(
                    "benchmarks.skillsbench_adapter.subprocess.run",
                    side_effect=subprocess.TimeoutExpired(["bench"], 10),
                ),
            ):
                with self.assertRaisesRegex(
                    SkillsBenchInfrastructureError,
                    "subprocess exceeded 10.0s",
                ):
                    run_skillsbench_agent(
                        instance,
                        None,
                        SimpleNamespace(model="openai/gpt-5.4-nano"),
                    )
            record_balance.assert_called_once_with(
                "after_agent_rollout_timeout",
                reservation_token="timeout-reservation",
            )

    def test_wsl_docker_env_adds_process_local_proxy_bind_shim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch.dict(os.environ, {"WSL_DISTRO_NAME": "Ubuntu"}, clear=False),
                patch(
                    "benchmarks.skillsbench_adapter.platform.release",
                    return_value="6.18.0-microsoft-standard-WSL2",
                ),
            ):
                env = _benchflow_subprocess_env(sandbox="docker", run_root=root)
            self.assertEqual("1", env["SKILLSBENCH_APPLY_WSL_DOCKER_COMPAT"])
            self.assertEqual(
                "1", env["SKILLSBENCH_BENCHFLOW_HARDENING_COMPAT"]
            )
            self.assertEqual(
                "1",
                env["SKILLSBENCH_BENCHFLOW_TRAJECTORY_PUBLISH_COMPAT"],
            )
            self.assertEqual(
                "1", env["SKILLSBENCH_BENCHFLOW_FAILURE_FORENSICS"]
            )
            shim_root = Path(env["PYTHONPATH"].split(os.pathsep)[0])
            shim = (shim_root / "sitecustomize.py").read_text(encoding="utf-8")
            self.assertIn("'host.docker.internal'", shim)
            self.assertIn("_kill_sandbox_user_procs_once_as_root", shim)
            fake_runtime = ModuleType("benchflow.providers.litellm_runtime")
            fake_runtime._docker_host_address = lambda: "172.17.0.1"
            fake_providers = ModuleType("benchflow.providers")
            fake_providers.litellm_runtime = fake_runtime
            fake_lockdown = ModuleType("benchflow.sandbox.lockdown")

            async def original_kill(*_args, **_kwargs):
                raise AssertionError("unpatched hardening function was called")

            async def original_harden(*_args, **_kwargs):
                return None

            fake_lockdown._kill_sandbox_user_procs = original_kill
            fake_lockdown.harden_before_verify = original_harden
            fake_sandbox = ModuleType("benchflow.sandbox")
            fake_sandbox.lockdown = fake_lockdown
            fake_verifier_core = ModuleType("benchflow.task.verifier_core")

            class Verifier:
                async def _verify_test_script(self, *_args, **_kwargs):
                    return None

            Verifier.__module__ = "benchflow.task.verifier_core"
            fake_verifier_core.Verifier = Verifier
            fake_task = ModuleType("benchflow.task")
            fake_task.verifier_core = fake_verifier_core
            fake_benchflow = ModuleType("benchflow")
            fake_benchflow.providers = fake_providers
            fake_benchflow.sandbox = fake_sandbox
            fake_benchflow.task = fake_task
            with (
                patch.dict(
                    sys.modules,
                    {
                        "benchflow": fake_benchflow,
                        "benchflow.providers": fake_providers,
                        "benchflow.providers.litellm_runtime": fake_runtime,
                        "benchflow.sandbox": fake_sandbox,
                        "benchflow.sandbox.lockdown": fake_lockdown,
                        "benchflow.task": fake_task,
                        "benchflow.task.verifier_core": fake_verifier_core,
                    },
                ),
                patch.dict(
                    os.environ,
                    {
                        "SKILLSBENCH_APPLY_WSL_DOCKER_COMPAT": "1",
                        "SKILLSBENCH_BENCHFLOW_HARDENING_COMPAT": "1",
                    },
                    clear=False,
                ),
            ):
                exec(compile(shim, str(shim_root / "sitecustomize.py"), "exec"), {})
            self.assertEqual("host.docker.internal", fake_runtime._docker_host_address())

            calls: list[tuple[str, str | None, int | None]] = []

            class FakeEnv:
                async def exec(self, command, *, user=None, timeout_sec=None):
                    calls.append((command, user, timeout_sec))
                    return SimpleNamespace(return_code=0)

            asyncio.run(
                fake_lockdown._kill_sandbox_user_procs(FakeEnv(), "agent")
            )
            self.assertEqual(1, len(calls))
            command, user, timeout_sec = calls[0]
            self.assertEqual("root", user)
            self.assertEqual(10, timeout_sec)
            self.assertIn("pkill -u agent", command)
            self.assertIn("pkill -9 -u agent", command)
            self.assertIn("! pgrep -u agent", command)

    def test_deepseek_callback_replays_exact_reasoning_for_split_tool_calls(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            callback = self._deepseek_callback_compat_harness(Path(tmp))

            response = {
                "id": "response_exact",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "provider-exact-reasoning",
                            "tool_calls": [
                                {"id": "call_00_exact", "type": "function"},
                                {"id": "call_01_exact", "type": "function"},
                            ],
                        }
                    }
                ]
            }
            asyncio.run(
                callback.async_post_call_success_hook(
                    {"model": "deepseek-v4-flash"}, None, response
                )
            )
            original_messages = [
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "provider-exact-reasoning",
                    "tool_calls": [
                        {"id": "call_00_exact", "type": "function"}
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_00_exact",
                    "content": "first result",
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_01_exact",
                            "type": "function",
                            "function": {
                                "name": "terminal",
                                "arguments": "modified-by-agent-serializer",
                            },
                        }
                    ],
                },
            ]
            data = {
                "model": "deepseek-v4-flash",
                "messages": original_messages,
            }
            repaired = asyncio.run(
                callback.async_pre_call_hook(None, None, data, "acompletion")
            )

            self.assertIsNotNone(repaired)
            self.assertEqual(
                "provider-exact-reasoning",
                repaired["messages"][2]["reasoning_content"],
            )
            self.assertEqual("", repaired["messages"][2]["content"])
            self.assertNotIn("reasoning_content", original_messages[2])
            self.assertIsNone(original_messages[2]["content"])

    def test_deepseek_callback_preserves_provider_empty_reasoning_field(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            callback = self._deepseek_callback_compat_harness(Path(tmp))
            response = {
                "id": "response_empty_reasoning",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "",
                            "tool_calls": [
                                {"id": "call_empty_reasoning", "type": "function"}
                            ],
                        }
                    }
                ],
            }
            asyncio.run(
                callback.async_post_call_success_hook(
                    {"model": "deepseek-v4-flash"}, None, response
                )
            )
            data = {
                "model": "deepseek-v4-flash",
                "messages": [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"id": "call_empty_reasoning", "type": "function"}
                        ],
                    }
                ],
            }
            repaired = asyncio.run(
                callback.async_pre_call_hook(None, None, data, "acompletion")
            )
            self.assertIn("reasoning_content", repaired["messages"][0])
            self.assertEqual("", repaired["messages"][0]["reasoning_content"])
            self.assertNotIn("reasoning_content", data["messages"][0])

            conflicting = {
                "model": "deepseek-v4-flash",
                "messages": [
                    {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": None,
                        "tool_calls": [
                            {"id": "call_empty_reasoning", "type": "function"}
                        ],
                    }
                ],
            }
            with self.assertRaisesRegex(RuntimeError, "conflicts"):
                asyncio.run(
                    callback.async_pre_call_hook(
                        None, None, conflicting, "acompletion"
                    )
                )

            nullable_response = {
                "id": "response_null_reasoning",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": None,
                            "tool_calls": [
                                {"id": "call_null_reasoning", "type": "function"}
                            ],
                        }
                    }
                ],
            }
            asyncio.run(
                callback.async_post_call_success_hook(
                    {"model": "deepseek-v4-flash"}, None, nullable_response
                )
            )
            nullable_data = {
                "model": "deepseek-v4-flash",
                "messages": [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"id": "call_null_reasoning", "type": "function"}
                        ],
                    }
                ],
            }
            nullable_repaired = asyncio.run(
                callback.async_pre_call_hook(
                    None, None, nullable_data, "acompletion"
                )
            )
            self.assertIn(
                "reasoning_content", nullable_repaired["messages"][0]
            )
            self.assertIsNone(
                nullable_repaired["messages"][0]["reasoning_content"]
            )

    def test_deepseek_callback_reasoning_replay_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            callback = self._deepseek_callback_compat_harness(Path(tmp))

            missing = {
                "model": "deepseek-v4-flash",
                "messages": [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"id": "call_unseen", "type": "function"}
                        ],
                    }
                ],
            }
            with self.assertRaisesRegex(RuntimeError, "provenance"):
                asyncio.run(
                    callback.async_pre_call_hook(
                        None, None, missing, "acompletion"
                    )
                )

            first = {
                "id": "response_first",
                "choices": [
                    {
                        "message": {
                            "reasoning_content": "first",
                            "tool_calls": [
                                {"id": "call_conflict", "type": "function"}
                            ],
                        }
                    }
                ]
            }
            second = {
                "id": "response_second",
                "choices": [
                    {
                        "message": {
                            "reasoning_content": "second",
                            "tool_calls": [
                                {"id": "call_conflict", "type": "function"}
                            ],
                        }
                    }
                ]
            }
            asyncio.run(
                callback.async_post_call_success_hook(
                    {"model": "deepseek-v4-flash"}, None, first
                )
            )
            with self.assertRaisesRegex(RuntimeError, "identity conflict"):
                asyncio.run(
                    callback.async_post_call_success_hook(
                        {"model": "deepseek-v4-flash"}, None, second
                    )
                )

            conflicting_history = {
                "model": "deepseek-v4-flash",
                "messages": [
                    {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "wrong",
                        "tool_calls": [
                            {"id": "call_conflict", "type": "function"}
                        ],
                    }
                ],
            }
            with self.assertRaisesRegex(RuntimeError, "conflicts"):
                asyncio.run(
                    callback.async_pre_call_hook(
                        None, None, conflicting_history, "acompletion"
                    )
                )

    def test_deepseek_callback_rejects_mixed_turns_and_commits_atomically(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            callback = self._deepseek_callback_compat_harness(Path(tmp))
            for response_id, tool_id in (
                ("response_a", "call_a"),
                ("response_b", "call_b"),
            ):
                asyncio.run(
                    callback.async_post_call_success_hook(
                        {"model": "deepseek-v4-flash"},
                        None,
                        {
                            "id": response_id,
                            "choices": [
                                {
                                    "message": {
                                        "reasoning_content": "same-text",
                                        "tool_calls": [
                                            {"id": tool_id, "type": "function"}
                                        ],
                                    }
                                }
                            ],
                        },
                    )
                )
            mixed = {
                "model": "deepseek-v4-flash",
                "messages": [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"id": "call_a", "type": "function"},
                            {"id": "call_b", "type": "function"},
                        ],
                    }
                ],
            }
            with self.assertRaisesRegex(RuntimeError, "ambiguous"):
                asyncio.run(
                    callback.async_pre_call_hook(None, None, mixed, "acompletion")
                )

            ordered_response = {
                "id": "response_ordered",
                "choices": [
                    {
                        "message": {
                            "reasoning_content": "ordered-reasoning",
                            "tool_calls": [
                                {"id": "call_first", "type": "function"},
                                {"id": "call_second", "type": "function"},
                            ],
                        }
                    }
                ],
            }
            asyncio.run(
                callback.async_post_call_success_hook(
                    {"model": "deepseek-v4-flash"}, None, ordered_response
                )
            )
            reordered = {
                "model": "deepseek-v4-flash",
                "messages": [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"id": "call_second", "type": "function"},
                            {"id": "call_first", "type": "function"},
                        ],
                    }
                ],
            }
            with self.assertRaisesRegex(RuntimeError, "tool order"):
                asyncio.run(
                    callback.async_pre_call_hook(
                        None, None, reordered, "acompletion"
                    )
                )

            reverse_split = {
                "model": "deepseek-v4-flash",
                "messages": [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"id": "call_second", "type": "function"}
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_second",
                        "content": "second",
                    },
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"id": "call_first", "type": "function"}
                        ],
                    },
                ],
            }
            with self.assertRaisesRegex(RuntimeError, "tool order"):
                asyncio.run(
                    callback.async_pre_call_hook(
                        None, None, reverse_split, "acompletion"
                    )
                )

            repeated_split = {
                "model": "deepseek-v4-flash",
                "messages": [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"id": "call_first", "type": "function"}
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_first",
                        "content": "first",
                    },
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"id": "call_first", "type": "function"}
                        ],
                    },
                ],
            }
            with self.assertRaisesRegex(RuntimeError, "repeated in history"):
                asyncio.run(
                    callback.async_pre_call_hook(
                        None, None, repeated_split, "acompletion"
                    )
                )

            before = dict(callback._deepseek_reasoning_by_tool_call)
            invalid_response = {
                "id": "response_invalid",
                "choices": [
                    {
                        "message": {
                            "reasoning_content": "new-reasoning",
                            "tool_calls": [
                                {"id": "call_valid_prefix", "type": "function"},
                                {"id": "", "type": "function"},
                            ],
                        }
                    }
                ],
            }
            with self.assertRaisesRegex(RuntimeError, "identity is invalid"):
                asyncio.run(
                    callback.async_post_call_success_hook(
                        {"model": "deepseek-v4-flash"}, None, invalid_response
                    )
                )
            self.assertEqual(before, callback._deepseek_reasoning_by_tool_call)

            missing_reasoning = {
                "id": "response_missing_reasoning",
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"id": "call_missing_reason", "type": "function"}
                            ]
                        }
                    }
                ],
            }
            with self.assertRaisesRegex(RuntimeError, "provenance is incomplete"):
                asyncio.run(
                    callback.async_post_call_success_hook(
                        {"model": "deepseek-v4-flash"},
                        None,
                        missing_reasoning,
                    )
                )
            self.assertEqual(before, callback._deepseek_reasoning_by_tool_call)

    def test_deepseek_callback_does_not_rewrite_other_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            callback = self._deepseek_callback_compat_harness(Path(tmp))
            data = {
                "model": "openai/gpt-test",
                "messages": [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {"id": "call_other", "type": "function"}
                        ],
                    }
                ],
            }
            self.assertIsNone(
                asyncio.run(
                    callback.async_pre_call_hook(
                        None, None, data, "acompletion"
                    )
                )
            )
            self.assertIsNone(data["messages"][0]["content"])

    def test_hardening_compat_proactive_raw_docker_is_scoped_and_diagnostic(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            harness = self._hardening_compat_harness(
                root, original_error=AssertionError()
            )
            verifier_calls = []

            async def lifecycle():
                result = await harness.harden(
                    harness.env, object(), "agent", workspace="/root"
                )
                verifier_calls.append(True)
                return result

            with (
                patch.dict(sys.modules, harness.modules),
                patch("subprocess.run", side_effect=harness.raw_run),
            ):
                result = asyncio.run(lifecycle())

            self.assertEqual(0, result.return_code)
            self.assertEqual([True], verifier_calls)
            self.assertEqual([], harness.compose_calls)
            self.assertEqual(3, len(harness.raw_calls))
            ps_args = harness.raw_calls[0][0]
            self.assertEqual(
                ["docker", "ps", "-q", "--no-trunc"], ps_args[:4]
            )
            self.assertIn("label=benchflow.owned=true", ps_args)
            self.assertIn(
                "label=com.docker.compose.project=" + harness.inner.session_id,
                ps_args,
            )
            self.assertIn(
                "label=com.docker.compose.service=main", ps_args
            )
            self.assertIn(
                "label=com.docker.compose.container-number=1", ps_args
            )
            self.assertIn("label=com.docker.compose.oneoff=False", ps_args)
            self.assertIn("status=running", ps_args)
            inspect_args = harness.raw_calls[1][0]
            self.assertEqual(
                [
                    "docker",
                    "inspect",
                    "--type",
                    "container",
                    "--format",
                    "{{json .Config.Labels}}|{{json .Config.User}}",
                    harness.container_id,
                ],
                inspect_args,
            )
            raw_args = harness.raw_calls[2][0]
            self.assertEqual(["docker", "exec", "-u", "root"], raw_args[:4])
            self.assertNotIn("-T", raw_args)
            self.assertEqual(harness.container_id, raw_args[4])
            self.assertEqual(["sh", "-c"], raw_args[5:7])

            self.assertNotIn(
                "_run_docker_compose_command", harness.inner.__dict__
            )
            self.assertIs(
                harness.original_run,
                type(harness.inner)._run_docker_compose_command,
            )
            agent_events = [
                json.loads(line)
                for line in (
                    harness.diagnostic_dir / "harden_before_verify_compat.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [
                    "raw_docker_identity_verified",
                    "raw_docker_command_succeeded",
                ],
                [event["event"] for event in agent_events],
            )
            self.assertEqual(
                "kill_sandbox_user_processes", agent_events[1]["step"]
            )
            identity_hash = agent_events[0]["identity_sha256"]
            self.assertEqual(64, len(identity_hash))
            int(identity_hash, 16)
            self.assertEqual(identity_hash, agent_events[1]["identity_sha256"])
            expected_hash = hashlib.sha256(
                json.dumps(
                    harness.command,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            self.assertEqual(expected_hash, agent_events[1]["command_sha256"])
            summary = json.loads(
                (
                    harness.diagnostic_dir / "harden_before_verify_compat.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual("completed", summary["outcome"])
            self.assertEqual(2, len(summary["events"]))
            if os.name == "posix":
                self.assertEqual(
                    0o700, harness.diagnostic_dir.stat().st_mode & 0o777
                )
                self.assertEqual(
                    0o600,
                    (
                        harness.diagnostic_dir
                        / "harden_before_verify_compat.jsonl"
                    ).stat().st_mode & 0o777,
                )
                self.assertEqual(
                    0o600,
                    (
                        harness.diagnostic_dir
                        / "harden_before_verify_compat.json"
                    ).stat().st_mode & 0o777,
                )
            diagnostic_text = json.dumps([agent_events, summary], sort_keys=True)
            self.assertNotIn(harness.inner.session_id, diagnostic_text)
            self.assertNotIn(harness.container_id, diagnostic_text)

    def test_hardening_diagnostics_are_host_only_and_do_not_follow_symlinks(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            harness = self._hardening_compat_harness(
                root, original_error=AssertionError("compose must not run")
            )
            harness.agent_dir.mkdir(exist_ok=True)
            harness.verifier_dir.mkdir(exist_ok=True)
            victim = root / "victim.txt"
            victim.write_text("sentinel", encoding="utf-8")
            agent_link = (
                harness.agent_dir / "harden_before_verify_compat.jsonl"
            )
            try:
                agent_link.symlink_to(victim)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")

            with (
                patch.dict(sys.modules, harness.modules),
                patch("subprocess.run", side_effect=harness.raw_run),
            ):
                result = asyncio.run(
                    harness.harden(harness.env, object(), "agent")
                )

            self.assertEqual(0, result.return_code)
            self.assertEqual("sentinel", victim.read_text(encoding="utf-8"))
            self.assertTrue(agent_link.is_symlink())
            self.assertTrue(
                (harness.diagnostic_dir / "harden_before_verify_compat.jsonl")
                .is_file()
            )
            self.assertTrue(
                (harness.diagnostic_dir / "harden_before_verify_compat.json")
                .is_file()
            )
            self.assertFalse(
                (harness.verifier_dir / "harden_before_verify_compat.json")
                .exists()
            )

    def test_hardening_diagnostic_append_rejects_host_only_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            harness = self._hardening_compat_harness(
                root, original_error=AssertionError("compose must not run")
            )
            harness.diagnostic_dir.mkdir(mode=0o700)
            victim = root / "victim.txt"
            victim.write_text("sentinel", encoding="utf-8")
            event_path = (
                harness.diagnostic_dir / "harden_before_verify_compat.jsonl"
            )
            old_summary_tmp = harness.diagnostic_dir / (
                "harden_before_verify_compat.json" + f".tmp.{os.getpid()}"
            )
            try:
                event_path.symlink_to(victim)
                old_summary_tmp.symlink_to(victim)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")

            with (
                patch.dict(sys.modules, harness.modules),
                patch("subprocess.run", side_effect=harness.raw_run),
            ):
                result = asyncio.run(
                    harness.harden(harness.env, object(), "agent")
                )

            self.assertEqual(0, result.return_code)
            self.assertEqual("sentinel", victim.read_text(encoding="utf-8"))
            self.assertTrue(event_path.is_symlink())
            self.assertTrue(old_summary_tmp.is_symlink())
            summary = json.loads(
                (
                    harness.diagnostic_dir / "harden_before_verify_compat.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual("completed", summary["outcome"])

    def test_hardening_compat_bypasses_v7_ordinal_9_and_10_compose_timeouts(
        self,
    ) -> None:
        cases = (
            ("ordinal_9_path", "printenv PATH"),
            (
                "ordinal_10_trusted_path",
                "python3 -c 'trusted-path-probe' 'input' 'safe' 'blocked'",
            ),
        )
        for name, payload in cases:
            with self.subTest(case=name):
                with tempfile.TemporaryDirectory() as tmp:
                    command = [
                        "exec", "-T", "-u", "root", "main", "sh", "-c",
                        payload,
                    ]
                    harness = self._hardening_compat_harness(
                        Path(tmp),
                        original_error=RuntimeError(
                            "Command timed out after 10 seconds"
                        ),
                        command=command,
                        timeout_sec=10,
                    )
                    with (
                        patch.dict(sys.modules, harness.modules),
                        patch("subprocess.run", side_effect=harness.raw_run),
                    ):
                        result = asyncio.run(
                            harness.harden(harness.env, object(), "agent")
                        )
                    self.assertEqual(0, result.return_code)
                    self.assertEqual([], harness.compose_calls)
                    self.assertEqual(3, len(harness.raw_calls))
                    self.assertEqual(
                        ["docker", "exec"], harness.raw_calls[2][0][:2]
                    )
                    self.assertEqual(payload, harness.raw_calls[2][0][-1])
                    self.assertNotIn(
                        "_run_docker_compose_command", harness.inner.__dict__
                    )

    def test_hardening_compat_routes_non_strict_command_to_compose(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            error = RuntimeError("compose rejected non-strict command")
            harness = self._hardening_compat_harness(
                Path(tmp),
                original_error=error,
                command=["ps", "-q", "main"],
            )
            with (
                patch.dict(sys.modules, harness.modules),
                patch("subprocess.run", side_effect=harness.raw_run),
                self.assertRaisesRegex(
                    RuntimeError, "compose rejected non-strict command"
                ) as caught,
            ):
                asyncio.run(harness.harden(harness.env, object(), "agent"))

            self.assertIs(error, caught.exception)
            self.assertEqual(1, len(harness.compose_calls))
            self.assertEqual(2, len(harness.raw_calls))
            self.assertNotIn(
                "_run_docker_compose_command", harness.inner.__dict__
            )
            events = [
                json.loads(line)
                for line in (
                    harness.diagnostic_dir / "harden_before_verify_compat.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [
                    "raw_docker_identity_verified",
                    "compose_command_failed",
                ],
                [event["event"] for event in events],
            )
            summary = json.loads(
                (
                    harness.diagnostic_dir / "harden_before_verify_compat.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual("failed", summary["outcome"])
            self.assertEqual("RuntimeError", summary["exception_type"])

    def test_hardening_raw_primary_requires_exact_root_shape_and_check_false(
        self,
    ) -> None:
        cases = (
            (
                "numeric_root",
                ["exec", "-T", "-u", "0", "main", "sh", "-c", "probe"],
                False,
            ),
            (
                "workdir",
                [
                    "exec", "-T", "-w", "/root", "-u", "root", "main",
                    "sh", "-c", "probe",
                ],
                False,
            ),
            (
                "missing_tty_disable",
                ["exec", "-u", "root", "main", "sh", "-c", "probe"],
                False,
            ),
            (
                "check_true",
                [
                    "exec", "-T", "-u", "root", "main", "sh", "-c",
                    "probe",
                ],
                True,
            ),
        )
        for name, command, check in cases:
            with self.subTest(case=name):
                with tempfile.TemporaryDirectory() as tmp:
                    harness = self._hardening_compat_harness(
                        Path(tmp),
                        original_error=None,
                        command=command,
                        timeout_sec=10,
                        check=check,
                    )
                    with (
                        patch.dict(sys.modules, harness.modules),
                        patch("subprocess.run", side_effect=harness.raw_run),
                    ):
                        result = asyncio.run(
                            harness.harden(harness.env, object(), "agent")
                        )
                    self.assertEqual(0, result.return_code)
                    self.assertEqual(1, len(harness.compose_calls))
                    self.assertEqual(2, len(harness.raw_calls))

    def test_hardening_compat_long_compose_exception_never_raw_retries(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            error = AssertionError()
            harness = self._hardening_compat_harness(
                Path(tmp), original_error=error, timeout_sec=180
            )
            with (
                patch.dict(sys.modules, harness.modules),
                patch("subprocess.run", side_effect=harness.raw_run),
                self.assertRaises(AssertionError) as caught,
            ):
                asyncio.run(harness.harden(harness.env, object(), "agent"))
            self.assertIs(error, caught.exception)
            self.assertEqual(1, len(harness.compose_calls))
            self.assertEqual(2, len(harness.raw_calls))
            self.assertFalse(any(
                call[0][:2] == ["docker", "exec"] for call in harness.raw_calls
            ))

    def test_hardening_compat_cancelled_error_is_recorded_and_restored(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness = self._hardening_compat_harness(
                Path(tmp),
                original_error=asyncio.CancelledError(),
                timeout_sec=180,
            )
            with (
                patch.dict(sys.modules, harness.modules),
                patch("subprocess.run", side_effect=harness.raw_run),
                self.assertRaises(asyncio.CancelledError),
            ):
                asyncio.run(harness.harden(harness.env, object(), "agent"))

            self.assertNotIn(
                "_run_docker_compose_command", harness.inner.__dict__
            )
            summary = json.loads(
                (
                    harness.diagnostic_dir / "harden_before_verify_compat.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual("failed", summary["outcome"])
            self.assertEqual("CancelledError", summary["exception_type"])
            self.assertEqual(
                "compose_command_failed", summary["events"][-1]["event"]
            )

    def test_hardening_raw_cancelled_error_is_bare_raised_and_restored(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness = self._hardening_compat_harness(
                Path(tmp),
                original_error=AssertionError("compose must not run"),
                raw_error=asyncio.CancelledError(),
            )
            with (
                patch.dict(sys.modules, harness.modules),
                patch("subprocess.run", side_effect=harness.raw_run),
                self.assertRaises(asyncio.CancelledError),
            ):
                asyncio.run(harness.harden(harness.env, object(), "agent"))

            self.assertEqual([], harness.compose_calls)
            self.assertEqual(3, len(harness.raw_calls))
            self.assertNotIn(
                "_run_docker_compose_command", harness.inner.__dict__
            )
            summary = json.loads(
                (
                    harness.diagnostic_dir / "harden_before_verify_compat.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual("failed", summary["outcome"])
            self.assertEqual("CancelledError", summary["exception_type"])
            self.assertEqual(
                "raw_docker_command_failed", summary["events"][-1]["event"]
            )

    def test_hardening_identity_cancelled_error_is_bare_raised(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness = self._hardening_compat_harness(
                Path(tmp),
                original_error=AssertionError("compose must not run"),
            )

            def cancel_identity(*_args, **_kwargs):
                raise asyncio.CancelledError()

            with (
                patch.dict(sys.modules, harness.modules),
                patch("subprocess.run", side_effect=cancel_identity),
                self.assertRaises(asyncio.CancelledError),
            ):
                asyncio.run(harness.harden(harness.env, object(), "agent"))

            self.assertEqual([], harness.compose_calls)
            self.assertNotIn(
                "_run_docker_compose_command", harness.inner.__dict__
            )
            summary = json.loads(
                (
                    harness.diagnostic_dir / "harden_before_verify_compat.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual("failed", summary["outcome"])
            self.assertEqual("CancelledError", summary["exception_type"])
            self.assertEqual(
                "raw_docker_identity_failed", summary["events"][-1]["event"]
            )

    def test_hardening_compat_rejects_ambiguous_container_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness = self._hardening_compat_harness(
                Path(tmp),
                original_error=AssertionError("compose must not run"),
                ps_stdout=("a1" * 32) + "\n" + ("b2" * 32) + "\n",
            )
            with (
                patch.dict(sys.modules, harness.modules),
                patch("subprocess.run", side_effect=harness.raw_run),
                self.assertRaisesRegex(
                    RuntimeError, "Docker identity validation failed"
                ) as caught,
            ):
                asyncio.run(harness.harden(harness.env, object(), "agent"))

            self.assertIsInstance(caught.exception.__cause__, RuntimeError)
            self.assertEqual([], harness.compose_calls)
            self.assertEqual(1, len(harness.raw_calls))
            events = [
                json.loads(line)
                for line in (
                    harness.diagnostic_dir / "harden_before_verify_compat.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(["raw_docker_identity_failed"], [
                event["event"] for event in events
            ])
            diagnostic_text = json.dumps(events, sort_keys=True)
            self.assertNotIn(harness.inner.session_id, diagnostic_text)
            self.assertNotIn("a1" * 32, diagnostic_text)

    def test_hardening_compat_revalidates_labels_and_raw_failure_restores(
        self,
    ) -> None:
        cases = (
            {
                "name": "inspect_identity_mismatch",
                "inspect_labels": {
                    "benchflow.owned": "true",
                    "com.docker.compose.project": "wrong-project",
                    "com.docker.compose.service": "main",
                    "com.docker.compose.container-number": "1",
                    "com.docker.compose.oneoff": "False",
                },
                "raw_returncode": 0,
                "raw_error": None,
                "expected_calls": 2,
                "expected_event": "raw_docker_identity_failed",
                "expected_error": "Docker identity validation failed",
            },
            {
                "name": "raw_timeout",
                "inspect_labels": None,
                "raw_returncode": 0,
                "raw_error": subprocess.TimeoutExpired(
                    ["docker", "exec"], 10
                ),
                "expected_calls": 3,
                "expected_event": "raw_docker_command_failed",
                "expected_error": "proactive raw Docker hardening failed",
            },
        )
        for case in cases:
            with self.subTest(case=case["name"]):
                with tempfile.TemporaryDirectory() as tmp:
                    harness = self._hardening_compat_harness(
                        Path(tmp),
                        original_error=AssertionError("compose must not run"),
                        inspect_labels=case["inspect_labels"],
                        raw_returncode=case["raw_returncode"],
                        raw_error=case["raw_error"],
                    )
                    with (
                        patch.dict(sys.modules, harness.modules),
                        patch("subprocess.run", side_effect=harness.raw_run),
                        self.assertRaisesRegex(
                            RuntimeError, case["expected_error"]
                        ),
                    ):
                        asyncio.run(
                            harness.harden(harness.env, object(), "agent")
                        )

                    self.assertEqual([], harness.compose_calls)
                    self.assertEqual(case["expected_calls"], len(harness.raw_calls))
                    self.assertNotIn(
                        "_run_docker_compose_command", harness.inner.__dict__
                    )
                    events = [
                        json.loads(line)
                        for line in (
                            harness.diagnostic_dir
                            / "harden_before_verify_compat.jsonl"
                        ).read_text(encoding="utf-8").splitlines()
                    ]
                    self.assertEqual(case["expected_event"], events[-1]["event"])
                    diagnostic_text = json.dumps(events, sort_keys=True)
                    self.assertNotIn(harness.inner.session_id, diagnostic_text)
                    self.assertNotIn(harness.container_id, diagnostic_text)
                    self.assertNotIn("wrong-project", diagnostic_text)
                    summary = json.loads(
                        (
                            harness.diagnostic_dir
                            / "harden_before_verify_compat.json"
                        ).read_text(encoding="utf-8")
                    )
                    self.assertEqual("failed", summary["outcome"])

    def test_hardening_compat_raw_primary_preserves_nonzero_exec_result(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness = self._hardening_compat_harness(
                Path(tmp),
                original_error=AssertionError("compose must not run"),
                raw_returncode=23,
            )
            with (
                patch.dict(sys.modules, harness.modules),
                patch("subprocess.run", side_effect=harness.raw_run),
            ):
                result = asyncio.run(
                    harness.harden(harness.env, object(), "agent")
                )
            self.assertEqual(23, result.return_code)
            self.assertIsNone(result.stderr)
            self.assertEqual([], harness.compose_calls)
            events = [
                json.loads(line)
                for line in (
                    harness.diagnostic_dir / "harden_before_verify_compat.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual("raw_docker_command_succeeded", events[-1]["event"])
            self.assertEqual(23, events[-1]["return_code"])

    def test_hardening_compat_hybrid_oracle_routes_ordinals_1_and_9_to_13_raw(
        self,
    ) -> None:
        def strict(payload: str) -> list[str]:
            return [
                "exec", "-T", "-u", "root", "main", "sh", "-c", payload,
            ]

        payloads = [f"official-hardening-{ordinal}" for ordinal in range(1, 14)]
        timeouts = [10, 180, 180, 30, 180, 180, 180, 180, 10, 10, 10, 5, 15]
        sequence = [
            (strict(payload), timeout)
            for payload, timeout in zip(payloads, timeouts, strict=True)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            harness = self._hardening_compat_harness(
                Path(tmp), original_error=None, command_sequence=sequence
            )
            with (
                patch.dict(sys.modules, harness.modules),
                patch("subprocess.run", side_effect=harness.raw_run),
            ):
                result = asyncio.run(
                    harness.harden(harness.env, object(), "agent")
                )

            self.assertEqual(0, result.return_code)
            self.assertEqual(payloads[1:8], [
                call[0][-1] for call in harness.compose_calls
            ])
            raw_exec_calls = [
                call for call in harness.raw_calls
                if call[0][:2] == ["docker", "exec"]
            ]
            self.assertEqual(
                [payloads[0], *payloads[8:13]],
                [call[0][-1] for call in raw_exec_calls],
            )
            self.assertEqual(8, len(harness.raw_calls))
            self.assertTrue(all(
                call[1].get("env") == {"DOCKER_HOST": "mock://benchflow"}
                for call in harness.raw_calls
            ))
            self.assertTrue(all(
                call[1].get("stderr") is subprocess.STDOUT
                for call in raw_exec_calls
            ))
            summary_path = (
                harness.diagnostic_dir / "harden_before_verify_compat.json"
            )
            self.assertTrue(summary_path.is_file())
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual("completed", summary["outcome"])
            raw_events = [
                event for event in summary["events"]
                if event["event"] == "raw_docker_command_succeeded"
            ]
            self.assertEqual(
                [1, 9, 10, 11, 12, 13],
                [event["exec_ordinal"] for event in raw_events],
            )
            self.assertEqual([0] * 6, [
                event["return_code"] for event in raw_events
            ])
            self.assertNotIn("exception_type", summary)
            self.assertNotIn(
                "_run_docker_compose_command", harness.inner.__dict__
            )

    def test_hardening_compat_requires_exact_docker_sandbox(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness = self._hardening_compat_harness(
                Path(tmp),
                original_error=AssertionError("compose must not run"),
                docker=False,
            )
            with (
                patch.dict(sys.modules, harness.modules),
                patch(
                    "subprocess.run",
                    side_effect=AssertionError("identity lookup must not run"),
                ),
                self.assertRaisesRegex(RuntimeError, "exact DockerSandbox"),
            ):
                asyncio.run(harness.harden(harness.env, object(), "agent"))
            self.assertEqual([], harness.compose_calls)
            self.assertEqual([], harness.raw_calls)

    def test_verifier_exec_compat_bypasses_v8_empty_compose_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trusted_env = {
                "PATH": "/trusted/verifier/bin",
                "PYTHONPATH": "/trusted/verifier/python",
            }
            chmod_command = [
                "exec", "-T", "-u", "root", "main", "sh", "-c",
                "chmod +x /verifier/test.sh",
            ]
            test_command = (
                "/verifier/test.sh > /logs/verifier/test-stdout.txt 2>&1"
            )
            harness = self._hardening_compat_harness(
                Path(tmp),
                original_error=None,
                long_compose_error=AssertionError(),
                verifier_env=trusted_env,
                verifier_exec_sequence=[
                    {
                        "command": "chmod +x /verifier/test.sh",
                        "env": None,
                        "user": "root",
                        "service": "main",
                        "timeout_sec": 10,
                    },
                    {
                        "command": test_command,
                        "cwd": None,
                        "env": dict(trusted_env),
                        "user": None,
                        "service": "main",
                        "timeout_sec": 240,
                    },
                ],
            )
            with (
                patch.dict(sys.modules, harness.modules),
                patch("subprocess.run", side_effect=harness.raw_run),
                patch(
                    "asyncio.create_subprocess_exec",
                    side_effect=harness.raw_create_subprocess_exec,
                ),
            ):
                result = asyncio.run(harness.verifier._verify_test_script())

            self.assertEqual(0, result.return_code)
            self.assertEqual([(chmod_command, False, 10)], harness.compose_calls)
            self.assertEqual(2, len(harness.raw_calls))
            self.assertEqual(1, len(harness.async_raw_calls))
            raw_args = harness.async_raw_calls[0][0]
            self.assertEqual(
                ["docker", "exec", harness.container_id],
                raw_args[:3],
            )
            self.assertEqual(["sh", "-c"], raw_args[3:5])
            self.assertTrue(raw_args[-1].startswith("mock-env-file:"))
            self.assertTrue(raw_args[-1].endswith(harness.verifier_test_command))
            self.assertNotIn("exec", harness.inner.__dict__)
            self.assertIs(
                harness.original_exec,
                type(harness.inner).exec,
            )

            event_path = (
                harness.diagnostic_dir / "verifier_test_exec_compat.jsonl"
            )
            summary_path = (
                harness.diagnostic_dir / "verifier_test_exec_compat.json"
            )
            events = [
                json.loads(line)
                for line in event_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [
                    "raw_verifier_identity_verified",
                    "raw_verifier_test_succeeded",
                ],
                [event["event"] for event in events],
            )
            self.assertEqual(0, events[-1]["return_code"])
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual("completed", summary["outcome"])
            diagnostic_text = json.dumps(summary, sort_keys=True)
            self.assertNotIn(harness.inner.session_id, diagnostic_text)
            self.assertNotIn(harness.container_id, diagnostic_text)
            self.assertNotIn(harness.verifier_test_command, diagnostic_text)
            self.assertFalse(
                (harness.verifier_dir / "verifier_test_exec_compat.json").exists()
            )

    def test_verifier_exec_compat_accepts_official_timeout_range(self) -> None:
        for timeout_sec in (240, 600, 900, 1350, 1800):
            with self.subTest(timeout_sec=timeout_sec):
                with tempfile.TemporaryDirectory() as tmp:
                    harness = self._hardening_compat_harness(
                        Path(tmp),
                        original_error=AssertionError("compose must not run"),
                        verifier_timeout_sec=timeout_sec,
                    )
                    with (
                        patch.dict(sys.modules, harness.modules),
                        patch("subprocess.run", side_effect=harness.raw_run),
                        patch(
                            "asyncio.create_subprocess_exec",
                            side_effect=harness.raw_create_subprocess_exec,
                        ),
                    ):
                        result = asyncio.run(
                            harness.verifier._verify_test_script()
                        )
                    self.assertEqual(0, result.return_code)
                    self.assertEqual([], harness.compose_calls)
                    self.assertEqual(2, len(harness.raw_calls))
                    self.assertEqual(1, len(harness.async_raw_calls))

    def test_verifier_exec_compat_preserves_persistent_env_without_leaking(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            secret = "diagnostic-must-not-contain-this-value"
            verifier_env = {
                "PATH": "/trusted/verifier/bin",
                "VERIFIER_TOKEN": secret,
            }
            harness = self._hardening_compat_harness(
                Path(tmp),
                original_error=AssertionError("compose must not run"),
                verifier_env=verifier_env,
                persistent_env={"LANG": "C", "PATH": "/persistent/bin"},
            )
            with (
                patch.dict(sys.modules, harness.modules),
                patch("subprocess.run", side_effect=harness.raw_run),
                patch(
                    "asyncio.create_subprocess_exec",
                    side_effect=harness.raw_create_subprocess_exec,
                ),
            ):
                result = asyncio.run(harness.verifier._verify_test_script())

            self.assertEqual(0, result.return_code)
            wrapped = harness.async_raw_calls[0][0][-1]
            encoded_env, command = wrapped.removeprefix(
                "mock-env-file:"
            ).split("\n", 1)
            self.assertEqual(
                {
                    "LANG": "C",
                    "PATH": "/trusted/verifier/bin",
                    "VERIFIER_TOKEN": secret,
                },
                json.loads(encoded_env),
            )
            self.assertEqual(harness.verifier_test_command, command)
            diagnostic_text = "\n".join(
                path.read_text(encoding="utf-8")
                for path in (
                    harness.diagnostic_dir
                    / "verifier_test_exec_compat.jsonl",
                    harness.diagnostic_dir
                    / "verifier_test_exec_compat.json",
                )
            )
            self.assertNotIn(secret, diagnostic_text)
            self.assertNotIn("VERIFIER_TOKEN", diagnostic_text)

    def test_verifier_exec_compat_scope_mutations_fail_closed(self) -> None:
        cases = (
            ("container_user", {"container_user": "1000"}, 2),
            ("verifier_user", {"verifier_user": "root"}, 0),
            ("default_user", {"default_user": "root"}, 0),
            ("empty_env", {"verifier_env": {}}, 0),
            ("wrong_type", {"verifier_type": "llm-judge"}, 0),
            ("wrong_service", {"verifier_service": "target"}, 0),
            ("indirect_sandbox", {}, 0),
        )
        for name, kwargs, identity_calls in cases:
            with self.subTest(case=name):
                with tempfile.TemporaryDirectory() as tmp:
                    harness = self._hardening_compat_harness(
                        Path(tmp),
                        original_error=AssertionError("compose must not run"),
                        **kwargs,
                    )
                    if name == "indirect_sandbox":
                        harness.verifier._sandbox = SimpleNamespace(
                            inner=harness.inner,
                            exec=harness.inner.exec,
                        )
                    with (
                        patch.dict(sys.modules, harness.modules),
                        patch("subprocess.run", side_effect=harness.raw_run),
                        patch(
                            "asyncio.create_subprocess_exec",
                            side_effect=harness.raw_create_subprocess_exec,
                        ),
                        self.assertRaisesRegex(
                            RuntimeError, "scope validation failed"
                        ),
                    ):
                        asyncio.run(harness.verifier._verify_test_script())
                    self.assertEqual([], harness.compose_calls)
                    self.assertEqual(identity_calls, len(harness.raw_calls))
                    self.assertEqual([], harness.async_raw_calls)
                    self.assertNotIn("exec", harness.inner.__dict__)

    def test_official_docker_sandbox_exec_builder_uses_implicit_user_and_env(
        self,
    ) -> None:
        try:
            from benchflow.sandbox.docker import DockerSandbox
        except ImportError:
            self.skipTest("BenchFlow is installed only in the WSL tool env")

        sandbox = object.__new__(DockerSandbox)
        sandbox.default_user = None
        sandbox._persistent_env = {}
        calls = []

        async def fake_compose(
            _self, command, check=True, timeout_sec=None
        ):
            calls.append((list(command), check, timeout_sec))
            return SimpleNamespace(stdout=None, stderr=None, return_code=0)

        sandbox._run_docker_compose_command = MethodType(
            fake_compose, sandbox
        )
        command = "/verifier/test.sh > /logs/verifier/test-stdout.txt 2>&1"
        secret = "official-builder-secret"
        result = asyncio.run(
            sandbox.exec(
                command,
                env={"VERIFIER_TOKEN": secret},
                user=None,
                service="main",
                timeout_sec=240,
            )
        )

        self.assertEqual(0, result.return_code)
        self.assertEqual(1, len(calls))
        argv, check, timeout_sec = calls[0]
        self.assertEqual(["exec", "-T", "main", "sh", "-c"], argv[:5])
        self.assertNotIn("-u", argv)
        self.assertNotEqual(command, argv[-1])
        self.assertTrue(argv[-1].endswith(command))
        self.assertNotIn(secret, argv[-1])
        self.assertFalse(check)
        self.assertEqual(240, timeout_sec)

    def test_verifier_exec_compat_near_matches_stay_on_compose(self) -> None:
        trusted_env = {"PATH": "/trusted/verifier/bin"}
        exact = {
            "command": "/verifier/test.sh > /logs/verifier/test-stdout.txt 2>&1",
            "cwd": None,
            "env": dict(trusted_env),
            "timeout_sec": 240,
            "user": None,
            "service": "main",
        }
        cases = (
            ("explicit_root", {"user": "root"}, True),
            ("workdir", {"cwd": "/root"}, True),
            ("other_service", {"service": "target"}, True),
            ("other_payload", {"command": "/verifier/other.sh"}, True),
            ("other_env", {"env": {"PATH": "/other"}}, True),
            ("timeout_mismatch", {"timeout_sec": 241}, False),
            ("timeout_too_short", {"timeout_sec": 15}, False),
            ("timeout_too_long", {"timeout_sec": 1801}, False),
        )
        for name, overrides, is_candidate in cases:
            with self.subTest(case=name):
                with tempfile.TemporaryDirectory() as tmp:
                    requested = {**exact, **overrides}
                    harness = self._hardening_compat_harness(
                        Path(tmp),
                        original_error=None,
                        verifier_env=trusted_env,
                        verifier_exec_sequence=[requested],
                    )
                    context = (
                        self.assertRaisesRegex(
                            RuntimeError, "rejected long command"
                        )
                        if is_candidate
                        else nullcontext()
                    )
                    with (
                        patch.dict(sys.modules, harness.modules),
                        patch("subprocess.run", side_effect=harness.raw_run),
                        patch(
                            "asyncio.create_subprocess_exec",
                            side_effect=harness.raw_create_subprocess_exec,
                        ),
                        context,
                    ):
                        result = asyncio.run(
                            harness.verifier._verify_test_script()
                        )
                    if not is_candidate:
                        self.assertEqual(0, result.return_code)
                    self.assertEqual(
                        0 if is_candidate else 1,
                        len(harness.compose_calls),
                    )
                    self.assertEqual(2, len(harness.raw_calls))
                    self.assertEqual([], harness.async_raw_calls)
                    self.assertNotIn(
                        "exec", harness.inner.__dict__
                    )

    def test_verifier_exec_compat_raw_nonzero_is_not_fabricated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness = self._hardening_compat_harness(
                Path(tmp),
                original_error=AssertionError("compose must not run"),
                raw_returncode=23,
            )
            with (
                patch.dict(sys.modules, harness.modules),
                patch("subprocess.run", side_effect=harness.raw_run),
                patch(
                    "asyncio.create_subprocess_exec",
                    side_effect=harness.raw_create_subprocess_exec,
                ),
            ):
                result = asyncio.run(harness.verifier._verify_test_script())
            self.assertEqual(23, result.return_code)
            self.assertIsNone(result.stderr)
            self.assertEqual([], harness.compose_calls)
            self.assertEqual(2, len(harness.raw_calls))
            self.assertEqual(1, len(harness.async_raw_calls))
            self.assertNotIn("exec", harness.inner.__dict__)
            summary = json.loads(
                (
                    harness.diagnostic_dir / "verifier_test_exec_compat.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual("completed", summary["outcome"])
            self.assertEqual(23, summary["events"][-1]["return_code"])

    def test_verifier_exec_compat_failures_never_compose_retry_and_restore(
        self,
    ) -> None:
        cases = (
            (
                "timeout",
                subprocess.TimeoutExpired(["docker", "exec"], 240),
                RuntimeError,
                "RuntimeError",
            ),
            ("oserror", OSError("docker unavailable"), RuntimeError, "RuntimeError"),
            (
                "cancelled",
                asyncio.CancelledError(),
                asyncio.CancelledError,
                "CancelledError",
            ),
        )
        for name, raw_error, expected_error, diagnostic_type in cases:
            with self.subTest(case=name):
                with tempfile.TemporaryDirectory() as tmp:
                    harness = self._hardening_compat_harness(
                        Path(tmp),
                        original_error=AssertionError("compose must not run"),
                        raw_error=raw_error,
                    )
                    with (
                        patch.dict(sys.modules, harness.modules),
                        patch("subprocess.run", side_effect=harness.raw_run),
                        patch(
                            "asyncio.create_subprocess_exec",
                            side_effect=harness.raw_create_subprocess_exec,
                        ),
                        self.assertRaises(expected_error),
                    ):
                        asyncio.run(harness.verifier._verify_test_script())
                    self.assertEqual([], harness.compose_calls)
                    self.assertEqual(2, len(harness.raw_calls))
                    self.assertEqual(1, len(harness.async_raw_calls))
                    self.assertNotIn("exec", harness.inner.__dict__)
                    if name in {"timeout", "cancelled"}:
                        self.assertEqual(1, len(harness.async_processes))
                        self.assertTrue(harness.async_processes[0].terminated)
                    else:
                        self.assertEqual([], harness.async_processes)
                    summary = json.loads(
                        (
                            harness.diagnostic_dir
                            / "verifier_test_exec_compat.json"
                        ).read_text(encoding="utf-8")
                    )
                    self.assertEqual("failed", summary["outcome"])
                    self.assertEqual(diagnostic_type, summary["exception_type"])
                    self.assertEqual(
                        "raw_verifier_test_failed",
                        summary["events"][-1]["event"],
                    )
                    self.assertEqual(
                        diagnostic_type,
                        summary["events"][-1]["exception_type"],
                    )

    def test_verifier_exec_compat_external_cancellation_reaps_raw_process(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness = self._hardening_compat_harness(
                Path(tmp),
                original_error=AssertionError("compose must not run"),
                block_async_raw=True,
            )

            async def cancel_in_flight() -> None:
                task = asyncio.create_task(
                    harness.verifier._verify_test_script()
                )
                while not harness.async_processes:
                    await asyncio.sleep(0)
                process = harness.async_processes[0]
                await process.communicate_started.wait()
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task

            with (
                patch.dict(sys.modules, harness.modules),
                patch("subprocess.run", side_effect=harness.raw_run),
                patch(
                    "asyncio.create_subprocess_exec",
                    side_effect=harness.raw_create_subprocess_exec,
                ),
            ):
                asyncio.run(cancel_in_flight())

            self.assertEqual([], harness.compose_calls)
            self.assertEqual(2, len(harness.raw_calls))
            self.assertEqual(1, len(harness.async_raw_calls))
            self.assertEqual(1, len(harness.async_processes))
            self.assertTrue(harness.async_processes[0].terminated)
            self.assertIsNotNone(harness.async_processes[0].returncode)
            self.assertNotIn("exec", harness.inner.__dict__)
            summary = json.loads(
                (
                    harness.diagnostic_dir / "verifier_test_exec_compat.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual("failed", summary["outcome"])
            self.assertEqual("CancelledError", summary["exception_type"])
            self.assertEqual(
                "raw_verifier_test_failed",
                summary["events"][-1]["event"],
            )

    def test_verifier_exec_compat_host_bind_and_identity_fail_closed(self) -> None:
        cases = ("host_bind", "ambiguous_identity")
        for name in cases:
            with self.subTest(case=name):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    harness = self._hardening_compat_harness(
                        root,
                        original_error=AssertionError("compose must not run"),
                        ps_stdout=(
                            ("a1" * 32) + "\n" + ("b2" * 32) + "\n"
                            if name == "ambiguous_identity"
                            else None
                        ),
                    )
                    if name == "host_bind":
                        harness.inner.rollout_paths.verifier_dir = (
                            root / "other" / "verifier"
                        )
                    with (
                        patch.dict(sys.modules, harness.modules),
                        patch("subprocess.run", side_effect=harness.raw_run),
                        patch(
                            "asyncio.create_subprocess_exec",
                            side_effect=harness.raw_create_subprocess_exec,
                        ),
                        self.assertRaisesRegex(
                            RuntimeError, "scope validation failed"
                        ),
                    ):
                        asyncio.run(harness.verifier._verify_test_script())
                    self.assertEqual([], harness.compose_calls)
                    self.assertEqual(
                        0 if name == "host_bind" else 1,
                        len(harness.raw_calls),
                    )
                    self.assertNotIn("exec", harness.inner.__dict__)
                    summary = json.loads(
                        (
                            harness.diagnostic_dir
                            / "verifier_test_exec_compat.json"
                        ).read_text(encoding="utf-8")
                    )
                    self.assertEqual("failed", summary["outcome"])
                    self.assertEqual(
                        "raw_verifier_identity_failed",
                        summary["events"][-1]["event"],
                    )

    def test_failure_forensics_retains_v2_and_v3_terminal_unscored_docker(
        self,
    ) -> None:
        cases = (
            {
                "phase": "verified",
                "verifier_error": "verifier crashed: ",
                "agent_error": None,
                "expected_reason": "verifier_infrastructure_failure",
            },
            {
                "phase": "verifying",
                "verifier_error": None,
                "agent_error": "",
                "expected_reason": "terminal_unscored_lifecycle_failure",
            },
        )
        for case in cases:
            with self.subTest(case=case):
                with tempfile.TemporaryDirectory() as tmp:
                    harness = self._failure_forensics_harness(
                        Path(tmp),
                        phase=case["phase"],
                        verifier_error=case["verifier_error"],
                        agent_error=case["agent_error"],
                    )
                    with patch.dict(sys.modules, harness.modules):
                        result = asyncio.run(harness.cleanup())
                    self.assertEqual("cleanup-result", result)
                    self.assertEqual([True], harness.cleanup_observations)
                    self.assertTrue(
                        harness.rollout._env.inner._keep_containers
                    )
                    marker = json.loads(
                        harness.marker.read_text(encoding="utf-8")
                    )
                    self.assertTrue(marker["retention_requested"])
                    self.assertTrue(marker["cleanup_completed"])
                    self.assertEqual("stopped", marker["expected_container_state"])
                    self.assertEqual(case["expected_reason"], marker["reason"])
                    self.assertEqual(
                        "shock-analysis-demand__forensics",
                        marker["compose_project"],
                    )

    def test_failure_forensics_does_not_retain_noneligible_rollouts(self) -> None:
        cases = (
            {"rewards": {"reward": 1.0}},
            {"agent_error": "real agent failure"},
            {"terminal": False},
            {"phase": "executed"},
            {"docker": False},
            {"skip_verify": True},
        )
        for case in cases:
            with self.subTest(case=case):
                with tempfile.TemporaryDirectory() as tmp:
                    harness = self._failure_forensics_harness(
                        Path(tmp), **case
                    )
                    with patch.dict(sys.modules, harness.modules):
                        result = asyncio.run(harness.cleanup())
                    self.assertEqual("cleanup-result", result)
                    self.assertEqual([False], harness.cleanup_observations)
                    self.assertFalse(
                        harness.rollout._env.inner._keep_containers
                    )
                    self.assertFalse(harness.marker.exists())

    def test_failure_forensics_retains_clean_terminal_timeout_without_reward(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness = self._failure_forensics_harness(
                Path(tmp),
                phase="verifying",
                terminal_kind="agent_timeout",
                agent_error="Agent prompt exceeded wall-clock budget 1800s",
            )
            with patch.dict(sys.modules, harness.modules):
                result = asyncio.run(harness.cleanup())
            self.assertEqual("cleanup-result", result)
            self.assertEqual([True], harness.cleanup_observations)
            marker = json.loads(harness.marker.read_text(encoding="utf-8"))
            self.assertEqual("agent_timeout", marker["terminal_kind"])
            self.assertTrue(marker["retention_requested"])

    def test_failure_forensics_retains_partial_timeout_without_reward(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness = self._failure_forensics_harness(
                Path(tmp),
                phase="verifying",
                terminal_kind="partial_timeout",
                agent_error="Agent prompt exceeded wall-clock budget 1800s",
            )
            with patch.dict(sys.modules, harness.modules):
                result = asyncio.run(harness.cleanup())
            self.assertEqual("cleanup-result", result)
            self.assertEqual([True], harness.cleanup_observations)
            marker = json.loads(harness.marker.read_text(encoding="utf-8"))
            self.assertEqual("partial_agent_timeout", marker["terminal_kind"])
            self.assertTrue(marker["retention_requested"])

    def test_failure_forensics_supports_direct_docker_sandbox(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness = self._failure_forensics_harness(
                Path(tmp),
                phase="verifying",
                agent_error="",
                wrapped_env=False,
            )
            with patch.dict(sys.modules, harness.modules):
                result = asyncio.run(harness.cleanup())
            self.assertEqual("cleanup-result", result)
            self.assertEqual([True], harness.cleanup_observations)
            self.assertTrue(harness.rollout._env._keep_containers)
            marker = json.loads(harness.marker.read_text(encoding="utf-8"))
            self.assertTrue(marker["retention_requested"])

    def test_failure_forensics_atomic_write_does_not_follow_fixed_symlinks(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            harness = self._failure_forensics_harness(
                root, phase="verifying", agent_error=""
            )
            victim = root / "victim.txt"
            victim.write_text("sentinel", encoding="utf-8")
            old_fixed_tmp = harness.marker.with_name(
                harness.marker.name + f".tmp.{os.getpid()}"
            )
            try:
                harness.marker.symlink_to(victim)
                old_fixed_tmp.symlink_to(victim)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")

            with patch.dict(sys.modules, harness.modules):
                result = asyncio.run(harness.cleanup())

            self.assertEqual("cleanup-result", result)
            self.assertEqual("sentinel", victim.read_text(encoding="utf-8"))
            self.assertTrue(harness.marker.is_file())
            self.assertFalse(harness.marker.is_symlink())
            self.assertTrue(old_fixed_tmp.is_symlink())
            marker = json.loads(harness.marker.read_text(encoding="utf-8"))
            self.assertTrue(marker["retention_requested"])

    def test_publish_compat_exact_match_continues_to_verifier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness = self._publish_compat_harness(
                Path(tmp), publish_error=AssertionError()
            )
            trajectory = [
                {"type": "tool_call", "status": "completed"},
                {"type": "agent_message", "text": "deliverable is ready"},
            ]
            original = json.loads(json.dumps(trajectory))
            verifier_calls = []

            async def lifecycle():
                await harness.publish(
                    harness.env, trajectory, harness.agent_dir
                )
                verifier_calls.append(True)
                return {"reward": 1.0}

            with patch.object(subprocess, "run", side_effect=harness.raw_run):
                reward = asyncio.run(lifecycle())
            self.assertEqual({"reward": 1.0}, reward)
            self.assertEqual([True], verifier_calls)
            self.assertEqual(original, trajectory)
            self.assertIs(harness.publish, harness.setup_publish)
            self.assertEqual([], harness.original_calls)
            self.assertEqual([], harness.probes)
            self.assertEqual(2, len(harness.raw_calls))
            diagnostic = json.loads(
                (harness.agent_dir / "trajectory_publish_compat.json").read_text(
                    encoding="utf-8"
                )
            )
            payload = (
                harness.agent_dir / "acp_trajectory.jsonl"
            ).read_bytes()
            self.assertEqual(
                "proactive_verified_bind_publish", diagnostic["recovery"]
            )
            self.assertEqual(len(payload), diagnostic["size"])
            self.assertEqual(
                hashlib.sha256(payload).hexdigest(), diagnostic["sha256"]
            )
            self.assertEqual("agent_message", diagnostic["terminal_kind"])
            self.assertEqual("raw_docker_exec", diagnostic["probe_transport"])
            self.assertEqual([], list(harness.agent_dir.glob("*.tmp*")))

    def test_publish_atomic_write_ignores_fixed_tmp_and_target_symlinks(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            harness = self._publish_compat_harness(
                root, publish_error=AssertionError()
            )
            victim = root / "victim.txt"
            victim.write_text("sentinel", encoding="utf-8")
            target = harness.agent_dir / "acp_trajectory.jsonl"
            diagnostic = harness.agent_dir / "trajectory_publish_compat.json"
            old_payload_tmp = target.with_name(
                target.name + f".tmp.{os.getpid()}"
            )
            old_diagnostic_tmp = diagnostic.with_name(diagnostic.name + ".tmp")
            try:
                target.symlink_to(victim)
                old_payload_tmp.symlink_to(victim)
                old_diagnostic_tmp.symlink_to(victim)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")
            trajectory = [
                {"type": "agent_message", "text": "deliverable is ready"}
            ]

            with patch.object(subprocess, "run", side_effect=harness.raw_run):
                asyncio.run(
                    harness.publish(harness.env, trajectory, harness.agent_dir)
                )

            self.assertEqual("sentinel", victim.read_text(encoding="utf-8"))
            self.assertTrue(target.is_file())
            self.assertFalse(target.is_symlink())
            self.assertTrue(diagnostic.is_file())
            self.assertTrue(old_payload_tmp.is_symlink())
            self.assertTrue(old_diagnostic_tmp.is_symlink())

    def test_publish_readback_rejects_symlink_swap_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            harness = self._publish_compat_harness(
                root, publish_error=AssertionError()
            )
            victim = root / "victim.txt"
            victim.write_text("sentinel", encoding="utf-8")
            capability_link = root / "symlink-capability"
            try:
                capability_link.symlink_to(victim)
                capability_link.unlink()
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")
            trajectory = [
                {"type": "agent_message", "text": "deliverable is ready"}
            ]
            publish_globals = harness.publish.__globals__
            original_atomic_write = publish_globals[
                "_compat_atomic_write_bytes"
            ]

            def write_then_swap(path, payload):
                original_atomic_write(path, payload)
                target = Path(path)
                if target.name == "acp_trajectory.jsonl":
                    target.unlink()
                    target.symlink_to(victim)

            with (
                patch.dict(
                    publish_globals,
                    {"_compat_atomic_write_bytes": write_then_swap},
                ),
                self.assertRaisesRegex(
                    RuntimeError, "trajectory host readback failed"
                ),
            ):
                asyncio.run(
                    harness.publish(harness.env, trajectory, harness.agent_dir)
                )

            self.assertEqual("sentinel", victim.read_text(encoding="utf-8"))
            self.assertEqual([], harness.raw_calls)
            self.assertEqual([], harness.original_calls)

    def test_publish_compat_supports_direct_docker_sandbox(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness = self._publish_compat_harness(
                Path(tmp),
                publish_error=AssertionError(),
                wrapped_env=False,
            )
            trajectory = [
                {"type": "agent_message", "text": "deliverable is ready"}
            ]
            with patch.object(subprocess, "run", side_effect=harness.raw_run):
                asyncio.run(
                    harness.publish(
                        harness.env, trajectory, harness.agent_dir
                    )
                )
            self.assertEqual([], harness.original_calls)
            self.assertEqual([], harness.probes)
            diagnostic = json.loads(
                (harness.agent_dir / "trajectory_publish_compat.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                "proactive_verified_bind_publish", diagnostic["recovery"]
            )

    def test_publish_compat_clean_timeout_proactively_reaches_official_verifier(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness = self._publish_compat_harness(
                Path(tmp),
                publish_error=AssertionError(),
            )
            trajectory = [
                {"type": "tool_call", "status": "completed"},
                {
                    "type": "agent_timeout",
                    "reason": "wall_clock_timeout",
                    "timeout_sec": 1800.0,
                    "pending_tool_call_ids": [],
                    "terminal_trajectory_complete": True,
                },
            ]
            original = json.loads(json.dumps(trajectory))
            verifier_calls = []

            async def lifecycle():
                await harness.publish(
                    harness.env, trajectory, harness.agent_dir
                )
                verifier_calls.append(True)
                return {"reward": 0.25}

            with patch.object(subprocess, "run", side_effect=harness.raw_run):
                reward = asyncio.run(lifecycle())
            self.assertEqual({"reward": 0.25}, reward)
            self.assertEqual([True], verifier_calls)
            self.assertEqual(original, trajectory)
            self.assertEqual([], harness.original_calls)
            self.assertEqual([], harness.probes)
            self.assertEqual(2, len(harness.raw_calls))
            self.assertEqual(["docker", "ps", "-q"], harness.raw_calls[0][0][:3])
            self.assertEqual(["docker", "exec"], harness.raw_calls[1][0][:2])
            diagnostic = json.loads(
                (harness.agent_dir / "trajectory_publish_compat.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual("agent_timeout", diagnostic["terminal_kind"])
            self.assertEqual("raw_docker_exec", diagnostic["probe_transport"])
            self.assertEqual(
                "proactive_verified_bind_publish", diagnostic["recovery"]
            )

    def test_publish_compat_partial_timeout_reaches_official_verifier(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness = self._publish_compat_harness(
                Path(tmp), publish_error=AssertionError()
            )
            trajectory = [
                {"type": "tool_call", "status": "pending"},
                {
                    "type": "agent_timeout",
                    "reason": "wall_clock_timeout",
                    "timeout_sec": 1800.0,
                    "pending_tool_call_ids": ["pending"],
                    "terminal_trajectory_complete": False,
                },
            ]
            with patch.object(subprocess, "run", side_effect=harness.raw_run):
                asyncio.run(
                    harness.publish(harness.env, trajectory, harness.agent_dir)
                )
            self.assertEqual([], harness.original_calls)
            diagnostic = json.loads(
                (harness.agent_dir / "trajectory_publish_compat.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                "partial_agent_timeout", diagnostic["terminal_kind"]
            )
            self.assertEqual(
                "proactive_verified_bind_publish", diagnostic["recovery"]
            )

    def test_publish_compat_raw_hash_failure_is_fail_closed(self) -> None:
        for raw_target_state in ("missing", "mismatch"):
            with self.subTest(raw_target_state=raw_target_state):
                with tempfile.TemporaryDirectory() as tmp:
                    harness = self._publish_compat_harness(
                        Path(tmp),
                        publish_error=AssertionError(),
                        raw_target_state=raw_target_state,
                    )
                    trajectory = [
                        {
                            "type": "agent_timeout",
                            "reason": "wall_clock_timeout",
                            "timeout_sec": 1800.0,
                            "pending_tool_call_ids": [],
                            "terminal_trajectory_complete": True,
                        }
                    ]
                    with (
                        patch.object(
                            subprocess, "run", side_effect=harness.raw_run
                        ),
                        self.assertRaisesRegex(
                            RuntimeError, "raw size/SHA256 verification failed"
                        ),
                    ):
                        asyncio.run(
                            harness.publish(
                                harness.env, trajectory, harness.agent_dir
                            )
                        )
                    self.assertEqual([], harness.original_calls)
                    self.assertEqual([], harness.probes)
                    self.assertFalse(
                        (
                            harness.agent_dir
                            / "trajectory_publish_compat.json"
                        ).exists()
                    )

    def test_publish_compat_nonmatching_bind_falls_back_to_original(self) -> None:
        for target_state in ("path_mismatch", "env_mismatch"):
            with self.subTest(target_state=target_state):
                with tempfile.TemporaryDirectory() as tmp:
                    error = AssertionError()
                    harness = self._publish_compat_harness(
                        Path(tmp),
                        publish_error=error,
                        target_state=target_state,
                    )
                    trajectory = [
                        {"type": "agent_message", "text": "done"}
                    ]
                    with self.assertRaises(AssertionError) as caught:
                        asyncio.run(
                            harness.publish(
                                harness.env, trajectory, harness.agent_dir
                            )
                        )
                    self.assertIs(error, caught.exception)
                    self.assertEqual(1, len(harness.original_calls))
                    self.assertEqual([], harness.raw_calls)
                    self.assertFalse(
                        (
                            harness.agent_dir
                            / "trajectory_publish_compat.json"
                        ).exists()
                    )

    def test_publish_compat_rethrows_without_docker_or_terminal_final(self) -> None:
        cases = (
            (False, [{"type": "agent_message", "text": "done"}]),
            (True, [{"type": "tool_call", "status": "completed"}]),
            (True, [{"type": "agent_message", "text": "   "}]),
        )
        for docker, trajectory in cases:
            with self.subTest(docker=docker, trajectory=trajectory):
                with tempfile.TemporaryDirectory() as tmp:
                    error = AssertionError()
                    harness = self._publish_compat_harness(
                        Path(tmp), publish_error=error, docker=docker
                    )
                    with self.assertRaises(AssertionError) as caught:
                        asyncio.run(
                            harness.publish(
                                harness.env, trajectory, harness.agent_dir
                            )
                        )
                    self.assertIs(error, caught.exception)
                    self.assertEqual(1, len(harness.original_calls))
                    self.assertEqual([], harness.raw_calls)

    def test_publish_compat_fallback_never_suppresses_original_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            error = RuntimeError("docker compose upload failed")
            harness = self._publish_compat_harness(
                Path(tmp), publish_error=error, docker=False
            )
            trajectory = [{"type": "agent_message", "text": "done"}]
            with self.assertRaisesRegex(
                RuntimeError, "docker compose upload failed"
            ) as caught:
                asyncio.run(
                    harness.publish(harness.env, trajectory, harness.agent_dir)
            )
            self.assertIs(error, caught.exception)
            self.assertEqual(1, len(harness.original_calls))
            self.assertEqual([], harness.probes)


if __name__ == "__main__":
    unittest.main()
