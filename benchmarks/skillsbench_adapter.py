"""SkillsBench execution adapter for the released SkillGen pipeline.

The adapter deliberately changes only the benchmark I/O boundary:

* one :class:`~models.TaskInstance` represents one pre-declared stochastic
  rollout of a single SkillsBench task package;
* the task is executed by the official BenchFlow harness and scored by the
  task's verifier;
* task-bundled skills are removed by ``--skill-mode no-skill``;
* a generated SkillGen skill is mounted as a custom runtime skill.

It does not pool examples across SkillsBench tasks, alter SkillGen's prompts,
or add positive examples during refinement.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

import pilot_budget_guard

from benchmarks.skillsbench_rollout_cache import (
    RolloutCacheRequest,
    SkillsBenchRolloutCacheError,
    attempt_directory_name,
    load_cached_trajectory,
    manifested_attempts,
    read_attempt_receipt,
    slot_lock,
    write_attempt_manifest,
    write_attempt_receipt,
    write_cached_trajectory,
)
from models import SkillItem, TaskInstance, Trajectory


ADAPTER_SCHEMA_VERSION = "skillsbench-skillgen-v1"
EXPECTED_BENCHFLOW_VERSION = "0.6.7"
PASS_REWARD = 1.0


def _is_wsl_docker_desktop_runtime() -> bool:
    """Return whether BenchFlow is running in WSL against Docker Desktop.

    BenchFlow 0.6.7 asks Docker for the bridge gateway and binds its host
    LiteLLM proxy to that address.  Docker Desktop reports ``172.17.0.1`` even
    though that address is owned by its VM rather than the WSL host, so the
    proxy fails with ``EADDRNOTAVAIL`` before any provider request.  WSL can
    instead bind all interfaces and containers can reach it through Docker
    Desktop's stable ``host.docker.internal`` name.
    """

    release = platform.release().lower()
    return bool(os.environ.get("WSL_DISTRO_NAME")) and "microsoft" in release


def _write_benchflow_wsl_compat_shim(root: Path) -> Path:
    """Write a process-local BenchFlow 0.6.7 Docker bind compatibility shim."""

    shim_root = root / "benchflow-compat"
    shim_root.mkdir(parents=True, exist_ok=False)
    shim = shim_root / "sitecustomize.py"
    shim.write_text(
        "# Auto-generated for this isolated SkillsBench rollout.\n"
        "import os as _os\n"
        "if _os.environ.get('SKILLSBENCH_APPLY_WSL_DOCKER_COMPAT') == '1':\n"
        "    import hashlib as _compat_hashlib\n"
        "    import secrets as _compat_secrets\n"
        "    import stat as _compat_stat\n"
        "    from pathlib import Path as _CompatPath\n"
        "    from benchflow.providers import litellm_runtime as _runtime\n"
        "    _runtime._docker_host_address = lambda: 'host.docker.internal'\n"
        "    _COMPAT_SECURE_POSIX = (\n"
        "        _os.name == 'posix'\n"
        "        and hasattr(_os, 'O_NOFOLLOW')\n"
        "        and hasattr(_os, 'O_DIRECTORY')\n"
        "    )\n"
        "    def _compat_checked_path(path):\n"
        "        _path = _CompatPath(path)\n"
        "        if (\n"
        "            not _path.is_absolute()\n"
        "            or not _path.name\n"
        "            or _path.name in ('.', '..')\n"
        "        ):\n"
        "            raise RuntimeError('compat host path is invalid')\n"
        "        return _path\n"
        "    def _compat_write_all(fd, payload):\n"
        "        _view = memoryview(payload)\n"
        "        while _view:\n"
        "            _written = _os.write(fd, _view)\n"
        "            if not isinstance(_written, int) or _written <= 0:\n"
        "                raise RuntimeError('compat host write made no progress')\n"
        "            _view = _view[_written:]\n"
        "    def _compat_open_posix_dir(path):\n"
        "        _flags = (\n"
        "            _os.O_RDONLY | _os.O_DIRECTORY | _os.O_NOFOLLOW\n"
        "            | getattr(_os, 'O_CLOEXEC', 0)\n"
        "        )\n"
        "        _fd = _os.open(str(path), _flags)\n"
        "        if not _compat_stat.S_ISDIR(_os.fstat(_fd).st_mode):\n"
        "            _os.close(_fd)\n"
        "            raise RuntimeError('compat host directory is not regular')\n"
        "        return _fd\n"
        "    def _compat_check_windows_dir(path):\n"
        "        _info = _os.lstat(path)\n"
        "        if (\n"
        "            _compat_stat.S_ISLNK(_info.st_mode)\n"
        "            or not _compat_stat.S_ISDIR(_info.st_mode)\n"
        "        ):\n"
        "            raise RuntimeError('compat host directory is unsafe')\n"
        "    def _compat_atomic_write_bytes(path, payload):\n"
        "        _path = _compat_checked_path(path)\n"
        "        if not isinstance(payload, bytes):\n"
        "            raise TypeError('compat atomic payload must be bytes')\n"
        "        _temp_name = (\n"
        "            f'.{_path.name}.tmp.{_os.getpid()}.'\n"
        "            f'{_compat_secrets.token_hex(16)}'\n"
        "        )\n"
        "        _fd = None\n"
        "        _dir_fd = None\n"
        "        _temp_path = _path.with_name(_temp_name)\n"
        "        try:\n"
        "            if _COMPAT_SECURE_POSIX:\n"
        "                _dir_fd = _compat_open_posix_dir(_path.parent)\n"
        "                _flags = (\n"
        "                    _os.O_WRONLY | _os.O_CREAT | _os.O_EXCL\n"
        "                    | _os.O_NOFOLLOW | getattr(_os, 'O_CLOEXEC', 0)\n"
        "                )\n"
        "                _fd = _os.open(\n"
        "                    _temp_name, _flags, 0o600, dir_fd=_dir_fd\n"
        "                )\n"
        "            elif _os.name == 'nt':\n"
        "                _compat_check_windows_dir(_path.parent)\n"
        "                _flags = (\n"
        "                    _os.O_WRONLY | _os.O_CREAT | _os.O_EXCL\n"
        "                    | getattr(_os, 'O_BINARY', 0)\n"
        "                )\n"
        "                _fd = _os.open(str(_temp_path), _flags, 0o600)\n"
        "            else:\n"
        "                raise RuntimeError('secure host writes require nofollow')\n"
        "            if not _compat_stat.S_ISREG(_os.fstat(_fd).st_mode):\n"
        "                raise RuntimeError('compat host temp is not regular')\n"
        "            if hasattr(_os, 'fchmod'):\n"
        "                _os.fchmod(_fd, 0o600)\n"
        "            _compat_write_all(_fd, payload)\n"
        "            _os.fsync(_fd)\n"
        "            _os.close(_fd)\n"
        "            _fd = None\n"
        "            if _COMPAT_SECURE_POSIX:\n"
        "                _os.replace(\n"
        "                    _temp_name, _path.name,\n"
        "                    src_dir_fd=_dir_fd, dst_dir_fd=_dir_fd,\n"
        "                )\n"
        "                _os.fsync(_dir_fd)\n"
        "            else:\n"
        "                _os.replace(_temp_path, _path)\n"
        "            _temp_name = None\n"
        "        finally:\n"
        "            if _fd is not None:\n"
        "                _os.close(_fd)\n"
        "            if _temp_name is not None:\n"
        "                try:\n"
        "                    if _COMPAT_SECURE_POSIX and _dir_fd is not None:\n"
        "                        _os.unlink(_temp_name, dir_fd=_dir_fd)\n"
        "                    else:\n"
        "                        _os.unlink(_temp_path)\n"
        "                except OSError:\n"
        "                    pass\n"
        "            if _dir_fd is not None:\n"
        "                _os.close(_dir_fd)\n"
        "    def _compat_atomic_write_text(path, text):\n"
        "        if not isinstance(text, str):\n"
        "            raise TypeError('compat atomic text must be str')\n"
        "        _compat_atomic_write_bytes(path, text.encode('utf-8'))\n"
        "    def _compat_append_bytes(path, payload):\n"
        "        _path = _compat_checked_path(path)\n"
        "        if not isinstance(payload, bytes):\n"
        "            raise TypeError('compat append payload must be bytes')\n"
        "        _fd = None\n"
        "        _dir_fd = None\n"
        "        try:\n"
        "            if _COMPAT_SECURE_POSIX:\n"
        "                _dir_fd = _compat_open_posix_dir(_path.parent)\n"
        "                _flags = (\n"
        "                    _os.O_WRONLY | _os.O_CREAT | _os.O_APPEND\n"
        "                    | _os.O_NOFOLLOW | getattr(_os, 'O_CLOEXEC', 0)\n"
        "                )\n"
        "                _fd = _os.open(\n"
        "                    _path.name, _flags, 0o600, dir_fd=_dir_fd\n"
        "                )\n"
        "            elif _os.name == 'nt':\n"
        "                _compat_check_windows_dir(_path.parent)\n"
        "                try:\n"
        "                    _existing = _os.lstat(_path)\n"
        "                except FileNotFoundError:\n"
        "                    _existing = None\n"
        "                if (\n"
        "                    _existing is not None\n"
        "                    and _compat_stat.S_ISLNK(_existing.st_mode)\n"
        "                ):\n"
        "                    raise RuntimeError('compat append target is unsafe')\n"
        "                _flags = (\n"
        "                    _os.O_WRONLY | _os.O_CREAT | _os.O_APPEND\n"
        "                    | getattr(_os, 'O_BINARY', 0)\n"
        "                )\n"
        "                _fd = _os.open(str(_path), _flags, 0o600)\n"
        "            else:\n"
        "                raise RuntimeError('secure host writes require nofollow')\n"
        "            if not _compat_stat.S_ISREG(_os.fstat(_fd).st_mode):\n"
        "                raise RuntimeError('compat append target is not regular')\n"
        "            if hasattr(_os, 'fchmod'):\n"
        "                _os.fchmod(_fd, 0o600)\n"
        "            _compat_write_all(_fd, payload)\n"
        "            _os.fsync(_fd)\n"
        "        finally:\n"
        "            if _fd is not None:\n"
        "                _os.close(_fd)\n"
        "            if _dir_fd is not None:\n"
        "                _os.close(_dir_fd)\n"
        "    def _compat_regular_file_digest(path):\n"
        "        _path = _compat_checked_path(path)\n"
        "        _fd = None\n"
        "        _dir_fd = None\n"
        "        try:\n"
        "            if _COMPAT_SECURE_POSIX:\n"
        "                _dir_fd = _compat_open_posix_dir(_path.parent)\n"
        "                _flags = (\n"
        "                    _os.O_RDONLY | _os.O_NOFOLLOW\n"
        "                    | getattr(_os, 'O_CLOEXEC', 0)\n"
        "                )\n"
        "                _fd = _os.open(_path.name, _flags, dir_fd=_dir_fd)\n"
        "            elif _os.name == 'nt':\n"
        "                _compat_check_windows_dir(_path.parent)\n"
        "                _existing = _os.lstat(_path)\n"
        "                if _compat_stat.S_ISLNK(_existing.st_mode):\n"
        "                    raise RuntimeError('compat read target is unsafe')\n"
        "                _fd = _os.open(\n"
        "                    str(_path),\n"
        "                    _os.O_RDONLY | getattr(_os, 'O_BINARY', 0),\n"
        "                )\n"
        "            else:\n"
        "                raise RuntimeError('secure host reads require nofollow')\n"
        "            _info = _os.fstat(_fd)\n"
        "            if not _compat_stat.S_ISREG(_info.st_mode):\n"
        "                raise RuntimeError('compat read target is not regular')\n"
        "            _digest = _compat_hashlib.sha256()\n"
        "            _size = 0\n"
        "            while True:\n"
        "                _chunk = _os.read(_fd, 1024 * 1024)\n"
        "                if not _chunk:\n"
        "                    break\n"
        "                _size += len(_chunk)\n"
        "                _digest.update(_chunk)\n"
        "            return _size, _digest.hexdigest()\n"
        "        finally:\n"
        "            if _fd is not None:\n"
        "                _os.close(_fd)\n"
        "            if _dir_fd is not None:\n"
        "                _os.close(_dir_fd)\n"
        "    def _compat_ensure_private_dir(path):\n"
        "        _path = _compat_checked_path(path)\n"
        "        if _COMPAT_SECURE_POSIX:\n"
        "            _parent_fd = _compat_open_posix_dir(_path.parent)\n"
        "            _dir_fd = None\n"
        "            try:\n"
        "                try:\n"
        "                    _os.mkdir(_path.name, 0o700, dir_fd=_parent_fd)\n"
        "                except FileExistsError:\n"
        "                    pass\n"
        "                _flags = (\n"
        "                    _os.O_RDONLY | _os.O_DIRECTORY | _os.O_NOFOLLOW\n"
        "                    | getattr(_os, 'O_CLOEXEC', 0)\n"
        "                )\n"
        "                _dir_fd = _os.open(\n"
        "                    _path.name, _flags, dir_fd=_parent_fd\n"
        "                )\n"
        "                if not _compat_stat.S_ISDIR(_os.fstat(_dir_fd).st_mode):\n"
        "                    raise RuntimeError('compat private path is not a dir')\n"
        "                _os.fchmod(_dir_fd, 0o700)\n"
        "                _os.fsync(_parent_fd)\n"
        "            finally:\n"
        "                if _dir_fd is not None:\n"
        "                    _os.close(_dir_fd)\n"
        "                _os.close(_parent_fd)\n"
        "        elif _os.name == 'nt':\n"
        "            _compat_check_windows_dir(_path.parent)\n"
        "            try:\n"
        "                _os.mkdir(_path, 0o700)\n"
        "            except FileExistsError:\n"
        "                pass\n"
        "            _compat_check_windows_dir(_path)\n"
        "        else:\n"
        "            raise RuntimeError('secure host dirs require nofollow')\n"
        "    if _os.environ.get('SKILLSBENCH_BENCHFLOW_HARDENING_COMPAT') == '1':\n"
        "        import asyncio as _asyncio\n"
        "        import hashlib as _hardening_hashlib\n"
        "        import json as _hardening_json\n"
        "        import subprocess as _hardening_subprocess\n"
        "        from pathlib import Path as _HardeningPath\n"
        "        from types import MethodType as _MethodType\n"
        "        import shlex as _shlex\n"
        "        from benchflow.sandbox import lockdown as _lockdown\n"
        "        _original_harden_before_verify = _lockdown.harden_before_verify\n"
        "        def _hardening_exec_step(command, ordinal):\n"
        "            _parts = [str(_part) for _part in command]\n"
        "            _payload = (\n"
        "                _parts[-1]\n"
        "                if len(_parts) >= 2 and _parts[-2] == '-c'\n"
        "                else ''\n"
        "            )\n"
        "            if 'pkill -u ' in _payload and '! pgrep -u ' in _payload:\n"
        "                return 'kill_sandbox_user_processes'\n"
        "            if _payload == getattr(_lockdown, '_CLEAR_VERIFIER_DIR_CMD', None):\n"
        "                return 'clear_verifier_output_dir'\n"
        "            if _payload == getattr(_lockdown, '_ENSURE_APP_DIR_CMD', None):\n"
        "                return 'ensure_legacy_app_dir'\n"
        "            if _payload.startswith('chown -R root:root '):\n"
        "                return 'freeze_workspace_ownership'\n"
        "            return f'hardening_compose_exec_{ordinal}'\n"
        "        def _parse_hardening_compose_exec(command):\n"
        "            if not isinstance(command, (list, tuple)):\n"
        "                return None\n"
        "            _parts = [str(_part) for _part in command]\n"
        "            if (\n"
        "                len(_parts) != 8\n"
        "                or _parts[:7]\n"
        "                != ['exec', '-T', '-u', 'root', 'main', 'sh', '-c']\n"
        "                or not _parts[7]\n"
        "            ):\n"
        "                return None\n"
        "            return ['-u', 'root'], 'main', _parts[5:]\n"
        "        def _hardening_diagnostic_paths(inner):\n"
        "            _env_vars = getattr(inner, '_env_vars', None)\n"
        "            if _env_vars is None:\n"
        "                return None, None\n"
        "            try:\n"
        "                _host_dirs = (\n"
        "                    _HardeningPath(_env_vars.host_agent_logs_path),\n"
        "                    _HardeningPath(_env_vars.host_verifier_logs_path),\n"
        "                    _HardeningPath(_env_vars.host_artifacts_path),\n"
        "                )\n"
        "            except (AttributeError, OSError, TypeError, ValueError):\n"
        "                return None, None\n"
        "            if (\n"
        "                any(not _path.is_absolute() for _path in _host_dirs)\n"
        "                or tuple(_path.name for _path in _host_dirs)\n"
        "                != ('agent', 'verifier', 'artifacts')\n"
        "                or len({_path.parent for _path in _host_dirs}) != 1\n"
        "            ):\n"
        "                return None, None\n"
        "            _diagnostic_dir = (\n"
        "                _host_dirs[0].parent / 'benchflow-compat-diagnostics'\n"
        "            )\n"
        "            try:\n"
        "                _compat_ensure_private_dir(_diagnostic_dir)\n"
        "            except Exception:\n"
        "                return None, None\n"
        "            return (\n"
        "                _diagnostic_dir / 'harden_before_verify_compat.jsonl',\n"
        "                _diagnostic_dir / 'harden_before_verify_compat.json',\n"
        "            )\n"
        "        def _append_hardening_diagnostic(state, inner, event):\n"
        "            _record = {'schema_version': 1, **event}\n"
        "            state['events'].append(_record)\n"
        "            _agent_path, _ = _hardening_diagnostic_paths(inner)\n"
        "            if _agent_path is None:\n"
        "                return\n"
        "            try:\n"
        "                _compat_append_bytes(\n"
        "                    _agent_path,\n"
        "                    (\n"
        "                        _hardening_json.dumps(_record, sort_keys=True)\n"
        "                        + '\\n'\n"
        "                    ).encode('utf-8'),\n"
        "                )\n"
        "            except Exception:\n"
        "                pass\n"
        "        def _write_hardening_summary(state, inner, outcome, exc=None):\n"
        "            if not state['events'] and exc is None:\n"
        "                return\n"
        "            _, _verifier_path = _hardening_diagnostic_paths(inner)\n"
        "            if _verifier_path is None:\n"
        "                return\n"
        "            _summary = {\n"
        "                'schema_version': 1,\n"
        "                'outcome': outcome,\n"
        "                'last_step': state.get('step'),\n"
        "                'events': state['events'],\n"
        "            }\n"
        "            if exc is not None:\n"
        "                _summary['exception_type'] = type(exc).__name__\n"
        "                _summary['exception_repr'] = repr(exc)[:2000]\n"
        "            try:\n"
        "                _compat_atomic_write_text(\n"
        "                    _verifier_path,\n"
        "                    _hardening_json.dumps(_summary, sort_keys=True) + '\\n',\n"
        "                )\n"
        "            except Exception:\n"
        "                pass\n"
        "        async def _resolve_hardening_docker_identity(inner):\n"
        "            from benchflow.sandbox import docker as _docker\n"
        "            _inner_type = type(inner)\n"
        "            if (\n"
        "                _inner_type.__module__ != 'benchflow.sandbox.docker'\n"
        "                or _inner_type.__name__ != 'DockerSandbox'\n"
        "            ):\n"
        "                raise RuntimeError(\n"
        "                    'hardening compat requires exact DockerSandbox'\n"
        "                )\n"
        "            _session_id = getattr(inner, 'session_id', None)\n"
        "            if not isinstance(_session_id, str) or not _session_id.strip():\n"
        "                raise RuntimeError(\n"
        "                    'hardening compat requires a non-empty session id'\n"
        "                )\n"
        "            _project = _docker._sanitize_docker_compose_project_name(\n"
        "                _session_id\n"
        "            )\n"
        "            if not isinstance(_project, str) or not _project:\n"
        "                raise RuntimeError(\n"
        "                    'hardening compat produced an invalid project identity'\n"
        "                )\n"
        "            _env_vars = getattr(inner, '_env_vars', None)\n"
        "            _expected_targets = {\n"
        "                'env_agent_logs_path': '/logs/agent',\n"
        "                'env_verifier_logs_path': '/logs/verifier',\n"
        "                'env_artifacts_path': '/logs/artifacts',\n"
        "            }\n"
        "            if _env_vars is None or any(\n"
        "                str(getattr(_env_vars, _name, '')).rstrip('/') != _target\n"
        "                for _name, _target in _expected_targets.items()\n"
        "            ):\n"
        "                raise RuntimeError(\n"
        "                    'hardening compat Docker log targets do not match'\n"
        "                )\n"
        "            _compose_env_factory = getattr(\n"
        "                inner, '_docker_compose_env', None\n"
        "            )\n"
        "            if not callable(_compose_env_factory):\n"
        "                raise RuntimeError(\n"
        "                    'hardening compat Docker environment unavailable'\n"
        "                )\n"
        "            try:\n"
        "                _compose_env = dict(_compose_env_factory())\n"
        "            except Exception as _compose_env_exc:\n"
        "                raise RuntimeError(\n"
        "                    'hardening compat Docker environment snapshot failed'\n"
        "                ) from _compose_env_exc\n"
        "            if any(\n"
        "                not isinstance(_key, str) or not isinstance(_value, str)\n"
        "                for _key, _value in _compose_env.items()\n"
        "            ):\n"
        "                raise RuntimeError(\n"
        "                    'hardening compat Docker environment was invalid'\n"
        "                )\n"
        "            def _run_sync():\n"
        "                _ps = _hardening_subprocess.run(\n"
        "                    [\n"
        "                        'docker', 'ps', '-q', '--no-trunc',\n"
        "                        '--filter', 'label=benchflow.owned=true',\n"
        "                        '--filter',\n"
        "                        f'label=com.docker.compose.project={_project}',\n"
        "                        '--filter',\n"
        "                        'label=com.docker.compose.service=main',\n"
        "                        '--filter',\n"
        "                        'label=com.docker.compose.container-number=1',\n"
        "                        '--filter',\n"
        "                        'label=com.docker.compose.oneoff=False',\n"
        "                        '--filter', 'status=running',\n"
        "                    ],\n"
        "                    env=_compose_env,\n"
        "                    stdin=_hardening_subprocess.DEVNULL,\n"
        "                    stdout=_hardening_subprocess.PIPE,\n"
        "                    stderr=_hardening_subprocess.STDOUT,\n"
        "                    text=True, timeout=10,\n"
        "                    check=False,\n"
        "                )\n"
        "                if int(_ps.returncode or 0) != 0:\n"
        "                    raise RuntimeError(\n"
        "                        'hardening compat container lookup failed'\n"
        "                    )\n"
        "                _container_ids = [\n"
        "                    _line.strip()\n"
        "                    for _line in (_ps.stdout or '').splitlines()\n"
        "                    if _line.strip()\n"
        "                ]\n"
        "                _valid_container_id = (\n"
        "                    len(_container_ids) == 1\n"
        "                    and len(_container_ids[0]) == 64\n"
        "                    and all(\n"
        "                        _char in '0123456789abcdef'\n"
        "                        for _char in _container_ids[0]\n"
        "                    )\n"
        "                )\n"
        "                if not _valid_container_id:\n"
        "                    raise RuntimeError(\n"
        "                        'hardening compat expected exactly one valid '\n"
        "                        'running main container'\n"
        "                    )\n"
        "                _container_id = _container_ids[0]\n"
        "                _inspect = _hardening_subprocess.run(\n"
        "                    [\n"
        "                        'docker', 'inspect', '--type', 'container',\n"
        "                        '--format',\n"
        "                        '{{json .Config.Labels}}|{{json .Config.User}}',\n"
        "                        _container_id,\n"
        "                    ],\n"
        "                    env=_compose_env,\n"
        "                    stdin=_hardening_subprocess.DEVNULL,\n"
        "                    stdout=_hardening_subprocess.PIPE,\n"
        "                    stderr=_hardening_subprocess.STDOUT,\n"
        "                    text=True, timeout=10,\n"
        "                    check=False,\n"
        "                )\n"
        "                if int(_inspect.returncode or 0) != 0:\n"
        "                    raise RuntimeError(\n"
        "                        'hardening compat container inspection failed'\n"
        "                    )\n"
        "                try:\n"
        "                    _labels_json, _container_user_json = (\n"
        "                        (_inspect.stdout or '').rsplit('|', 1)\n"
        "                    )\n"
        "                    _labels = _hardening_json.loads(_labels_json)\n"
        "                    _container_user = _hardening_json.loads(\n"
        "                        _container_user_json\n"
        "                    )\n"
        "                except Exception as _label_exc:\n"
        "                    raise RuntimeError(\n"
        "                        'hardening compat container labels were invalid'\n"
        "                    ) from _label_exc\n"
        "                _expected_labels = {\n"
        "                    'benchflow.owned': 'true',\n"
        "                    'com.docker.compose.project': _project,\n"
        "                    'com.docker.compose.service': 'main',\n"
        "                    'com.docker.compose.container-number': '1',\n"
        "                    'com.docker.compose.oneoff': 'False',\n"
        "                }\n"
        "                if (\n"
        "                    not isinstance(_labels, dict)\n"
        "                    or not isinstance(_container_user, str)\n"
        "                    or any(\n"
        "                    str(_labels.get(_key, '')) != _value\n"
        "                    for _key, _value in _expected_labels.items()\n"
        "                    )\n"
        "                ):\n"
        "                    raise RuntimeError(\n"
        "                        'hardening compat container identity mismatch'\n"
        "                    )\n"
        "                _identity_sha256 = _hardening_hashlib.sha256(\n"
        "                    (\n"
        "                        _project + '\\0main\\0' + _container_id\n"
        "                        + '\\0' + _container_user\n"
        "                    ).encode('utf-8')\n"
        "                ).hexdigest()\n"
        "                return {\n"
        "                    'container_id': _container_id,\n"
        "                    'docker_env': _compose_env,\n"
        "                    'identity_sha256': _identity_sha256,\n"
        "                    'service': 'main',\n"
        "                    'container_user': _container_user,\n"
        "                }\n"
        "            return await _asyncio.to_thread(_run_sync)\n"
        "        async def _raw_docker_exec_once(\n"
        "            identity, parsed, *, timeout_sec\n"
        "        ):\n"
        "            from benchflow.sandbox._base import ExecResult as _ExecResult\n"
        "            _raw_options, _service, _argv = parsed\n"
        "            if _service != identity.get('service'):\n"
        "                raise RuntimeError(\n"
        "                    'hardening compat raw service identity mismatch'\n"
        "                )\n"
        "            if (\n"
        "                not isinstance(timeout_sec, (int, float))\n"
        "                or isinstance(timeout_sec, bool)\n"
        "                or not 0 < timeout_sec <= 15\n"
        "            ):\n"
        "                raise RuntimeError(\n"
        "                    'raw Docker hardening timeout was out of scope'\n"
        "                )\n"
        "            def _run_sync():\n"
        "                try:\n"
        "                    _raw = _hardening_subprocess.run(\n"
        "                        [\n"
        "                            'docker', 'exec', *_raw_options,\n"
        "                            identity['container_id'], *_argv,\n"
        "                        ],\n"
        "                        env=identity['docker_env'],\n"
        "                        stdin=_hardening_subprocess.DEVNULL,\n"
        "                        stdout=_hardening_subprocess.PIPE,\n"
        "                        stderr=_hardening_subprocess.STDOUT,\n"
        "                        text=True, timeout=timeout_sec, check=False,\n"
        "                    )\n"
        "                except _hardening_subprocess.TimeoutExpired as _timeout_exc:\n"
        "                    raise RuntimeError(\n"
        "                        'raw Docker hardening command timed out'\n"
        "                    ) from _timeout_exc\n"
        "                _return_code = int(_raw.returncode or 0)\n"
        "                return _ExecResult(\n"
        "                    stdout=_raw.stdout or None,\n"
        "                    stderr=None,\n"
        "                    return_code=_return_code,\n"
        "                )\n"
        "            return await _asyncio.to_thread(_run_sync)\n"
        "        async def _harden_before_verify_with_scoped_exec_recovery(\n"
        "            env, *args, **kwargs\n"
        "        ):\n"
        "            _inner = getattr(env, 'inner', env)\n"
        "            _inner_type = type(_inner)\n"
        "            if (\n"
        "                _inner_type.__module__ != 'benchflow.sandbox.docker'\n"
        "                or _inner_type.__name__ != 'DockerSandbox'\n"
        "            ):\n"
        "                raise RuntimeError(\n"
        "                    'hardening compat requires exact DockerSandbox'\n"
        "                )\n"
        "            _original_run = getattr(\n"
        "                _inner, '_run_docker_compose_command', None\n"
        "            )\n"
        "            if not callable(_original_run):\n"
        "                raise RuntimeError(\n"
        "                    'hardening compat compose executor unavailable'\n"
        "                )\n"
        "            _had_instance_run = (\n"
        "                '_run_docker_compose_command' in getattr(_inner, '__dict__', {})\n"
        "            )\n"
        "            _instance_run = getattr(_inner, '__dict__', {}).get(\n"
        "                '_run_docker_compose_command'\n"
        "            )\n"
        "            _state = {'ordinal': 0, 'step': 'identity', 'events': []}\n"
        "            def _record_identity_failure(_identity_exc):\n"
        "                _append_hardening_diagnostic(\n"
        "                    _state, _inner,\n"
        "                    {\n"
        "                        'event': 'raw_docker_identity_failed',\n"
        "                        'step': 'identity',\n"
        "                        'exception_type': type(_identity_exc).__name__,\n"
        "                        'exception_repr': repr(_identity_exc)[:2000],\n"
        "                    },\n"
        "                )\n"
        "            try:\n"
        "                _identity = await _resolve_hardening_docker_identity(_inner)\n"
        "            except _asyncio.CancelledError as _identity_exc:\n"
        "                _record_identity_failure(_identity_exc)\n"
        "                _write_hardening_summary(\n"
        "                    _state, _inner, 'failed', _identity_exc\n"
        "                )\n"
        "                raise\n"
        "            except Exception as _identity_exc:\n"
        "                _record_identity_failure(_identity_exc)\n"
        "                _identity_failure = RuntimeError(\n"
        "                    'hardening compat Docker identity validation failed'\n"
        "                )\n"
        "                _write_hardening_summary(\n"
        "                    _state, _inner, 'failed', _identity_failure\n"
        "                )\n"
        "                raise _identity_failure from _identity_exc\n"
        "            _append_hardening_diagnostic(\n"
        "                _state, _inner,\n"
        "                {\n"
        "                    'event': 'raw_docker_identity_verified',\n"
        "                    'step': 'identity',\n"
        "                    'identity_sha256': _identity['identity_sha256'],\n"
        "                },\n"
        "            )\n"
        "            async def _run_with_scoped_raw_primary(\n"
        "                _self, command, check=True, timeout_sec=None\n"
        "            ):\n"
        "                _state['ordinal'] += 1\n"
        "                _state['step'] = _hardening_exec_step(\n"
        "                    command, _state['ordinal']\n"
        "                )\n"
        "                _command_hash = _hardening_hashlib.sha256(\n"
        "                    _hardening_json.dumps(\n"
        "                        [str(_part) for _part in command],\n"
        "                        separators=(',', ':'),\n"
        "                    ).encode('utf-8')\n"
        "                ).hexdigest()\n"
        "                _parsed = _parse_hardening_compose_exec(command)\n"
        "                _raw_primary = (\n"
        "                    _parsed is not None\n"
        "                    and check is False\n"
        "                    and isinstance(timeout_sec, (int, float))\n"
        "                    and not isinstance(timeout_sec, bool)\n"
        "                    and 0 < timeout_sec <= 15\n"
        "                )\n"
        "                if not _raw_primary:\n"
        "                    try:\n"
        "                        return await _original_run(\n"
        "                            command, check=check, timeout_sec=timeout_sec\n"
        "                        )\n"
        "                    except BaseException as _compose_exc:\n"
        "                        _append_hardening_diagnostic(\n"
        "                            _state, _inner,\n"
        "                            {\n"
        "                                'event': 'compose_command_failed',\n"
        "                                'step': _state['step'],\n"
        "                                'exec_ordinal': _state['ordinal'],\n"
        "                                'command_sha256': _command_hash,\n"
        "                                'identity_sha256': (\n"
        "                                    _identity['identity_sha256']\n"
        "                                ),\n"
        "                                'exception_type': (\n"
        "                                    type(_compose_exc).__name__\n"
        "                                ),\n"
        "                                'exception_repr': (\n"
        "                                    repr(_compose_exc)[:2000]\n"
        "                                ),\n"
        "                            },\n"
        "                        )\n"
        "                        raise\n"
        "                def _record_raw_failure(_raw_exc):\n"
        "                    _append_hardening_diagnostic(\n"
        "                        _state,\n"
        "                        _inner,\n"
        "                        {\n"
        "                            'event': 'raw_docker_command_failed',\n"
        "                            'step': _state['step'],\n"
        "                            'exec_ordinal': _state['ordinal'],\n"
        "                            'command_sha256': _command_hash,\n"
        "                            'identity_sha256': (\n"
        "                                _identity['identity_sha256']\n"
        "                            ),\n"
        "                            'exception_type': type(_raw_exc).__name__,\n"
        "                            'exception_repr': repr(_raw_exc)[:2000],\n"
        "                        },\n"
        "                    )\n"
        "                try:\n"
        "                    _result = await _raw_docker_exec_once(\n"
        "                        _identity, _parsed, timeout_sec=timeout_sec\n"
        "                    )\n"
        "                except _asyncio.CancelledError as _raw_exc:\n"
        "                    _record_raw_failure(_raw_exc)\n"
        "                    raise\n"
        "                except Exception as _raw_exc:\n"
        "                    _record_raw_failure(_raw_exc)\n"
        "                    raise RuntimeError(\n"
        "                        'proactive raw Docker hardening failed at '\n"
        "                        f'{_state[\"step\"]}: '\n"
        "                        f'{type(_raw_exc).__name__}'\n"
        "                    ) from _raw_exc\n"
        "                _append_hardening_diagnostic(\n"
        "                    _state,\n"
        "                    _inner,\n"
        "                    {\n"
        "                        'event': 'raw_docker_command_succeeded',\n"
        "                        'step': _state['step'],\n"
        "                        'exec_ordinal': _state['ordinal'],\n"
        "                        'command_sha256': _command_hash,\n"
        "                        'identity_sha256': (\n"
        "                            _identity['identity_sha256']\n"
        "                        ),\n"
        "                        'return_code': int(_result.return_code),\n"
        "                    },\n"
        "                )\n"
        "                return _result\n"
        "            _inner._run_docker_compose_command = _MethodType(\n"
        "                _run_with_scoped_raw_primary, _inner\n"
        "            )\n"
        "            _harden_exc = None\n"
        "            try:\n"
        "                _result = await _original_harden_before_verify(\n"
        "                    env, *args, **kwargs\n"
        "                )\n"
        "                return _result\n"
        "            except BaseException as _exc:\n"
        "                _harden_exc = _exc\n"
        "                raise\n"
        "            finally:\n"
        "                if _had_instance_run:\n"
        "                    _inner._run_docker_compose_command = _instance_run\n"
        "                else:\n"
        "                    try:\n"
        "                        delattr(_inner, '_run_docker_compose_command')\n"
        "                    except AttributeError:\n"
        "                        pass\n"
        "                _write_hardening_summary(\n"
        "                    _state, _inner,\n"
        "                    'failed' if _harden_exc is not None else 'completed',\n"
        "                    _harden_exc,\n"
        "                )\n"
        "        _lockdown.harden_before_verify = (\n"
        "            _harden_before_verify_with_scoped_exec_recovery\n"
        "        )\n"
        "        from benchflow.task import verifier_core as _verifier_core\n"
        "        _original_verify_test_script = (\n"
        "            _verifier_core.Verifier._verify_test_script\n"
        "        )\n"
        "        _VERIFIER_TEST_COMMAND = (\n"
        "            '/verifier/test.sh > /logs/verifier/test-stdout.txt 2>&1'\n"
        "        )\n"
        "        def _verifier_exec_diagnostic_paths(inner):\n"
        "            _event_path, _ = _hardening_diagnostic_paths(inner)\n"
        "            if _event_path is None:\n"
        "                return None, None\n"
        "            return (\n"
        "                _event_path.parent / 'verifier_test_exec_compat.jsonl',\n"
        "                _event_path.parent / 'verifier_test_exec_compat.json',\n"
        "            )\n"
        "        def _append_verifier_exec_diagnostic(state, inner, event):\n"
        "            _record = {'schema_version': 1, **event}\n"
        "            state['events'].append(_record)\n"
        "            _event_path, _ = _verifier_exec_diagnostic_paths(inner)\n"
        "            if _event_path is None:\n"
        "                return\n"
        "            try:\n"
        "                _compat_append_bytes(\n"
        "                    _event_path,\n"
        "                    (\n"
        "                        _hardening_json.dumps(_record, sort_keys=True)\n"
        "                        + '\\n'\n"
        "                    ).encode('utf-8'),\n"
        "                )\n"
        "            except Exception:\n"
        "                pass\n"
        "        def _write_verifier_exec_summary(\n"
        "            state, inner, outcome, exc=None\n"
        "        ):\n"
        "            if not state['events'] and exc is None:\n"
        "                return\n"
        "            _, _summary_path = _verifier_exec_diagnostic_paths(inner)\n"
        "            if _summary_path is None:\n"
        "                return\n"
        "            _summary = {\n"
        "                'schema_version': 1,\n"
        "                'outcome': outcome,\n"
        "                'events': state['events'],\n"
        "            }\n"
        "            if exc is not None:\n"
        "                _summary['exception_type'] = type(exc).__name__\n"
        "            try:\n"
        "                _compat_atomic_write_text(\n"
        "                    _summary_path,\n"
        "                    _hardening_json.dumps(_summary, sort_keys=True)\n"
        "                    + '\\n',\n"
        "                )\n"
        "            except Exception:\n"
        "                pass\n"
        "        def _verifier_timeout_matches(timeout_sec, expected_timeout):\n"
        "            return (\n"
        "                isinstance(timeout_sec, (int, float))\n"
        "                and not isinstance(timeout_sec, bool)\n"
        "                and isinstance(expected_timeout, (int, float))\n"
        "                and not isinstance(expected_timeout, bool)\n"
        "                and 15 < timeout_sec <= 1800\n"
        "                and float(timeout_sec) == float(expected_timeout)\n"
        "            )\n"
        "        def _validate_verifier_test_scope(verifier, sandbox, inner):\n"
        "            _verifier_type = type(verifier)\n"
        "            if (\n"
        "                _verifier_type.__module__ != 'benchflow.task.verifier_core'\n"
        "                or _verifier_type.__name__ != 'Verifier'\n"
        "            ):\n"
        "                raise RuntimeError(\n"
        "                    'verifier exec compat requires exact Verifier'\n"
        "                )\n"
        "            if sandbox is not inner:\n"
        "                raise RuntimeError(\n"
        "                    'verifier exec compat requires direct sandbox'\n"
        "                )\n"
        "            _inner_type = type(inner)\n"
        "            if (\n"
        "                _inner_type.__module__ != 'benchflow.sandbox.docker'\n"
        "                or _inner_type.__name__ != 'DockerSandbox'\n"
        "            ):\n"
        "                raise RuntimeError(\n"
        "                    'verifier exec compat requires exact DockerSandbox'\n"
        "                )\n"
        "            _env_vars = getattr(inner, '_env_vars', None)\n"
        "            _rollout_paths = getattr(inner, 'rollout_paths', None)\n"
        "            try:\n"
        "                _host_verifier = _HardeningPath(\n"
        "                    _env_vars.host_verifier_logs_path\n"
        "                )\n"
        "                _rollout_verifier = _HardeningPath(\n"
        "                    _rollout_paths.verifier_dir\n"
        "                )\n"
        "            except (AttributeError, OSError, TypeError, ValueError) as _exc:\n"
        "                raise RuntimeError(\n"
        "                    'verifier exec compat host bind unavailable'\n"
        "                ) from _exc\n"
        "            _normalized_host = _os.path.normcase(\n"
        "                _os.path.abspath(_os.fspath(_host_verifier))\n"
        "            )\n"
        "            _normalized_rollout = _os.path.normcase(\n"
        "                _os.path.abspath(_os.fspath(_rollout_verifier))\n"
        "            )\n"
        "            if (\n"
        "                not _host_verifier.is_absolute()\n"
        "                or not _rollout_verifier.is_absolute()\n"
        "                or _host_verifier.name != 'verifier'\n"
        "                or _rollout_verifier.name != 'verifier'\n"
        "                or _normalized_host != _normalized_rollout\n"
        "            ):\n"
        "                raise RuntimeError(\n"
        "                    'verifier exec compat host bind mismatch'\n"
        "                )\n"
        "            _verifier_config = getattr(\n"
        "                getattr(getattr(verifier, '_task', None), 'config', None),\n"
        "                'verifier', None,\n"
        "            )\n"
        "            _expected_timeout = getattr(\n"
        "                _verifier_config, 'timeout_sec', None\n"
        "            )\n"
        "            _expected_env = getattr(_verifier_config, 'env', None)\n"
        "            if (\n"
        "                getattr(_verifier_config, 'type', None)\n"
        "                != 'test-script'\n"
        "                or getattr(_verifier_config, 'service', None)\n"
        "                != 'main'\n"
        "            ):\n"
        "                raise RuntimeError(\n"
        "                    'verifier exec compat expected main test-script'\n"
        "                )\n"
        "            if (\n"
        "                getattr(_verifier_config, 'user', None) is not None\n"
        "                or getattr(inner, 'default_user', None) is not None\n"
        "            ):\n"
        "                raise RuntimeError(\n"
        "                    'verifier exec compat expected implicit Docker user'\n"
        "                )\n"
        "            if (\n"
        "                not isinstance(_expected_env, dict)\n"
        "                or not _expected_env\n"
        "                or any(\n"
        "                    not isinstance(_key, str)\n"
        "                    or not isinstance(_value, str)\n"
        "                    for _key, _value in _expected_env.items()\n"
        "                )\n"
        "            ):\n"
        "                raise RuntimeError(\n"
        "                    'verifier exec compat expected trusted verifier env'\n"
        "                )\n"
        "            if not _verifier_timeout_matches(\n"
        "                _expected_timeout, _expected_timeout\n"
        "            ):\n"
        "                raise RuntimeError(\n"
        "                    'verifier exec compat timeout was out of scope'\n"
        "                )\n"
        "            return (\n"
        "                _expected_timeout,\n"
        "                dict(_expected_env) if _expected_env else None,\n"
        "            )\n"
        "        async def _raw_docker_verifier_test_once(\n"
        "            identity, wrapped_command, *, timeout_sec\n"
        "        ):\n"
        "            from benchflow.sandbox._base import ExecResult as _ExecResult\n"
        "            if identity.get('service') != 'main':\n"
        "                raise RuntimeError(\n"
        "                    'verifier exec compat raw service identity mismatch'\n"
        "                )\n"
        "            if not isinstance(wrapped_command, str) or not wrapped_command:\n"
        "                raise RuntimeError(\n"
        "                    'verifier exec compat wrapped command was invalid'\n"
        "                )\n"
        "            if (\n"
        "                not isinstance(timeout_sec, (int, float))\n"
        "                or isinstance(timeout_sec, bool)\n"
        "                or not 15 < timeout_sec <= 1800\n"
        "            ):\n"
        "                raise RuntimeError(\n"
        "                    'raw Docker verifier timeout was out of scope'\n"
        "                )\n"
        "            try:\n"
        "                _process = await _asyncio.create_subprocess_exec(\n"
        "                    'docker', 'exec', identity['container_id'],\n"
        "                    'sh', '-c', wrapped_command,\n"
        "                    env=identity['docker_env'],\n"
        "                    stdin=_asyncio.subprocess.DEVNULL,\n"
        "                    stdout=_asyncio.subprocess.PIPE,\n"
        "                    stderr=_asyncio.subprocess.STDOUT,\n"
        "                )\n"
        "            except OSError as _os_exc:\n"
        "                raise RuntimeError(\n"
        "                    'raw Docker verifier test command could not start'\n"
        "                ) from _os_exc\n"
        "            async def _terminate_process():\n"
        "                if _process.returncode is not None:\n"
        "                    return\n"
        "                try:\n"
        "                    _process.terminate()\n"
        "                except ProcessLookupError:\n"
        "                    return\n"
        "                try:\n"
        "                    await _asyncio.wait_for(\n"
        "                        _process.communicate(), timeout=5\n"
        "                    )\n"
        "                except TimeoutError:\n"
        "                    try:\n"
        "                        _process.kill()\n"
        "                    except ProcessLookupError:\n"
        "                        pass\n"
        "                    await _process.communicate()\n"
        "            try:\n"
        "                _stdout_bytes, _stderr_bytes = await _asyncio.wait_for(\n"
        "                    _process.communicate(), timeout=timeout_sec\n"
        "                )\n"
        "            except TimeoutError:\n"
        "                await _terminate_process()\n"
        "                raise RuntimeError(\n"
        "                    f'Command timed out after {timeout_sec} seconds'\n"
        "                ) from None\n"
        "            except _asyncio.CancelledError:\n"
        "                try:\n"
        "                    await _terminate_process()\n"
        "                except Exception:\n"
        "                    pass\n"
        "                raise\n"
        "            _stdout = (\n"
        "                _stdout_bytes.decode(errors='replace')\n"
        "                if _stdout_bytes else None\n"
        "            )\n"
        "            _stderr = (\n"
        "                _stderr_bytes.decode(errors='replace')\n"
        "                if _stderr_bytes else None\n"
        "            )\n"
        "            return _ExecResult(\n"
        "                stdout=_stdout, stderr=_stderr,\n"
        "                return_code=int(_process.returncode or 0),\n"
        "            )\n"
        "        async def _verify_test_script_with_scoped_raw_primary(\n"
        "            self, *args, **kwargs\n"
        "        ):\n"
        "            _sandbox = getattr(self, '_sandbox', None)\n"
        "            _inner = getattr(_sandbox, 'inner', _sandbox)\n"
        "            _state = {'events': []}\n"
        "            def _record_scope_failure(_scope_exc):\n"
        "                _append_verifier_exec_diagnostic(\n"
        "                    _state, _inner,\n"
        "                    {\n"
        "                        'event': 'raw_verifier_identity_failed',\n"
        "                        'exception_type': type(_scope_exc).__name__,\n"
        "                    },\n"
        "                )\n"
        "            try:\n"
        "                _expected_timeout, _expected_env = (\n"
        "                    _validate_verifier_test_scope(\n"
        "                        self, _sandbox, _inner\n"
        "                    )\n"
        "                )\n"
        "                _original_exec = getattr(_inner, 'exec', None)\n"
        "                if not callable(_original_exec):\n"
        "                    raise RuntimeError(\n"
        "                        'verifier exec compat sandbox executor unavailable'\n"
        "                    )\n"
        "                _wrap_with_env = getattr(\n"
        "                    _inner, '_wrap_command_with_env_file', None\n"
        "                )\n"
        "                _resolve_user = getattr(_inner, '_resolve_user', None)\n"
        "                _merge_env = getattr(_inner, '_merge_env', None)\n"
        "                if not callable(_wrap_with_env):\n"
        "                    raise RuntimeError(\n"
        "                        'verifier exec compat env wrapper unavailable'\n"
        "                    )\n"
        "                if not callable(_resolve_user) or not callable(_merge_env):\n"
        "                    raise RuntimeError(\n"
        "                        'verifier exec compat semantic helpers unavailable'\n"
        "                    )\n"
        "                _identity = await _resolve_hardening_docker_identity(_inner)\n"
        "                if _identity.get('container_user') not in ('', 'root', '0'):\n"
        "                    raise RuntimeError(\n"
        "                        'verifier exec compat container default user '\n"
        "                        'was not root'\n"
        "                    )\n"
        "            except _asyncio.CancelledError as _scope_exc:\n"
        "                _record_scope_failure(_scope_exc)\n"
        "                _write_verifier_exec_summary(\n"
        "                    _state, _inner, 'failed', _scope_exc\n"
        "                )\n"
        "                raise\n"
        "            except Exception as _scope_exc:\n"
        "                _record_scope_failure(_scope_exc)\n"
        "                _scope_failure = RuntimeError(\n"
        "                    'verifier exec compat scope validation failed'\n"
        "                )\n"
        "                _write_verifier_exec_summary(\n"
        "                    _state, _inner, 'failed', _scope_failure\n"
        "                )\n"
        "                raise _scope_failure from _scope_exc\n"
        "            _append_verifier_exec_diagnostic(\n"
        "                _state, _inner,\n"
        "                {\n"
        "                    'event': 'raw_verifier_identity_verified',\n"
        "                    'identity_sha256': _identity['identity_sha256'],\n"
        "                },\n"
        "            )\n"
        "            _had_instance_exec = (\n"
        "                'exec' in getattr(_inner, '__dict__', {})\n"
        "            )\n"
        "            _instance_exec = getattr(_inner, '__dict__', {}).get('exec')\n"
        "            async def _exec_with_verifier_raw_primary(\n"
        "                _self, command, cwd=None, env=None, timeout_sec=None,\n"
        "                user=None, service='main'\n"
        "            ):\n"
        "                _hash_input = (\n"
        "                    command\n"
        "                    if isinstance(command, str)\n"
        "                    else f'<{type(command).__name__}>'\n"
        "                )\n"
        "                _command_hash = _hardening_hashlib.sha256(\n"
        "                    _hash_input.encode('utf-8')\n"
        "                ).hexdigest()\n"
        "                _current_verifier_config = getattr(\n"
        "                    getattr(\n"
        "                        getattr(self, '_task', None), 'config', None\n"
        "                    ),\n"
        "                    'verifier', None,\n"
        "                )\n"
        "                _candidate = _verifier_timeout_matches(\n"
        "                    timeout_sec, _expected_timeout\n"
        "                )\n"
        "                if not _candidate:\n"
        "                    try:\n"
        "                        return await _original_exec(\n"
        "                            command, cwd=cwd, env=env,\n"
        "                            timeout_sec=timeout_sec, user=user,\n"
        "                            service=service,\n"
        "                        )\n"
        "                    except BaseException as _semantic_exc:\n"
        "                        _append_verifier_exec_diagnostic(\n"
        "                            _state, _inner,\n"
        "                            {\n"
        "                                'event': 'verifier_semantic_exec_failed',\n"
        "                                'command_sha256': _command_hash,\n"
        "                                'identity_sha256': (\n"
        "                                    _identity['identity_sha256']\n"
        "                                ),\n"
        "                                'exception_type': (\n"
        "                                    type(_semantic_exc).__name__\n"
        "                                ),\n"
        "                            },\n"
        "                        )\n"
        "                        raise\n"
        "                _current_config_env = getattr(\n"
        "                    _current_verifier_config, 'env', None\n"
        "                )\n"
        "                _current_expected_env = (\n"
        "                    dict(_current_config_env)\n"
        "                    if isinstance(_current_config_env, dict)\n"
        "                    and _current_config_env\n"
        "                    else None\n"
        "                )\n"
        "                _candidate_valid = (\n"
        "                    isinstance(command, str)\n"
        "                    and command == _VERIFIER_TEST_COMMAND\n"
        "                    and cwd is None\n"
        "                    and user is None\n"
        "                    and service == 'main'\n"
        "                    and isinstance(env, dict)\n"
        "                    and bool(env)\n"
        "                    and all(\n"
        "                        isinstance(_key, str)\n"
        "                        and isinstance(_value, str)\n"
        "                        for _key, _value in env.items()\n"
        "                    )\n"
        "                    and getattr(\n"
        "                        _current_verifier_config, 'type', None\n"
        "                    ) == 'test-script'\n"
        "                    and getattr(\n"
        "                        _current_verifier_config, 'service', None\n"
        "                    ) == 'main'\n"
        "                    and getattr(\n"
        "                        _current_verifier_config, 'user', None\n"
        "                    ) is None\n"
        "                    and getattr(_inner, 'default_user', None) is None\n"
        "                    and env == _expected_env\n"
        "                    and _current_expected_env == _expected_env\n"
        "                )\n"
        "                if not _candidate_valid:\n"
        "                    _candidate_exc = RuntimeError(\n"
        "                        'verifier exec compat rejected long command'\n"
        "                    )\n"
        "                    _append_verifier_exec_diagnostic(\n"
        "                        _state, _inner,\n"
        "                        {\n"
        "                            'event': 'verifier_raw_candidate_rejected',\n"
        "                            'command_sha256': _command_hash,\n"
        "                            'identity_sha256': (\n"
        "                                _identity['identity_sha256']\n"
        "                            ),\n"
        "                            'exception_type': 'RuntimeError',\n"
        "                        },\n"
        "                    )\n"
        "                    raise _candidate_exc\n"
        "                def _record_raw_failure(_raw_exc):\n"
        "                    _append_verifier_exec_diagnostic(\n"
        "                        _state, _inner,\n"
        "                        {\n"
        "                            'event': 'raw_verifier_test_failed',\n"
        "                            'command_sha256': _command_hash,\n"
        "                            'identity_sha256': (\n"
        "                                _identity['identity_sha256']\n"
        "                            ),\n"
        "                            'exception_type': type(_raw_exc).__name__,\n"
        "                        },\n"
        "                    )\n"
        "                try:\n"
        "                    _resolved_user = _resolve_user(user)\n"
        "                    if _resolved_user is not None:\n"
        "                        raise RuntimeError(\n"
        "                            'verifier exec compat resolved non-root-default '\n"
        "                            'user unexpectedly'\n"
        "                        )\n"
        "                    _merged_env = _merge_env(env)\n"
        "                    if (\n"
        "                        _merged_env is not None\n"
        "                        and (\n"
        "                            not isinstance(_merged_env, dict)\n"
        "                            or any(\n"
        "                                not isinstance(_key, str)\n"
        "                                or not isinstance(_value, str)\n"
        "                                for _key, _value in _merged_env.items()\n"
        "                            )\n"
        "                        )\n"
        "                    ):\n"
        "                        raise RuntimeError(\n"
        "                            'verifier exec compat merged env was invalid'\n"
        "                        )\n"
        "                    _wrapped_command = (\n"
        "                        _wrap_with_env(_merged_env, command)\n"
        "                        if _merged_env else command\n"
        "                    )\n"
        "                    _result = await _raw_docker_verifier_test_once(\n"
        "                        _identity, _wrapped_command,\n"
        "                        timeout_sec=timeout_sec\n"
        "                    )\n"
        "                except _asyncio.CancelledError as _raw_exc:\n"
        "                    _record_raw_failure(_raw_exc)\n"
        "                    raise\n"
        "                except RuntimeError as _raw_exc:\n"
        "                    _record_raw_failure(_raw_exc)\n"
        "                    raise\n"
        "                except Exception as _raw_exc:\n"
        "                    _record_raw_failure(_raw_exc)\n"
        "                    raise RuntimeError(\n"
        "                        'proactive raw Docker verifier test failed'\n"
        "                    ) from _raw_exc\n"
        "                _append_verifier_exec_diagnostic(\n"
        "                    _state, _inner,\n"
        "                    {\n"
        "                        'event': 'raw_verifier_test_succeeded',\n"
        "                        'command_sha256': _command_hash,\n"
        "                        'identity_sha256': (\n"
        "                            _identity['identity_sha256']\n"
        "                        ),\n"
        "                        'return_code': int(_result.return_code),\n"
        "                    },\n"
        "                )\n"
        "                return _result\n"
        "            _inner.exec = _MethodType(\n"
        "                _exec_with_verifier_raw_primary, _inner\n"
        "            )\n"
        "            _verify_exc = None\n"
        "            try:\n"
        "                return await _original_verify_test_script(\n"
        "                    self, *args, **kwargs\n"
        "                )\n"
        "            except BaseException as _exc:\n"
        "                _verify_exc = _exc\n"
        "                raise\n"
        "            finally:\n"
        "                if _had_instance_exec:\n"
        "                    _inner.exec = _instance_exec\n"
        "                else:\n"
        "                    try:\n"
        "                        delattr(_inner, 'exec')\n"
        "                    except AttributeError:\n"
        "                        pass\n"
        "                _write_verifier_exec_summary(\n"
        "                    _state, _inner,\n"
        "                    'failed' if _verify_exc is not None else 'completed',\n"
        "                    _verify_exc,\n"
        "                )\n"
        "        _verifier_core.Verifier._verify_test_script = (\n"
        "            _verify_test_script_with_scoped_raw_primary\n"
        "        )\n"
        "        async def _kill_sandbox_user_procs_once_as_root(env, sandbox_user):\n"
        "            _user = _shlex.quote(str(sandbox_user))\n"
        "            _command = (\n"
        "                f'pkill -u {_user} 2>/dev/null || true; sleep 1; '\n"
        "                f'pkill -9 -u {_user} 2>/dev/null || true; sleep 1; '\n"
        "                f'! pgrep -u {_user} > /dev/null 2>&1'\n"
        "            )\n"
        "            _result = await env.exec(\n"
        "                _command, user='root', timeout_sec=10\n"
        "            )\n"
        "            _rc = getattr(\n"
        "                _result, 'return_code', getattr(_result, 'returncode', 0)\n"
        "            )\n"
        "            if int(_rc or 0) != 0:\n"
        "                raise RuntimeError(\n"
        "                    'Verifier hardening failed: sandbox-user processes survived'\n"
        "                )\n"
        "        _lockdown._kill_sandbox_user_procs = (\n"
        "            _kill_sandbox_user_procs_once_as_root\n"
        "        )\n"
        "    if _os.environ.get('SKILLSBENCH_BENCHFLOW_TRAJECTORY_PUBLISH_COMPAT') == '1':\n"
        "        import asyncio as _publish_asyncio\n"
        "        import hashlib as _hashlib\n"
        "        import json as _json\n"
        "        import subprocess as _publish_subprocess\n"
        "        from pathlib import Path as _Path\n"
        "        import benchflow.rollout as _rollout\n"
        "        from benchflow.rollout import _setup as _rollout_setup\n"
        "        from benchflow.sandbox import docker as _publish_docker\n"
        "        from benchflow.trajectories.types import (\n"
        "            redact_acp_trajectory_jsonl as _redact_acp_trajectory_jsonl,\n"
        "        )\n"
        "        _original_publish_trajectory = (\n"
        "            _rollout._publish_trajectory_for_verifier\n"
        "        )\n"
        "        def _publishable_terminal_kind(trajectory):\n"
        "            if not trajectory:\n"
        "                return None\n"
        "            _last = trajectory[-1]\n"
        "            if not isinstance(_last, dict):\n"
        "                return None\n"
        "            if (\n"
        "                _last.get('type') == 'agent_message'\n"
        "                and isinstance(_last.get('text'), str)\n"
        "                and bool(_last['text'].strip())\n"
        "            ):\n"
        "                return 'agent_message'\n"
        "            _timeout_sec = _last.get('timeout_sec')\n"
        "            _pending = _last.get('pending_tool_call_ids')\n"
        "            _terminal_complete = _last.get(\n"
        "                'terminal_trajectory_complete'\n"
        "            )\n"
        "            if (\n"
        "                _last.get('type') == 'agent_timeout'\n"
        "                and _last.get('reason') == 'wall_clock_timeout'\n"
        "                and isinstance(_pending, list)\n"
        "                and all(isinstance(_item, str) for _item in _pending)\n"
        "                and isinstance(_terminal_complete, bool)\n"
        "                and (_terminal_complete == (not bool(_pending)))\n"
        "                and isinstance(_timeout_sec, (int, float))\n"
        "                and not isinstance(_timeout_sec, bool)\n"
        "                and _timeout_sec > 0\n"
        "            ):\n"
        "                return (\n"
        "                    'agent_timeout'\n"
        "                    if _terminal_complete\n"
        "                    else 'partial_agent_timeout'\n"
        "                )\n"
        "            return None\n"
        "        async def _raw_docker_payload_probe(inner, size, sha256):\n"
        "            _project = (\n"
        "                _publish_docker._sanitize_docker_compose_project_name(\n"
        "                    inner.session_id\n"
        "                )\n"
        "            )\n"
        "            _probe_command = (\n"
        "                'wc -c < /logs/agent/acp_trajectory.jsonl && '\n"
        "                'sha256sum /logs/agent/acp_trajectory.jsonl'\n"
        "            )\n"
        "            def _run_sync():\n"
        "                _ps = _publish_subprocess.run(\n"
        "                    [\n"
        "                        'docker', 'ps', '-q',\n"
        "                        '--filter',\n"
        "                        f'label=com.docker.compose.project={_project}',\n"
        "                        '--filter',\n"
        "                        'label=com.docker.compose.service=main',\n"
        "                        '--filter', 'status=running',\n"
        "                    ],\n"
        "                    stdin=_publish_subprocess.DEVNULL,\n"
        "                    capture_output=True, text=True, timeout=10, check=False,\n"
        "                )\n"
        "                if _ps.returncode != 0:\n"
        "                    return None\n"
        "                _container_ids = [\n"
        "                    _line.strip()\n"
        "                    for _line in (_ps.stdout or '').splitlines()\n"
        "                    if _line.strip()\n"
        "                ]\n"
        "                _valid_container_id = (\n"
        "                    len(_container_ids) == 1\n"
        "                    and 12 <= len(_container_ids[0]) <= 64\n"
        "                    and all(\n"
        "                        _char in '0123456789abcdefABCDEF'\n"
        "                        for _char in _container_ids[0]\n"
        "                    )\n"
        "                )\n"
        "                if not _valid_container_id:\n"
        "                    return None\n"
        "                _raw = _publish_subprocess.run(\n"
        "                    [\n"
        "                        'docker', 'exec', _container_ids[0],\n"
        "                        'sh', '-c', _probe_command,\n"
        "                    ],\n"
        "                    stdin=_publish_subprocess.DEVNULL,\n"
        "                    capture_output=True, text=True, timeout=10, check=False,\n"
        "                )\n"
        "                return _raw\n"
        "            _raw = await _publish_asyncio.to_thread(_run_sync)\n"
        "            if _raw is None or int(_raw.returncode or 0) != 0:\n"
        "                return None\n"
        "            _parts = str(_raw.stdout or '').split()\n"
        "            if (\n"
        "                len(_parts) < 2\n"
        "                or _parts[0] != str(size)\n"
        "                or _parts[1].lower() != sha256\n"
        "            ):\n"
        "                return None\n"
        "            return {\n"
        "                'probe_transport': 'raw_docker_exec',\n"
        "                'probe_return_code': int(_raw.returncode or 0),\n"
        "            }\n"
        "        def _exact_docker_bind_target(env, agent_dir):\n"
        "            _inner = getattr(env, 'inner', env)\n"
        "            _inner_type = type(_inner)\n"
        "            if (\n"
        "                _inner_type.__module__ != 'benchflow.sandbox.docker'\n"
        "                or _inner_type.__name__ != 'DockerSandbox'\n"
        "            ):\n"
        "                return None\n"
        "            _env_vars = getattr(_inner, '_env_vars', None)\n"
        "            if _env_vars is None:\n"
        "                return None\n"
        "            try:\n"
        "                _host_agent_dir = _Path(\n"
        "                    _env_vars.host_agent_logs_path\n"
        "                )\n"
        "                _expected_agent_dir = _Path(agent_dir)\n"
        "            except (AttributeError, OSError, TypeError, ValueError):\n"
        "                return None\n"
        "            if (\n"
        "                not _host_agent_dir.is_absolute()\n"
        "                or not _expected_agent_dir.is_absolute()\n"
        "                or _host_agent_dir != _expected_agent_dir\n"
        "                or _host_agent_dir.name != 'agent'\n"
        "                or str(getattr(_env_vars, 'env_agent_logs_path', '')).rstrip('/')\n"
        "                != '/logs/agent'\n"
        "            ):\n"
        "                return None\n"
        "            return _inner, _expected_agent_dir\n"
        "        async def _write_and_verify_docker_bind_payload(\n"
        "            inner, trajectory, _expected_agent_dir\n"
        "        ):\n"
        "            _expected = (\n"
        "                _redact_acp_trajectory_jsonl(trajectory) + '\\n'\n"
        "            ).encode('utf-8')\n"
        "            _host_path = _expected_agent_dir / 'acp_trajectory.jsonl'\n"
        "            try:\n"
        "                _compat_atomic_write_bytes(_host_path, _expected)\n"
        "            except Exception as _write_exc:\n"
        "                raise RuntimeError(\n"
        "                    'proactive Docker-bind trajectory host write failed: '\n"
        "                    f'{type(_write_exc).__name__}: {_write_exc}'\n"
        "                ) from _write_exc\n"
        "            _size = len(_expected)\n"
        "            _sha256 = _hashlib.sha256(_expected).hexdigest()\n"
        "            try:\n"
        "                _actual_size, _actual_sha256 = (\n"
        "                    _compat_regular_file_digest(_host_path)\n"
        "                )\n"
        "            except Exception as _read_exc:\n"
        "                raise RuntimeError(\n"
        "                    'proactive Docker-bind trajectory host readback failed'\n"
        "                ) from _read_exc\n"
        "            if (\n"
        "                _actual_size != _size\n"
        "                or _actual_sha256 != _sha256\n"
        "            ):\n"
        "                raise RuntimeError(\n"
        "                    'proactive Docker-bind trajectory host bytes mismatch'\n"
        "                )\n"
        "            try:\n"
        "                _raw_verified = await _raw_docker_payload_probe(\n"
        "                    inner, _size, _sha256\n"
        "                )\n"
        "            except Exception as _probe_exc:\n"
        "                raise RuntimeError(\n"
        "                    'proactive Docker-bind trajectory raw probe failed: '\n"
        "                    f'{type(_probe_exc).__name__}: {_probe_exc}'\n"
        "                ) from _probe_exc\n"
        "            if _raw_verified is None:\n"
        "                raise RuntimeError(\n"
        "                    'proactive Docker-bind trajectory raw size/SHA256 '\n"
        "                    'verification failed'\n"
        "                )\n"
        "            return {\n"
        "                'size': _size, 'sha256': _sha256, **_raw_verified,\n"
        "            }\n"
        "        async def _publish_trajectory_with_proactive_verified_bind(\n"
        "            env, trajectory, agent_dir\n"
        "        ):\n"
        "            _terminal_kind = _publishable_terminal_kind(trajectory)\n"
        "            if _terminal_kind is None:\n"
        "                return await _original_publish_trajectory(\n"
        "                    env, trajectory, agent_dir\n"
        "                )\n"
        "            _target = _exact_docker_bind_target(env, agent_dir)\n"
        "            if _target is None:\n"
        "                return await _original_publish_trajectory(\n"
        "                    env, trajectory, agent_dir\n"
        "                )\n"
        "            _inner, _expected_agent_dir = _target\n"
        "            _verified = await _write_and_verify_docker_bind_payload(\n"
        "                _inner, trajectory, _expected_agent_dir\n"
        "            )\n"
        "            _diagnostic = _Path(agent_dir) / (\n"
        "                'trajectory_publish_compat.json'\n"
        "            )\n"
        "            try:\n"
        "                _compat_atomic_write_text(\n"
        "                    _diagnostic,\n"
        "                    _json.dumps(\n"
        "                        {\n"
        "                            'schema_version': 2,\n"
        "                            'recovery': (\n"
        "                                'proactive_verified_bind_publish'\n"
        "                            ),\n"
        "                            'terminal_kind': _terminal_kind,\n"
        "                            **_verified,\n"
        "                        },\n"
        "                        sort_keys=True,\n"
        "                    ) + '\\n',\n"
        "                )\n"
        "            except Exception as _diagnostic_exc:\n"
        "                raise RuntimeError(\n"
        "                    'proactive Docker-bind trajectory diagnostic write failed'\n"
        "                ) from _diagnostic_exc\n"
        "            return None\n"
        "        _rollout._publish_trajectory_for_verifier = (\n"
        "            _publish_trajectory_with_proactive_verified_bind\n"
        "        )\n"
        "        _rollout_setup._publish_trajectory_for_verifier = (\n"
        "            _publish_trajectory_with_proactive_verified_bind\n"
        "        )\n"
        "    if _os.environ.get('SKILLSBENCH_BENCHFLOW_FAILURE_FORENSICS') == '1':\n"
        "        import json as _forensics_json\n"
        "        from pathlib import Path as _ForensicsPath\n"
        "        import benchflow.rollout as _forensics_rollout\n"
        "        _original_rollout_cleanup = _forensics_rollout.Rollout.cleanup\n"
        "        def _forensics_terminal_kind(trajectory):\n"
        "            if not isinstance(trajectory, list) or not trajectory:\n"
        "                return None\n"
        "            _last = trajectory[-1]\n"
        "            if not isinstance(_last, dict):\n"
        "                return None\n"
        "            if (\n"
        "                _last.get('type') == 'agent_message'\n"
        "                and isinstance(_last.get('text'), str)\n"
        "                and bool(_last['text'].strip())\n"
        "            ):\n"
        "                return 'agent_message'\n"
        "            _timeout_sec = _last.get('timeout_sec')\n"
        "            _pending = _last.get('pending_tool_call_ids')\n"
        "            _terminal_complete = _last.get(\n"
        "                'terminal_trajectory_complete'\n"
        "            )\n"
        "            if (\n"
        "                _last.get('type') == 'agent_timeout'\n"
        "                and _last.get('reason') == 'wall_clock_timeout'\n"
        "                and isinstance(_pending, list)\n"
        "                and all(isinstance(_item, str) for _item in _pending)\n"
        "                and isinstance(_terminal_complete, bool)\n"
        "                and (_terminal_complete == (not bool(_pending)))\n"
        "                and isinstance(_timeout_sec, (int, float))\n"
        "                and not isinstance(_timeout_sec, bool)\n"
        "                and _timeout_sec > 0\n"
        "            ):\n"
        "                return (\n"
        "                    'agent_timeout'\n"
        "                    if _terminal_complete\n"
        "                    else 'partial_agent_timeout'\n"
        "                )\n"
        "            return None\n"
        "        def _write_forensics_marker(path, payload):\n"
        "            try:\n"
        "                _compat_atomic_write_text(\n"
        "                    path,\n"
        "                    _forensics_json.dumps(payload, sort_keys=True) + '\\n',\n"
        "                )\n"
        "            except Exception:\n"
        "                pass\n"
        "        async def _cleanup_with_failed_verifier_forensics(self):\n"
        "            _marker = None\n"
        "            _marker_payload = None\n"
        "            _env = getattr(self, '_env', None)\n"
        "            _inner = getattr(_env, 'inner', _env)\n"
        "            _inner_type = type(_inner)\n"
        "            _trajectory = getattr(self, '_trajectory', None)\n"
        "            _rewards = getattr(self, '_rewards', None)\n"
        "            _reward = (\n"
        "                _rewards.get('reward')\n"
        "                if isinstance(_rewards, dict)\n"
        "                else None\n"
        "            )\n"
        "            _has_numeric_reward = (\n"
        "                isinstance(_reward, (int, float))\n"
        "                and not isinstance(_reward, bool)\n"
        "            )\n"
        "            _verifier_error = getattr(self, '_verifier_error', None)\n"
        "            _agent_error = getattr(self, '_error', None)\n"
        "            _empty_agent_error = (\n"
        "                _agent_error is None\n"
        "                or (\n"
        "                    isinstance(_agent_error, str)\n"
        "                    and not _agent_error.strip()\n"
        "                )\n"
        "            )\n"
        "            _phase = str(getattr(self, '_phase', '') or '')\n"
        "            _terminal_kind = _forensics_terminal_kind(_trajectory)\n"
        "            _config = getattr(self, '_config', None)\n"
        "            _retain = (\n"
        "                _inner_type.__module__ == 'benchflow.sandbox.docker'\n"
        "                and _inner_type.__name__ == 'DockerSandbox'\n"
        "                and _phase in ('verifying', 'verified')\n"
        "                and not bool(getattr(_config, 'skip_verify', False))\n"
        "                and _terminal_kind is not None\n"
        "                and not _has_numeric_reward\n"
        "                and (\n"
        "                    _verifier_error is not None\n"
        "                    or _empty_agent_error\n"
        "                    or _terminal_kind in (\n"
        "                        'agent_timeout', 'partial_agent_timeout'\n"
        "                    )\n"
        "                )\n"
        "            )\n"
        "            if _retain:\n"
        "                _inner._keep_containers = True\n"
        "                _env_vars = getattr(_inner, '_env_vars', None)\n"
        "                try:\n"
        "                    _marker = _ForensicsPath(\n"
        "                        _env_vars.host_agent_logs_path\n"
        "                    ) / 'retained_container_forensics.json'\n"
        "                except (AttributeError, OSError, TypeError, ValueError):\n"
        "                    _marker = None\n"
        "                try:\n"
        "                    from benchflow.sandbox import docker as _forensics_docker\n"
        "                    _project = (\n"
        "                        _forensics_docker\n"
        "                        ._sanitize_docker_compose_project_name(\n"
        "                            _inner.session_id\n"
        "                        )\n"
        "                    )\n"
        "                except Exception:\n"
        "                    _project = str(getattr(_inner, 'session_id', '') or '')\n"
        "                _marker_payload = {\n"
        "                    'schema_version': 1,\n"
        "                    'policy': 'retain_terminal_unscored_docker_rollout',\n"
        "                    'retention_requested': True,\n"
        "                    'cleanup_completed': False,\n"
        "                    'expected_container_state': 'stopped',\n"
        "                    'compose_project': _project,\n"
        "                    'phase_before_cleanup': _phase,\n"
        "                    'terminal_kind': _terminal_kind,\n"
        "                    'reason': (\n"
        "                        'verifier_infrastructure_failure'\n"
        "                        if _verifier_error is not None\n"
        "                        else 'terminal_unscored_lifecycle_failure'\n"
        "                    ),\n"
        "                    'verifier_error': (\n"
        "                        str(_verifier_error)[:2000]\n"
        "                        if _verifier_error is not None\n"
        "                        else None\n"
        "                    ),\n"
        "                    'agent_error_empty': _empty_agent_error,\n"
        "                }\n"
        "                if _marker is not None:\n"
        "                    _write_forensics_marker(_marker, _marker_payload)\n"
        "            _cleanup_exc = None\n"
        "            try:\n"
        "                return await _original_rollout_cleanup(self)\n"
        "            except Exception as _exc:\n"
        "                _cleanup_exc = _exc\n"
        "                raise\n"
        "            finally:\n"
        "                if _marker is not None and _marker_payload is not None:\n"
        "                    _marker_payload['cleanup_completed'] = (\n"
        "                        _cleanup_exc is None\n"
        "                    )\n"
        "                    if _cleanup_exc is not None:\n"
        "                        _marker_payload['cleanup_exception_type'] = (\n"
        "                            type(_cleanup_exc).__name__\n"
        "                        )\n"
        "                        _marker_payload['cleanup_exception_repr'] = (\n"
        "                            repr(_cleanup_exc)[:2000]\n"
        "                        )\n"
        "                    _write_forensics_marker(_marker, _marker_payload)\n"
        "        _forensics_rollout.Rollout.cleanup = (\n"
        "            _cleanup_with_failed_verifier_forensics\n"
        "        )\n"
        "    if _os.environ.get('SKILLSBENCH_DEEPSEEK_OPENHANDS_COMPAT') == '1':\n"
        "        from benchflow.agents import registry as _registry\n"
        "        _cfg = _registry.AGENTS.get('openhands')\n"
        "        if _cfg is not None:\n"
        "            _old = \"printf '}}'; } > ~/.openhands/agent_settings.json\"\n"
        "            _new = (\"printf ',\\\"model_canonical_name\\\":\"\n"
        "                    \"\\\"deepseek/deepseek-v4-flash\\\"}}'; } > \"\n"
        "                    \"~/.openhands/agent_settings.json\")\n"
        "            if _old not in _cfg.launch_cmd:\n"
        "                raise RuntimeError('OpenHands launch shim anchor drifted')\n"
        "            _cfg.launch_cmd = _cfg.launch_cmd.replace(_old, _new, 1)\n"
        "            _registry.AGENT_LAUNCH['openhands'] = _cfg.launch_cmd\n"
        "        from benchflow.providers import litellm_logging as _logging\n"
        "        _original_source = _logging.callback_module_source\n"
        "        def _deepseek_compat_source():\n"
        "            _source = _original_source()\n"
        "            _anchor = (\"        _gate_opencode_skill_catalog(data)\\n\"\n"
        "                       \"\\n        cleaned = data\\n\")\n"
        "            _replacement = (\"        _gate_opencode_skill_catalog(data)\\n\"\n"
        "                \"\\n        cleaned = data\\n\"\n"
        "                \"        _model = str(data.get('model') or '')\\n\"\n"
        "                \"        if 'deepseek' in _model and isinstance(data.get('messages'), list):\\n\"\n"
        "                \"            _messages = []\\n\"\n"
        "                \"            _seen_tool_ids = set()\\n\"\n"
        "                \"            _turn_last_positions = {}\\n\"\n"
        "                \"            for _message in data['messages']:\\n\"\n"
        "                \"                if isinstance(_message, dict) and _message.get('role') == 'assistant' and _message.get('tool_calls'):\\n\"\n"
        "                \"                    _message = dict(_message)\\n\"\n"
        "                \"                    if _message.get('content') is None:\\n\"\n"
        "                \"                        _message['content'] = ''\\n\"\n"
        "                \"                    _tool_ids = []\\n\"\n"
        "                \"                    for _tool_call in _message['tool_calls']:\\n\"\n"
        "                \"                        if not isinstance(_tool_call, dict) or not isinstance(_tool_call.get('id'), str) or not _tool_call['id']:\\n\"\n"
        "                \"                            raise RuntimeError('DeepSeek reasoning replay tool identity is invalid')\\n\"\n"
        "                \"                        _tool_ids.append(_tool_call['id'])\\n\"\n"
        "                \"                    if len(set(_tool_ids)) != len(_tool_ids):\\n\"\n"
        "                \"                        raise RuntimeError('DeepSeek reasoning replay tool identity is duplicated')\\n\"\n"
        "                \"                    if any(_tool_id in _seen_tool_ids for _tool_id in _tool_ids):\\n\"\n"
        "                \"                        raise RuntimeError('DeepSeek reasoning replay tool identity is repeated in history')\\n\"\n"
        "                \"                    _reasoning_by_tool = getattr(self, '_deepseek_reasoning_by_tool_call', {})\\n\"\n"
        "                \"                    if not isinstance(_reasoning_by_tool, dict):\\n\"\n"
        "                \"                        raise RuntimeError('DeepSeek reasoning replay registry is invalid')\\n\"\n"
        "                \"                    _records = [_reasoning_by_tool.get(_tool_id) for _tool_id in _tool_ids]\\n\"\n"
        "                \"                    if any(not isinstance(_record, tuple) or len(_record) != 3 or not isinstance(_record[0], str) or not _record[0] or not isinstance(_record[1], tuple) or not _record[1] or any(not isinstance(_member, str) or not _member for _member in _record[1]) or (not isinstance(_record[2], str) and _record[2] is not None) for _record in _records):\\n\"\n"
        "                \"                        raise RuntimeError('DeepSeek reasoning replay provenance is missing or malformed')\\n\"\n"
        "                \"                    _turn_ids = {_record[0] for _record in _records}\\n\"\n"
        "                \"                    _turn_tool_ids = {_record[1] for _record in _records}\\n\"\n"
        "                \"                    _turn_reasoning = {_record[2] for _record in _records}\\n\"\n"
        "                \"                    if len(_turn_ids) != 1 or len(_turn_tool_ids) != 1 or len(_turn_reasoning) != 1:\\n\"\n"
        "                \"                        raise RuntimeError('DeepSeek reasoning replay provenance is ambiguous')\\n\"\n"
        "                \"                    _full_tool_ids = next(iter(_turn_tool_ids))\\n\"\n"
        "                \"                    if not isinstance(_full_tool_ids, tuple) or any(_tool_id not in _full_tool_ids for _tool_id in _tool_ids):\\n\"\n"
        "                \"                        raise RuntimeError('DeepSeek reasoning replay turn membership is invalid')\\n\"\n"
        "                \"                    _positions = [_full_tool_ids.index(_tool_id) for _tool_id in _tool_ids]\\n\"\n"
        "                \"                    _turn_record = _records[0]\\n\"\n"
        "                \"                    _last_position = _turn_last_positions.get(_turn_record, -1)\\n\"\n"
        "                \"                    if _positions != sorted(_positions) or any(_position <= _last_position for _position in _positions):\\n\"\n"
        "                \"                        raise RuntimeError('DeepSeek reasoning replay tool order is invalid')\\n\"\n"
        "                \"                    _seen_tool_ids.update(_tool_ids)\\n\"\n"
        "                \"                    _turn_last_positions[_turn_record] = _positions[-1]\\n\"\n"
        "                \"                    _exact_reasoning = next(iter(_turn_reasoning))\\n\"\n"
        "                \"                    if not isinstance(_exact_reasoning, str) and _exact_reasoning is not None:\\n\"\n"
        "                \"                        raise RuntimeError('DeepSeek reasoning replay content is invalid')\\n\"\n"
        "                \"                    if 'reasoning_content' in _message and _message['reasoning_content'] != _exact_reasoning:\\n\"\n"
        "                \"                        raise RuntimeError('DeepSeek reasoning replay content conflicts with provider provenance')\\n\"\n"
        "                \"                    _message['reasoning_content'] = _exact_reasoning\\n\"\n"
        "                \"                _messages.append(_message)\\n\"\n"
        "                \"            cleaned = dict(cleaned)\\n\"\n"
        "                \"            cleaned['messages'] = _messages\\n\"\n"
        "                \"        if 'deepseek' in _model and cleaned.get('max_completion_tokens') is not None:\\n\"\n"
        "                \"            cleaned = dict(cleaned)\\n\"\n"
        "                \"            cleaned['max_tokens'] = cleaned.pop('max_completion_tokens')\\n\")\n"
        "            if _anchor not in _source:\n"
        "                raise RuntimeError('LiteLLM callback shim anchor drifted')\n"
        "            _source = _source.replace(_anchor, _replacement, 1)\n"
        "            _success_anchor = (\"    async def async_log_success_event(\"\n"
        "                               \"self, kwargs, response_obj, start_time, end_time):\\n\")\n"
        "            _success_replacement = (\n"
        "                \"    async def async_post_call_success_hook(self, data, user_api_key_dict, response):\\n\"\n"
        "                \"        if isinstance(data, dict) and 'deepseek' in str(data.get('model') or ''):\\n\"\n"
        "                \"            _response = _jsonable(response)\\n\"\n"
        "                \"            if not isinstance(_response, dict):\\n\"\n"
        "                \"                raise RuntimeError('DeepSeek response provenance is not an object')\\n\"\n"
        "                \"            _choices = _response.get('choices')\\n\"\n"
        "                \"            if not isinstance(_choices, list) or len(_choices) != 1 or not isinstance(_choices[0], dict):\\n\"\n"
        "                \"                raise RuntimeError('DeepSeek response provenance has ambiguous choices')\\n\"\n"
        "                \"            _response_message = _choices[0].get('message')\\n\"\n"
        "                \"            if not isinstance(_response_message, dict):\\n\"\n"
        "                \"                raise RuntimeError('DeepSeek response provenance has no message')\\n\"\n"
        "                \"            _tool_calls = _response_message.get('tool_calls')\\n\"\n"
        "                \"            if _tool_calls:\\n\"\n"
        "                \"                if not isinstance(_tool_calls, list):\\n\"\n"
        "                \"                    raise RuntimeError('DeepSeek response tool calls are invalid')\\n\"\n"
        "                \"                _reasoning = _response_message.get('reasoning_content')\\n\"\n"
        "                \"                _response_id = _response.get('id')\\n\"\n"
        "                \"                if 'reasoning_content' not in _response_message or (not isinstance(_reasoning, str) and _reasoning is not None) or not isinstance(_response_id, str) or not _response_id:\\n\"\n"
        "                \"                    raise RuntimeError('DeepSeek response reasoning provenance is incomplete')\\n\"\n"
        "                \"                _tool_ids = []\\n\"\n"
        "                \"                for _tool_call in _tool_calls:\\n\"\n"
        "                \"                    if not isinstance(_tool_call, dict) or not isinstance(_tool_call.get('id'), str) or not _tool_call['id']:\\n\"\n"
        "                \"                        raise RuntimeError('DeepSeek response tool identity is invalid')\\n\"\n"
        "                \"                    _tool_ids.append(_tool_call['id'])\\n\"\n"
        "                \"                if len(set(_tool_ids)) != len(_tool_ids):\\n\"\n"
        "                \"                    raise RuntimeError('DeepSeek response tool identity is duplicated')\\n\"\n"
        "                \"                _reasoning_by_tool = getattr(self, '_deepseek_reasoning_by_tool_call', {})\\n\"\n"
        "                \"                if not isinstance(_reasoning_by_tool, dict):\\n\"\n"
        "                \"                    raise RuntimeError('DeepSeek reasoning replay registry is invalid')\\n\"\n"
        "                \"                if len(set(_reasoning_by_tool).union(_tool_ids)) > 4096:\\n\"\n"
        "                \"                    raise RuntimeError('DeepSeek reasoning replay registry capacity exceeded')\\n\"\n"
        "                \"                _record = (_response_id, tuple(_tool_ids), _reasoning)\\n\"\n"
        "                \"                for _tool_id in _tool_ids:\\n\"\n"
        "                \"                    _prior = _reasoning_by_tool.get(_tool_id)\\n\"\n"
        "                \"                    if _prior is not None and _prior != _record:\\n\"\n"
        "                \"                        raise RuntimeError('DeepSeek reasoning replay identity conflict')\\n\"\n"
        "                \"                _updated = dict(_reasoning_by_tool)\\n\"\n"
        "                \"                for _tool_id in _tool_ids:\\n\"\n"
        "                \"                    _updated[_tool_id] = _record\\n\"\n"
        "                \"                self._deepseek_reasoning_by_tool_call = _updated\\n\"\n"
        "                \"        return None\\n\\n\"\n"
        "                + _success_anchor\n"
        "            )\n"
        "            if _success_anchor not in _source:\n"
        "                raise RuntimeError('LiteLLM success callback shim anchor drifted')\n"
        "            return _source.replace(\n"
        "                _success_anchor, _success_replacement, 1\n"
        "            )\n"
        "        _logging.callback_module_source = _deepseek_compat_source\n"
        "        _runtime.callback_module_source = _deepseek_compat_source\n",
        encoding="utf-8",
    )
    return shim_root


def _benchflow_subprocess_env(*, sandbox: str, run_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    if sandbox != "docker" or not _is_wsl_docker_desktop_runtime():
        return env
    shim_root = _write_benchflow_wsl_compat_shim(run_root)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(shim_root) if not existing else str(shim_root) + os.pathsep + existing
    )
    env["SKILLSBENCH_APPLY_WSL_DOCKER_COMPAT"] = "1"
    env["SKILLSBENCH_BENCHFLOW_HARDENING_COMPAT"] = "1"
    env["SKILLSBENCH_BENCHFLOW_TRAJECTORY_PUBLISH_COMPAT"] = "1"
    env["SKILLSBENCH_BENCHFLOW_FAILURE_FORENSICS"] = "1"
    env["SKILLSBENCH_DEEPSEEK_OPENHANDS_COMPAT"] = "1"
    return env


class SkillsBenchAdapterError(RuntimeError):
    """Base class for errors raised at the SkillsBench I/O boundary."""


class SkillsBenchInfrastructureError(SkillsBenchAdapterError):
    """A run cannot be interpreted as an agent success or failure.

    Examples include a missing BenchFlow executable, container failure,
    verifier crash, malformed artifacts, and a subprocess timeout. These
    errors must stop the experiment rather than enter SkillGen as negative
    examples.
    """


def _normalise_text(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n")


def parse_task_markdown(task_path: str | Path) -> tuple[dict[str, Any], str]:
    """Return ``(frontmatter, prompt_body)`` from a native task.md package."""

    path = Path(task_path)
    if path.is_dir():
        path = path / "task.md"
    if not path.is_file():
        raise FileNotFoundError(f"SkillsBench task.md not found: {path}")

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"SkillsBench task.md lacks YAML frontmatter: {path}")

    closing = next(
        (idx for idx, line in enumerate(lines[1:], start=1) if line.strip() == "---"),
        None,
    )
    if closing is None:
        raise ValueError(f"SkillsBench task.md has unterminated frontmatter: {path}")

    frontmatter = yaml.safe_load("\n".join(lines[1:closing])) or {}
    if not isinstance(frontmatter, dict):
        raise ValueError(f"SkillsBench task.md frontmatter must be a mapping: {path}")
    prompt = "\n".join(lines[closing + 1 :]).strip()
    if not prompt:
        raise ValueError(f"SkillsBench task prompt is empty: {path}")
    return frontmatter, prompt


def validate_task_package(task_dir: str | Path) -> Path:
    """Validate the minimum native SkillsBench task-package structure."""

    task = Path(task_dir).expanduser().resolve()
    required = (
        task / "task.md",
        task / "environment" / "Dockerfile",
        task / "verifier" / "test.sh",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Invalid SkillsBench task package; missing: " + ", ".join(missing)
        )
    parse_task_markdown(task)
    return task


@lru_cache(maxsize=256)
def _cached_task_digest(resolved_task_dir: str) -> str:
    task = Path(resolved_task_dir)
    digest = hashlib.sha256()
    files = sorted(
        (path for path in task.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(task).as_posix(),
    )
    if not files:
        raise ValueError(f"SkillsBench task directory has no files: {task}")
    for path in files:
        rel = path.relative_to(task).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(rel).to_bytes(8, "big"))
        digest.update(rel)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return "sha256:" + digest.hexdigest()


def task_package_digest(task_dir: str | Path) -> str:
    """Hash every file in a task package using stable relative paths."""

    task = validate_task_package(task_dir)
    return _cached_task_digest(str(task))


def resolve_task_dir(metadata: Mapping[str, Any]) -> Path:
    """Resolve a prepared task in the current runtime environment.

    ``SKILLSBENCH_ROOT`` may relocate a prepared dataset to another checkout
    of the same pinned repository (for example, from Windows to WSL).  The
    package digest remains the authority for whether the relocation is safe.
    """

    task_relpath = metadata.get("skillsbench_task_relpath")
    root_override = os.environ.get("SKILLSBENCH_ROOT")
    if root_override and task_relpath:
        task = Path(root_override).expanduser() / str(task_relpath)
    else:
        raw = metadata.get("skillsbench_task_dir")
        if not raw:
            raise SkillsBenchInfrastructureError(
                "Instance metadata is missing skillsbench_task_dir"
            )
        task = Path(str(raw)).expanduser()

    try:
        task = validate_task_package(task)
    except (FileNotFoundError, ValueError) as exc:
        raise SkillsBenchInfrastructureError(str(exc)) from exc

    expected = metadata.get("skillsbench_task_digest")
    if expected:
        actual = task_package_digest(task)
        if actual != expected:
            raise SkillsBenchInfrastructureError(
                "SkillsBench task package changed after dataset preparation: "
                f"expected {expected}, got {actual} ({task})"
            )
    return task


def _looks_like_windows_absolute_path(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:[\\/]", value)) or value.startswith("\\\\")


def resolve_jobs_root(metadata: Mapping[str, Any]) -> Path:
    """Resolve the BenchFlow artifact root without accepting foreign paths."""

    raw = os.environ.get("SKILLSBENCH_JOBS_ROOT") or metadata.get(
        "skillsbench_jobs_root"
    )
    if not raw:
        raw = "./artifacts/skillsbench/benchflow_jobs"
    raw_text = str(raw)
    if os.name != "nt" and _looks_like_windows_absolute_path(raw_text):
        raise SkillsBenchInfrastructureError(
            "Prepared jobs_root is a Windows path but the run is executing on "
            "Linux/WSL. Re-prepare in WSL or set SKILLSBENCH_JOBS_ROOT to an "
            "absolute POSIX path."
        )
    return Path(raw_text).expanduser().resolve()


def _safe_slug(value: str, *, fallback: str) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
    slug = re.sub(r"-+", "-", slug)
    return (slug or fallback)[:63].rstrip("-")


def materialize_generated_skill(skill: SkillItem, root: str | Path) -> Path:
    """Write one BenchFlow-compatible ``<root>/<name>/SKILL.md`` bundle."""

    target_root = Path(root).resolve()
    name = _safe_slug(
        f"skillgen-{skill.task_name or skill.skill_id[:12]}",
        fallback="skillgen-generated",
    )
    skill_dir = target_root / name
    skill_dir.mkdir(parents=True, exist_ok=False)

    description = " ".join(
        _normalise_text(skill.contextual_abstract).strip().split()
    )[:500]
    if not description:
        description = "Guidance distilled by the released SkillGen method."
    body = _normalise_text(skill.body).strip()
    if not body:
        raise SkillsBenchInfrastructureError("SkillGen produced an empty skill body")

    frontmatter = (
        "---\n"
        f"name: {json.dumps(name, ensure_ascii=False)}\n"
        f"description: {json.dumps(description, ensure_ascii=False)}\n"
        "---\n\n"
    )
    (skill_dir / "SKILL.md").write_text(frontmatter + body + "\n", encoding="utf-8")
    return target_root


def build_benchflow_command(
    *,
    bench_executable: str,
    task_dir: Path,
    jobs_dir: Path,
    agent: str,
    model: str,
    sandbox: str,
    skills_dir: Path | None,
) -> list[str]:
    """Build the shell-free BenchFlow command for exactly one rollout."""

    command = [
        bench_executable,
        "eval",
        "run",
        "--tasks-dir",
        str(task_dir),
        "--agent",
        agent,
        "--model",
        model,
        "--sandbox",
        sandbox,
        "--skill-mode",
        "with-skill" if skills_dir is not None else "no-skill",
        "--concurrency",
        "1",
        # One pre-declared SkillGen rollout slot must correspond to one model
        # attempt. BenchFlow 0.6.7 otherwise retries retryable infrastructure
        # errors twice, obscuring cost and leaving multiple result artifacts.
        "--retry-attempts",
        "0",
        "--jobs-dir",
        str(jobs_dir),
        "--quiet",
    ]
    if skills_dir is not None:
        command.extend(["--skills-dir", str(skills_dir)])
    if agent == "openhands" and model == "deepseek/deepseek-v4-flash":
        command.extend(["--agent-env", "LLM_REASONING_EFFORT=high"])
    return command


def _resolve_executable(name: str) -> str:
    candidate = Path(name).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    resolved = shutil.which(name)
    if resolved:
        return resolved
    raise SkillsBenchInfrastructureError(
        f"BenchFlow executable not found: {name!r}. "
        f"Install the pinned runtime with: uv tool install benchflow=={EXPECTED_BENCHFLOW_VERSION}"
    )


@lru_cache(maxsize=8)
def _validate_benchflow_executable(executable: str) -> None:
    """Fail before a paid rollout if the resolved BenchFlow version drifts."""

    try:
        completed = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SkillsBenchInfrastructureError(
            f"Could not inspect BenchFlow executable {executable!r}: {exc}"
        ) from exc
    version_text = f"{completed.stdout}\n{completed.stderr}"
    if completed.returncode != 0 or EXPECTED_BENCHFLOW_VERSION not in version_text:
        raise SkillsBenchInfrastructureError(
            "BenchFlow version check failed: expected "
            f"{EXPECTED_BENCHFLOW_VERSION}, returncode={completed.returncode}, "
            f"output={version_text.strip()[-500:]!r}"
        )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SkillsBenchInfrastructureError(f"Cannot parse {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SkillsBenchInfrastructureError(f"Expected a JSON object in {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SkillsBenchInfrastructureError(f"Missing BenchFlow trajectory: {path}")
    records: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("record is not a JSON object")
            records.append(record)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SkillsBenchInfrastructureError(
            f"Cannot parse {path} as JSONL: {exc}"
        ) from exc
    return records


def _is_clean_terminal_wall_clock_timeout(
    result: Mapping[str, Any], events: Sequence[Mapping[str, Any]]
) -> bool:
    """Return whether timeout evidence describes a clean terminal timeout."""

    return _scored_wall_clock_timeout_kind(result, events) == "normal_timeout"


def _scored_wall_clock_timeout_kind(
    result: Mapping[str, Any], events: Sequence[Mapping[str, Any]]
) -> str | None:
    """Classify BenchFlow's signed wall-clock timeout evidence.

    BenchFlow deliberately runs the official verifier after every agent prompt
    timeout, including a timeout with an in-flight ACP tool call.  The latter
    is a partial trajectory, but its current workspace still has an official
    benchmark score and must not be retried merely because the agent ran out
    of its task-defined wall-clock budget.
    """

    info = result.get("agent_timeout_info")
    summary = result.get("trajectory_summary")
    if (
        result.get("error_category") != "timeout"
        or not isinstance(info, Mapping)
        or not isinstance(summary, Mapping)
        or not events
    ):
        return None
    last = events[-1]
    if not isinstance(last, Mapping):
        return None
    diagnostic_timeout = info.get("timeout_sec")
    event_timeout = last.get("timeout_sec")
    numeric_timeouts = (
        isinstance(diagnostic_timeout, (int, float))
        and not isinstance(diagnostic_timeout, bool)
        and diagnostic_timeout > 0
        and isinstance(event_timeout, (int, float))
        and not isinstance(event_timeout, bool)
        and event_timeout > 0
    )
    diagnostic_pending = info.get("pending_tool_call_ids")
    event_pending = last.get("pending_tool_call_ids")
    diagnostic_complete = info.get("terminal_trajectory_complete")
    event_complete = last.get("terminal_trajectory_complete")
    if not (
        numeric_timeouts
        and float(diagnostic_timeout) == float(event_timeout)
        and info.get("reason") == "wall_clock_timeout"
        and info.get("terminal_event_recorded") is True
        and last.get("type") == "agent_timeout"
        and last.get("reason") == "wall_clock_timeout"
        and isinstance(diagnostic_pending, list)
        and all(isinstance(item, str) for item in diagnostic_pending)
        and isinstance(event_pending, list)
        and diagnostic_pending == event_pending
        and isinstance(diagnostic_complete, bool)
        and isinstance(event_complete, bool)
        and diagnostic_complete == event_complete
        and diagnostic_complete == (not bool(diagnostic_pending))
    ):
        return None
    expected_partial = not diagnostic_complete
    if summary.get("partial_trajectory") is not expected_partial:
        return None
    return "normal_timeout" if diagnostic_complete else "partial_timeout"


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        preferred = content.get("text")
        if preferred is None:
            preferred = content.get("content")
        if preferred is not None:
            return _content_to_text(preferred)
        return json.dumps(content, ensure_ascii=False, sort_keys=True)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            value = _content_to_text(item)
            if value:
                parts.append(value)
        return "\n".join(parts)
    return str(content or "")


def acp_events_to_skillgen_messages(
    events: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Convert BenchFlow's normalized ACP events to SkillGen messages."""

    messages: list[dict[str, Any]] = []
    for event in events:
        event_type = event.get("type")
        if event_type == "user_message":
            text = _normalise_text(event.get("text"))
            if text:
                messages.append({"role": "user", "content": text})
        elif event_type in {"agent_message", "agent_thought"}:
            text = _normalise_text(event.get("text"))
            if text:
                messages.append(
                    {
                        "role": "assistant",
                        "content": text,
                        "event_type": str(event_type),
                    }
                )
        elif event_type == "tool_call":
            kind = _normalise_text(event.get("kind") or "tool")
            title = _normalise_text(event.get("title") or kind)
            status = _normalise_text(event.get("status") or "unknown")
            output = _content_to_text(event.get("content"))
            content = f"title={title}\nstatus={status}"
            if output:
                content += "\n" + output
            messages.append({"role": "tool", "name": kind, "content": content})
        elif event_type == "agent_timeout":
            messages.append(
                {
                    "role": "tool",
                    "name": "agent_timeout",
                    "content": _normalise_text(event.get("reason") or "agent timeout"),
                }
            )
    return messages


def _last_agent_output(messages: Sequence[Mapping[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("content"):
            return str(message["content"])
    return ""


def _find_rollout_artifacts(jobs_dir: Path) -> tuple[Path, Path]:
    results = sorted(jobs_dir.rglob("result.json")) if jobs_dir.exists() else []
    if len(results) != 1:
        raise SkillsBenchInfrastructureError(
            f"Expected exactly one result.json under {jobs_dir}, found {len(results)}"
        )
    result_path = results[0]
    trajectory_path = result_path.parent / "trajectory" / "acp_trajectory.jsonl"
    return result_path, trajectory_path


def _validate_treatment(result: Mapping[str, Any], *, with_skill: bool) -> None:
    expected_mode = "with-skill" if with_skill else "no-skill"
    if result.get("skill_mode") != expected_mode:
        raise SkillsBenchInfrastructureError(
            f"BenchFlow treatment mismatch: expected skill_mode={expected_mode!r}, "
            f"got {result.get('skill_mode')!r}"
        )
    source = result.get("skill_source")
    if with_skill:
        if source != "custom_runtime" or result.get("include_task_skills") is not False:
            raise SkillsBenchInfrastructureError(
                "Generated-skill rollout did not use an isolated custom runtime skill "
                f"(skill_source={source!r}, include_task_skills={result.get('include_task_skills')!r})"
            )
    elif source != "none" or result.get("include_task_skills") is not False:
        raise SkillsBenchInfrastructureError(
            "Baseline rollout was contaminated by a skill "
            f"(skill_source={source!r}, include_task_skills={result.get('include_task_skills')!r})"
        )


def _validate_result_identity(
    result: Mapping[str, Any],
    *,
    instance: TaskInstance,
    expected_model: str | None,
    expected_agent: str | None,
) -> None:
    """Reject a valid-looking artifact produced for another task or route."""

    expected_task = str(
        (instance.metadata or {}).get("skillsbench_task_id") or ""
    )
    if expected_task and result.get("task_name") != expected_task:
        raise SkillsBenchInfrastructureError(
            "BenchFlow task identity mismatch: "
            f"expected {expected_task!r}, got {result.get('task_name')!r}"
        )
    if expected_model is not None and result.get("model") != expected_model:
        raise SkillsBenchInfrastructureError(
            "BenchFlow model identity mismatch: "
            f"expected {expected_model!r}, got {result.get('model')!r}"
        )
    if expected_agent is not None and result.get("agent") != expected_agent:
        raise SkillsBenchInfrastructureError(
            "BenchFlow agent identity mismatch: "
            f"expected {expected_agent!r}, got {result.get('agent')!r}"
        )


def _trajectory_from_artifacts(
    *,
    instance: TaskInstance,
    skill_bundle: SkillItem | None,
    result_path: Path,
    trajectory_path: Path,
    process_returncode: int,
    run_root: Path,
    elapsed: float,
    expected_model: str | None = None,
    expected_agent: str | None = None,
) -> Trajectory:
    result = _read_json(result_path)
    _validate_result_identity(
        result,
        instance=instance,
        expected_model=expected_model,
        expected_agent=expected_agent,
    )
    _validate_treatment(result, with_skill=skill_bundle is not None)

    error = result.get("error")
    verifier_error = result.get("verifier_error")
    rewards = result.get("rewards")
    reward = rewards.get("reward") if isinstance(rewards, dict) else None
    events = _read_jsonl(trajectory_path)
    timeout_outcome = _scored_wall_clock_timeout_kind(result, events)
    scored_timeout = timeout_outcome is not None
    numeric_reward = isinstance(reward, (int, float)) and not isinstance(reward, bool)
    if (
        process_returncode != 0
        or (error and not scored_timeout)
        or verifier_error
        or not numeric_reward
    ):
        raise SkillsBenchInfrastructureError(
            "BenchFlow did not produce a valid scored rollout: "
            f"returncode={process_returncode}, error={error!r}, "
            f"verifier_error={verifier_error!r}, reward={reward!r}; artifacts={run_root}"
        )

    score = float(reward)
    if not 0.0 <= score <= 1.0:
        raise SkillsBenchInfrastructureError(
            f"SkillsBench reward outside [0, 1]: {score} ({result_path})"
        )
    messages = acp_events_to_skillgen_messages(events)
    success = score == PASS_REWARD
    agent_result = result.get("agent_result") or {}
    total_tokens = agent_result.get("total_tokens") if isinstance(agent_result, dict) else None
    n_tool_calls = result.get("n_tool_calls") or 0

    agent_config = {
        "model": result.get("model"),
        "inference_model": result.get("model"),
        "agent": result.get("agent"),
        "agent_name": result.get("agent_name"),
        "skill_id": skill_bundle.skill_id if skill_bundle else None,
        "skill_mode": result.get("skill_mode"),
        "skill_source": result.get("skill_source"),
        "benchflow_version": EXPECTED_BENCHFLOW_VERSION,
    }
    metadata = {
        "benchmark": "skillsbench",
        "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
        "task_name": result.get("task_name"),
        "rollout_name": result.get("rollout_name"),
        "reward": score,
        "passed": success,
        "n_tool_calls": n_tool_calls,
        "n_skill_invocations": result.get("n_skill_invocations"),
        "real_run_checks": {
            "has_model_tokens": isinstance(total_tokens, (int, float)) and total_tokens > 0,
            "has_tool_calls": isinstance(n_tool_calls, (int, float)) and n_tool_calls > 0,
            "has_verifier_reward": True,
        },
        "skill_mode": result.get("skill_mode"),
        "skill_source": result.get("skill_source"),
        "include_task_skills": result.get("include_task_skills"),
        # The frozen package digest is computed by this adapter over every
        # task file. BenchFlow's own task_digest uses a different algorithm;
        # retain it as provenance instead of confusing it with the authority
        # used by the cache key and dataset manifest.
        "task_digest": instance.metadata.get("skillsbench_task_digest")
        or result.get("task_digest"),
        "benchflow_task_digest": result.get("task_digest"),
        "benchflow_result_path": str(result_path),
        "benchflow_trajectory_path": str(trajectory_path),
        "benchflow_run_root": str(run_root),
        "benchflow_returncode": process_returncode,
        "benchflow_outcome": timeout_outcome or "scored",
        "agent_timed_out": scored_timeout,
        "agent_timeout_info": (
            dict(result["agent_timeout_info"])
            if scored_timeout
            else None
        ),
        "partial_trajectory": bool(
            (result.get("trajectory_summary") or {}).get("partial_trajectory")
        ),
        "seed_control": "unsupported_by_benchflow_cli",
        "rollout_replica_kind": "stochastic_same_task",
    }
    error_summary = None
    if not success:
        error_summary = (
            f"SkillsBench verifier reward={score:.6g}; full pass requires reward=1.0."
        )

    timing = result.get("timing") if isinstance(result.get("timing"), dict) else {}
    latency = timing.get("total") if isinstance(timing, dict) else None
    if not isinstance(latency, (int, float)):
        latency = elapsed

    return Trajectory(
        trajectory_id=str(uuid.uuid4()),
        instance_id=instance.instance_id,
        agent_config=agent_config,
        messages=messages,
        final_output=_last_agent_output(messages),
        success=success,
        score=score,
        error_summary=error_summary,
        token_usage=dict(agent_result) if isinstance(agent_result, dict) else None,
        latency=float(latency),
        timestamp=str(result.get("finished_at") or ""),
        metadata=metadata,
    )


def _rollout_cache_request(
    *,
    instance: TaskInstance,
    task_dir: Path,
    model: str,
    agent: str,
    sandbox: str,
    skill_bundle: SkillItem | None,
) -> RolloutCacheRequest:
    metadata = instance.metadata or {}
    task_digest = str(
        metadata.get("skillsbench_task_digest") or task_package_digest(task_dir)
    )
    return RolloutCacheRequest.build(
        instance=instance,
        task_digest=task_digest,
        model=model,
        agent=agent,
        sandbox=sandbox,
        skill=skill_bundle,
        adapter_schema_version=ADAPTER_SCHEMA_VERSION,
        benchflow_version=EXPECTED_BENCHFLOW_VERSION,
    )


def _cache_call(operation, *args, **kwargs):
    """Expose cache trust failures as benchmark infrastructure failures."""

    try:
        return operation(*args, **kwargs)
    except SkillsBenchRolloutCacheError as exc:
        raise SkillsBenchInfrastructureError(str(exc)) from exc


@contextmanager
def _cache_slot_lock(jobs_root: Path, request: RolloutCacheRequest):
    try:
        with slot_lock(jobs_root, request) as lock_path:
            yield lock_path
    except SkillsBenchRolloutCacheError as exc:
        raise SkillsBenchInfrastructureError(str(exc)) from exc


def _trajectory_from_run_root(
    *,
    run_root: Path,
    instance: TaskInstance,
    skill_bundle: SkillItem | None,
    model: str,
    agent: str,
    process_returncode: int,
    elapsed: float,
) -> Trajectory:
    result_path, trajectory_path = _find_rollout_artifacts(run_root / "jobs")
    return _trajectory_from_artifacts(
        instance=instance,
        skill_bundle=skill_bundle,
        result_path=result_path,
        trajectory_path=trajectory_path,
        process_returncode=process_returncode,
        run_root=run_root,
        elapsed=elapsed,
        expected_model=model,
        expected_agent=agent,
    )


def _recover_manifested_trajectory(
    *,
    jobs_root: Path,
    request: RolloutCacheRequest,
    instance: TaskInstance,
    skill_bundle: SkillItem | None,
    model: str,
    agent: str,
) -> Trajectory | None:
    """Recover a paid result durable on disk before the central cache write."""

    attempts = _cache_call(manifested_attempts, jobs_root, request)
    for run_root in attempts:
        receipt = _cache_call(read_attempt_receipt, run_root)
        if receipt is not None and receipt[0] != 0:
            # A known completed infrastructure/preflight failure is eligible
            # for a fresh attempt, but never for the result cache.
            continue
        try:
            trajectory = _trajectory_from_run_root(
                run_root=run_root,
                instance=instance,
                skill_bundle=skill_bundle,
                model=model,
                agent=agent,
                process_returncode=0,
                elapsed=receipt[1] if receipt is not None else 0.0,
            )
        except SkillsBenchInfrastructureError as exc:
            if receipt is None:
                raise SkillsBenchInfrastructureError(
                    "A prior paid slot has no completion receipt and no valid "
                    "recoverable result. It may still be running; inspect before "
                    f"retrying: {run_root}"
                ) from exc
            raise
        _cache_call(
            write_cached_trajectory,
            jobs_root,
            request,
            trajectory,
            source={
                "kind": "manifest_recovery",
                "run_root": str(run_root),
            },
        )
        return trajectory
    return None


def bootstrap_skillsbench_rollout_cache(
    *,
    instance: TaskInstance,
    skill_bundle: SkillItem | None,
    config,
    run_root: str | Path,
) -> Trajectory:
    """Explicitly import one legacy artifact into its exact frozen slot.

    Legacy run directories predate request manifests, so the caller must make
    the slot-to-directory mapping explicit.  Task package, task name, route,
    treatment, official reward, verifier status, and artifact structure are
    all revalidated before the entry is published.
    """

    metadata = instance.metadata or {}
    task_dir = resolve_task_dir(metadata)
    agent = str(metadata.get("skillsbench_agent") or "").strip()
    sandbox = str(metadata.get("skillsbench_sandbox") or "docker").strip()
    if not agent:
        raise SkillsBenchInfrastructureError(
            "Instance metadata is missing skillsbench_agent"
        )
    model = str(config.model)
    jobs_root = resolve_jobs_root(metadata)
    source_root = Path(run_root).expanduser().resolve()
    try:
        source_root.relative_to(jobs_root)
    except ValueError as exc:
        raise SkillsBenchInfrastructureError(
            f"Bootstrap artifact is outside the configured jobs root: {source_root}"
        ) from exc
    request = _rollout_cache_request(
        instance=instance,
        task_dir=task_dir,
        model=model,
        agent=agent,
        sandbox=sandbox,
        skill_bundle=skill_bundle,
    )
    cached = _cache_call(load_cached_trajectory, jobs_root, request)
    if cached is not None:
        return cached

    receipt = _cache_call(read_attempt_receipt, source_root)
    process_returncode, elapsed = receipt if receipt is not None else (0, 0.0)
    trajectory = _trajectory_from_run_root(
        run_root=source_root,
        instance=instance,
        skill_bundle=skill_bundle,
        model=model,
        agent=agent,
        process_returncode=process_returncode,
        elapsed=elapsed,
    )
    _cache_call(
        write_cached_trajectory,
        jobs_root,
        request,
        trajectory,
        source={
            "kind": "explicit_legacy_bootstrap",
            "run_root": str(source_root),
            "legacy_without_request_manifest": not (
                source_root / "skillgen_cache_request.json"
            ).is_file(),
        },
    )
    return trajectory


def run_skillsbench_agent(instance: TaskInstance, skill_bundle: SkillItem | None, config) -> Trajectory:
    """Execute or idempotently reuse one isolated official BenchFlow rollout."""

    metadata = instance.metadata or {}
    task_dir = resolve_task_dir(metadata)
    agent = str(metadata.get("skillsbench_agent") or "").strip()
    sandbox = str(metadata.get("skillsbench_sandbox") or "docker").strip()
    if not agent:
        raise SkillsBenchInfrastructureError(
            "Instance metadata is missing skillsbench_agent"
        )

    jobs_root = resolve_jobs_root(metadata)
    jobs_root.mkdir(parents=True, exist_ok=True)
    model = str(config.model)
    request = _rollout_cache_request(
        instance=instance,
        task_dir=task_dir,
        model=model,
        agent=agent,
        sandbox=sandbox,
        skill_bundle=skill_bundle,
    )
    cached = _cache_call(load_cached_trajectory, jobs_root, request)
    if cached is not None:
        return cached
    recovered = _recover_manifested_trajectory(
        jobs_root=jobs_root,
        request=request,
        instance=instance,
        skill_bundle=skill_bundle,
        model=model,
        agent=agent,
    )
    if recovered is not None:
        return recovered

    # Only require the runtime executable after all free cache/recovery paths
    # have missed.  A cache hit never invokes BenchFlow or the budget guard.
    bench_executable = _resolve_executable(
        str(metadata.get("skillsbench_bench_executable") or "bench")
    )
    _validate_benchflow_executable(bench_executable)
    run_name = _safe_slug(instance.instance_id, fallback="rollout")
    timeout_sec = float(metadata.get("skillsbench_subprocess_timeout_sec") or 7200)
    if timeout_sec <= 0:
        raise SkillsBenchInfrastructureError(
            "skillsbench_subprocess_timeout_sec must be positive"
        )

    with _cache_slot_lock(jobs_root, request):
        # Recheck after acquiring the cross-process lock.
        cached = _cache_call(load_cached_trajectory, jobs_root, request)
        if cached is not None:
            return cached
        recovered = _recover_manifested_trajectory(
            jobs_root=jobs_root,
            request=request,
            instance=instance,
            skill_bundle=skill_bundle,
            model=model,
            agent=agent,
        )
        if recovered is not None:
            return recovered

        attempt_id = uuid.uuid4().hex[:12]
        run_root = jobs_root / attempt_directory_name(
            run_name, request, attempt_id
        )
        run_root.mkdir(parents=False, exist_ok=False)
        write_attempt_manifest(run_root, request)
        bench_jobs_dir = run_root / "jobs"

        try:
            skills_dir: Path | None = None
            if skill_bundle is not None:
                skills_dir = materialize_generated_skill(
                    skill_bundle, run_root / "generated-skills"
                )
            command = build_benchflow_command(
                bench_executable=bench_executable,
                task_dir=task_dir,
                jobs_dir=bench_jobs_dir,
                agent=agent,
                model=model,
                sandbox=sandbox,
                skills_dir=skills_dir,
            )
        except Exception:
            write_attempt_receipt(
                run_root, process_returncode=-2, elapsed=0.0
            )
            raise

        started = time.monotonic()
        try:
            reservation_token = pilot_budget_guard.before_agent_rollout()
        except Exception:
            # No subprocess has started, so this attempt is known-safe to
            # supersede after the budget condition is resolved.
            write_attempt_receipt(
                run_root,
                process_returncode=-3,
                elapsed=time.monotonic() - started,
            )
            raise
        settlement_kind = "after_agent_rollout"
        try:
            try:
                completed = subprocess.run(
                    command,
                    cwd=str(task_dir),
                    env=_benchflow_subprocess_env(sandbox=sandbox, run_root=run_root),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout_sec,
                    check=False,
                    shell=False,
                )
            except subprocess.TimeoutExpired as exc:
                # Deliberately omit a completion receipt. The next invocation will
                # recover a valid artifact if one became durable, otherwise fail
                # closed because provider completion is ambiguous.
                settlement_kind = "after_agent_rollout_timeout"
                raise SkillsBenchInfrastructureError(
                    f"BenchFlow subprocess exceeded {timeout_sec}s; artifacts={run_root}"
                ) from exc
            except OSError as exc:
                settlement_kind = "after_agent_rollout_launch_error"
                write_attempt_receipt(
                    run_root,
                    process_returncode=-1,
                    elapsed=time.monotonic() - started,
                )
                raise SkillsBenchInfrastructureError(
                    f"Could not launch BenchFlow: {exc}; artifacts={run_root}"
                ) from exc
            elapsed = time.monotonic() - started
            write_attempt_receipt(
                run_root,
                process_returncode=completed.returncode,
                elapsed=elapsed,
            )
            trajectory = _trajectory_from_run_root(
                run_root=run_root,
                instance=instance,
                skill_bundle=skill_bundle,
                model=model,
                agent=agent,
                process_returncode=completed.returncode,
                elapsed=elapsed,
            )
            _cache_call(
                write_cached_trajectory,
                jobs_root,
                request,
                trajectory,
                source={"kind": "benchflow", "run_root": str(run_root)},
            )
            # The paid result is durable before settlement can unwind the
            # pipeline. Resume will therefore reuse it without another API call.
            return trajectory
        finally:
            # A successful official balance audit atomically releases this
            # rollout's persistent reservation. If the audit fails, the token
            # remains in the ledger so a later resume fails closed. Preserve a
            # primary verifier/infrastructure exception while doing so.
            active_exception = sys.exc_info()[0] is not None
            try:
                pilot_budget_guard.record_balance(
                    settlement_kind,
                    reservation_token=reservation_token,
                )
            except Exception:
                if not active_exception:
                    raise
