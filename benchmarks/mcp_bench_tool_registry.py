"""MCP-Bench tool-name registry + deterministic plan validator.

Motivation
----------
In the static planning variant we cannot execute real MCP servers, so the LLM
judge is our only evaluator. Empirically we observed that both baseline and
skill runs frequently *hallucinate* tool names (e.g. HF's real tools are
kebab-case `search-models`, `get-model-info`, `get-daily-papers`, but agents
write `list_models`, `listModels`, `getModel`, etc.). The judge's scoring of
such hallucinations is inconsistent - sometimes 0.45, sometimes 0.40 - which
adds noise and can cause a skill to slip through McNemar on train yet
systematically regress on test (or vice-versa).

This module provides a *deterministic* per-server tool allowlist built from
every occurrence of a tool-like identifier inside the three split JSONs
(``task_description``, ``dependency_analysis``, ``fuzzy_description``),
covering snake_case, camelCase, and kebab-case conventions. The registry is
a strict superset of all real MCP tool names authors reference, so any
identifier outside it is almost certainly hallucinated.

Public API
----------

    build_registry() -> ToolRegistry                 # cached
    validate_plan(plan: dict, task_meta: dict) -> dict

The returned dict fields are stable and designed to be embedded into the
grader's output under the ``tool_validation`` key:

    {
      "n_steps":           int,
      "tools":             [<tool strings per step>],
      "classifications":   [<"known"|"wrong_server"|"pooled"|"hallucinated">],
      "n_hallucinated":    int,
      "n_wrong_server":    int,
      "miss_count":        int,   # hallucinated + wrong_server
      "miss_rate":         float, # miss_count / max(1, n_steps)
      "hallucinations":    [{"step","server","tool","reason"}, ...],
      "wrong_server_usage":[{"step","server","tool","true_server"}, ...],
      "invalid_servers":   [{"step","server"}, ...],
    }
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from .mcp_bench_adapter import SERVER_DESCRIPTIONS, load_mcp_bench_raw

# Regex catalogue

# Tool-name identifiers must start with a lowercase letter, and be either
# snake_case (contains '_'), kebab-case (contains '-'), or camelCase
# (contains an internal uppercase). This excludes plain English words and
# single-word lowercase identifiers (which are too ambiguous to attribute).
_RX_KEBAB = re.compile(r"\b([a-z][a-z0-9]*(?:-[a-z0-9]+)+)\b")
_RX_SNAKE = re.compile(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b")
_RX_CAMEL = re.compile(r"\b([a-z][a-z0-9]*[A-Z][A-Za-z0-9]+)\b")

# ``Server Name:toolIdent`` - authoritative attribution, since the task
# author explicitly names the server. Allows at most 3 TitleCase words in
# the server portion (enough to cover all 28 MCP-Bench servers).
_RX_SERVER_COLON = re.compile(
    r"([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){0,3})\s*:\s*([a-z][A-Za-z0-9_-]{2,})"
)

# Free-form tool patterns that are still high-confidence:
#   - ``toolName(`` (function-call syntax)
#   - after an imperative verb: ``call|use|invoke|run|execute|fetch`` + toolName
# Used to enrich ``tools_authoritative`` for single-server tasks where the
# author doesn't bother with the ``Server:`` prefix.
_RX_FUNC_CALL = re.compile(r"\b([a-z][A-Za-z0-9_-]{2,})\s*\(")
_RX_VERB_TOOL = re.compile(
    r"\b(?:call|calls|called|calling|use|using|invoke|invoked|run|runs|running|"
    r"execute|executed|executing)\s+([a-z][A-Za-z0-9_-]{2,})\b",
    re.IGNORECASE,
)

# Hand-curated stop-lists
# English compound tokens that match the regex but are never tool names.

_STOP_KEBAB = {
    "open-source", "long-term", "real-time", "end-to-end", "step-by-step",
    "trade-off", "self-contained", "follow-up", "ready-to", "high-level",
    "low-level", "one-stop", "big-picture", "in-depth", "top-down",
    "bottom-up", "well-known", "state-of-the-art", "cross-platform",
    "wide-ranging", "day-by-day", "up-to", "out-of", "one-line",
    "best-practice", "user-facing", "high-frequency", "long-running",
    "short-lived", "stand-alone", "fall-through", "pull-request",
    "co-located", "object-oriented", "data-driven", "hand-crafted",
    "fine-tuning", "use-case", "use-cases", "cross-server", "intra-server",
    "multi-server", "single-server", "multi-park", "cross-validation",
    "cross-checked", "knee-deep", "human-readable", "cursor-based",
    "decision-making", "leg-by-leg", "model-dataset", "pre-trained",
    "spam-detection", "spam-classification", "text-classification",
    "example-count", "train-size", "visitor-center", "water-and-toilet",
    "apache-2", "built-in", "code-hosting", "bike-shop", "coffee-shop",
    "mass-spec",
}
_STOP_SHORT = {"get", "set", "call", "do", "if", "on", "in", "to", "of"}


def _clean_token(t: str) -> bool:
    if len(t) < 4:
        return False
    if t in _STOP_KEBAB or t in _STOP_SHORT:
        return False
    return True


# Common tool-name verb prefixes / infixes. Used to distinguish likely tool
# names from parameter / field names when promoting free-form tokens into
# the authoritative catalogue.
_TOOL_VERBS = (
    "get", "set", "list", "search", "find", "fetch", "read", "download",
    "upload", "create", "delete", "update", "convert", "calculate", "compute",
    "validate", "send", "post", "run", "query", "lookup", "resolve", "summarize",
    "extract", "parse", "generate", "invoke", "check", "view", "scale", "project",
    "change",
)


def _looks_like_tool(tok: str) -> bool:
    """Heuristic: does this token look like a tool name rather than a field?

    A tool name usually starts with (or ends with / contains) an imperative
    verb. Parameter names like ``model_id``, ``dataset_size``, ``arxiv_ids``
    are overwhelmingly noun-only.
    """
    lo = tok.lower()
    # normalise separators so ``getModel`` / ``get_model`` / ``get-model``
    # are all inspected the same way
    parts = re.split(r"[_\-]", lo)
    # also split camelCase
    camel_parts: list[str] = []
    for p in parts:
        camel_parts.extend(re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1_\2", p)))
    for v in _TOOL_VERBS:
        for p in camel_parts:
            if p == v:
                return True
    return False


def _extract_tokens(text: str) -> set[str]:
    toks: set[str] = set()
    for rx in (_RX_KEBAB, _RX_SNAKE, _RX_CAMEL):
        toks.update(rx.findall(text or ""))
    return {t for t in toks if _clean_token(t)}


# Server-name canonicalisation

_SERVER_CANON = {s.lower(): s for s in SERVER_DESCRIPTIONS}
_SERVER_CANON_FLAT = {s.lower().replace(" ", ""): s for s in SERVER_DESCRIPTIONS}


def _canonicalise_server(raw: str, task_servers: list[str] | None = None) -> str | None:
    if not raw:
        return None
    key = raw.strip().lower()
    if key in _SERVER_CANON:
        return _SERVER_CANON[key]
    flat = key.replace(" ", "").replace("-", "").replace("_", "")
    if flat in _SERVER_CANON_FLAT:
        return _SERVER_CANON_FLAT[flat]
    # Try matching as a prefix of a task-declared server (handles
    # abbreviations like "BioMCP" vs "BioMCP server").
    if task_servers:
        for s in task_servers:
            if s.lower() == key or s.lower().startswith(key) or key.startswith(s.lower()):
                return s
    return None


# Registry


class ToolRegistry:
    """Per-server allowlist + multi-server pool of legitimate tool tokens.

    Two levels of strictness are exposed:

    - ``tools_authoritative[s]`` - tools attributed to server *s* via an explicit
      ``"Server Name:tool"`` reference in at least one task. These are the high-
      confidence canonical tool names, suitable for surfacing to the agent as a
      tool catalogue.
    - ``per_server[s]`` - ``tools_authoritative[s]`` plus every tool-like token
      that co-occurs with server *s* in single-server task text. Used by the
      hallucination check so that tokens mentioned anywhere in *s*'s tasks are
      still accepted (minimises false positives from the deterministic guard).
    """

    def __init__(
        self,
        per_server: dict[str, set[str]],
        tools_authoritative: dict[str, set[str]],
        pool: set[str],
    ) -> None:
        self.per_server = {s: frozenset(v) for s, v in per_server.items()}
        self.tools_authoritative = {s: frozenset(v) for s, v in tools_authoritative.items()}
        self.pool = frozenset(pool)
        self._inverse: dict[str, list[str]] = {}
        for srv, toks in per_server.items():
            for tk in toks:
                self._inverse.setdefault(tk, []).append(srv)

    def classify(self, tool: str, task_servers: list[str]) -> tuple[str, str | None]:
        """Return (classification, note) for a single tool.

        Classifications:
          - "known"        - tool is in allowlist of at least one task server
          - "wrong_server" - tool is attributed to some OTHER server
          - "pooled"       - tool is in the multi-server unattributed pool
          - "hallucinated" - tool is unknown to all of MCP-Bench's text
        """
        if not tool:
            return "hallucinated", "empty tool name"
        for s in task_servers:
            if tool in self.per_server.get(s, frozenset()):
                return "known", None
        # Attributed to some other server?
        owners = self._inverse.get(tool)
        if owners:
            other = [o for o in owners if o not in task_servers]
            if other:
                return "wrong_server", other[0]
        if tool in self.pool:
            return "pooled", None
        return "hallucinated", None

    def sizes(self) -> dict[str, int]:
        return {s: len(v) for s, v in self.per_server.items()}


def _build() -> ToolRegistry:
    per_server: dict[str, set[str]] = {s: set() for s in SERVER_DESCRIPTIONS}
    tools_auth: dict[str, set[str]] = {s: set() for s in SERVER_DESCRIPTIONS}
    pool: set[str] = set()

    for split in ("single", "multi_2server", "multi_3server"):
        try:
            tasks = load_mcp_bench_raw(split)
        except FileNotFoundError:
            continue
        for t in tasks:
            servers = [s for s in (t.get("servers") or []) if s]
            text = "\n".join(
                str(t.get(k) or "")
                for k in ("task_description", "dependency_analysis", "fuzzy_description")
            )
            # 1. Authoritative Server:tool attributions - the only place we
            #    can be sure an identifier really is a tool rather than a
            #    parameter name, field name, or English compound word.
            for m in _RX_SERVER_COLON.finditer(text):
                svr_raw, tool = m.group(1), m.group(2)
                if not _clean_token(tool):
                    continue
                canon = _canonicalise_server(svr_raw, servers)
                if canon:
                    tools_auth.setdefault(canon, set()).add(tool)
                    per_server.setdefault(canon, set()).add(tool)
            # 2. High-confidence free-form tool mentions in single-server
            #    tasks: function-call syntax ``toolName(`` and imperative
            #    ``call/use/invoke toolName`` patterns. These cannot appear
            #    in multi-server text without ambiguity about which server
            #    owns the tool, so we only promote for |servers|==1.
            if len(servers) == 1:
                home = servers[0]
                for rx in (_RX_FUNC_CALL, _RX_VERB_TOOL):
                    for m in rx.finditer(text):
                        tok = m.group(1)
                        if _clean_token(tok) and _looks_like_tool(tok):
                            tools_auth.setdefault(home, set()).add(tok)
                            per_server.setdefault(home, set()).add(tok)
            # 3. Free-form tool-like tokens (all kebab/snake/camel) - feed
            #    the ``per_server`` allowlist (hallucination check is
            #    lenient by design) or the global pool for multi-server
            #    tasks where we cannot attribute.
            tokens = _extract_tokens(text)
            if len(servers) == 1:
                per_server[servers[0]].update(tokens)
            else:
                pool.update(tokens)

    return ToolRegistry(per_server, tools_auth, pool)


@lru_cache(maxsize=1)
def build_registry() -> ToolRegistry:
    return _build()


# Plan validator

_HALLUCINATION_REASON = "tool name not found in any MCP-Bench task text"


def validate_plan(plan: dict[str, Any] | None, task_meta: dict[str, Any]) -> dict[str, Any]:
    """Validate each plan step's (server, tool) pair against the registry.

    Returns a structured report; all counts default to 0 if ``plan`` is None
    or malformed, so the caller can safely embed the result without
    conditionals.
    """
    empty = {
        "n_steps": 0,
        "tools": [],
        "classifications": [],
        "n_hallucinated": 0,
        "n_wrong_server": 0,
        "miss_count": 0,
        "miss_rate": 0.0,
        "hallucinations": [],
        "wrong_server_usage": [],
        "invalid_servers": [],
    }
    if not isinstance(plan, dict):
        return empty
    steps = plan.get("plan") or plan.get("steps") or []
    if not isinstance(steps, list) or not steps:
        return empty

    task_servers = [s for s in (task_meta.get("servers") or []) if s]
    distractions = set(task_meta.get("distraction_servers") or [])
    reg = build_registry()

    tools: list[str] = []
    classifications: list[str] = []
    hallucinations: list[dict[str, Any]] = []
    wrong_server: list[dict[str, Any]] = []
    invalid_servers: list[dict[str, Any]] = []

    for idx, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            continue
        server_raw = str(step.get("server") or "").strip()
        tool_raw = str(step.get("tool") or "").strip()
        step_no = step.get("step") or idx
        tools.append(tool_raw)

        canon_server = _canonicalise_server(server_raw, task_servers)
        if canon_server is None or canon_server not in task_servers:
            invalid_servers.append(
                {
                    "step": step_no,
                    "server": server_raw,
                    "reason": (
                        "distractor server used" if canon_server in distractions
                        else "server not in task's available set"
                    ),
                }
            )

        cls, note = reg.classify(tool_raw, task_servers)
        classifications.append(cls)
        if cls == "hallucinated":
            hallucinations.append(
                {
                    "step": step_no,
                    "server": server_raw,
                    "tool": tool_raw,
                    "reason": _HALLUCINATION_REASON,
                }
            )
        elif cls == "wrong_server":
            wrong_server.append(
                {
                    "step": step_no,
                    "server": server_raw,
                    "tool": tool_raw,
                    "true_server": note,
                }
            )

    n_steps = len(tools)
    n_hallucinated = len(hallucinations)
    n_wrong_server = len(wrong_server)
    miss_count = n_hallucinated + n_wrong_server
    miss_rate = miss_count / n_steps if n_steps else 0.0

    return {
        "n_steps": n_steps,
        "tools": tools,
        "classifications": classifications,
        "n_hallucinated": n_hallucinated,
        "n_wrong_server": n_wrong_server,
        "miss_count": miss_count,
        "miss_rate": miss_rate,
        "hallucinations": hallucinations,
        "wrong_server_usage": wrong_server,
        "invalid_servers": invalid_servers,
    }


__all__ = [
    "ToolRegistry",
    "build_registry",
    "validate_plan",
]
