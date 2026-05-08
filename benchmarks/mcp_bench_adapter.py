"""MCP-Bench adapter (static planning variant).

MCP-Bench (https://github.com/Accenture/mcp-bench) benchmarks tool-using LLM
agents across 28 MCP servers with three splits: single-server, 2-server, and
3-server combinations. Each task has:

    - task_description  : concrete, step-by-step requirements (the "rubric")
    - fuzzy_description : the ambiguous user-facing prompt shown to the agent
    - dependency_analysis : expected tool-call plan (used only by the judge)
    - servers           : MCP servers available for this task
    - distraction_servers: (single-server split only) unrelated servers

In SkillGen we run the *static planning* variant: the agent never touches real
MCP servers. Instead, it receives the server list + one-line descriptions and
is asked to produce a structured plan (ordered tool calls) together with a
final synthesis. The grader (mcp_bench_grader.py) runs mcp-bench's six-
dimension LLM-as-judge rubric on that plan, normalises to a 0-1 score, and
thresholds it to a pass/fail for SkillGen's failure-driven pipeline.

This matches the abstraction level of mind2web_adapter / spreadsheetbench_
adapter: a pure data-loading layer that plugs into the existing trajectory
loop without spawning servers or requiring external API keys.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_TASKS_ROOT = Path(__file__).parent / "external" / "mcp-bench" / "tasks"

SPLIT_FILES: dict[str, str] = {
    "single": "mcpbench_tasks_single_runner_format.json",
    "multi_2server": "mcpbench_tasks_multi_2server_runner_format.json",
    "multi_3server": "mcpbench_tasks_multi_3server_runner_format.json",
}

# One-line descriptions for the 28 MCP servers, copied verbatim from the
# upstream README (Accenture/mcp-bench). Used to give the agent enough domain
# context to plan tool calls without needing the live servers' schemas.
SERVER_DESCRIPTIONS: dict[str, str] = {
    "BioMCP": "Biomedical research data, clinical trials, and health information",
    "Bibliomantic": "I Ching divination, hexagrams, and mystical guidance",
    "Call for Papers": "Academic conference submissions and call announcements",
    "Car Price Evaluator": "Vehicle valuation and automotive market analysis",
    "Context7": "Project context management and documentation services",
    "DEX Paprika": "Cryptocurrency DeFi analytics and decentralized exchange data",
    "FruityVice": "Comprehensive fruit nutrition information and dietary data",
    "Game Trends": "Gaming industry statistics and trend analysis",
    "Google Maps": "Location services, geocoding, and mapping functionality",
    "Huge Icons": "Icon search, management, and design resources",
    "Hugging Face": "Machine learning models, datasets, and AI capabilities",
    "Math MCP": "Mathematical calculations and computational operations",
    "Medical Calculator": "Clinical calculation tools and medical formulas",
    "Metropolitan Museum": "Art collection database and museum information",
    "Movie Recommender": "Film recommendations and movie metadata",
    "NASA Data": "Space mission data and astronomical information",
    "National Parks": "US National Parks information and visitor services",
    "NixOS": "Package management and system configuration tools",
    "OKX Exchange": "Cryptocurrency trading data and market information",
    "OpenAPI Explorer": "API specification exploration and testing tools",
    "OSINT Intelligence": "Open source intelligence gathering and analysis",
    "Paper Search": "Academic paper search across multiple research databases",
    "Reddit": "Social media content and community discussions",
    "Scientific Computing": "Advanced mathematical computations and data analysis",
    "Time MCP": "Date, time utilities, and timezone conversions",
    "Unit Converter": "Measurement conversions across different unit systems",
    "Weather Data": "Weather forecasts and meteorological information",
    "Wikipedia": "Encyclopedia content search and retrieval",
}


# Loading

def load_mcp_bench_raw(split: str, tasks_root: str | Path | None = None) -> list[dict[str, Any]]:
    """Return the flat list of raw task dicts for one split.

    Each returned dict is enriched with `servers` (the parent group's server
    list) and `combination_type` so downstream code does not need the original
    nested structure.
    """
    if split not in SPLIT_FILES:
        raise ValueError(f"Unknown MCP-Bench split: {split!r}. Expected one of {list(SPLIT_FILES)}")
    root = Path(tasks_root) if tasks_root else _TASKS_ROOT
    path = root / SPLIT_FILES[split]
    if not path.exists():
        raise FileNotFoundError(
            f"MCP-Bench task file not found: {path}. "
            "Did you clone Accenture/mcp-bench into external/mcp-bench?"
        )
    payload = json.loads(path.read_text())

    flat: list[dict[str, Any]] = []
    for group in payload.get("server_tasks") or []:
        servers = group.get("servers") or ([group["server_name"]] if group.get("server_name") else [])
        combo_type = group.get("combination_type") or split
        combo_name = group.get("combination_name") or group.get("server_name") or ""
        for t in group.get("tasks") or []:
            t = dict(t)
            t["servers"] = servers
            t["combination_type"] = combo_type
            t["combination_name"] = combo_name
            flat.append(t)
    return flat


# Prompt rendering

_AGENT_PREAMBLE = """\
You are a tool-using planning agent for the MCP-Bench benchmark (static \
planning variant - no live MCP servers are available at run time). Your job is \
to carefully read the user's request, figure out which MCP servers and tools \
you would call, and produce a structured plan together with the final answer \
you would return to the user.

Scoring is done by an LLM judge against a concrete reference plan. The judge \
checks six dimensions: task fulfilment, grounding, tool appropriateness, \
parameter accuracy, dependency awareness, and parallelism/efficiency. Prefer \
precise tool selection and minimal redundant calls over verbose prose.\
"""

_AGENT_OUTPUT_SPEC = """\
Output format (respond with EXACTLY one fenced JSON object, no extra prose):

```json
{
  "plan": [
    {
      "step": 1,
      "server": "<one of the available servers>",
      "tool": "<tool / operation name on that server>",
      "parameters": {"<param>": "<value>"},
      "depends_on": [],
      "purpose": "<one sentence>"
    }
  ],
  "final_answer": "<the answer you would return to the user, synthesising the \
tool results you expect to obtain>"
}
```

Rules:
- `server` MUST be chosen from the Available servers list below.
- `tool` should be a realistic operation name for that server. Real MCP \
servers use a mix of naming conventions (snake_case, camelCase, kebab-case); \
prefer the exact name a well-designed MCP server for that domain would \
publish. Grader runs a deterministic check: inventing a tool name that does \
not match that server's real surface will be penalised.
- Steps that can run in parallel should share the same `depends_on` set \
(or `[]` if they depend only on the user input).
- `final_answer` must stand on its own: it is what the real user would read.
"""


def _render_server_block(servers: list[str], distractions: list[str]) -> str:
    lines = ["Available servers (USE ONLY THESE):"]
    for s in servers:
        desc = SERVER_DESCRIPTIONS.get(s, "(no description)")
        lines.append(f"  - {s}: {desc}")
    if distractions:
        lines.append("")
        lines.append("Distractor servers (present in the benchmark but IRRELEVANT; do NOT plan calls to these):")
        for s in distractions:
            desc = SERVER_DESCRIPTIONS.get(s, "(no description)")
            lines.append(f"  - {s}: {desc}")
    return "\n".join(lines)


def _render_tool_catalogue(servers: list[str]) -> str:
    """List the per-server canonical tool names so the agent can ground its
    plan on real operations. Drawn from ``mcp_bench_tool_registry`` (which
    scans every task in the benchmark corpus) so tool names are verified
    kebab-/snake-/camelCase strings that actually exist. Without this, the
    grader's deterministic tool-name check penalises every agent equally
    regardless of plan quality.
    """
    # Imported locally to avoid a cycle if the registry later imports the
    # adapter (it currently imports only SERVER_DESCRIPTIONS + loader).
    from .mcp_bench_tool_registry import build_registry

    reg = build_registry()
    lines = ["Tool catalogue (use these exact tool names - spelling and casing MATTER):"]
    for s in servers:
        tools = sorted(reg.tools_authoritative.get(s, frozenset()))
        if tools:
            preview = ", ".join(tools)
            lines.append(f"  - {s}: [{preview}]")
        else:
            lines.append(f"  - {s}: (tool list unavailable; use a plausible MCP operation name)")
    return "\n".join(lines)


def convert_mcp_bench_task(task: dict[str, Any], *, split: str) -> dict[str, Any]:
    """Render a single raw task dict into a SkillGen TaskInstance-shaped dict."""
    servers = list(task.get("servers") or [])
    distractions = list(task.get("distraction_servers") or [])
    fuzzy = (task.get("fuzzy_description") or task.get("task_description") or "").strip()
    concrete = (task.get("task_description") or "").strip()
    dep = (task.get("dependency_analysis") or "").strip()
    task_id = task.get("task_id") or ""
    combo_name = task.get("combination_name") or (servers[0] if servers else "")

    prompt = "\n\n".join([
        _AGENT_PREAMBLE,
        f"User request:\n{fuzzy}",
        _render_server_block(servers, distractions),
        _render_tool_catalogue(servers),
        _AGENT_OUTPUT_SPEC,
    ])

    return {
        "instance_id": task_id or f"{split}__{combo_name}__unknown",
        "input": prompt,
        # No ground-truth string; judging is open-ended via mcp_bench_grader.
        "ground_truth": None,
        "metadata": {
            "benchmark": "mcp_bench",
            "split": split,
            "task_id": task_id,
            "combination_name": combo_name,
            "combination_type": task.get("combination_type") or split,
            "servers": servers,
            "distraction_servers": distractions,
            "fuzzy_description": fuzzy,
            "task_description": concrete,
            "dependency_analysis": dep,
        },
    }


def load_mcp_bench_split(
    split: str,
    *,
    tasks_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load a split and return instance-shaped dicts ready for JSON dumping."""
    raw = load_mcp_bench_raw(split, tasks_root=tasks_root)
    return [convert_mcp_bench_task(t, split=split) for t in raw]


__all__ = [
    "SPLIT_FILES",
    "SERVER_DESCRIPTIONS",
    "load_mcp_bench_raw",
    "load_mcp_bench_split",
    "convert_mcp_bench_task",
]
