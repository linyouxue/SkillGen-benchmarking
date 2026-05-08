"""ToolBench adapter (static planning variant).

ToolBench (https://github.com/OpenBMB/ToolBench, ICLR'24) benchmarks tool-using
LLM agents across 16464 RapidAPI endpoints. Each query ships with:

    - query       : the user-facing natural-language request
    - query_id    : integer identifier
    - api_list    : list of APIs available to the agent, each with
                    {category_name, tool_name, api_name,
                     api_description, required_parameters, optional_parameters,
                     method, (optional) template_response}
    - relevant APIs (train-only): ground-truth (tool_name, api_name) pairs

Upstream ships the real 200-query test files inside a ~4GB Google-Drive archive
(`data.zip` -> `data/test_instruction/G{1,2,3}_{instruction,category,tool}.json`).
The repo itself only carries 5 + 3 + 2 example queries under `data_example/`.

In SkillGen we run the **static planning** variant (same pattern as mcp_bench):
the agent never calls RapidAPI; it reads the `api_list` + descriptions and
produces a structured JSON plan + final answer. The grader
(`toolbench_grader.py`) does a 5-dimension LLM-as-judge pass and thresholds to
pass/fail for the failure-driven pipeline.

This file is a pure data loader + prompt renderer; it never spawns servers or
hits the network. Test fixtures are picked up in this order:

    1. `external/ToolBench/data/test_instruction/{subset}.json`   (real 200-eval)
    2. `external/ToolBench/data/instruction/{G{n}}_query.json`    (full training set)
    3. `external/ToolBench/data_example/instruction/{G{n}}_query.json` (5/3/2 examples)
    4. `$TOOLBENCH_DATA_DIR` env var  (user-supplied local checkout)

Override via `subset=` for test-split loading, `pool=` for train-split loading.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).parent
_DEFAULT_TB_ROOT = _REPO_ROOT / "external" / "ToolBench"


# Official test subsets (200 queries each after data.zip unpack).
TEST_SUBSETS: tuple[str, ...] = (
    "G1_instruction",
    "G1_category",
    "G1_tool",
    "G2_instruction",
    "G2_category",
    "G3_instruction",
)

# Training pools (large JSONs inside data.zip -> data/instruction/).
TRAIN_POOLS: dict[str, str] = {
    "G1": "G1_query.json",
    "G2": "G2_query.json",
    "G3": "G3_query.json",
}


def _toolbench_root() -> Path:
    override = os.environ.get("TOOLBENCH_DATA_DIR")
    if override:
        return Path(override)
    return _DEFAULT_TB_ROOT


def _candidate_paths(*, subset: str | None, pool: str | None) -> list[Path]:
    """Return prioritised list of paths to look for."""
    root = _toolbench_root()
    paths: list[Path] = []
    if subset:
        paths.append(root / "data" / "test_instruction" / f"{subset}.json")
        # If the user passed e.g. "G1_instruction" but only has the
        # data_example checkout, fall back to G1_query.json (5 items).
        g = subset.split("_", 1)[0]
        if g in TRAIN_POOLS:
            paths.append(root / "data" / "instruction" / TRAIN_POOLS[g])
            paths.append(root / "data_example" / "instruction" / TRAIN_POOLS[g])
    if pool:
        if pool not in TRAIN_POOLS:
            raise ValueError(f"Unknown ToolBench pool {pool!r}. Use one of {list(TRAIN_POOLS)}.")
        paths.append(root / "data" / "instruction" / TRAIN_POOLS[pool])
        paths.append(root / "data_example" / "instruction" / TRAIN_POOLS[pool])
    return paths


def load_toolbench_raw(
    *,
    subset: str | None = None,
    pool: str | None = None,
    path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return the raw list of ToolBench query dicts for one subset or pool.

    Exactly one of `subset`, `pool`, or `path` must be provided. With `subset`
    we first look for the real 200-query eval file, then fall back to the
    corresponding train `G{n}_query.json`, then the tiny `data_example` file,
    so smoke tests work out of the box even without the 4GB `data.zip`.
    """
    if sum(bool(x) for x in (subset, pool, path)) != 1:
        raise ValueError("Pass exactly one of `subset`, `pool`, or `path`.")

    candidates: list[Path] = [Path(path)] if path else _candidate_paths(subset=subset, pool=pool)

    for p in candidates:
        if p.is_file():
            data = json.loads(p.read_text())
            if not isinstance(data, list):
                raise ValueError(f"{p} is not a JSON array of queries.")
            return data

    # Nothing found - surface a helpful error pointing to the archive.
    root = _toolbench_root()
    key = subset or pool or str(path)
    raise FileNotFoundError(
        f"ToolBench data not found for {key!r}. Tried:\n"
        + "\n".join(f"  - {p}" for p in candidates)
        + "\n\nFixes:\n"
        "  1) Download data.zip from upstream's Google Drive and unzip\n"
        f"     `data/` into {root}/data/, OR\n"
        "  2) Point TOOLBENCH_DATA_DIR at a local ToolBench checkout, OR\n"
        "  3) For smoke-tests only, use pool=\"G1\" (falls back to the 5 examples\n"
        "     in data_example/instruction/G1_query.json)."
    )


# Prompt rendering

_AGENT_PREAMBLE = """\
You are a tool-using planning agent for the ToolBench benchmark (OpenBMB, \
ICLR'24). You are operating in the STATIC PLANNING variant - no RapidAPI \
endpoint is actually reachable at run time. Your job: read the user's request, \
decide which of the listed APIs you would call and in what order, and produce \
a structured plan plus the final answer you would return once those calls \
succeeded.

Judging is done by an LLM against a five-dimension rubric: task fulfilment, \
tool appropriateness, parameter accuracy, plan coherence (dependency order), \
and answer groundedness (the final answer must be consistent with what the \
planned tools can actually return; don't fabricate concrete numbers / URLs \
that no planned call produces). Prefer precise tool selection and minimal \
redundant calls.\
"""

_AGENT_OUTPUT_SPEC = """\
Output format - respond with EXACTLY ONE fenced JSON object, no extra prose:

```json
{
  "plan": [
    {
      "step": 1,
      "category": "<category_name from the API list>",
      "tool": "<tool_name from the API list>",
      "api": "<api_name from the API list>",
      "parameters": {"<param>": "<value>"},
      "depends_on": [],
      "purpose": "<one sentence>"
    }
  ],
  "final_answer": "<the synthesised answer you would return to the user once \
the above calls had resolved>"
}
```

Rules:
- (category, tool, api) MUST be a triple from the Available APIs list below.
  Do NOT invent APIs.
- `parameters` keys must come from each API's `required_parameters` (and, \
when justified, `optional_parameters`). Supply plausible concrete values \
given the user's request.
- Steps that can run in parallel share a `depends_on` set (or `[]` for calls \
that depend only on the user input).
- `final_answer` must stand on its own for the end user and must not assert \
concrete facts that none of the planned calls would plausibly return.
"""


def _format_params(params: list[dict[str, Any]] | None) -> str:
    if not params:
        return "(none)"
    lines: list[str] = []
    for p in params:
        name = p.get("name", "?")
        t = p.get("type", "")
        desc = (p.get("description") or "").strip().replace("\n", " ")
        default = p.get("default", "")
        extra = f" type={t}" if t else ""
        if default not in ("", None):
            extra += f" default={default!r}"
        lines.append(f"    * {name}{extra}: {desc[:200]}")
    return "\n".join(lines)


def _render_api_block(api_list: list[dict[str, Any]]) -> str:
    """Render the available-APIs section of the prompt."""
    lines = ["Available APIs (USE ONLY THESE triples (category, tool, api)):"]
    # Group by (category, tool) for readability.
    by_tool: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for api in api_list:
        key = (api.get("category_name", "?"), api.get("tool_name", "?"))
        by_tool.setdefault(key, []).append(api)
    for (cat, tool), apis in by_tool.items():
        lines.append(f"- category: {cat} | tool: {tool}")
        for api in apis:
            name = api.get("api_name", "?")
            method = api.get("method", "")
            desc = (api.get("api_description") or "").strip().replace("\n", " ")
            header = f"  * api: {name}"
            if method:
                header += f"  (HTTP {method})"
            if desc:
                header += f" - {desc[:240]}"
            lines.append(header)
            req = api.get("required_parameters") or []
            opt = api.get("optional_parameters") or []
            if req:
                lines.append("    required_parameters:")
                lines.append(_format_params(req))
            if opt:
                lines.append("    optional_parameters:")
                lines.append(_format_params(opt))
    return "\n".join(lines)


def convert_toolbench_query(
    query: dict[str, Any],
    *,
    subset: str,
    split_source: str,
) -> dict[str, Any]:
    """Convert one raw ToolBench query dict into a SkillGen TaskInstance dict.

    `subset` is the logical subset label ("G1_instruction" / "G1" / ...) and
    `split_source` records which source file this came from, for debugging.
    """
    api_list = list(query.get("api_list") or [])
    user_query = (query.get("query") or "").strip()
    query_id = query.get("query_id")
    relevant = query.get("relevant APIs") or query.get("relevant_APIs") or []

    prompt = "\n\n".join([
        _AGENT_PREAMBLE,
        f"User request:\n{user_query}",
        _render_api_block(api_list),
        _AGENT_OUTPUT_SPEC,
    ])

    stable_id = f"toolbench__{subset}__qid{query_id}"

    return {
        "instance_id": stable_id,
        "input": prompt,
        "ground_truth": None,
        "metadata": {
            "benchmark": "toolbench",
            "subset": subset,
            "query_id": query_id,
            "query": user_query,
            "api_list": api_list,
            "relevant_apis": relevant,
            "split_source": split_source,
        },
    }


def load_toolbench_split(
    *,
    subset: str | None = None,
    pool: str | None = None,
    path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load queries and return instance-shaped dicts ready for JSON dumping."""
    raw = load_toolbench_raw(subset=subset, pool=pool, path=path)
    logical = subset or pool or (Path(path).stem if path else "toolbench")
    # Record the actual source file we ended up reading.
    source = None
    for p in ([Path(path)] if path else _candidate_paths(subset=subset, pool=pool)):
        if p.is_file():
            source = str(p)
            break
    return [convert_toolbench_query(q, subset=logical, split_source=source or "") for q in raw]


__all__ = [
    "TEST_SUBSETS",
    "TRAIN_POOLS",
    "load_toolbench_raw",
    "load_toolbench_split",
    "convert_toolbench_query",
]
