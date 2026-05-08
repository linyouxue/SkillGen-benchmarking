"""ScienceWorld offline-plan adapter for SkillGen.

ScienceWorld is an interactive text-based science environment (allenai/ScienceWorld).
Here we adapt it to SkillGen's one-shot text-generation framework by operating
in an **offline plan-generation variant**:

    - Agent is given: the task description + initial scene observation + action
      grammar hint.
    - Agent outputs: a numbered ordered list of actions that would solve the task.
    - Grader compares the plan against the gold action sequence (shipped with the
      repo in `external/scienceworld/goldpaths/goldsequences-...json`), focusing
      on coverage of score-changing "key actions".

This mirrors how ALFWorld is already wired in this repo (see `data/alfworld/`):
both benchmarks are interactive environments that we reduce to static plan
generation for the purposes of the single-skill pipeline.

The adapter ONLY reads the gold path JSON; it does NOT require the
`scienceworld` PyPI package or a running JVM. Splitting uses the benchmark's
built-in `fold` field (train / dev / test) which guarantees task-variation
disjointness between our train and test pools.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

_REPO_ROOT_DEFAULT = Path(__file__).resolve().parent
_GOLD_ZIP_PATH_DEFAULT = _REPO_ROOT_DEFAULT / "external" / "scienceworld" / "goldpaths"

# The zip expands into a single monolithic JSON with a stable name.
_GOLD_JSON_NAME = "goldsequences-0-1-2-3-4-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23-24-25-26-27-28-29.json"


# Action-grammar hint
#
# ScienceWorld uses a constrained text-game grammar. The agent won't know it
# a priori (unlike in real MCP settings where a tool list is discoverable), so
# we ship a compact one-pager. These verbs are taken from Wang et al. 2022,
# Appendix A (official action list); we keep the canonical ones most tasks use.
_ACTION_GRAMMAR_HINT = """\
ScienceWorld actions follow a fixed grammar. The most common verbs are:
  navigation   : `go to <room>`, `open door to <room>`, `close door to <room>`
  inspection   : `look around`, `look at <obj>`, `read <obj>`
  manipulation : `pick up <obj>`, `put down <obj>`, `move <obj> to <container>`,
                 `open <obj>`, `close <obj>`, `activate <obj>`, `deactivate <obj>`
  focus        : `focus on <obj>`   (mandatory to declare the "target of measurement")
  measurement  : `use <instrument> on <obj>` (e.g. `use thermometer on ice`)
  chemistry    : `mix <obj1> and <obj2>`, `pour <obj1> into <obj2>`,
                 `dunk <obj> in <liquid>`
  thermodynamics: `heat <obj>`, `cool <obj>`, `freeze <obj>`
  answer       : `wait`, `wait1` (let time pass), `answer <X>` (categorical task)

Tips:
  - Always `look around` first if the scene is unclear.
  - For measurement tasks you MUST `focus on <obj>` before reading the result.
  - Paths between rooms often require `open door to <room>` before `go to <room>`.
"""


# Loading

def _find_goldpath_json(goldpath_dir: str | Path | None = None) -> Path:
    root = Path(goldpath_dir) if goldpath_dir else _GOLD_ZIP_PATH_DEFAULT
    candidate = root / _GOLD_JSON_NAME
    if candidate.exists():
        return candidate
    # Fallback: any goldsequences-*.json under that dir.
    for p in sorted(root.glob("goldsequences-*.json")):
        return p
    raise FileNotFoundError(
        f"ScienceWorld gold path JSON not found under {root}. "
        f"Did you unzip `goldpaths-all.zip`? Expected file name: {_GOLD_JSON_NAME}"
    )


_GOLD_CACHE: dict[str, Any] | None = None


def load_goldpaths_raw(goldpath_dir: str | Path | None = None) -> dict[str, Any]:
    """Load the ScienceWorld gold-paths JSON once and cache it (442 MB)."""
    global _GOLD_CACHE
    if _GOLD_CACHE is not None:
        return _GOLD_CACHE
    path = _find_goldpath_json(goldpath_dir)
    with open(path) as f:
        _GOLD_CACHE = json.load(f)
    return _GOLD_CACHE


# Key-action extraction

def _as_score(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() == "true"
    return False


def _extract_key_actions(path: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the subset of gold steps where the episode score changed.

    A "key action" is one whose `score` differs from the previous step's score
    (meaningful per the env's reward function) or that completes the episode.
    We keep just `{action, score, is_completed}` to stay compact; full
    observation text would balloon file size.

    Note: the gold-paths JSON stores `score` and `isCompleted` as STRINGS
    (e.g. "0.0", "false"), not Python numbers/booleans, so we coerce.
    """
    if not path:
        return []
    prev_score = 0.0
    keys: list[dict[str, Any]] = []
    for step in path:
        s = _as_score(step.get("score", 0))
        done = _as_bool(step.get("isCompleted"))
        if s != prev_score or done:
            keys.append({
                "action": step.get("action") or "",
                "score": s,
                "is_completed": done,
            })
        prev_score = s
    if not keys:  # edge case: no scoring steps - fall back to final step.
        last = path[-1]
        keys.append({
            "action": last.get("action") or "",
            "score": _as_score(last.get("score", 0)),
            "is_completed": _as_bool(last.get("isCompleted")),
        })
    return keys


def _initial_observation(path: list[dict[str, Any]]) -> str:
    """Return the first observation in the gold path (usually the `look around`
    output), which serves as the initial scene the agent gets to see."""
    if not path:
        return ""
    return str(path[0].get("observation") or "")[:2000]


# Instance conversion

def _render_prompt(task_description: str, initial_obs: str) -> str:
    parts = [
        "You are solving a ScienceWorld task in the offline planning variant "
        "(no live environment available). You must produce a single numbered "
        "plan of text-game actions that would solve the task. Grading compares "
        "your plan against the reference gold trajectory, with emphasis on "
        "covering the score-changing key actions.\n",
        f"## Task\n{task_description.strip()}\n",
        f"## Initial scene (what `look around` would return)\n{initial_obs.strip() or '(not provided)'}\n",
        "## Action grammar\n" + _ACTION_GRAMMAR_HINT.rstrip() + "\n",
        (
            "## Output format\n"
            "Respond with EXACTLY a numbered plan, one action per line, and nothing else:\n"
            "```\n"
            "1. <action>\n"
            "2. <action>\n"
            "...\n"
            "N. <action>\n"
            "```\n"
            "Use the action grammar literally. Prefer short, concrete actions. "
            "Include measurement/focus/answer steps required by the task. "
            "Do NOT add prose commentary."
        ),
    ]
    return "\n".join(parts)


def convert_goldpath_sequence(
    task_id: str,
    task_name: str,
    seq: dict[str, Any],
) -> dict[str, Any]:
    """Convert one gold-sequence record into a SkillGen TaskInstance dict."""
    variation_idx = seq.get("variationIdx")
    fold = (seq.get("fold") or "").lower() or "unknown"
    task_description = (seq.get("taskDescription") or "").strip()
    gold_path = seq.get("path") or []
    initial_obs = _initial_observation(gold_path)
    key_actions = _extract_key_actions(gold_path)
    gold_actions_full = [step.get("action") or "" for step in gold_path]

    instance_id = f"sw_t{task_id}_v{variation_idx}_{fold}"
    prompt = _render_prompt(task_description, initial_obs)

    # Ground-truth string for SkillGen bookkeeping (short, human-readable).
    gt_summary = f"Task: {task_description}\nKey actions: " + " -> ".join(
        (ka["action"] or "") for ka in key_actions[:15]
    )

    return {
        "instance_id": instance_id,
        "input": prompt,
        "ground_truth": gt_summary,
        "metadata": {
            "benchmark": "scienceworld",
            "task_id": str(task_id),
            "task_name": task_name,
            "variation_idx": variation_idx,
            "fold": fold,
            "task_description": task_description,
            "initial_observation": initial_obs,
            "key_actions": key_actions,
            "gold_actions": gold_actions_full,
            "gold_path_length": len(gold_path),
        },
    }


def iter_goldpath_sequences(
    *,
    folds: Iterable[str] | None = ("train", "test"),
    goldpath_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Flatten gold paths into a list of converted TaskInstance dicts.

    `folds`: which subsets of {train, dev, test} to include. Default is
    train+test (disjoint by construction via ScienceWorld's own split).
    """
    data = load_goldpaths_raw(goldpath_dir=goldpath_dir)
    wanted = set((f or "").lower() for f in (folds or ())) or {"train", "test"}
    out: list[dict[str, Any]] = []
    for task_id, task_data in data.items():
        task_name = task_data.get("taskName") or f"task-{task_id}"
        for seq in task_data.get("goldActionSequences") or []:
            fold = (seq.get("fold") or "").lower()
            if fold not in wanted:
                continue
            out.append(convert_goldpath_sequence(task_id, task_name, seq))
    return out


__all__ = [
    "load_goldpaths_raw",
    "iter_goldpath_sequences",
    "convert_goldpath_sequence",
]
