"""Single-skill on-disk store.

The post-refactor pipeline produces exactly ONE skill per (task, dataset) and
therefore no longer needs a multi-skill repository with overlap/merge logic.
This module is the minimal replacement: save a CandidateSkill or SkillItem to
disk, load the active skill back, and (for back-compat with older callers)
expose a lookup helper that returns the single active skill.

On-disk layout
--------------
<skill_dir>/
    <skill_id>.json            # the SkillItem JSON
    <skill_id>_helpers.py      # generated iff the skill has `scripts`
    skill_analysis.json        # optional; copied from induction if requested
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from pathlib import Path

from models import CandidateSkill, SkillItem, SkillStatus


def _skill_path(skill_dir: Path, skill_id: str) -> Path:
    return skill_dir / f"{skill_id}.json"


def _helpers_path(skill_dir: Path, skill_id: str) -> Path:
    return skill_dir / f"{skill_id}_helpers.py"


def _write_helpers_py(skill: SkillItem, skill_dir: Path) -> None:
    """Write a standalone Python module with the skill's helper functions."""
    if not skill.scripts:
        return
    lines: list[str] = [
        '"""',
        f"Auto-generated helper functions for skill: {skill.skill_id}",
        f"Contextual abstract: {skill.contextual_abstract}",
        '"""',
        "",
    ]
    if skill.requirements:
        lines += [
            "# -- Required packages --",
            "# Install with:",
            f"#   pip install {' '.join(skill.requirements)}",
            "",
        ]
    lines.append("# -- Helper functions --")
    lines.append("")
    for func_src in skill.scripts:
        lines.append(func_src.strip())
        lines.append("")

    _helpers_path(skill_dir, skill.skill_id).write_text("\n".join(lines), encoding="utf-8")


def save_skill(skill: SkillItem, skill_dir: str | Path) -> Path:
    """Persist a SkillItem (and its helper scripts, if any) to disk."""
    out = Path(skill_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = _skill_path(out, skill.skill_id)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(asdict(skill), fh, indent=2, default=str)
    _write_helpers_py(skill, out)
    return path


def candidate_to_skill(candidate: CandidateSkill, *,
                       analysis_id: str | None = None,
                       dataset_id: str | None = None,
                       task_name: str | None = None) -> SkillItem:
    """Promote a CandidateSkill to a SkillItem (fresh UUID)."""
    return SkillItem(
        skill_id=str(uuid.uuid4()),
        body=candidate.body,
        contextual_abstract=candidate.contextual_abstract,
        scripts=list(candidate.scripts),
        requirements=list(candidate.requirements),
        references=list(candidate.references),
        reference_docs=list(getattr(candidate, "reference_docs", []) or []),
        status=SkillStatus.ACTIVE,
        analysis_id=analysis_id or candidate.analysis_id,
        dataset_id=dataset_id,
        task_name=task_name,
        token_count=len((candidate.body or "").split()),
    )


def finalize_skill(
    candidate: CandidateSkill,
    skill_dir: str | Path,
    *,
    analysis_id: str | None = None,
    dataset_id: str | None = None,
    task_name: str | None = None,
) -> SkillItem:
    """Convenience: promote a candidate and write it to disk."""
    skill = candidate_to_skill(
        candidate,
        analysis_id=analysis_id,
        dataset_id=dataset_id,
        task_name=task_name,
    )
    save_skill(skill, skill_dir)
    return skill


def list_skills(skill_dir: str | Path) -> list[SkillItem]:
    """Return all SkillItem records stored under `skill_dir`."""
    out = Path(skill_dir)
    if not out.exists():
        return []
    skills: list[SkillItem] = []
    for path in out.glob("*.json"):
        if path.name.startswith("_") or path.name in {"skill_analysis.json",
                                                      "skill_analysis_summary.json",
                                                      "token_usage.json"}:
            continue
        with path.open(encoding="utf-8") as fh:
            raw = json.load(fh)
        if not isinstance(raw, dict) or "skill_id" not in raw:
            # Defensive: skip any non-skill JSON (e.g. token_usage.json, logs,
            # aggregator outputs) that happens to live in the skill dir.
            continue
        skills.append(_skill_from_raw(raw))
    return skills


def _skill_from_raw(raw: dict) -> SkillItem:
    # Back-compat defaults for older skill JSONs
    raw.setdefault("requirements", [])
    raw.setdefault("references", [])
    raw.setdefault("scripts", [])
    raw.setdefault("reference_docs", [])   # progressive-disclosure docs (new field)
    raw.setdefault("contextual_abstract", raw.get("semantic_aim", "") or "")
    # ``json.dump(default=str)`` persists the enum as its value.  Rehydrate it
    # here so resumed runs have the same in-memory type as fresh runs.
    raw["status"] = SkillStatus(raw.get("status", SkillStatus.ACTIVE.value))
    raw.setdefault("behavioral_aim", "")
    raw.setdefault("semantic_aim", raw.get("contextual_abstract", ""))
    raw.pop("source_cluster_ids", None)  # field no longer exists
    # Drop any unknown fields defensively (schema evolves across runs).
    import dataclasses
    known = {f.name for f in dataclasses.fields(SkillItem)}
    raw = {k: v for k, v in raw.items() if k in known}
    return SkillItem(**raw)


def load_skill(skill_dir: str | Path, skill_id: str | None = None) -> SkillItem:
    """Load a skill from disk.

    If `skill_id` is given, load that specific file. Otherwise return the first
    ACTIVE skill. Raises `FileNotFoundError` if nothing is found.
    """
    out = Path(skill_dir)
    if not out.exists():
        raise FileNotFoundError(f"Skill dir does not exist: {out}")

    if skill_id:
        path = _skill_path(out, skill_id)
        if not path.exists():
            raise FileNotFoundError(f"Skill not found: {path}")
        with path.open(encoding="utf-8") as fh:
            return _skill_from_raw(json.load(fh))

    skills = list_skills(out)
    if not skills:
        raise FileNotFoundError(f"No skill JSON files under: {out}")
    active = [s for s in skills if s.status == SkillStatus.ACTIVE]
    return (active or skills)[0]


def primary_skill(skill_dir: str | Path) -> SkillItem | None:
    """Return the primary (active) skill, or None if the dir has none."""
    try:
        return load_skill(skill_dir)
    except FileNotFoundError:
        return None
