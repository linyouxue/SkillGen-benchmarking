"""SocialMaze adapter - Find the Spy / Review Decision Prediction / User
Profile Inference, unified behind one benchmark='socialmaze' tag plus a
`task` sub-field.

We cover 3 of the 6 SocialMaze tasks:

  * fts  - Find the Spy: 4-player word-description game; pick which
           player has the outlier word.
  * rdp  - Review Decision Prediction: given ICLR paper + reviews,
           predict Accept vs Reject.
  * upi  - User Profile Inference: given a user's product comments,
           infer (age_group, gender). Two flavours:
              - upi_persona : same-user comments (intra-user)
              - upi_entity  : same-product comments from a primary
                              demographic (inter-user).

Each `convert_*_item` returns a SkillGen-shaped dict (instance_id, input,
ground_truth, metadata). The metadata carries `benchmark="socialmaze"`
and `task=<task_name>` so `trajectory.py` can dispatch to the grader.

Pool generation for FTS and UPI is in `prepare_socialmaze.py` - this
module is pure loading + prompt rendering.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).parent
_SOCIALMAZE_ROOT = _REPO_ROOT / "external" / "social-maze"


# Find the Spy

_FTS_SYSTEM_PREAMBLE = (
    "You are a skilled player in a word description game. "
    "Your task is to identify which player is the 'spy' based on their descriptions.\n\n"
    "Game Rules:\n"
    "1. There are 4 players in the game.\n"
    "2. 3 players received the same word (normal players).\n"
    "3. 1 player received a different but related word (the spy).\n"
    "4. Each player describes their word without saying it directly.\n"
    "5. You need to determine who is the spy based on these descriptions.\n"
)


def _fts_user_prompt(scenario: dict[str, Any]) -> str:
    """Render the player-1-perspective prompt used by fts_eva."""
    player1_word = scenario["player_words"]["1"]
    rounds_blob = []
    for round_idx, round_data in enumerate(scenario.get("statements") or []):
        rounds_blob.append(f"Round {round_idx + 1}:")
        for stmt in round_data.get("statements", []):
            rounds_blob.append(stmt["statement"])
        rounds_blob.append("")  # blank line between rounds
    rounds_text = "\n".join(rounds_blob).rstrip()

    return (
        f"You are Player 1, and your word is: \"{player1_word}\".\n\n"
        "Analyze the descriptions carefully. Look for subtle differences "
        "that might reveal who has a different word.\n\n"
        f"{rounds_text}\n\n"
        "Based on these descriptions, which player is the spy (Player 1, 2, 3, or 4)?\n"
        "Explain your reasoning step by step, then provide your final answer "
        "in the format: 'Final Spy: Player X'"
    )


def convert_fts_item(scenario: dict[str, Any], idx: int | None = None) -> dict[str, Any]:
    scenario_id = scenario.get("scenario_id") or f"fts_{idx}"
    prompt = (
        _FTS_SYSTEM_PREAMBLE + "\n" + _fts_user_prompt(scenario)
    )
    return {
        "instance_id": scenario_id,
        "input": prompt,
        "ground_truth": str(scenario["spy_player"]),
        "metadata": {
            "benchmark": "socialmaze",
            "task": "fts",
            "scenario_id": scenario_id,
            "spy_player": str(scenario["spy_player"]),
            "normal_word": scenario.get("normal_word"),
            "spy_word": scenario.get("spy_word"),
            "num_rounds": scenario.get("num_rounds"),
        },
    }


# Review Decision Prediction

_RDP_SYSTEM_PREAMBLE = (
    "You are an expert reviewer for a prestigious academic conference. Your task "
    "is to evaluate a research paper and determine whether it should be accepted "
    "or rejected for publication.\n\n"
    "Important context:\n"
    "- You have access to the paper's title, abstract, reviewer comments, and "
    "author responses\n"
    "- The paper should be judged by the standards of a top-tier conference\n"
)


_RDP_MAX_REVIEW_CHARS = 6000   # ~=1.5k tokens per section
_RDP_MAX_REBUTTAL_CHARS = 4000


def _normalise_rdp_decision(raw: str) -> str:
    """Map the raw dataset decision (e.g. 'Accept (Poster)', 'Reject') to
    a canonical 'Accept' or 'Reject'. Returns '' for unknown decisions.
    """
    if not raw:
        return ""
    low = raw.lower()
    if any(t in low for t in ("accept", "spotlight", "poster", "oral")):
        return "Accept"
    if any(t in low for t in ("reject", "desk-reject", "declined")):
        return "Reject"
    return ""


def _rdp_user_prompt(item: dict[str, Any]) -> str:
    statements = item.get("statements") or []
    # statements is a list of {"roundN": text}. Rounds are positional:
    # round1 = paper info, round2 = reviewer comments, round3 = author response.
    def _round_text(rd: dict[str, Any]) -> str:
        if not rd:
            return ""
        _k = next(iter(rd.keys()))
        return str(rd.get(_k) or "")

    paper = _round_text(statements[0]) if len(statements) >= 1 else ""
    reviews = _round_text(statements[1]) if len(statements) >= 2 else ""
    rebuttal = _round_text(statements[2]) if len(statements) >= 3 else ""

    parts = ["Please analyse the following research paper and determine whether "
             "it should be accepted or rejected for publication at a top-tier "
             "conference.\n"]
    if paper:
        parts.append("## Paper Information\n\n" + paper.strip() + "\n")
    if reviews:
        snippet = reviews.strip()
        if len(snippet) > _RDP_MAX_REVIEW_CHARS:
            snippet = snippet[:_RDP_MAX_REVIEW_CHARS] + "\n...[reviewer comments truncated]..."
        parts.append("## Reviewer Comments\n\n" + snippet + "\n")
    if rebuttal:
        snippet = rebuttal.strip()
        if len(snippet) > _RDP_MAX_REBUTTAL_CHARS:
            snippet = snippet[:_RDP_MAX_REBUTTAL_CHARS] + "\n...[author response truncated]..."
        parts.append("## Author Response\n\n" + snippet + "\n")
    parts.append(
        "Based on all the information provided, carefully analyse whether this "
        "paper should be accepted or rejected for publication.\n\n"
        "First, provide your reasoning. Then on the last line write exactly:\n"
        "Final Decision: Accept\n"
        "OR\n"
        "Final Decision: Reject"
    )
    return "\n".join(parts)


def convert_rdp_item(item: dict[str, Any], idx: int | None = None) -> dict[str, Any] | None:
    gt = _normalise_rdp_decision(item.get("decision") or "")
    if not gt:
        return None  # drop items with unknown decisions
    instance_id = item.get("id") or f"rdp_{idx}"
    prompt = _RDP_SYSTEM_PREAMBLE + "\n" + _rdp_user_prompt(item)
    return {
        "instance_id": instance_id,
        "input": prompt,
        "ground_truth": gt,
        "metadata": {
            "benchmark": "socialmaze",
            "task": "rdp",
            "paper_id": item.get("id"),
            "source": item.get("source"),
            "raw_decision": item.get("decision"),
            "normalised_decision": gt,
        },
    }


# User Profile Inference

_UPI_SYSTEM_PERSONA = (
    "Your job is to analyse multiple reviews written by the same person and "
    "determine their likely age group and gender based on their writing style, "
    "interests, and perspectives.\n\n"
    "Focus on identifying the most likely demographic profile from the text "
    "patterns, interests, and perspectives in the comments."
)

_UPI_SYSTEM_ENTITY = (
    "Your job is to analyse multiple reviews for a product and determine the "
    "most likely demographics of the primary user group who wrote these reviews.\n\n"
    "Focus on identifying the MAJORITY demographic group based on text patterns "
    "and content of the reviews."
)

_UPI_OUTPUT_SPEC = (
    "First explain your reasoning, then on the final two lines write exactly:\n"
    "Age Group: [18-34 OR 35-54 OR 55+]\n"
    "Gender: [Male OR Female OR Non-binary]"
)


def _upi_persona_user_prompt(group: dict[str, Any]) -> str:
    lines = ["Reviews by the same user:\n"]
    for comment in group.get("comments", []):
        product = comment.get("product", "something")
        text = comment.get("comment", "")
        lines.append(f'On {product}: "{text}"')
    lines.append("")
    lines.append(
        "Analyse these reviews carefully. What are the likely demographic "
        "characteristics of this user?\n"
    )
    lines.append(_UPI_OUTPUT_SPEC)
    return "\n".join(lines)


def _upi_entity_user_prompt(scenario: dict[str, Any]) -> str:
    lines = [f"Product: {scenario.get('product_name', '(unknown)')}\n", "Reviews:\n"]
    for comment in scenario.get("comments", []):
        text = comment.get("comment", "")
        lines.append(f'"{text}"')
    lines.append("")
    lines.append(
        "Analyse these reviews carefully. What is the PRIMARY demographic "
        "group writing these reviews?\n"
    )
    lines.append(_UPI_OUTPUT_SPEC)
    return "\n".join(lines)


def convert_upi_persona_item(
    group: dict[str, Any], idx: int | None = None
) -> dict[str, Any]:
    group_id = group.get("group_id", idx)
    age = group["demographics"]["age_group"]
    gender = group["demographics"]["gender"]
    prompt = _UPI_SYSTEM_PERSONA + "\n\n" + _upi_persona_user_prompt(group)
    return {
        "instance_id": f"upi_persona_{group_id}",
        "input": prompt,
        "ground_truth": json.dumps({"age_group": age, "gender": gender}),
        "metadata": {
            "benchmark": "socialmaze",
            "task": "upi",
            "upi_variant": "persona",
            "group_id": group_id,
            "age_group": age,
            "gender": gender,
            "n_comments": len(group.get("comments") or []),
        },
    }


def convert_upi_entity_item(
    scenario_key: str, scenario: dict[str, Any]
) -> dict[str, Any]:
    primary = scenario.get("primary_user_group") or {}
    age = primary.get("primary_age_group") or primary.get("age_group")
    gender = primary.get("primary_gender") or primary.get("gender")
    prompt = _UPI_SYSTEM_ENTITY + "\n\n" + _upi_entity_user_prompt(scenario)
    return {
        "instance_id": f"upi_entity_{scenario_key}",
        "input": prompt,
        "ground_truth": json.dumps({"age_group": age, "gender": gender}),
        "metadata": {
            "benchmark": "socialmaze",
            "task": "upi",
            "upi_variant": "entity",
            "scenario_id": scenario_key,
            "product_name": scenario.get("product_name"),
            "age_group": age,
            "gender": gender,
            "n_comments": len(scenario.get("comments") or []),
        },
    }


# Raw loaders (shipped data)


def load_fts_shipped(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load FTS scenarios from the shipped fts_dataset_eval.json.

    Returns the list of raw scenario dicts (NOT the converted instances).
    """
    p = Path(path) if path else _SOCIALMAZE_ROOT / "find_the_spy" / "data" / "fts_dataset_eval.json"
    if not p.exists():
        return []
    return json.loads(p.read_text())


def load_rdp_shipped(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load RDP debate items from the shipped debate.json."""
    p = Path(path) if path else _SOCIALMAZE_ROOT / "review_decision_prediction" / "data" / "debate.json"
    if not p.exists():
        return []
    return json.loads(p.read_text())


def load_upi_persona_shipped(path: str | Path | None = None) -> list[dict[str, Any]]:
    p = Path(path) if path else _SOCIALMAZE_ROOT / "user_profile_inference" / "data" / "user_persona.json"
    if not p.exists():
        return []
    d = json.loads(p.read_text())
    return list(d.get("profile_groups") or [])


def load_upi_entity_shipped(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    p = Path(path) if path else _SOCIALMAZE_ROOT / "user_profile_inference" / "data" / "user_entity.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    return dict(d.get("scenarios") or {})


__all__ = [
    # FTS
    "convert_fts_item",
    "load_fts_shipped",
    # RDP
    "convert_rdp_item",
    "load_rdp_shipped",
    # UPI
    "convert_upi_persona_item",
    "convert_upi_entity_item",
    "load_upi_persona_shipped",
    "load_upi_entity_shipped",
]
