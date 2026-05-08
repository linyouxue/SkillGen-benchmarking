"""Prompts for skill routing and selection."""

# Per-instance skill-application gate

SKILL_ROUTER_SYSTEM = (
    "You are a cautious skill-routing gate. Given one task and ONE candidate "
    "skill, decide whether injecting this skill's instructions is likely to "
    "help the agent solve this specific task. Be conservative: if the skill "
    "does not clearly match the task's requirements, prefer NOT to apply it. "
    "Applying an unrelated skill typically causes the agent to regress on "
    "tasks it would otherwise solve correctly."
)

SKILL_ROUTER_PROMPT = """\
## Task
{task_input}

## Candidate Skill
- ID: {skill_id}
- Contextual abstract: {contextual_abstract}
- Routing summary:
{routing_summary}

## Decision rules
1. Read the task and identify what specific capability it requires.
2. Compare with the skill's contextual abstract and any "Failure lessons" anti-triggers.
3. Apply the skill ONLY if it clearly addresses the task's requirement.
4. When uncertain, choose "no". A false-positive (applying an unrelated skill) \
typically causes regression; a false-negative just leaves the agent to solve \
the task on its own.

Reply with JSON only:
{{"apply": true_or_false, "reason": "<one short sentence>"}}
"""


# Multi-skill selection (used by eval_repo.py for repo-scale eval)

SKILL_SELECT_SYSTEM = (
    "You are a cautious skill selector. Given one task and several candidate "
    "skills retrieved from a skill repository, decide which (if any) is the "
    "best fit. When no skill clearly matches, return 'none'."
)

SKILL_SELECT_PROMPT = """\
## Task
{task_input}

## Retrieved Candidate Skills
{skills_list}

Select the single most relevant skill for this task. If none of the skills \
clearly apply, reply with `selected_skill_id = "none"`. Applying an unrelated \
skill is much worse than skipping, so be conservative.

Reply with JSON only:
{{"selected_skill_id": "<id or 'none'>", "reason": "<one short sentence>"}}
"""
