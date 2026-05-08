"""Prompts: failure and success trajectory summarisation."""

SUMMARISE_FAILURE_PROMPT = """\
Below is an agent's reasoning trace for a task it answered incorrectly.

## Task Input
{input}

## Agent Trace
{trace}

## Error Summary
{error_summary}

Write a concise (2-3 sentence) description of *why* the agent failed.
Focus on the root cause: a missing reasoning step, wrong tool use, misunderstood \
constraint, etc.

IMPORTANT - describe the failure as a capability / behavior gap, not as a lexical \
artifact of the user's message. Do NOT fixate on verbatim tokens, placeholders, \
or dataset-specific formatting. Abstract away surface notation to the underlying \
intent.
"""

SUMMARISE_SUCCESS_PROMPT = """\
Below is an agent's reasoning trace for a task it answered correctly.

## Task Input
{input}

## Agent Trace
{trace}

## Ground Truth (if known)
{ground_truth}

Write a concise (2-3 sentence) description of *how* the agent solved this task.
Focus on the key decision, structure, or technique that made the answer correct - \
not on the surface words of the task. Prefer verbs describing what the agent *did* \
(e.g. "enumerated cases by parity", "decomposed the problem into two subproblems \
and solved them independently", "verified intermediate result before committing").
"""
