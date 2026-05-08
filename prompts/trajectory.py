"""Prompts for the agent runner and LLM evaluator."""

EVALUATE_SYSTEM = "You are a strict but fair evaluator. Decide if the agent's output is correct."

EVALUATE_PROMPT = """\
## Task Input
{input}

## Ground Truth
{ground_truth}

## Agent Output
{output}

## Task Type
{task_type}

{rubric_section}

Judge whether the agent output is correct using the criteria for this task type:
- binary: the output must be factually correct and match the key content of the ground truth. \
  Minor wording differences are fine; focus on whether the agent identified the right answer.
- scored: rate the quality of the output from 0.0 to 1.0 relative to the ground truth.
- open_ended: decide if the output adequately addresses the task goal. \
  The output does NOT need to match the ground truth word-for-word - judge whether the \
  agent's plan/answer is reasonable, covers the key steps or concepts, and would plausibly \
  accomplish the stated goal.

Respond in JSON:
{{"pass": true/false, "score": <float 0-1>, "explanation": "<brief reason>"}}
"""

AGENT_SYSTEM = """\
You are a helpful agent. Solve the given task step by step.
Always provide a clear final answer at the end."""

AGENT_SYSTEM_WITH_TOOLS = """\
You are a capable agent equipped with tools. Solve the given task step by step.

Tool usage guidelines:
- Use `web_search` when you need current facts, documentation, or domain knowledge \
not covered by the skill guidance.
- Use skill helper functions (prefixed `skill_`) when they can validate, transform, \
or compute something relevant to the task - call them proactively, not as a last resort.
- You may call multiple tools in sequence; each result will be returned before you continue.
- After gathering information from tools, synthesise a final answer.

Always provide a clear, complete final answer at the end of your reasoning."""

AGENT_PROMPT = """\
{skill_section}{reference_section}
## Task
{input}

Think step by step, then provide your final answer.
"""

SKILL_SECTION_TEMPLATE = """\
# Skill (read before answering)
The guidance below was distilled from prior attempts on this kind of task. \
Follow it where it applies; if a rule clearly does not match the task, ignore \
that specific rule and solve the task on its own.

{skill_body}

---
"""
