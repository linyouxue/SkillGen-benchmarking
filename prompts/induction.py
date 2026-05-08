"""Prompts: contextual abstract, cluster patterns, and contrastive pair analysis."""

# Contextual abstract

CONTEXTUAL_ABSTRACT_SYSTEM = """\
You are an expert task analyst. Given a sample of instances from a dataset, \
you describe - in 3-6 sentences - what this task / dataset is about at the \
capability level. The description should help a model that has never seen the \
dataset understand what kind of reasoning or tooling is required.
"""

CONTEXTUAL_ABSTRACT_PROMPT = """\
## Task Name
{task_name}

## Task Type
{task_type}

## Sample Instances (up to {n} shown)
{samples}

Write a contextual abstract of this task. Cover:
- What the agent is ultimately asked to produce (answer type, format, etc.)
- What domain knowledge or reasoning skills are typically required
- What kinds of inputs vary across instances
- The overall "shape" of the problem - e.g. "short math proofs with integer \
answers", "multi-step tool-use customer-support dialogues", "competitive \
programming problems with hidden test cases".

Do NOT enumerate specific instances. Do NOT copy wording from the samples. \
Produce a self-contained paragraph (3-6 sentences) that a downstream agent \
can read instead of the raw dataset.

Respond in JSON:
{{
  "contextual_abstract": "<3-6 sentence abstract>"
}}
"""


# Cluster-level pattern synthesis

FAILURE_PATTERN_SYSTEM = """\
You are a failure-analysis agent. Given several failure summaries drawn from the \
SAME cluster, you distil them into one actionable failure pattern the downstream \
skill-writer can guard against.
"""

FAILURE_PATTERN_PROMPT = """\
## Cluster of Failure Summaries
{summaries}

Describe the SHARED failure pattern across these cases. Write 4-8 sentences that:
1. Name the root-cause capability gap (one phrase).
2. Explain HOW the agent goes wrong (typical mis-step sequence).
3. List at least 2 observable signals that identify a task as being in this pattern.
4. State the concrete fix / behavior the agent should have exhibited instead.

Do not fixate on dataset-specific tokens; describe the pattern as a transferable \
capability gap.

Respond in JSON:
{{
  "pattern_name": "<short noun phrase>",
  "root_cause": "<1-2 sentences>",
  "typical_mistake_sequence": "<1-2 sentences>",
  "observable_signals": ["<signal 1>", "<signal 2>"],
  "correct_behavior": "<1-2 sentences>"
}}
"""


SUCCESS_PATTERN_SYSTEM = """\
You are a success-analysis agent. Given several success summaries drawn from the \
SAME cluster, you distil them into one reusable solving technique.
"""

SUCCESS_PATTERN_PROMPT = """\
## Cluster of Success Summaries
{summaries}

Describe the SHARED solving technique across these cases. Write 4-8 sentences that:
1. Name the technique (one phrase).
2. Describe the concrete procedure / decision sequence the agent used.
3. List at least 2 observable signals that tell you a task should use this technique.
4. Note any intermediate checks or sanity verifications that made the solution robust.

Respond in JSON:
{{
  "technique_name": "<short noun phrase>",
  "procedure": "<2-3 sentences describing the approach>",
  "observable_signals": ["<signal 1>", "<signal 2>"],
  "robustness_checks": "<1-2 sentences>"
}}
"""


# Contrastive pairs (failure vs nearest success)

CONTRASTIVE_SYSTEM = """\
You are a contrastive-analysis agent. Given ONE failing trajectory and ONE \
successful trajectory, you decide whether they target the SAME TYPE of task \
(i.e. the same underlying solving pattern is expected). Only when they are the \
same type do you explain what the success did that the failure missed.
"""

CONTRASTIVE_PROMPT = """\
## Failing Instance
### Task
{failure_input}
### Failure summary
{failure_summary}
### Failure trace (excerpt)
{failure_trace}

## Successful Instance
### Task
{success_input}
### Success summary
{success_summary}
### Success trace (excerpt)
{success_trace}

## Step 1 - Same-type judgement
Decide whether the two tasks belong to the same TYPE of problem - i.e. a solver \
that handles one should plausibly handle the other using the same high-level \
approach. Be strict: surface-level lexical overlap is not enough.

## Step 2 - Contrastive analysis (ONLY if same_type is true)
Write a concise analysis of what the successful trajectory did that the failing \
one missed. Focus on:
- The key step / decision the success got right
- The point at which the failure went off track
- The transferable rule that, if followed, would flip the failure to a success

If same_type is false, leave `analysis` as an empty string.

Respond in JSON:
{{
  "same_type": true | false,
  "same_type_reason": "<one short sentence>",
  "analysis": "<3-5 sentences if same_type; empty string otherwise>"
}}
"""
