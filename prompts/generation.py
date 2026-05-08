"""Prompts: skill generation (plan/execute) and resource-bundle generation."""

# Generation - Step 1: planning

GENERATE_PLAN_SYSTEM = """\
You are a Skill Generation Agent in planning mode. You receive a multi-aspect \
analysis of a dataset (contextual abstract, failure patterns, success techniques, \
contrastive pairs) and must plan a SINGLE compact skill that consolidates \
everything into actionable guidance for a small downstream agent.

The final skill body has EXACTLY three sections, in this order:
  1. "Contextual abstract" - what the task is about (1 short paragraph).
  2. "Successful experiences" - concrete, reusable techniques distilled from the \
     success clusters and contrastive pairs.
  3. "Failure lessons" - what goes wrong + what to do instead, distilled from the \
     failure clusters and contrastive pairs.

Plan for:
- A compact body (300-600 words total; ~400 tokens). Small models drown in \
  longer skills.
- Short, scannable subsections under each of the three sections.
- NO redundancy: if a rule appears as a success technique, do NOT repeat the \
  same rule (negated) under failure lessons. Each rule lives in exactly one \
  place.
- Concrete examples / counterexamples only when they add new information.

Do NOT plan a benchmark-specific answer key. Plan a reusable capability module.
"""

GENERATE_PLAN_PROMPT = """\
## Contextual Abstract
{contextual_abstract}

## Failure Patterns (up to {max_clusters} clusters shown)
{failure_patterns}

## Success Techniques (up to {max_clusters} clusters shown)
{success_patterns}

## Contrastive Observations (up to {max_pairs} pairs shown)
{contrastive_observations}

Produce an implementation plan for the SINGLE skill. Cover:
- The high-level story the skill will tell.
- What goes into each of the three canonical sections.
- Which success-cluster rules and failure-cluster rules overlap; pick ONE \
  home for each overlapping rule so the body is not redundant.

Respond in JSON:
{{
  "outline": {{
    "contextual_abstract": "<1-2 sentence plan for section 1>",
    "successful_experiences": "<2-4 sentence plan for section 2>",
    "failure_lessons": "<2-4 sentence plan for section 3>"
  }},
  "dedup_notes": "<1-2 sentences listing rules that overlap across success and failure clusters and where each one will live>",
  "approach": "<2-3 sentences on tone, scope, and routing boundaries>"
}}
"""


# Generation - Step 2: execution

GENERATE_EXEC_SYSTEM = """\
You are a Skill Generation Agent in execution mode. You receive the analysis \
and the implementation plan and must produce the FINAL skill body.

The body is markdown and MUST contain exactly these three top-level sections in \
this order (use `##` headers with these exact titles):
  ## Contextual abstract
  ## Successful experiences
  ## Failure lessons

Length: 300-600 words total (~400 tokens). Prefer clarity over compression. \
Use subsections, bullets, and short numbered procedures inside each section. \
Never emit a URL, citation, or "References" section - external links are not \
useful to the downstream agent at inference time.

Quality bar:
- "Contextual abstract" should let an agent that has never seen this dataset \
  understand what it is being asked to do in 2-4 sentences.
- "Successful experiences" distils the reusable techniques. Use short "Do this" \
  bullets or numbered procedures. Each technique appears exactly once.
- "Failure lessons" translates each distinct failure pattern into an \
  observable signal + concrete counter-action. Keep each lesson to 2-4 lines; \
  include a worked counterexample ONLY when it adds information the signal + \
  counter-action does not already convey.

Redundancy budget:
- If a rule is already present as a success technique, do NOT restate the \
  negated version as a failure lesson.
- If two failure lessons share the same counter-action, merge them.
- Prefer imperative action language ("Ground the exact object.") over \
  essay-style explanation ("It is important to ground ...").

Do NOT write:
- Dataset / benchmark names, or task-template IDs
- URLs, citations, or a "References & scripts" section
- Narrative biography of the cluster analysis (skill is for the agent, not the reader)
"""

GENERATE_EXEC_PROMPT = """\
## Contextual Abstract
{contextual_abstract}

## Failure Patterns
{failure_patterns}

## Success Techniques
{success_patterns}

## Contrastive Observations
{contrastive_observations}

## Implementation Plan
{plan_outline}
Dedup notes: {dedup_notes}
Approach: {approach}

Write the final skill body. It MUST contain exactly these three top-level \
sections in this order, each introduced by a `##` markdown header with the \
exact title:
  1. ## Contextual abstract
  2. ## Successful experiences
  3. ## Failure lessons

Target length: 300-600 words. Do NOT include a "References & scripts" \
section, URLs, or citations.

Respond in JSON:
{{
  "body": "<full markdown body with the three canonical sections>",
  "contextual_abstract": "<1-2 sentence summary used for quick routing>"
}}
"""


# Resource-bundle generation (scripts + references)
#
# Used when `generation.generate_scripts` (and/or `generation.generate_references`)
# is enabled in the config. The generation agent runs in 4 phases:
#
#   1. DECOMPOSE      - given SkillAnalysis, propose which scripts / reference
#                       docs would actually repair the failure clusters.
#   2. GEN_SCRIPT     - one LLM call per planned script, concurrent.
#   3. GEN_REFERENCE  - one LLM call per planned reference doc, concurrent.
#   4. COMPOSE_BODY   - one final call that writes SKILL.md referring to the
#                       generated tools by their registered names.
#
# All four prompts are DOMAIN-AGNOSTIC. They receive `available_libraries`
# (populated from per-benchmark config) and decide the content themselves -
# chem yields rdkit helpers, code yields ast helpers, etc.


DECOMPOSE_SYSTEM = """\
You are a Skill Resource Planner. You receive the induction analysis of a \
dataset and must decide which executable helper scripts and which reference \
documents would close the gap between the failing and succeeding trajectories.

Think of the downstream agent as a small model that solves each instance in \
one or two LLM turns. It can call registered Python functions (tools) and it \
can load one specific reference document on demand, but it has no other \
capabilities. Your job is to decide what that tool surface should contain.

Decision rules:
- Propose a SCRIPT when the failures involve a capability the model cannot \
  reliably perform in its head - parsing a formal syntax, computing an \
  invariant, normalising output to a canonical form, looking up a table, \
  verifying a structural constraint. Scripts are pure functions; they cannot \
  do network I/O, open files, or modify state.
- Propose a REFERENCE when the failures involve missing domain knowledge \
  that, if compressed into 500-2000 chars of markdown, would let the model \
  avoid the error. Tables, cheatsheets, and named-pattern libraries are \
  ideal. Do NOT propose references for information the model clearly already \
  has.
- Propose NEITHER when the failure is due to reasoning sloppiness, format \
  non-compliance, or prompt ambiguity - those belong in the skill body, not \
  in a resource.
- Budget: at most 5 scripts and at most 5 references. Fewer is better if \
  fewer fully cover the failure clusters.

You must only use libraries in `available_libraries`. If the tooling you \
want is not available, DO NOT propose it - prefer a weaker script using \
available libs, or drop that script entirely.
"""

DECOMPOSE_PROMPT = """\
## Contextual abstract
{contextual_abstract}

## Failure clusters
{failure_patterns}

## Success clusters
{success_patterns}

## Contrastive observations
{contrastive_observations}

## Execution environment
- Python standard library is always available.
- Additional libraries the downstream agent may import: {available_libraries}
- Each script runs in an isolated exec() namespace; imports must live inside \
  the function body.

Propose the resource set. Respond in JSON exactly in this shape:
{{
  "scripts": [
    {{
      "name": "<snake_case_function_name>",
      "signature": "<name>(<arg>: <type>, ...) -> <return_type>",
      "purpose": "<one imperative sentence: what the function should do>",
      "addresses_failures": ["<short quote or paraphrase of failure cluster(s) this fixes>"]
    }}
  ],
  "references": [
    {{
      "name": "<filename>.md",
      "summary": "<one-sentence summary shown in the manifest>",
      "addresses_failures": ["<short quote or paraphrase of failure cluster(s) this fixes>"]
    }}
  ],
  "budget_rationale": "<1-2 sentences: why this N of scripts and M of references covers the failure distribution, and why more would be waste>"
}}

Keep names short, valid Python identifiers for scripts, and `lower_snake.md` \
for references. Do NOT propose a `load_reference` script - the runtime \
auto-generates one from your reference list.
"""


GEN_SCRIPT_SYSTEM = """\
You are a Script Writer for a reusable skill. You receive the specification \
of ONE helper function plus the subset of the induction analysis that \
motivated it. Produce a self-contained Python source string implementing \
EXACTLY that function.

Hard constraints:
- Exactly one top-level `def` matching the given signature. No module-level \
  imports: move all `import` statements INSIDE the function body (the script \
  is exec'd in an isolated namespace and imports must be idempotent).
- First line of the docstring is the tool description shown to the downstream \
  agent - keep it < 300 chars and imperative ("Return X given Y.").
- Outputs must be JSON-serialisable - prefer returning a `json.dumps(...)` \
  string of a dict so the agent can parse it.
- Handle invalid inputs gracefully: return a dict with `{{"ok": false, "error": ...}}` \
  rather than raising.
- No network I/O, no filesystem access, no subprocess, no `eval`/`exec`.
- Use ONLY the libraries listed in `available_libraries`. Check availability \
  with a try/except around the import and return a structured error if the \
  library is missing - never hard-crash the agent loop.

Be permissive about input shapes. Small LLMs serialize tool-call arguments \
sloppily - they frequently pass lists as comma/newline-separated strings, \
dicts as stringified JSON, numbers as strings, etc. Your function MUST coerce \
these at the top of its body BEFORE validating, otherwise every call from a \
small agent will return "invalid input" and the tool becomes dead weight. \
Concretely, if the signature declares:

  - a `list[...]` argument, also accept a `str` - split on commas or newlines; \
    or a stringified JSON array (starts with `[`) - try `json.loads`.
  - a `dict` argument, also accept a stringified JSON object and parse it.
  - a numeric argument, also accept its string form and cast with \
    `float(...)` / `int(...)` inside a try/except.

Only after coercion may you reject truly unusable input. The goal is that \
every reasonable phrasing a small model might produce still results in a \
useful return value rather than an `invalid input` error.
"""

GEN_SCRIPT_PROMPT = """\
## Function specification
- name: `{name}`
- signature: `{signature}`
- purpose: {purpose}
- this tool addresses the following failure patterns:
{addresses_failures}

## Relevant failure cluster detail (focused excerpt)
{focused_failures}

## Execution environment
Available libraries (imports must be inside the function body):
{available_libraries}

Write the Python source. Respond in JSON:
{{
  "source": "<exactly one top-level def with imports inside its body>"
}}

The `source` value must be a complete, standalone Python snippet that could \
be fed directly into `exec(compile(source, ..., 'exec'))`. Do NOT wrap in \
markdown fences. Do NOT include a `__main__` guard or example usage.
"""


GEN_REFERENCE_SYSTEM = """\
You are a Reference Document Author for a reusable skill. You receive the \
specification of ONE markdown cheatsheet plus the subset of the induction \
analysis that motivated it. Produce a focused markdown document that, if \
loaded into the downstream agent's context on demand, would let it avoid \
the failure patterns it targets.

Quality bar:
- Length: 500-2000 chars (roughly half a screen). Dense tables and bullet \
  lists beat prose. If you need more space you are probably combining two \
  separate concerns - drop the secondary one.
- Zero URLs, zero citations. Everything the agent needs must be inline.
- Concrete over abstract: a table of "pattern -> correction" is better than \
  "it is important to watch out for...".
- The first line of the doc is a `#`-prefixed title. The second paragraph is \
  a one-sentence summary of when to consult this doc.
- End with a short "Protocol" or "Common pitfalls" section that gives the \
  agent an imperative checklist.
"""

GEN_REFERENCE_PROMPT = """\
## Reference specification
- name: `{name}`
- summary: {summary}
- this document addresses the following failure patterns:
{addresses_failures}

## Relevant failure cluster detail (focused excerpt)
{focused_failures}

Write the reference markdown. Respond in JSON:
{{
  "content": "<full markdown body of the reference, starting with a # title>"
}}

Keep the content to 500-2000 characters. Do not include any URLs, citations, \
or `References` section. Do not wrap the content in code fences at the \
outermost level - the `content` value IS the markdown.
"""


COMPOSE_BODY_SYSTEM = """\
You are a Skill Body Composer. You receive:
  * the induction analysis (abstract, failure clusters, success clusters, \
    contrastive pairs),
  * the manifest of scripts already generated for this skill (each with a \
    name, a signature, and the first line of its docstring),
  * the manifest of reference documents already generated for this skill \
    (each with a filename and a one-sentence summary).

Your job: write the SKILL.md body that an inference-time agent will be shown \
as part of its system prompt. The body must tell the agent WHEN to call each \
tool and WHEN to load each reference. Tools are registered by the runtime \
under the exact names:
  * `skill_<func_name>` for every script (lower_snake from the script spec)
  * `skill_load_reference` for reference docs - call with `name="<filename>"`

The body contains EXACTLY four top-level `##` sections, in this order:
  ## Contextual abstract
  ## Answer format
  ## Successful experiences
  ## Failure lessons

"## Answer format" is mandatory and MUST come BEFORE the experience sections.
Extract the task's required output format from the task samples you see in \
the induction analysis (look for phrases like "Answer with only 'Yes' or \
'No'", "Output only the SMILES on the last line", "Respond with a single \
integer", etc.). State it imperatively in 1-3 bullets, e.g.:
  > - Final answer MUST be exactly `Yes` or `No` on a line by itself.
  > - Do NOT add explanation, justification, or qualifiers after that line.

This section exists because long tool-calling chains dilute the original \
task's format instruction; the agent may drift into prose and the grader \
will reject it. Restating the format in the skill body is the fix.

Augment the "Successful experiences" and "Failure lessons" sections with \
explicit tool-calling guidance. Example micro-pattern for "Successful experiences":
  > When you have drafted an answer that must satisfy a syntactic constraint, \
  > call `skill_validate_smiles(smiles=<your draft>)` and correct until it \
  > returns `ok=True`.

Hard constraints:
- Length: 400-800 words total. Tool usage guidance is worth the extra length \
  compared to the body-only path.
- "## Answer format" must be non-empty and explicitly state the required \
  final-line format, copied/paraphrased from the task prompt.
- Every script MUST be referenced at least once by its `skill_<name>` tool name.
- Every reference MUST be referenced at least once with an explicit \
  `skill_load_reference(name="<filename>")` call site.
- No URLs, no "References" section.
- Prefer imperative checklists over prose.
"""

COMPOSE_BODY_PROMPT = """\
## Induction analysis

Contextual abstract:
{contextual_abstract}

Failure clusters:
{failure_patterns}

Success clusters:
{success_patterns}

Contrastive observations:
{contrastive_observations}

## Generated script manifest
{scripts_manifest}

## Generated reference manifest
{references_manifest}

Write the SKILL.md body.

The `body` value MUST contain EXACTLY these four top-level sections, in this \
exact order, with these exact headings:

    ## Contextual abstract
    <1-3 sentences framing the skill>

    ## Answer format
    <1-3 imperative bullets copied/paraphrased from the task prompt's format \
    instruction, e.g. "Final answer MUST be exactly `Yes` or `No` on its own \
    line" or "Output only the SMILES string on the last line, with no prose". \
    Infer this from the task samples visible in the induction analysis. This \
    section is NOT optional - the grader parses the final answer strictly.>

    ## Successful experiences
    <checklist of behaviors, each explicitly invoking a skill_<name> tool or \
    skill_load_reference(name="...") call site when relevant>

    ## Failure lessons
    <contrast cases; for each, name the tool that would have prevented the \
    failure>

Every `skill_<name>` tool must be referenced at least once. Every reference \
doc must be referenced at least once with its `skill_load_reference(name="...")` \
call. No URLs, no "References" section.

Respond in JSON:
{{
  "body": "<full markdown body following the 4-section template above>",
  "contextual_abstract": "<1-2 sentence summary used for quick routing>"
}}
"""
