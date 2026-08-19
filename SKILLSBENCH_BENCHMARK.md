# Released SkillGen on SkillsBench

This branch evaluates the released SkillGen method on SkillsBench without
adding the positive-example retrieval or Skill-repair extensions explored in
our separate method-development work.

## Experimental question

SkillsBench v1.1 contains 87 heterogeneous task packages. We process every
task independently, as requested for the benchmarking study. A task supplies
one fixed prompt and environment, so repeated executions are stochastic
rollout replicas of that task—not different task instances. The experiment
therefore measures task-specific experience reuse and sparse-trajectory
robustness; it does not establish cross-task generalization.

For each task:

1. Pre-declare fixed construction and sealed-test rollout IDs.
2. Hide the task's curated skills and collect no-skill construction rollouts.
3. Run the released SkillGen pipeline unchanged: failure/success induction,
   nearest success contrast, generation, and summary/feedback refinement.
4. Do not add rollouts adaptively to obtain both classes; do not pool other
   tasks; do not manually create examples; do not inject paired exemplars into
   later refinement rounds.
5. Evaluate the selected/deprecated result on fresh sealed-test rollout IDs.
6. Optionally evaluate the official curated SkillsBench skill as a separate
   reference ceiling. It must never enter SkillGen construction.

Because the official implementation reuses its construction baselines for
internal verification, `construction.json` is not an induction/verification
task split. `sealed_test.json` is the genuinely fresh external evaluation.

## Pinned runtime

- SkillGen: upstream commit `3c4537bb12ac287ceb1b5d410b491206089fdcb7`
- SkillsBench: tag `v1.1`, commit
  `b63b7b2850226b6aa4fb5929a8c1ac7bc4d9a6af`
- BenchFlow: `0.6.7`
- Python: 3.12+
- Linux/WSL2: required by the pinned BenchFlow 0.6.7 runtime (`fcntl` is
  imported by its rollout implementation; native Windows startup fails)
- Docker Engine/daemon: required for local `--sandbox docker`

Run every step below inside the same Linux/WSL2 distribution and checkout.
Do not prepare JSON paths with native Windows Python and then consume them in
WSL. If the checkout is relocated, set `SKILLSBENCH_ROOT` to the pinned
SkillsBench repository root and `SKILLSBENCH_JOBS_ROOT` to an absolute path in
the current runtime before preflight and execution.

Install BenchFlow inside Linux/WSL2:

```bash
uv tool install "benchflow==0.6.7"
bench --version
docker info
```

Clone the pinned task source outside this repository:

```bash
git clone --depth 1 --branch v1.1 https://github.com/benchflow-ai/skillsbench.git external/skillsbench-v1.1
```

## Prepare one task

The counts are deliberately explicit: sampling until both successes and
failures appear would hide the low-data failure mode we want to measure.

```bash
python3 scripts/prepare_skillsbench.py \
  external/skillsbench-v1.1/tasks/edit-pdf \
  --construction-rollouts 10 \
  --test-rollouts 10 \
  --agent codex \
  --sandbox docker
```

This writes:

```text
data/skillsbench/edit-pdf/
  construction.json
  sealed_test.json
  protocol_manifest.json
```

To freeze the complete pinned suite with the same pre-declared counts while
keeping every task in a separate directory:

```bash
python3 scripts/prepare_skillsbench_suite.py \
  external/skillsbench-v1.1/tasks \
  --construction-rollouts 10 \
  --test-rollouts 10 \
  --agent codex \
  --sandbox docker
```

Use repeated `--include <task-id>` flags for a pre-declared pilot subset. Do
not choose tasks after observing SkillGen outcomes.

Every repeated rollout has a unique `instance_id`; otherwise the released
SkillGen implementation collapses replicas by ID during verification. The
manifest states explicitly that these IDs represent same-task stochastic
replicas and that BenchFlow's ordinary CLI provides no common random seed.

Run the offline preflight before any paid request:

```bash
python3 scripts/check_skillsbench_environment.py \
  data/skillsbench/edit-pdf/construction.json
```

It checks the task digest, pinned BenchFlow version, Docker daemon, and unique
IDs. It does not read API keys or call a model.

## Run construction and sealed evaluation

First ensure the model strings in `config.skillsbench.yaml` are compatible
with the group-standard BenchFlow agent/provider. Keep the same inference
model for baseline and verification; otherwise model capability is confounded
with the skill treatment.

First print and inspect a no-API plan (the default):

```bash
python3 scripts/run_skillsbench_task.py \
  --task-id edit-pdf \
  --construction-dataset data/skillsbench/edit-pdf/construction.json \
  --sealed-test-dataset data/skillsbench/edit-pdf/sealed_test.json \
  --config config.skillsbench.yaml \
  --task-package external/skillsbench-v1.1/tasks/edit-pdf \
  --run-root artifacts/skillsbench/task_runs
```

After the group fixes the agent/model/counts and approves the request budget,
repeat the exact command with `--execute`. This is the only flag that opens
the construction and sealed-test model-call path. The task runner writes an
atomic protocol-hashed `status.json`, reuses completed sealed trajectories on
resume, and never feeds sealed results back into refinement.

If a paid stage fails or the process stops while request persistence is
uncertain, ordinary `--execute` fails closed. Inspect `status.json` and the
BenchFlow/run artifacts first; only then add `--retry-paid` to authorize
possibly repeated calls. This guards the unavoidable crash window between a
provider response and the local atomic checkpoint.

For a manual audit, the equivalent released entry points remain `main.py` and
`eval_skill.py`; add `--keep-blank` to the latter for the verifier-primary
SkillsBench summary.

`--keep-blank` is required for the verifier-primary SkillsBench result because
a valid file-producing rollout can have an empty final chat message. It does
not change the rollout or verifier result; it only prevents the upstream
sealed-evaluation summarizer from deleting an already-scored pair. From the
same saved trajectories, also report the upstream default blank-DROP summary
offline as a sensitivity analysis; never rerun paid trajectories for it.

The released construction pipeline can return no Skill when all construction
rollouts pass. A suite orchestrator must still run the sealed baseline once
and report the deployed intervention as empty (`skill == baseline`, repair=0,
regression=0, net gain=0, method status `not_applicable_no_failure`). Do not
invent a placeholder Skill or skip such tasks from the all-task panel.

Use the actual `eval_skill.py --help` flags above; the upstream README's older
`--skill-path/--test-data` example does not match the released CLI.

## Required reporting

Report all 87 tasks, not only tasks where SkillGen can form positive/negative
pairs. The applicability funnel should include:

- runnable task and valid verifier;
- baseline rollout completion;
- at least one failure signal;
- both successes and failures;
- at least one same-type contrastive pair;
- candidate generated;
- verification gate active;
- sealed test completed.

For each task, retain the number of successes/failures, unique trajectory
ratio, cluster counts, contrastive-pair count, validation net gain, gate
status, sealed baseline/skill pass rates, repair, regression, and cost. Treat
the task—not an individual repeated rollout—as the independent unit in
aggregate statistics.

Use two result panels:

1. **All-task/intention-to-evaluate:** includes no-failure, no-positive,
   deprecated, and runtime-inapplicable outcomes.
2. **Applicable subset:** tasks where the released method produced and
   activated a skill, clearly labelled as conditional analysis.

Do not convert container, verifier, API, or malformed-artifact errors into
negative examples. The adapter stops on these conditions and keeps the
BenchFlow artifacts for diagnosis.

## Released-code audit notes

These are recorded rather than silently repaired in the primary benchmark:

- The released pipeline performs induction and internal verification over the
  same baseline-evaluated instance pool, although the paper describes
  disjoint construction subsets.
- Its `best_candidate` points to a candidate object that refinement mutates in
  place, so the saved body may reflect a later round rather than a true
  snapshot of the best round.

If the group later approves paper-faithful bug-fix variants, report them as
separate variants; do not mix them into the released-implementation result.
