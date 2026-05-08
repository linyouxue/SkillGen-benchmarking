"""
Skill A/B evaluation: compare agent performance with vs. without a skill.

Usage examples
--------------
# Test the skill from the latest skill_repo run on 40 held-out instances:
python eval_skill.py \
    --skill-repo ./skill_repo/2026-04-15_17-52-53 \
    --dataset    data/alfworld/test.json \
    --n          40 \
    --seed       99

# Test two models (weak + strong) to see how the skill helps each:
python eval_skill.py \
    --skill-repo ./skill_repo/2026-04-15_17-52-53 \
    --dataset    data/alfworld/test.json \
    --models     "meta-llama/llama-3.2-3b-instruct" "openai/gpt-5.4-mini" \
    --n          40

# Test a specific skill by ID:
python eval_skill.py \
    --skill-repo ./skill_repo/2026-04-15_17-52-53 \
    --skill-id   2d88ca80-97f4-4b5d-a72a-54ac3d32462d \
    --dataset    data/alfworld/test.json \
    --n          30
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

import llm
from models import TaskInstance, TaskType, SkillItem, SkillStatus, Trajectory
from trajectory import collect_trajectories, AgentConfig
from artifacts import write_trajectories


# Data structures

@dataclass
class ConditionResult:
    model: str
    with_skill: bool
    skill_id: Optional[str]
    n_instances: int
    n_success: int
    success_rate: float
    latency_mean: float
    trajectories: list = field(default_factory=list, repr=False)


@dataclass
class PairedResult:
    model: str
    skill_id: str
    n_instances: int
    n_baseline_pass: int
    n_skill_pass: int
    repair: int        # baseline fail -> skill pass
    regression: int    # baseline pass -> skill fail
    both_pass: int
    both_fail: int
    repair_rate: float
    regression_rate: float
    net_gain: int
    baseline_acc: float
    skill_acc: float
    # blank-filter bookkeeping (only populated when drop_blank=True)
    n_paired_raw: int = 0          # pairs before blank filtering (both sides present)
    n_blank_baseline: int = 0      # pairs where baseline final_output is empty
    n_blank_skill: int = 0         # pairs where skill final_output is empty
    n_blank_either: int = 0        # pairs where at least one side is empty (= dropped)
    n_blank_both: int = 0          # pairs where both sides are empty


# Helpers

def load_dataset(path: str) -> tuple[list[TaskInstance], TaskType]:
    with open(path) as f:
        d = json.load(f)
    task_type = TaskType(d.get("task_type", "open_ended"))
    instances = [
        TaskInstance(
            instance_id=item["instance_id"],
            input=item["input"],
            ground_truth=item.get("ground_truth"),
            metadata=item.get("metadata", {}),
        )
        for item in d["instances"]
    ]
    return instances, task_type


def load_skill(repo_path: str, skill_id: Optional[str] = None) -> SkillItem:
    from skill_store import load_skill as _store_load
    return _store_load(repo_path, skill_id=skill_id)


def run_condition(
    instances: list[TaskInstance],
    task_type: TaskType,
    model: str,
    judge_model: str,
    skill: Optional[SkillItem],
    max_workers: int,
    enable_web_search: bool = False,
    execute_scripts: bool = False,
) -> ConditionResult:
    config = AgentConfig(
        model=model,
        judge_model=judge_model,
        temperature=0.0,
        # Tools are only meaningful when a skill bundle is present
        enable_web_search=enable_web_search and skill is not None,
        execute_scripts=execute_scripts and skill is not None,
    )
    tool_tag = ""
    if skill:
        parts = []
        if enable_web_search:
            parts.append("search")
        if execute_scripts:
            parts.append("exec")
        tool_tag = f" [{'+'.join(parts)}]" if parts else ""
    label = f"{'with skill' + tool_tag if skill else 'no skill'} | {model.split('/')[-1]}"
    trajs = collect_trajectories(
        instances, task_type,
        skill=skill,
        config=config,
        max_workers=max_workers,
        progress_desc=label,
    )
    n_success = sum(1 for t in trajs if t.success)
    latencies = [t.latency for t in trajs if t.latency is not None]
    return ConditionResult(
        model=model,
        with_skill=skill is not None,
        skill_id=skill.skill_id if skill else None,
        n_instances=len(trajs),
        n_success=n_success,
        success_rate=n_success / len(trajs) if trajs else 0.0,
        latency_mean=sum(latencies) / len(latencies) if latencies else 0.0,
        trajectories=trajs,
    )


def load_baseline_condition(
    path: str | Path,
    model: str,
    expected_instance_ids: set[str] | None = None,
) -> ConditionResult:
    """Load a previously-saved baseline trajectory JSONL and wrap it as a
    ``ConditionResult`` so downstream code can treat it identically to a
    freshly-produced baseline.

    Parameters
    ----------
    path:
        JSONL written by ``write_trajectories`` (one record per line).
    model:
        Expected inference model for these trajectories. We fail fast if the
        file was produced by a different model - pairing would be nonsense.
    expected_instance_ids:
        If provided, the loaded set must cover every instance the eval is
        about to run on; otherwise we raise. Prevents silently pairing a
        smaller baseline against a larger skill run.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Baseline trajectory file not found: {path}"
        )

    trajs: list[Trajectory] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            traj_model = (
                d.get("inference_model")
                or (d.get("agent_config") or {}).get("inference_model")
                or (d.get("agent_config") or {}).get("model")
            )
            if traj_model and traj_model != model:
                raise ValueError(
                    f"Baseline trajectories in {path} were produced by "
                    f"model={traj_model!r}, but eval is for model={model!r}. "
                    f"Reusing a mismatched baseline would corrupt paired stats."
                )
            trajs.append(
                Trajectory(
                    trajectory_id=d["trajectory_id"],
                    instance_id=d["instance_id"],
                    agent_config=d.get("agent_config", {}),
                    messages=d.get("messages", []),
                    final_output=d.get("final_output"),
                    success=d.get("success"),
                    score=d.get("score"),
                    error_summary=d.get("error_summary"),
                    token_usage=d.get("token_usage"),
                    latency=d.get("latency"),
                    timestamp=d.get("timestamp"),
                    metadata=d.get("metadata", {}),
                )
            )

    if expected_instance_ids is not None:
        got = {t.instance_id for t in trajs}
        missing = expected_instance_ids - got
        extra = got - expected_instance_ids
        if missing:
            raise ValueError(
                f"Loaded baseline covers {len(got)} instances but {len(missing)} "
                f"expected instance_ids are missing (first 5: "
                f"{sorted(missing)[:5]}). Shared baseline must match current "
                f"--dataset / --seed / --n exactly."
            )
        if extra:
            # Trim extras rather than fail - a superset is safe as long as we
            # only pair on the intersection.
            trajs = [t for t in trajs if t.instance_id in expected_instance_ids]

    n_success = sum(1 for t in trajs if t.success)
    latencies = [t.latency for t in trajs if t.latency is not None]
    return ConditionResult(
        model=model,
        with_skill=False,
        skill_id=None,
        n_instances=len(trajs),
        n_success=n_success,
        success_rate=n_success / len(trajs) if trajs else 0.0,
        latency_mean=sum(latencies) / len(latencies) if latencies else 0.0,
        trajectories=trajs,
    )


def _is_blank(traj) -> bool:
    """True if the trajectory has no usable final_output.

    Blank traj usually means one of:
      * API-level failure (router returned empty, network hiccup)
      * Model emitted only whitespace / EOS immediately

    We treat them as ungraded so they don't pollute the paired comparison.
    """
    fo = getattr(traj, "final_output", None) or ""
    return not fo.strip()


def paired_analysis(
    baseline_result: ConditionResult,
    skill_result: ConditionResult,
    drop_blank: bool = True,
) -> PairedResult:
    b_map = {t.instance_id: t for t in baseline_result.trajectories}
    s_map = {t.instance_id: t for t in skill_result.trajectories}

    repair = regression = both_pass = both_fail = 0
    paired_raw = 0
    n_blank_b = n_blank_s = n_blank_either = n_blank_both = 0

    for iid, bt in b_map.items():
        st = s_map.get(iid)
        if st is None:
            continue
        paired_raw += 1

        b_blank = _is_blank(bt)
        s_blank = _is_blank(st)
        if b_blank: n_blank_b += 1
        if s_blank: n_blank_s += 1
        if b_blank and s_blank: n_blank_both += 1
        if b_blank or s_blank:
            n_blank_either += 1
            if drop_blank:
                continue  # ungraded pair, skip from paired stats

        bp, sp = bool(bt.success), bool(st.success)
        if bp and sp:
            both_pass += 1
        elif not bp and not sp:
            both_fail += 1
        elif not bp and sp:
            repair += 1
        else:
            regression += 1

    paired = both_pass + both_fail + repair + regression
    n_baseline_pass = both_pass + regression
    n_skill_pass = both_pass + repair
    baseline_acc = n_baseline_pass / paired if paired else 0.0
    skill_acc = n_skill_pass / paired if paired else 0.0
    repair_rate = repair / (both_fail + repair) if (both_fail + repair) else 0.0
    regression_rate = regression / (both_pass + regression) if (both_pass + regression) else 0.0

    return PairedResult(
        model=baseline_result.model,
        skill_id=skill_result.skill_id or "",
        n_instances=paired,
        n_baseline_pass=n_baseline_pass,
        n_skill_pass=n_skill_pass,
        repair=repair,
        regression=regression,
        both_pass=both_pass,
        both_fail=both_fail,
        repair_rate=repair_rate,
        regression_rate=regression_rate,
        net_gain=repair - regression,
        baseline_acc=baseline_acc,
        skill_acc=skill_acc,
        n_paired_raw=paired_raw,
        n_blank_baseline=n_blank_b,
        n_blank_skill=n_blank_s,
        n_blank_either=n_blank_either,
        n_blank_both=n_blank_both,
    )


def print_summary(
    skill: SkillItem,
    results: list[tuple[ConditionResult, ConditionResult, PairedResult]],
) -> None:
    width = 72
    print("\n" + "=" * width)
    print("  SKILL A/B EVALUATION REPORT")
    print("=" * width)
    print(f"  Skill ID   : {skill.skill_id}")
    print(f"  Abstract   : {skill.contextual_abstract or skill.semantic_aim}")
    print(f"  Status     : {skill.status}")
    print("=" * width)

    for base_r, skill_r, paired in results:
        model_short = base_r.model.split("/")[-1]
        print(f"\n  Model: {base_r.model}")
        print(f"  {'':4s}{'Condition':<28} {'N':>4} {'Pass':>5} {'Acc':>7} {'Latency':>9}")
        print(f"  {'':4s}{'-'*52}")
        print(f"  {'':4s}{'Without skill':<28} {base_r.n_instances:>4} "
              f"{base_r.n_success:>5} {base_r.success_rate:>6.1%} "
              f"{base_r.latency_mean:>8.1f}s")
        print(f"  {'':4s}{'With skill':<28} {skill_r.n_instances:>4} "
              f"{skill_r.n_success:>5} {skill_r.success_rate:>6.1%} "
              f"{skill_r.latency_mean:>8.1f}s")

        delta = paired.skill_acc - paired.baseline_acc
        sign = "^" if delta > 0 else ("v" if delta < 0 else "-")
        print(f"\n  {'':4s}Accuracy delta : {sign} {abs(delta):.1%}  (baseline={paired.baseline_acc:.1%}, skill={paired.skill_acc:.1%})")
        print(f"  {'':4s}Repair     : {paired.repair}/{paired.both_fail + paired.repair} "
              f"  ({paired.repair_rate:.1%} of baseline failures fixed)")
        print(f"  {'':4s}Regression : {paired.regression}/{paired.both_pass + paired.regression} "
              f"  ({paired.regression_rate:.1%} of baseline successes broken)")
        print(f"  {'':4s}Net gain   : {paired.net_gain:+d} instances")

        # Blank-filter audit
        if paired.n_blank_either > 0:
            print(f"\n  {'':4s}Blank pairs dropped: {paired.n_blank_either}/{paired.n_paired_raw} "
                  f"(baseline-blank={paired.n_blank_baseline}, "
                  f"skill-blank={paired.n_blank_skill}, "
                  f"both-blank={paired.n_blank_both})")

        # Confusion matrix
        print(f"\n  {'':4s}Paired outcome matrix (N={paired.n_instances} after filter):")
        print(f"  {'':4s}{'':22s} {'Skill pass':>12} {'Skill fail':>12}")
        print(f"  {'':4s}{'Baseline pass':<22} {paired.both_pass:>12} {paired.regression:>12}")
        print(f"  {'':4s}{'Baseline fail':<22} {paired.repair:>12} {paired.both_fail:>12}")

    print("\n" + "=" * width + "\n")


# CLI

def parse_args():
    p = argparse.ArgumentParser(description="Skill A/B evaluation")
    p.add_argument("--skill-repo", required=True,
                   help="Path to a timestamped skill repo folder")
    p.add_argument("--skill-id", default=None,
                   help="Specific skill UUID to evaluate (default: first active skill)")
    p.add_argument("--dataset", required=True,
                   help="Path to dataset JSON (same format as pipeline input)")
    p.add_argument("--n", type=int, default=50,
                   help="Number of instances to evaluate on (default: 50)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for instance sampling (default: 42)")
    p.add_argument("--models", nargs="+",
                   default=["openai/gpt-5.4-nano"],
                   help="Agent models to test (default: gpt-5.4-nano)")
    p.add_argument("--judge-model", default="openai/gpt-5.4-mini",
                   help="Judge model for evaluation (default: gpt-5.4-mini)")
    p.add_argument("--max-workers", type=int, default=16)
    p.add_argument("--output", default=None,
                   help="Optional path to save JSON results")
    p.add_argument("--enable-web-search", action="store_true",
                   help="Enable web search tool on the skill-augmented condition. "
                        "DISABLED by default to keep the baseline/skill comparison fair "
                        "(baseline never gets tools, so the skill side must match).")
    p.add_argument("--enable-execute-scripts", action="store_true",
                   help="Enable real script execution on the skill-augmented condition. "
                        "DISABLED by default for the same fairness reason as web search.")
    p.add_argument("--no-web-search", action="store_true",
                   help=argparse.SUPPRESS)
    p.add_argument("--no-execute-scripts", action="store_true",
                   help=argparse.SUPPRESS)
    p.add_argument("--drop-blank", dest="drop_blank", action="store_true", default=True,
                   help="Exclude paired instances where EITHER side produced an empty "
                        "final_output (likely API / network hiccup or model stall). "
                        "Enabled by default; use --keep-blank to disable.")
    p.add_argument("--keep-blank", dest="drop_blank", action="store_false",
                   help="Include blank-output pairs in paired stats (legacy behaviour).")
    p.add_argument("--save-trajectories", dest="save_trajectories", action="store_true",
                   default=True,
                   help="Persist per-instance trajectories to JSONL alongside --output "
                        "(enabled by default, needed for blank audit).")
    p.add_argument("--no-save-trajectories", dest="save_trajectories", action="store_false",
                   help=argparse.SUPPRESS)
    p.add_argument("--baseline-trajectories", default=None,
                   help="Path to a shared baseline trajectory JSONL (produced by "
                        "a previous eval's --save-trajectories). If provided, the "
                        "baseline condition is LOADED from disk instead of re-run, "
                        "so every ablation / variant is paired against the same "
                        "fixed baseline. Must match --dataset/--seed/--n/--models. "
                        "For multi-model runs, pass a directory containing files "
                        "named '<model-slug>_baseline.jsonl', or the JSONL itself "
                        "when evaluating a single model.")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading skill from: {args.skill_repo}")
    skill = load_skill(args.skill_repo, args.skill_id)
    print(f"  -> {skill.skill_id}: {skill.semantic_aim}")

    # A skill is considered "rejected" if (a) training marked it DEPRECATED
    # because no refinement round passed the verification gate, or (b) its body
    # is empty. In both cases we must NOT evaluate it - applying an overfit or
    # empty skill would just add noise. We run the baseline once and reuse it
    # as the skill trajectories so paired stats correctly report net_gain=0.
    skill_rejected = (
        skill.status == SkillStatus.DEPRECATED
        or not (skill.body or "").strip()
    )
    if skill_rejected:
        reason = []
        if skill.status == SkillStatus.DEPRECATED:
            reason.append("status=DEPRECATED (no verification round passed)")
        if not (skill.body or "").strip():
            reason.append("empty body")
        print(
            f"  [!] Skill REJECTED: {'; '.join(reason)}.\n"
            f"  [!] Skipping skill-condition eval; reporting skill==baseline (net_gain=0)."
        )

    print(f"\nLoading dataset: {args.dataset}")
    all_instances, task_type = load_dataset(args.dataset)
    print(f"  -> {len(all_instances)} instances, task_type={task_type.value}")

    rng = random.Random(args.seed)
    if args.n < len(all_instances):
        instances = rng.sample(all_instances, args.n)
        print(f"  -> Sampled {args.n} instances (seed={args.seed})")
    else:
        instances = all_instances
        print(f"  -> Using all {len(instances)} instances")

    # Tool flags: DISABLED by default so baseline vs skill comparisons stay fair.
    # Use --enable-web-search / --enable-execute-scripts to opt in explicitly.
    # The legacy --no-web-search / --no-execute-scripts flags are accepted but a no-op.
    enable_web_search = args.enable_web_search
    execute_scripts = args.enable_execute_scripts
    # If the skill actually ships scripts, mock execution would make those
    # tools inert (returning the placeholder "[Script content for ... - not
    # executed in this mode]" string) and would silently invalidate the
    # skill-vs-baseline comparison. Auto-enable real exec in that case so a
    # user who forgot `--enable-execute-scripts` doesn't just measure mock
    # noise. Baseline still has no tools, so the comparison stays fair.
    if not execute_scripts and getattr(skill, "scripts", None) \
            and not args.no_execute_scripts:
        execute_scripts = True
        print("  (auto-enabled script exec: skill ships "
              f"{len(skill.scripts)} scripts; use --no-execute-scripts to force mock)")
    if args.no_web_search or args.no_execute_scripts:
        print("  (note: --no-web-search / --no-execute-scripts override the "
              "auto-enable above; scripts will be mocked)")

    print(f"\nModels to test     : {args.models}")
    print(f"Judge model        : {args.judge_model}")
    print(f"Skill web_search   : {'enabled' if enable_web_search else 'disabled'}")
    print(f"Skill script exec  : {'enabled (real exec)' if execute_scripts else 'disabled (mock)'}")
    print(f"Blank-pair policy  : {'DROP (paired_n shrinks, cleanest)' if args.drop_blank else 'KEEP (legacy)'}")
    print()

    all_results = []
    output_records = []

    # Where to save per-trajectory JSONL (sibling of --output).
    traj_dir = None
    if args.save_trajectories and args.output:
        traj_dir = Path(args.output).with_suffix("")  # drop .json
        traj_dir = Path(str(traj_dir) + "_trajectories")
        traj_dir.mkdir(parents=True, exist_ok=True)

    import llm as _llm
    _llm.reset_token_stats()

    shared_baseline_root = Path(args.baseline_trajectories) if args.baseline_trajectories else None
    if shared_baseline_root is not None and not shared_baseline_root.exists():
        raise FileNotFoundError(
            f"--baseline-trajectories path does not exist: {shared_baseline_root}"
        )

    expected_ids = {inst.instance_id for inst in instances}

    for model in args.models:
        print(f"-- Model: {model} --")

        model_slug = model.replace("/", "_")

        baseline_traj_path: Optional[Path] = None
        if shared_baseline_root is not None:
            if shared_baseline_root.is_dir():
                candidate = shared_baseline_root / f"{model_slug}_baseline.jsonl"
                if not candidate.exists():
                    raise FileNotFoundError(
                        f"Shared baseline directory {shared_baseline_root} does "
                        f"not contain {candidate.name}. Expected one JSONL per "
                        f"model with the '<model-slug>_baseline.jsonl' naming."
                    )
                baseline_traj_path = candidate
            else:
                baseline_traj_path = shared_baseline_root

        if baseline_traj_path is not None:
            print(f"  (reusing shared baseline: {baseline_traj_path})")
            base_result = load_baseline_condition(
                baseline_traj_path, model, expected_instance_ids=expected_ids,
            )
        else:
            with _llm.stage_scope(f"eval_baseline::{model_slug}"):
                base_result = run_condition(
                    instances, task_type, model, args.judge_model,
                    skill=None, max_workers=args.max_workers,
                    enable_web_search=False, execute_scripts=False,
                )

        if skill_rejected:
            # Reuse baseline trajectories for the "skill" condition. Paired
            # stats will show skill_acc == baseline_acc, repair == regression
            # == 0, net_gain == 0 - the intended outcome for a rejected skill.
            print(f"  -> Skill rejected; reusing baseline for skill condition.")
            skill_result = base_result
        else:
            with _llm.stage_scope(f"eval_skill::{model_slug}"):
                skill_result = run_condition(
                    instances, task_type, model, args.judge_model,
                    skill=skill, max_workers=args.max_workers,
                    enable_web_search=enable_web_search,
                    execute_scripts=execute_scripts,
                )

        paired = paired_analysis(base_result, skill_result, drop_blank=args.drop_blank)
        all_results.append((base_result, skill_result, paired))

        if traj_dir is not None:
            slug = model.replace("/", "_")
            write_trajectories(traj_dir / f"{slug}_baseline.jsonl", base_result.trajectories)
            write_trajectories(traj_dir / f"{slug}_with_skill.jsonl", skill_result.trajectories)

        output_records.append({
            "model": model,
            "skill_id": skill.skill_id,
            "n_instances": paired.n_instances,
            "baseline_acc": paired.baseline_acc,
            "skill_acc": paired.skill_acc,
            "delta_acc": paired.skill_acc - paired.baseline_acc,
            "repair": paired.repair,
            "regression": paired.regression,
            "repair_rate": paired.repair_rate,
            "regression_rate": paired.regression_rate,
            "net_gain": paired.net_gain,
            "blank_filter": {
                "drop_blank": args.drop_blank,
                "n_paired_raw": paired.n_paired_raw,
                "n_blank_baseline": paired.n_blank_baseline,
                "n_blank_skill": paired.n_blank_skill,
                "n_blank_either": paired.n_blank_either,
                "n_blank_both": paired.n_blank_both,
            },
            "tools": {
                "enable_web_search": enable_web_search,
                "execute_scripts": execute_scripts,
            },
        })

    print_summary(skill, all_results)

    if args.output:
        out = {
            "skill_id": skill.skill_id,
            "semantic_aim": skill.semantic_aim,
            "skill_status": skill.status.value if hasattr(skill.status, "value") else str(skill.status),
            "skill_rejected": bool(skill_rejected),
            "dataset": args.dataset,
            "n_sampled": len(instances),
            "seed": args.seed,
            "drop_blank": args.drop_blank,
            "results": output_records,
        }
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"Results saved to: {args.output}")
        if traj_dir is not None:
            print(f"Trajectories   : {traj_dir}/")

        token_path = Path(args.output).with_suffix(".token_usage.json")
        try:
            _llm.dump_token_stats(token_path)
            rows = _llm.get_token_stats()
            grand = sum(r["total_tokens"] for r in rows)
            pt = sum(r["prompt_tokens"] for r in rows)
            ct = sum(r["completion_tokens"] for r in rows)
            print(
                f"Token usage    : prompt={pt:,} | completion={ct:,} | total={grand:,} "
                f"-> {token_path}"
            )
        except Exception as exc:
            print(f"Warning: failed to dump token usage stats: {exc}")


if __name__ == "__main__":
    main()
