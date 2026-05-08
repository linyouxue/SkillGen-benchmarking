"""Prepare tau-bench train/test splits in SkillGen's dataset format.

tau-bench ships two domains under ``external/tau-bench/tau_bench/envs/``:

* ``retail``  - has ``tasks_train`` (500), ``tasks_dev`` (20), ``tasks_test`` (115)
* ``airline`` - has ``tasks_test`` only (50 tasks, variable named ``TASKS``)

Unlike MCP-Bench, tau-bench already ships disjoint train / dev / test pools,
so this script just draws a random subsample from each pool and writes two
JSON files compatible with ``main.py`` / ``eval_skill.py``.

Every written instance carries the full runtime config (user simulator
model/provider, ``max_env_steps``, ...) in ``metadata``, so downstream
``trajectory.py`` can spin up the interactive tau-bench env without
additional CLI flags.

Examples
--------
Retail 30/30 (default), user simulator = gpt-5.4-nano::

    python prepare_tau_bench.py --domain retail --train-n 30 --test-n 30

Airline test-only (no train split exists)::

    python prepare_tau_bench.py --domain airline --test-n 30

Match the tau-bench paper user simulator (gpt-4o)::

    python prepare_tau_bench.py --domain retail --train-n 80 --test-n 50 \
        --user-model gpt-4o --user-provider openai
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.tau_bench_adapter import load_tau_bench_dataset

_VALID_SPLITS = {
    "retail": ("train", "dev", "test"),
    "airline": ("test",),
}


def _instance_to_dict(instance) -> dict:
    d = dataclasses.asdict(instance)
    return {
        "instance_id": d["instance_id"],
        "input": d["input"],
        "ground_truth": d.get("ground_truth"),
        "metadata": d.get("metadata", {}),
    }


def _write_dataset(
    *,
    domain: str,
    task_split: str,
    split_label: str,
    instances: list,
    seed: int,
    out_path: Path,
    user_strategy: str,
    user_model: str,
    user_provider: str,
    max_env_steps: int,
) -> None:
    dataset_id = (
        f"tau_bench_{domain}_{split_label}_n{len(instances)}_seed{seed}"
    )
    task_name = f"tau-bench-{domain}-{split_label}"
    dataset = {
        "dataset_id": dataset_id,
        "task_name": task_name,
        "task_type": "binary",
        "instances": [_instance_to_dict(inst) for inst in instances],
        "metadata": {
            "benchmark": "tau_bench",
            "domain": domain,
            "task_split": task_split,
            "split_label": split_label,
            "n": len(instances),
            "seed": seed,
            "user_strategy": user_strategy,
            "user_model": user_model,
            "user_provider": user_provider,
            "max_env_steps": max_env_steps,
            "source": "sierra-research/tau-bench",
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2))
    print(f"  Wrote {len(instances)} instances -> {out_path}")


def _draw_split(
    *,
    domain: str,
    task_split: str,
    n: int,
    seed: int,
    user_strategy: str,
    user_model: str,
    user_provider: str,
    max_env_steps: int,
):
    dataset = load_tau_bench_dataset(
        domain=domain,
        task_split=task_split,
        user_strategy=user_strategy,
        user_model=user_model,
        user_provider=user_provider,
        max_env_steps=max_env_steps,
        n=None if n <= 0 else n,
        seed=seed,
        shuffle=True,
    )
    return dataset.instances


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare tau-bench train/test splits for SkillGen"
    )
    parser.add_argument(
        "--domain", choices=sorted(_VALID_SPLITS), default="retail",
    )
    parser.add_argument(
        "--train-n", type=int, default=None,
        help="Number of tasks from tasks_train (retail only). Omit to skip.",
    )
    parser.add_argument(
        "--test-n", type=int, default=None,
        help="Number of tasks from tasks_test. Omit to skip.",
    )
    parser.add_argument(
        "--dev-n", type=int, default=None,
        help="Number of tasks from tasks_dev (retail only). Omit to skip.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="data/tau_bench")

    parser.add_argument("--user-strategy", default="llm",
                        help="tau-bench user strategy (llm | react | verify | reflection | human)")
    parser.add_argument("--user-model", default="gpt-5.4-nano",
                        help="Model used to simulate the customer (litellm id).")
    parser.add_argument("--user-provider", default="openai",
                        help="LiteLLM provider override for the user simulator.")
    parser.add_argument("--max-env-steps", type=int, default=30)

    args = parser.parse_args()

    if args.train_n is None and args.test_n is None and args.dev_n is None:
        parser.error("provide at least one of --train-n / --test-n / --dev-n")

    valid_splits = _VALID_SPLITS[args.domain]
    for label, n in (("train", args.train_n), ("dev", args.dev_n), ("test", args.test_n)):
        if n is None:
            continue
        if label not in valid_splits:
            parser.error(
                f"domain={args.domain} has no '{label}' split "
                f"(available: {', '.join(valid_splits)})"
            )

    out_dir = Path(args.out_dir)
    common = dict(
        user_strategy=args.user_strategy,
        user_model=args.user_model,
        user_provider=args.user_provider,
        max_env_steps=args.max_env_steps,
    )

    print(f"== tau-bench prepare ({args.domain}) ==")
    print(f"  user simulator : {args.user_strategy} / {args.user_model} (provider={args.user_provider})")
    print(f"  max env steps  : {args.max_env_steps}")
    print(f"  out dir        : {out_dir}")

    for label, n, task_split in (
        ("train", args.train_n, "train"),
        ("dev", args.dev_n, "dev"),
        ("test", args.test_n, "test"),
    ):
        if n is None:
            continue
        instances = _draw_split(
            domain=args.domain, task_split=task_split, n=n, seed=args.seed, **common,
        )
        out_path = out_dir / f"{label}_{args.domain}_n{len(instances)}_seed{args.seed}.json"
        _write_dataset(
            domain=args.domain,
            task_split=task_split,
            split_label=label,
            instances=instances,
            seed=args.seed,
            out_path=out_path,
            **common,
        )


if __name__ == "__main__":
    main()
