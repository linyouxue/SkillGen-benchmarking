"""Prepare MCP-Bench train/test splits in SkillGen's dataset format.

MCP-Bench ships three task files under `external/mcp-bench/tasks/`:
  - single         (56 tasks across 28 servers)
  - multi_2server  (30 tasks across 30 combinations)
  - multi_3server  (18 tasks across 18 combinations)

This script picks a split, optionally subsamples, and writes a JSON file
compatible with `main.py` / `eval_skill.py`. When `--train-n` and `--test-n`
are both given, the script emits two disjoint files using a deterministic seed.

Usage (single split, no train/test disjointing):
    python prepare_mcp_bench.py --split single --n 56 -o data/mcp_bench/single.json

Train/test split (deterministic, disjoint on task_id):
    python prepare_mcp_bench.py --split single \
        --train-n 40 --test-n 16 --seed 42 \
        --out-dir data/mcp_bench

Supports all three splits: `single | multi_2server | multi_3server`.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.mcp_bench_adapter import SPLIT_FILES, load_mcp_bench_split


_ALL_SPLITS = tuple(SPLIT_FILES)  # ("single", "multi_2server", "multi_3server")


def _load_split_instances(split: str, tasks_root: str | None) -> list[dict]:
    """Load instances for one split, or concatenate all three when split=='all'.

    With `split=='all'`, instance-level `metadata.split` still reflects the
    original sub-split (single/multi_2server/multi_3server) so per-split
    stratification and inspection remain possible downstream.
    """
    if split == "all":
        merged: list[dict] = []
        for s in _ALL_SPLITS:
            merged.extend(load_mcp_bench_split(s, tasks_root=tasks_root))
        return merged
    return load_mcp_bench_split(split, tasks_root=tasks_root)


def _write_dataset(
    *,
    instances: list[dict],
    split: str,
    seed: int,
    split_label: str,
    out_path: Path,
    n_total: int,
) -> None:
    dataset_id = f"mcp_bench_{split}_{split_label}_n{len(instances)}_seed{seed}"
    task_name = f"mcp_bench_{split}_{split_label}"
    dataset = {
        "dataset_id": dataset_id,
        "task_name": task_name,
        "task_type": "open_ended",
        "instances": instances,
        "metadata": {
            "benchmark": "mcp_bench",
            "split": split,
            "split_label": split_label,
            "n": len(instances),
            "seed": seed,
            "pool_total": n_total,
            "source": "Accenture/mcp-bench",
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2))
    print(f"  Wrote {len(instances)} instances -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MCP-Bench tasks for SkillGen")
    parser.add_argument(
        "--split",
        choices=list(SPLIT_FILES) + ["all"],
        default="single",
        help="single | multi_2server | multi_3server | all (concatenate all three)",
    )
    parser.add_argument("--n", type=int, default=None,
                        help="(single-file mode) Number of tasks to sample. 0 or omit = all.")
    parser.add_argument("--train-n", type=int, default=None,
                        help="(train/test mode) Size of the train split.")
    parser.add_argument("--test-n", type=int, default=None,
                        help="(train/test mode) Size of the test split (disjoint from train).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output", default=None,
                        help="(single-file mode) Output JSON path.")
    parser.add_argument("--out-dir", default="data/mcp_bench",
                        help="(train/test mode) Directory to write train/test JSONs to.")
    parser.add_argument("--tasks-root", default=None,
                        help="Override for external/mcp-bench/tasks (useful in CI).")
    args = parser.parse_args()

    instances = _load_split_instances(args.split, tasks_root=args.tasks_root)
    n_total = len(instances)
    if args.split == "all":
        from collections import Counter
        sub = Counter(i["metadata"]["split"] for i in instances)
        print(f"Loaded {n_total} MCP-Bench tasks (split=all): "
              + ", ".join(f"{k}={v}" for k, v in sub.items()))
    else:
        print(f"Loaded {n_total} MCP-Bench tasks for split={args.split}")

    # Train/test disjoint mode
    if args.train_n is not None or args.test_n is not None:
        if args.train_n is None or args.test_n is None:
            parser.error("--train-n and --test-n must be provided together.")
        if args.train_n + args.test_n > n_total:
            parser.error(
                f"train_n ({args.train_n}) + test_n ({args.test_n}) exceeds "
                f"pool size ({n_total}) for split={args.split}."
            )
        rng = random.Random(args.seed)
        pool = list(instances)
        rng.shuffle(pool)
        train = pool[: args.train_n]
        test = pool[args.train_n : args.train_n + args.test_n]

        # Safety: task_id must be disjoint.
        train_ids = {t["metadata"]["task_id"] for t in train}
        test_ids = {t["metadata"]["task_id"] for t in test}
        overlap = train_ids & test_ids
        if overlap:
            raise RuntimeError(f"Train/test task_id overlap: {sorted(overlap)[:5]}...")

        out_dir = Path(args.out_dir)
        _write_dataset(
            instances=train, split=args.split, seed=args.seed,
            split_label="train",
            out_path=out_dir / f"train_{args.split}_n{args.train_n}_seed{args.seed}.json",
            n_total=n_total,
        )
        _write_dataset(
            instances=test, split=args.split, seed=args.seed,
            split_label="test",
            out_path=out_dir / f"test_{args.split}_n{args.test_n}_seed{args.seed}.json",
            n_total=n_total,
        )
        print(f"Disjoint train/test written. Train={len(train)}, Test={len(test)}.")
        return

    # Single-file mode
    if args.n and 0 < args.n < n_total:
        rng = random.Random(args.seed)
        instances = rng.sample(instances, args.n)
    out_path = Path(
        args.output or f"data/mcp_bench/{args.split}_n{len(instances)}_seed{args.seed}.json"
    )
    _write_dataset(
        instances=instances, split=args.split, seed=args.seed,
        split_label="all",
        out_path=out_path,
        n_total=n_total,
    )


if __name__ == "__main__":
    main()
