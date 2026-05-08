"""Prepare ScienceWorld train/test splits for SkillGen (offline plan variant).

ScienceWorld ships 29 distinct tasks (task IDs 0-29 with one gap at 5), each
with many variations, pre-assigned to `train` / `dev` / `test` folds in the
official gold-paths JSON. This script:

  - Reads the gold paths via `scienceworld_adapter.iter_goldpath_sequences`.
  - Stratified-samples a train pool from fold=train and a test pool from
    fold=test (disjointness is guaranteed by the official fold labels, not by
    our sampling).
  - Emits two JSON files in SkillGen's `TaskDataset` format.

Usage:
    # Pilot: 150 train / 100 test
    python prepare_scienceworld.py --train-n 150 --test-n 100 --seed 42

    # Full-size (if you ever want it):
    python prepare_scienceworld.py --train-n 500 --test-n 200 --seed 42

Output paths (under `--out-dir`, default `data/scienceworld/`):
    train_n{N}_seed{seed}.json
    test_n{N}_seed{seed}.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.scienceworld_adapter import iter_goldpath_sequences


def _stratified_sample(pool: list[dict], n: int, seed: int) -> list[dict]:
    """Sample `n` instances with approximately equal representation across the
    29 tasks. If some task doesn't have enough candidates, fill the remainder
    randomly from the overall pool.
    """
    rng = random.Random(seed)
    if n >= len(pool):
        return list(pool)

    by_task: dict[str, list[dict]] = defaultdict(list)
    for inst in pool:
        by_task[inst["metadata"]["task_id"]].append(inst)
    for tid in by_task:
        rng.shuffle(by_task[tid])

    tasks = sorted(by_task.keys(), key=lambda t: int(t))
    # Round-robin pick one per task until we reach n.
    picked: list[dict] = []
    cursors = {t: 0 for t in tasks}
    while len(picked) < n:
        made_progress = False
        for t in tasks:
            if cursors[t] < len(by_task[t]):
                picked.append(by_task[t][cursors[t]])
                cursors[t] += 1
                made_progress = True
                if len(picked) >= n:
                    break
        if not made_progress:
            break  # exhausted everything
    # Shuffle final order so baseline trajectories don't run one task back-to-back.
    rng.shuffle(picked)
    return picked[:n]


def _write_dataset(*, instances: list[dict], split_label: str, n: int, seed: int,
                    out_path: Path, pool_total: int) -> None:
    dataset_id = f"scienceworld_{split_label}_n{len(instances)}_seed{seed}"
    task_name = f"scienceworld_{split_label}"
    dataset = {
        "dataset_id": dataset_id,
        "task_name": task_name,
        "task_type": "open_ended",
        "instances": instances,
        "metadata": {
            "benchmark": "scienceworld",
            "split_label": split_label,
            "n": len(instances),
            "seed": seed,
            "pool_total": pool_total,
            "source": "allenai/ScienceWorld",
            "sampling": "stratified_by_task_id",
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2))
    print(f"  Wrote {len(instances)} instances -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ScienceWorld splits for SkillGen")
    parser.add_argument("--train-n", type=int, default=150)
    parser.add_argument("--test-n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="data/scienceworld")
    parser.add_argument("--goldpath-dir", default=None,
                        help="Override path to directory containing goldsequences-*.json.")
    args = parser.parse_args()

    print(f"Loading ScienceWorld gold paths (this reads a ~442 MB JSON once)...")
    train_pool = iter_goldpath_sequences(folds=("train",), goldpath_dir=args.goldpath_dir)
    test_pool = iter_goldpath_sequences(folds=("test",), goldpath_dir=args.goldpath_dir)
    print(f"  train pool: {len(train_pool)} sequences across "
          f"{len({i['metadata']['task_id'] for i in train_pool})} tasks")
    print(f"  test pool : {len(test_pool)} sequences across "
          f"{len({i['metadata']['task_id'] for i in test_pool})} tasks")

    if args.train_n > len(train_pool):
        parser.error(f"--train-n ({args.train_n}) exceeds train pool ({len(train_pool)}).")
    if args.test_n > len(test_pool):
        parser.error(f"--test-n ({args.test_n}) exceeds test pool ({len(test_pool)}).")

    train_inst = _stratified_sample(train_pool, args.train_n, seed=args.seed)
    test_inst = _stratified_sample(test_pool, args.test_n, seed=args.seed ^ 0xDEAD)

    # Safety: no instance_id overlap (guaranteed by fold label, but double-check).
    tr_ids = {i["instance_id"] for i in train_inst}
    te_ids = {i["instance_id"] for i in test_inst}
    overlap = tr_ids & te_ids
    if overlap:
        raise RuntimeError(f"Train/test instance_id overlap: {sorted(overlap)[:5]}...")

    out_dir = Path(args.out_dir)
    _write_dataset(
        instances=train_inst, split_label="train",
        n=args.train_n, seed=args.seed, pool_total=len(train_pool),
        out_path=out_dir / f"train_n{args.train_n}_seed{args.seed}.json",
    )
    _write_dataset(
        instances=test_inst, split_label="test",
        n=args.test_n, seed=args.seed, pool_total=len(test_pool),
        out_path=out_dir / f"test_n{args.test_n}_seed{args.seed}.json",
    )
    print(f"Done. Train={len(train_inst)}, Test={len(test_inst)}. "
          f"Disjoint: {len(overlap) == 0}.")


if __name__ == "__main__":
    main()
