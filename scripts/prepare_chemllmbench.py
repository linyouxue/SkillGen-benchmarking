"""Prepare ChemLLMBench tasks into SkillGen's standard TaskInstance JSON format.

Output: one pair of JSONs per task - `<task>_train.json` and `<task>_test.json` -
under `data/chemllmbench/`.

Splits are disjoint by instance_id. For `name_prediction` we honour the
official `label  in  {train, test}` marker; for all other tasks the raw
file is a single pool from which we sample a disjoint train/test split with
a fixed seed.

Usage:
    # single task
    python prepare_chemllmbench.py --task property_prediction --train-n 30 --test-n 10 --seed 42

    # all 8 tasks at once
    python prepare_chemllmbench.py --task all --train-n 30 --test-n 10 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.chemllmbench_adapter import (
    CHEM_TASKS,
    CONVERTERS,
    LOADERS,
    TASK_TYPES,
    default_data_dir,
)


def _split_name_prediction(records, train_n, test_n, seed):
    """Honour the official label=train/test marker, then sub-sample."""
    rng = random.Random(seed)
    train_pool = [r for r in records if (r.get("label") or "").lower() == "train"]
    test_pool = [r for r in records if (r.get("label") or "").lower() == "test"]
    if len(train_pool) < train_n or len(test_pool) < test_n:
        raise RuntimeError(
            f"name_prediction: need >={train_n} train and >={test_n} test, "
            f"got {len(train_pool)}/{len(test_pool)}"
        )
    rng.shuffle(train_pool)
    rng.shuffle(test_pool)
    return train_pool[:train_n], test_pool[:test_n]


def _split_generic(records, train_n, test_n, seed):
    """Uniform random disjoint split."""
    rng = random.Random(seed)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    need = train_n + test_n
    if len(records) < need:
        raise RuntimeError(
            f"Only {len(records)} records available; need {need} (train={train_n}, test={test_n})"
        )
    train_idx = idx[:train_n]
    test_idx = idx[train_n:train_n + test_n]
    return [records[i] for i in train_idx], [records[i] for i in test_idx]


def _split_stratified_property(records, train_n, test_n, seed):
    """Stratify property_prediction by (subset, label) so both splits see all sub-datasets."""
    rng = random.Random(seed)
    buckets: dict[tuple, list] = {}
    for r in records:
        key = (r["subset"], r["label"])
        buckets.setdefault(key, []).append(r)
    for k in buckets:
        rng.shuffle(buckets[k])

    train, test = [], []
    # Round-robin over buckets to keep balance.
    keys = sorted(buckets.keys())
    cursor = {k: 0 for k in keys}
    for target, n in (("train", train_n), ("test", test_n)):
        filled = 0
        while filled < n:
            progress = False
            for k in keys:
                if filled >= n:
                    break
                if cursor[k] < len(buckets[k]):
                    (train if target == "train" else test).append(buckets[k][cursor[k]])
                    cursor[k] += 1
                    filled += 1
                    progress = True
            if not progress:
                break
    if len(train) < train_n or len(test) < test_n:
        raise RuntimeError(
            f"property_prediction: could only gather {len(train)} train / {len(test)} test"
        )
    return train, test


def _split_stratified_by_subset(records, train_n, test_n, seed, key="subset"):
    """Stratify records by a single field (used for yield_prediction, reagent_selection)."""
    rng = random.Random(seed)
    buckets: dict[str, list] = {}
    for r in records:
        buckets.setdefault(r.get(key, "_"), []).append(r)
    for k in buckets:
        rng.shuffle(buckets[k])

    train, test = [], []
    keys = sorted(buckets.keys())
    cursor = {k: 0 for k in keys}
    for target, n in (("train", train_n), ("test", test_n)):
        filled = 0
        while filled < n:
            progress = False
            for k in keys:
                if filled >= n:
                    break
                if cursor[k] < len(buckets[k]):
                    (train if target == "train" else test).append(buckets[k][cursor[k]])
                    cursor[k] += 1
                    filled += 1
                    progress = True
            if not progress:
                break
    return train, test


def _split_for_task(task, records, train_n, test_n, seed):
    if task == "name_prediction":
        return _split_name_prediction(records, train_n, test_n, seed)
    if task == "property_prediction":
        return _split_stratified_property(records, train_n, test_n, seed)
    if task in ("yield_prediction", "reagent_selection"):
        return _split_stratified_by_subset(records, train_n, test_n, seed)
    return _split_generic(records, train_n, test_n, seed)


def _build_instances(task, records):
    converter = CONVERTERS[task]
    return [converter(rec, idx) for idx, rec in enumerate(records)]


def prepare_one_task(task, train_n, test_n, seed, out_dir, data_dir):
    loader = LOADERS[task]
    converter = CONVERTERS[task]
    records = loader(data_dir)
    if not records:
        raise RuntimeError(f"No records loaded for task={task} (data_dir={data_dir})")

    train_records, test_records = _split_for_task(task, records, train_n, test_n, seed)
    train_instances = [converter(r, i) for i, r in enumerate(train_records)]
    test_instances = [converter(r, i + train_n) for i, r in enumerate(test_records)]

    # Sanity check: no instance_id collision.
    train_ids = {x["instance_id"] for x in train_instances}
    test_ids = {x["instance_id"] for x in test_instances}
    assert not (train_ids & test_ids), f"instance_id collision in {task}"

    task_type = TASK_TYPES[task]
    for split, instances in (("train", train_instances), ("test", test_instances)):
        payload = {
            "dataset_id": f"chemllmbench_{task}",
            "task_name": f"chemllmbench_{task}_{split}",
            "task_type": task_type,
            "instances": instances,
            "metadata": {
                "source": "ChemFoundationModels/ChemLLMBench",
                "chem_task": task,
                "split": split,
                "n": len(instances),
                "seed": seed,
            },
        }
        path = Path(out_dir) / f"{task}_{split}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        print(f"  [{task}:{split}] {len(instances)} instances -> {path}")

    return len(train_instances), len(test_instances)


def main():
    ap = argparse.ArgumentParser(description="Prepare ChemLLMBench tasks for SkillGen")
    ap.add_argument("--task", required=True, choices=CHEM_TASKS + ["all"])
    ap.add_argument("--train-n", type=int, default=30)
    ap.add_argument("--test-n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="data/chemllmbench")
    ap.add_argument("--data-dir", default=None,
                    help="Path to ChemLLMBench/data/ (default: external/chemllmbench/data)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    if not data_dir.exists():
        raise SystemExit(
            f"Data dir not found: {data_dir}\n"
            "Run: git clone https://github.com/ChemFoundationModels/ChemLLMBench.git external/chemllmbench"
        )

    tasks = CHEM_TASKS if args.task == "all" else [args.task]
    total_train = total_test = 0
    print(f"ChemLLMBench data dir: {data_dir}")
    print(f"Output dir          : {args.out_dir}")
    print(f"Per-task: train={args.train_n}, test={args.test_n}, seed={args.seed}\n")
    for task in tasks:
        print(f"-> {task}")
        try:
            tr, te = prepare_one_task(task, args.train_n, args.test_n, args.seed,
                                       args.out_dir, data_dir)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            continue
        total_train += tr
        total_test += te
    print(f"\nTotal: {total_train} train + {total_test} test across {len(tasks)} task(s).")


if __name__ == "__main__":
    main()
