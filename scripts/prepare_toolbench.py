"""Prepare ToolBench train/test splits in SkillGen's dataset format.

Two source layouts are supported:

  - Real 200-query test_instruction set (needs upstream data.zip):
        external/ToolBench/data/test_instruction/{G1_instruction,
        G1_category, G1_tool, G2_instruction, G2_category, G3_instruction}.json

  - Large training pool (also needs data.zip):
        external/ToolBench/data/instruction/{G1,G2,G3}_query.json

  - Tiny demo fallback that ships in the repo (5 / 3 / 2 queries):
        external/ToolBench/data_example/instruction/{G1,G2,G3}_query.json

Typical usage (G1_instruction eval subset - recommended default):

    # One-shot file (all queries):
    python prepare_toolbench.py --subset G1_instruction \
        -o data/toolbench/G1_instruction.json

    # Disjoint train/test split (on query_id) sampled from one subset:
    python prepare_toolbench.py --subset G1_instruction \
        --train-n 120 --test-n 50 --seed 42

    # If you only have the data_example fallback (5 queries), fall back to
    # the G1 train pool which will auto-degrade to 5 items:
    python prepare_toolbench.py --pool G1 -o data/toolbench/G1_demo.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.toolbench_adapter import (
    TEST_SUBSETS,
    TRAIN_POOLS,
    load_toolbench_split,
)


def _write_dataset(
    *,
    instances: list[dict],
    subset_label: str,
    seed: int,
    split_label: str,
    out_path: Path,
    n_total: int,
) -> None:
    dataset_id = f"toolbench_{subset_label}_{split_label}_n{len(instances)}_seed{seed}"
    task_name = f"toolbench_{subset_label}_{split_label}"
    dataset = {
        "dataset_id": dataset_id,
        "task_name": task_name,
        "task_type": "open_ended",
        "instances": instances,
        "metadata": {
            "benchmark": "toolbench",
            "subset": subset_label,
            "split_label": split_label,
            "n": len(instances),
            "seed": seed,
            "pool_total": n_total,
            "source": "OpenBMB/ToolBench",
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2))
    print(f"  Wrote {len(instances)} instances -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ToolBench queries for SkillGen")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--subset", choices=list(TEST_SUBSETS),
                     help="Evaluation subset to load (200 queries each when "
                          "data.zip is available).")
    src.add_argument("--pool", choices=list(TRAIN_POOLS),
                     help="Training pool (G1/G2/G3) to load.")
    src.add_argument("--path", default=None,
                     help="Arbitrary ToolBench-format JSON file to load.")

    parser.add_argument("--n", type=int, default=None,
                        help="(single-file mode) Number of queries to sample. 0/omit = all.")
    parser.add_argument("--train-n", type=int, default=None,
                        help="(train/test mode) Size of the train split.")
    parser.add_argument("--test-n", type=int, default=None,
                        help="(train/test mode) Size of the test split (disjoint from train).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output", default=None,
                        help="(single-file mode) Output JSON path.")
    parser.add_argument("--out-dir", default="data/toolbench",
                        help="(train/test mode) Directory to write train/test JSONs to.")
    args = parser.parse_args()

    if args.subset:
        label = args.subset
    elif args.pool:
        label = args.pool
    elif args.path:
        label = Path(args.path).stem
    else:
        label = "toolbench"

    instances = load_toolbench_split(
        subset=args.subset, pool=args.pool, path=args.path,
    )
    n_total = len(instances)
    print(f"Loaded {n_total} ToolBench queries for {label!r}")
    if n_total and instances[0]["metadata"]["split_source"]:
        print(f"  source: {instances[0]['metadata']['split_source']}")

    # Train/test disjoint mode
    if args.train_n is not None or args.test_n is not None:
        if args.train_n is None or args.test_n is None:
            parser.error("--train-n and --test-n must be provided together.")
        if args.train_n + args.test_n > n_total:
            parser.error(
                f"train_n ({args.train_n}) + test_n ({args.test_n}) exceeds "
                f"pool size ({n_total}) for {label!r}. Did you download the "
                "upstream data.zip into external/ToolBench/data/?"
            )
        rng = random.Random(args.seed)
        pool = list(instances)
        rng.shuffle(pool)
        train = pool[: args.train_n]
        test = pool[args.train_n : args.train_n + args.test_n]

        train_ids = {t["metadata"]["query_id"] for t in train}
        test_ids = {t["metadata"]["query_id"] for t in test}
        overlap = train_ids & test_ids
        if overlap:
            raise RuntimeError(f"Train/test query_id overlap: {sorted(overlap)[:5]}...")

        out_dir = Path(args.out_dir)
        _write_dataset(
            instances=train, subset_label=label, seed=args.seed,
            split_label="train",
            out_path=out_dir / f"train_{label}_n{args.train_n}_seed{args.seed}.json",
            n_total=n_total,
        )
        _write_dataset(
            instances=test, subset_label=label, seed=args.seed,
            split_label="test",
            out_path=out_dir / f"test_{label}_n{args.test_n}_seed{args.seed}.json",
            n_total=n_total,
        )
        print(f"Disjoint train/test written. Train={len(train)}, Test={len(test)}.")
        return

    # Single-file mode
    if args.n and 0 < args.n < n_total:
        rng = random.Random(args.seed)
        instances = rng.sample(instances, args.n)
    out_path = Path(
        args.output or f"data/toolbench/{label}_n{len(instances)}_seed{args.seed}.json"
    )
    _write_dataset(
        instances=instances, subset_label=label, seed=args.seed,
        split_label="all", out_path=out_path, n_total=n_total,
    )


if __name__ == "__main__":
    main()
