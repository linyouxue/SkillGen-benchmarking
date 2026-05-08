"""Download and prepare benchmark datasets for SkillGen evaluation.

Supported benchmarks:
  1. MATH (DigitalLearningGmbH/MATH-lighteval) - mathematical reasoning
  2. GSM8K (openai/gsm8k) - grade school math
  3. MBPP (google-research-datasets/mbpp) - code generation
  4. HotpotQA (hotpotqa/hotpot_qa) - multi-hop reasoning
  5. ARC-Challenge (allenai/ai2_arc) - science reasoning
  6. AIME (allenai/aime-2022-2025) - olympiad-style math reasoning
  7. LiveCodeBench (livecodebench/code_generation_lite) - competitive programming
  8. SpreadsheetBench (KAKA22/SpreadsheetBench) - real-world spreadsheet manipulation

Usage:
  python prepare_benchmarks.py --benchmark math --n 50 -o math_50.json
  python prepare_benchmarks.py --benchmark livecodebench --livecodebench-version release_v6 --n 50
  python prepare_benchmarks.py --benchmark spreadsheetbench --spreadsheetbench-variant full_912 --n 50
  python prepare_benchmarks.py --benchmark all --n 50
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.livecodebench_adapter import (
    LIVECODEBENCH_VERSION_TAGS,
    convert_livecodebench_item,
    load_livecodebench_records,
)
from benchmarks.spreadsheetbench_adapter import (
    SPREADSHEETBENCH_VARIANTS,
    convert_spreadsheetbench_item,
    ensure_spreadsheetbench_data,
    load_spreadsheetbench_records,
)

BENCHMARKS = {
    "math": {
        "hf_id": "DigitalLearningGmbH/MATH-lighteval",
        "split": "test",
        "task_type": "binary",
        "domain": "mathematical_reasoning",
    },
    "gsm8k": {
        "hf_id": "openai/gsm8k",
        "hf_config": "main",
        "split": "test",
        "task_type": "binary",
        "domain": "grade_school_math",
    },
    "mbpp": {
        "hf_id": "google-research-datasets/mbpp",
        "hf_config": "sanitized",
        "split": "test",
        "task_type": "binary",
        "domain": "code_generation",
    },
    "hotpotqa": {
        "hf_id": "hotpotqa/hotpot_qa",
        "hf_config": "distractor",
        "split": "validation",
        "task_type": "binary",
        "domain": "multi_hop_reasoning",
    },
    "arc": {
        "hf_id": "allenai/ai2_arc",
        "hf_config": "ARC-Challenge",
        "split": "test",
        "task_type": "binary",
        "domain": "science_reasoning",
    },
    "pubmedqa": {
        "hf_id": "qiaojin/PubMedQA",
        "hf_config": "pqa_labeled",
        "split": "train",  # PubMedQA ships its 1000-example expert-labeled pool as the "train" split on HF.
        "task_type": "binary",
        "domain": "biomedical_qa",
    },
    "aime": {
        "hf_id": "allenai/aime-2022-2025",
        "split": "train",
        "task_type": "binary",
        "domain": "olympiad_math",
    },
    "livecodebench": {
        "hf_id": "livecodebench/code_generation_lite",
        "task_type": "binary",
        "domain": "competitive_programming",
    },
    "spreadsheetbench": {
        "hf_id": "KAKA22/SpreadsheetBench",
        "task_type": "binary",
        "domain": "spreadsheet_manipulation",
    },
}


def _convert_math(item, idx):
    # Extract boxed answer if present, otherwise use full solution
    sol = item.get("solution", "")
    answer = sol
    if "\\boxed{" in sol:
        start = sol.rfind("\\boxed{") + 7
        depth, end = 1, start
        while end < len(sol) and depth > 0:
            if sol[end] == "{": depth += 1
            elif sol[end] == "}": depth -= 1
            end += 1
        answer = sol[start:end-1]
    return {
        "instance_id": str(idx),
        "input": item["problem"],
        "ground_truth": answer,
        "metadata": {"level": item.get("level", ""), "type": item.get("type", ""),
                      "full_solution": sol},
    }


def _convert_gsm8k(item, idx):
    answer = item["answer"].split("####")[-1].strip() if "####" in item["answer"] else item["answer"]
    return {
        "instance_id": str(idx),
        "input": item["question"],
        "ground_truth": answer,
        "metadata": {"full_solution": item["answer"]},
    }


def _convert_mbpp(item, idx):
    return {
        "instance_id": str(idx),
        "input": f"{item['prompt']}\n\nWrite a Python function to solve this. Provide only the function code.",
        "ground_truth": item["code"],
        "metadata": {"task_id": item["task_id"],
                      "test_list": item["test_list"]},
    }


def _convert_hotpotqa(item, idx):
    # Provide context paragraphs for distractor setting
    context_parts = []
    for title, sents in zip(item["context"]["title"], item["context"]["sentences"]):
        context_parts.append(f"### {title}\n" + " ".join(sents))
    context = "\n\n".join(context_parts[:5])  # limit context length
    return {
        "instance_id": str(idx),
        "input": f"Answer the following question based on the context.\n\n{context}\n\nQuestion: {item['question']}\n\nProvide a short, direct answer.",
        "ground_truth": item["answer"],
        "metadata": {"type": item.get("type", ""), "level": item.get("level", "")},
    }


def _convert_pubmedqa(item, idx):
    # PubMedQA's context is a dict of parallel lists (contexts + labels); join them
    # into a labelled block so the agent can see each abstract section.
    ctx = item.get("context") or {}
    contexts = list(ctx.get("contexts") or [])
    labels = list(ctx.get("labels") or [])
    if labels and len(labels) == len(contexts):
        context_block = "\n\n".join(f"### {lab}\n{txt}" for lab, txt in zip(labels, contexts))
    else:
        context_block = "\n\n".join(contexts)

    prompt = (
        "Answer the biomedical research question using ONLY the provided abstract.\n\n"
        f"## Abstract\n{context_block}\n\n"
        f"## Question\n{item['question']}\n\n"
        "On the last line, write exactly one of: yes, no, maybe (lowercase, no punctuation)."
    )
    return {
        "instance_id": str(item.get("pubid") or idx),
        "input": prompt,
        "ground_truth": (item.get("final_decision") or "").strip().lower(),
        "metadata": {
            "benchmark": "pubmedqa",
            "pubid": item.get("pubid"),
            "final_decision": (item.get("final_decision") or "").strip().lower(),
            "long_answer": item.get("long_answer"),
            "meshes": list((ctx.get("meshes") or [])),
            "labels": labels,
        },
    }


def _convert_arc(item, idx):
    choices = item["choices"]
    labels = choices["label"]
    texts = choices["text"]
    formatted = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
    return {
        "instance_id": str(idx),
        "input": f"{item['question']}\n\n{formatted}\n\nAnswer with the letter only.",
        "ground_truth": item["answerKey"],
        "metadata": {},
    }


def _normalise_aime_answer(raw_answer):
    answer = str(raw_answer).strip()
    if "~user" in answer:
        answer = answer.split("~user", 1)[0].strip()
    if "|" in answer:
        answer = answer.split("|", 1)[0].strip()
    if "," in answer:
        answer = answer.split(",", 1)[0].strip()
    if ";" in answer:
        answer = answer.split(";", 1)[0].strip()
    return answer


def _convert_aime(item, idx):
    prompt = (
        f"{item['problem']}\n\n"
        "Solve the problem carefully. "
        "Your final answer must be a single integer from 000 to 999. "
        "On the last line, write only that integer."
    )
    return {
        "instance_id": str(idx),
        "input": prompt,
        "ground_truth": _normalise_aime_answer(item["answer"]),
        "metadata": {
            "benchmark": "aime",
            "year": item.get("year"),
            "contest": item.get("contest"),
            "problem_number": item.get("problem_number"),
            "url": item.get("url"),
            "raw_answer": item.get("answer"),
        },
    }


CONVERTERS = {
    "math": _convert_math,
    "gsm8k": _convert_gsm8k,
    "mbpp": _convert_mbpp,
    "hotpotqa": _convert_hotpotqa,
    "arc": _convert_arc,
    "aime": _convert_aime,
    "pubmedqa": _convert_pubmedqa,
    "livecodebench": convert_livecodebench_item,
    "spreadsheetbench": convert_spreadsheetbench_item,
}


def prepare_benchmark(
    name: str,
    n: int = 50,
    seed: int = 42,
    output: str | None = None,
    year: int | None = None,
    livecodebench_version: str = "release_latest",
    spreadsheetbench_variant: str = "full_912",
):
    cfg = BENCHMARKS[name]
    print(f"Loading {name} ({cfg['hf_id']})...")

    spreadsheet_data_dir = None
    if name == "livecodebench":
        items = load_livecodebench_records(livecodebench_version)
        print(f"  Loaded {len(items)} problems from {livecodebench_version}")
    elif name == "spreadsheetbench":
        spreadsheet_data_dir = ensure_spreadsheetbench_data(spreadsheetbench_variant)
        items = load_spreadsheetbench_records(spreadsheetbench_variant)
        print(f"  Loaded {len(items)} problems from {spreadsheetbench_variant} (data: {spreadsheet_data_dir})")
    else:
        from datasets import load_dataset

        kwargs = {"split": cfg["split"]}
        if "hf_config" in cfg:
            kwargs["name"] = cfg["hf_config"]

        ds = load_dataset(cfg["hf_id"], **kwargs)
        items = list(ds)

    total_before_sampling = len(items)

    if year is not None:
        items = [item for item in items if item.get("year") == year]
        print(f"  Filtered to year {year}: {len(items)} total")
        total_before_sampling = len(items)

    if n > 0 and n < len(items):
        random.seed(seed)
        items = random.sample(items, n)
        print(f"  Sampled {n} from {total_before_sampling} total")

    converter = CONVERTERS[name]
    if name == "spreadsheetbench":
        instances = [converter(item, i, spreadsheet_data_dir) for i, item in enumerate(items)]
    else:
        instances = [converter(item, i) for i, item in enumerate(items)]

    if name == "livecodebench":
        default_out = f"bench_{name}_{livecodebench_version}.json"
    elif name == "spreadsheetbench":
        default_out = f"bench_{name}_{spreadsheetbench_variant}.json"
    else:
        default_out = f"bench_{name}.json"
    out_path = output or default_out

    if name == "livecodebench":
        dataset_id = f"{name}_{livecodebench_version}"
        task_name = f"{name}_{livecodebench_version}_{cfg['domain']}"
    elif name == "spreadsheetbench":
        dataset_id = f"{name}_{spreadsheetbench_variant}"
        task_name = f"{name}_{spreadsheetbench_variant}_{cfg['domain']}"
    else:
        dataset_id = name
        task_name = f"{name}_{cfg['domain']}"

    extra_meta: dict = {}
    if name == "livecodebench":
        extra_meta["livecodebench_version"] = livecodebench_version
    elif name == "spreadsheetbench":
        extra_meta["spreadsheetbench_variant"] = spreadsheetbench_variant
        if spreadsheet_data_dir is not None:
            extra_meta["spreadsheet_data_dir"] = str(spreadsheet_data_dir)

    result = {
        "dataset_id": dataset_id,
        "task_name": task_name,
        "task_type": cfg["task_type"],
        "instances": instances,
        "metadata": {
            "source": cfg["hf_id"],
            "domain": cfg["domain"],
            "n": len(instances),
            "year_filter": year,
            **extra_meta,
        },
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(instances)} instances to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare benchmark datasets")
    parser.add_argument("--benchmark", required=True,
                        choices=list(BENCHMARKS.keys()) + ["all"])
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--year", type=int, default=None,
                        help="Optional year filter for datasets that expose a year field (e.g. AIME)")
    parser.add_argument(
        "--livecodebench-version",
        default="release_latest",
        choices=LIVECODEBENCH_VERSION_TAGS,
        help="LiveCodeBench release tag or window alias (e.g. release_v6, release_latest, v5, v4_v6)",
    )
    parser.add_argument(
        "--spreadsheetbench-variant",
        default="full_912",
        choices=SPREADSHEETBENCH_VARIANTS,
        help="SpreadsheetBench variant to download (full_912 or verified_400)",
    )
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()

    if args.benchmark == "all":
        for name in BENCHMARKS:
            prepare_benchmark(
                name,
                args.n,
                args.seed,
                year=args.year,
                livecodebench_version=args.livecodebench_version,
                spreadsheetbench_variant=args.spreadsheetbench_variant,
            )
    else:
        prepare_benchmark(
            args.benchmark,
            args.n,
            args.seed,
            args.output,
            year=args.year,
            livecodebench_version=args.livecodebench_version,
            spreadsheetbench_variant=args.spreadsheetbench_variant,
        )


if __name__ == "__main__":
    main()
