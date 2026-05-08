"""Entry point: load a dataset and run the skill discovery pipeline."""

from __future__ import annotations
import argparse, json
from models import TaskInstance, TaskDataset, TaskType
from logging_utils import configure_logging
from pipeline import run_pipeline


def load_dataset(path: str) -> TaskDataset:
    """Load a JSON dataset. Expected format:
    {
      "dataset_id": "...",
      "task_name": "...",
      "task_type": "binary" | "scored" | "open_ended",
      "instances": [
        {"instance_id": "...", "input": "...", "ground_truth": "..."},
        ...
      ]
    }
    """
    with open(path) as f:
        data = json.load(f)
    instances = [
        TaskInstance(
            instance_id=d["instance_id"],
            input=d["input"],
            ground_truth=d.get("ground_truth"),
            metadata=d.get("metadata", {}),
        )
        for d in data["instances"]
    ]
    return TaskDataset(
        dataset_id=data.get("dataset_id", "default"),
        task_name=data.get("task_name", "unnamed"),
        task_type=TaskType(data["task_type"]),
        instances=instances,
        metadata=data.get("metadata", {}),
    )


def main():
    parser = argparse.ArgumentParser(description="Failure-driven skill discovery")
    parser.add_argument("dataset", help="Path to dataset JSON file")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument(
        "--generate-scripts",
        action="store_true",
        default=None,
        help=(
            "Ask the generation agent to produce callable Python helper functions "
            "alongside each skill. Each skill with scripts gets a companion "
            "<skill_id>_helpers.py file in the skill repo. "
            "Overrides generation.generate_scripts in config.yaml."
        ),
    )
    parser.add_argument(
        "--resume",
        metavar="RUN_DIR",
        default=None,
        help=(
            "Resume a crashed run from its artifact directory "
            "(e.g. ./artifacts/runs/20260415-103000). "
            "Re-uses the saved baseline trajectories and (if present) the saved "
            "induction analysis so you don't have to repeat the expensive early stages."
        ),
    )
    parser.add_argument("--verbose-http", action="store_true",
                        help="Show raw HTTP request logs from SDK clients")
    args = parser.parse_args()

    log = configure_logging(verbose_http=args.verbose_http)
    dataset = load_dataset(args.dataset)
    log.info(
        "Loaded dataset '%s' with %d instances (%s)",
        dataset.task_name,
        len(dataset.instances),
        dataset.task_type.value,
    )

    skill = run_pipeline(
        dataset.instances,
        dataset.task_type,
        config_path=args.config,
        dataset_id=dataset.dataset_id,
        task_name=dataset.task_name,
        dataset_metadata=dataset.metadata,
        generate_scripts=args.generate_scripts,
        resume_dir=args.resume,
    )
    if skill is None:
        log.info("Done. No skill was produced (no failures to learn from).")
    else:
        log.info("Done. Generated skill id=%s", skill.skill_id)


if __name__ == "__main__":
    main()
