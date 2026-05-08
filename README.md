# SkillGen

Code for the paper: **SkillGen**: Verified Inference-Time Agent Skill Synthesis

## Requirements

```bash
pip install -r requirements.txt
```

Set the following environment variables:

```bash
export OPENROUTER_API_KEY="sk-or-..."   # LLM chat (OpenRouter)
export OPENAI_API_KEY="sk-..."          # embeddings (OpenAI)
```

## Quick Start

```bash
# Train: discover a skill on the training split
python main.py data/aime/train.json --config config.yaml

# Eval: test the discovered skill on the held-out split
python eval_skill.py \
    --skill-path ./skill_output/<run>/skill.json \
    --test-data  data/aime/test.json \
    --config     config.yaml
```

Results are written to `./skill_output/<timestamp>/skill.json`.

## Dataset Format

```json
{
  "dataset_id": "my_task",
  "task_name": "My Task",
  "task_type": "binary",
  "instances": [
    {"instance_id": "1", "input": "task description", "ground_truth": "expected answer"}
  ]
}
```

`task_type` options: `binary` (LLM judge pass/fail), `scored` (numeric threshold), `open_ended` (LLM quality judge).

The bundled datasets under `data/` are the exact splits used in our experiments and are ready to use directly. To prepare additional splits, use the scripts under `scripts/`.

## Preparing Additional Benchmark Datasets

```bash
# Generic (PubMedQA, AIME, ...)
python scripts/prepare_benchmarks.py --benchmark pubmedqa --n 500 -o data/pubmedqa.json

# ChemLLMBench
python scripts/prepare_chemllmbench.py -o data/chemllmbench.json

# MCP-Bench (requires: git clone https://github.com/Accenture/mcp-bench.git external/mcp-bench)
python scripts/prepare_mcp_bench.py --split single --train-n 40 --test-n 16

# ScienceWorld (requires: clone allenai/ScienceWorld and unzip goldpaths-all.zip)
python scripts/prepare_scienceworld.py --train-n 150 --test-n 100

# SocialMaze
python scripts/prepare_socialmaze.py --variant fts -o data/socialmaze_fts.json

# ToolBench (requires: clone OpenBMB/ToolBench)
python scripts/prepare_toolbench.py --subset G1_instruction --train-n 120 --test-n 50

# TAU-Bench
python scripts/prepare_tau_bench.py -o data/tau_bench.json
```

## Configuration

All hyperparameters are in `config.yaml`. The most commonly adjusted fields:

| Field | Default | Description |
|-------|---------|-------------|
| `models.baseline_agent` | `gpt-5.4-nano` | Agent model for baseline and verification |
| `models.generation_execute` | `gpt-5.4-mini` | Skill generation model |
| `pipeline.max_refine_rounds` | `8` | Max generation/refinement iterations |
| `pipeline.max_workers` | `16` | Parallel worker threads |
| `verification.sample_size` | `120` | Verification split size |
| `clustering.method` | `kmeans` | Clustering algorithm |

## File Structure

```
skillgen_paper/
  main.py              - CLI entry point
  pipeline.py          - End-to-end pipeline orchestration
  models.py            - Data models (TaskInstance, Trajectory, SkillItem, ...)
  llm.py               - LLM / embedding / web-search client wrappers
  trajectory.py        - Agent runner and LLM-judge evaluator
  clustering.py        - Failure/success trajectory clustering
  effectiveness.py     - Paired baseline-vs-skill verification
  skill_store.py       - Skill serialization and persistence
  router.py            - Per-instance skill-application gate
  eval_skill.py        - Held-out test evaluation
  artifacts.py         - Checkpoint and artifact helpers
  logging_utils.py     - Logging setup
  config.yaml          - Hyperparameters
  requirements.txt     - Python dependencies
  agents/
    induction.py       - Induction Agent
    generation.py      - Generation Agent
    verification.py    - Verification Agent
  prompts/             - LLM prompt templates
  benchmarks/          - Per-benchmark adapters and graders
  scripts/             - Dataset preparation scripts
  data/                - Benchmark splits used in experiments 
    aime/              - AIME olympiad math  
    pubmedqa/          - PubMedQA biomedical QA  
    mcp_bench/         - MCP-Bench tool-use planning  
    mind2web/          - Mind2Web web navigation  
    scienceworld/      - ScienceWorld science tasks  
    socialmaze/        - SocialMaze social reasoning  
    toolbench/         - ToolBench API-use planning  
```


## 📖 Citation

If you find this repository useful, please cite our paper:

```
@article{ma2026skillgen,
  title={{SkillGen}: Verified Inference-Time Agent Skill Synthesis},
  author={Ma, Yuchen and Huang, Yue and Bao, Han and Zhuang, Haomin and Shukla, Swadheen and Galley, Michel and Zhang, Xiangliang and Feuerriegel, Stefan},
  journal={arXiv preprint},
  year={2026}
}

```
