"""ChemLLMBench adapter.

Covers the 8 task families shipped with ChemFoundationModels/ChemLLMBench
(https://github.com/ChemFoundationModels/ChemLLMBench):

  1. name_prediction       IUPAC <-> SMILES
  2. property_prediction   BBBP / BACE / HIV / Tox21 / ClinTox (binary Yes/No)
  3. reaction_prediction   USPTO Mixed, forward reaction
  4. retrosynthesis        USPTO-50k, retro reaction
  5. yield_prediction      Buchwald-Hartwig / Suzuki-Miyaura (binary: high-yield or not)
  6. molecule_design       description -> SMILES (ChEBI-20)
  7. molecule_captioning   SMILES -> description (ChEBI-20)
  8. reagent_selection     pick ligand / reactant / solvent from a candidate list

For each task this module exposes:

    load_<task>_records(data_dir) -> list[dict]
        Read the raw files the repo ships (CSV, JSON, or NPZ) and emit a
        normalised dict with the fields needed for TaskInstance construction.

    convert_<task>_record(record, idx) -> dict
        Turn one raw record into the TaskInstance dict consumed by
        prepare_chemllmbench.py.

The `metadata.benchmark` field is always "chemllmbench" so downstream graders
(trajectory.py dispatch) can route correctly. `metadata.chem_task` identifies
the subtask family.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path


# Constants

CHEM_TASKS = [
    "name_prediction",
    "property_prediction",
    "reaction_prediction",
    "retrosynthesis",
    "yield_prediction",
    "molecule_design",
    "molecule_captioning",
    "reagent_selection",
]

# Property-prediction sub-datasets; each has its own prompt framing and label column.
_PROPERTY_SUBSETS = {
    "BBBP": {
        "file": "BBBP_test.csv",
        "smiles_col": "smiles",
        "label_col": "p_np",                    # 0/1 -> No/Yes
        "prompt_intro": (
            "You are an expert chemist. Given the SMILES string of a molecule, "
            "predict whether it penetrates the blood-brain barrier. "
            "Answer with only 'Yes' or 'No'."
        ),
    },
    "BACE": {
        "file": "BACE_test.csv",
        "smiles_col": "mol",
        "label_col": "Class",                   # 0/1
        "prompt_intro": (
            "You are an expert chemist. Given the SMILES string of a molecule, "
            "predict whether it inhibits Beta-site Amyloid Precursor Protein Cleaving "
            "Enzyme 1 (BACE1). Answer with only 'Yes' or 'No'."
        ),
    },
    "HIV": {
        "file": "HIV_test.csv",
        "smiles_col": "smiles",
        "label_col": "HIV_active",              # 0/1
        "prompt_intro": (
            "You are an expert chemist. Given the SMILES string of a molecule, "
            "predict whether it inhibits HIV replication. "
            "Answer with only 'Yes' or 'No'."
        ),
    },
    "Tox21": {
        "file": "Tox_test.csv",
        "smiles_col": "smiles",
        "label_col": "NR-AR",                   # use androgen-receptor endpoint (standard default)
        "prompt_intro": (
            "You are an expert chemist. Given the SMILES string of a molecule, "
            "predict whether it is toxic via the NR-AR pathway. "
            "Answer with only 'Yes' or 'No'."
        ),
    },
    "ClinTox": {
        "file": "ClinTox_test.csv",
        "smiles_col": "smiles",
        "label_col": "CT_TOX",                  # 0/1
        "prompt_intro": (
            "You are an expert chemist. Given the SMILES string of a molecule, "
            "predict whether it failed clinical trials due to toxicity. "
            "Answer with only 'Yes' or 'No'."
        ),
    },
}

# Reagent-selection sub-families (Suzuki coupling variants).
_REAGENT_SUBSETS = {
    "ligand":   "ligand_selection.json",
    "reactant": "reactant_selection.json",
    "solvent":  "solvent_selection.json",
}

# Yield-prediction sub-families.
_YIELD_SUBSETS = {
    "BH": "BH_sample_100_test.npz",     # Buchwald-Hartwig
    "SU": "SU_sample_100_test.npz",     # Suzuki-Miyaura
}


def default_data_dir() -> Path:
    """Return the path to the ChemLLMBench data/ dir, assumed at external/chemllmbench/data/."""
    return Path("external/chemllmbench/data").resolve()


# Loaders (return list[dict] of raw records with subset-aware fields)

def _load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_name_prediction_records(data_dir: Path) -> list[dict]:
    """name_prediction/llm_test.csv - 600 rows with label in {train,test}."""
    path = data_dir / "name_prediction" / "llm_test.csv"
    records = _load_csv(path)
    out = []
    for r in records:
        out.append({
            "cid": r.get("CID", ""),
            "smiles": r["smiles"],
            "iupac": r["iupac"],
            "formula": r.get("formula", ""),
            "label": r.get("label", ""),
        })
    return out


def load_property_prediction_records(data_dir: Path) -> list[dict]:
    """Concatenate all 5 property sub-datasets, tagging each row with its subset."""
    out = []
    base = data_dir / "property_prediction"
    for subset, spec in _PROPERTY_SUBSETS.items():
        path = base / spec["file"]
        if not path.exists():
            continue
        for row in _load_csv(path):
            smi = (row.get(spec["smiles_col"]) or "").strip()
            raw = (row.get(spec["label_col"]) or "").strip()
            if not smi or raw == "":
                continue
            try:
                label = int(float(raw))
            except ValueError:
                continue
            out.append({"subset": subset, "smiles": smi, "label": label})
    return out


def load_reaction_prediction_records(data_dir: Path) -> list[dict]:
    """reaction_prediction/uspto_test.csv - forward reaction prediction."""
    path = data_dir / "reaction_prediction" / "uspto_test.csv"
    return [
        {"reactants": r["reactant"], "product": r["product"]}
        for r in _load_csv(path)
    ]


def load_retrosynthesis_records(data_dir: Path) -> list[dict]:
    """retro/uspto50k_retro_test.csv - retrosynthesis."""
    path = data_dir / "retro" / "uspto50k_retro_test.csv"
    return [
        {"product": r["products_smiles"], "reactants": r["reactants_smiles"]}
        for r in _load_csv(path)
    ]


def load_yield_prediction_records(data_dir: Path) -> list[dict]:
    """yield_prediction/{BH,SU}_sample_100_test.npz.

    Each NPZ has key 'data_df' with shape (N, 2): [reaction_smiles, 'Yes'|'No'].
    """
    import numpy as np
    out = []
    base = data_dir / "yield_prediction"
    for subset, fname in _YIELD_SUBSETS.items():
        path = base / fname
        if not path.exists():
            continue
        arr = np.load(path, allow_pickle=True)["data_df"]
        for reaction, label in arr:
            out.append({
                "subset": subset,
                "reaction": str(reaction),
                "label": str(label).strip(),   # 'Yes' or 'No'
            })
    return out


def load_molecule_design_records(data_dir: Path) -> list[dict]:
    path = data_dir / "molecule_design" / "molecule_design_test.csv"
    return [
        {"description": r["description"], "smiles": r["SMILES"]}
        for r in _load_csv(path)
    ]


def load_molecule_captioning_records(data_dir: Path) -> list[dict]:
    path = data_dir / "molecule_captioning" / "molecule_captioning_test.csv"
    return [
        {"smiles": r["SMILES"], "description": r["description"]}
        for r in _load_csv(path)
    ]


def load_reagent_selection_records(data_dir: Path) -> list[dict]:
    """Concatenate ligand / reactant / solvent selection JSONs."""
    out = []
    base = data_dir / "reagent_selection"
    for subset, fname in _REAGENT_SUBSETS.items():
        path = base / fname
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            items = json.load(fh)
        for item in items:
            out.append({
                "subset": subset,
                "task": item.get("task", ""),
                "candidate_rank": item.get("candidate_rank", []),
                "candidate_over_30_yield": item.get("candidate_over_30_yield", []),
                "yield_details": item.get("yield_details", []),
            })
    return out


LOADERS = {
    "name_prediction":      load_name_prediction_records,
    "property_prediction":  load_property_prediction_records,
    "reaction_prediction":  load_reaction_prediction_records,
    "retrosynthesis":       load_retrosynthesis_records,
    "yield_prediction":     load_yield_prediction_records,
    "molecule_design":      load_molecule_design_records,
    "molecule_captioning":  load_molecule_captioning_records,
    "reagent_selection":    load_reagent_selection_records,
}


# Converters (raw record -> TaskInstance dict)

def _make_instance(
    task: str,
    idx: int,
    input_text: str,
    ground_truth,
    *,
    extra_meta: dict | None = None,
) -> dict:
    return {
        "instance_id": f"{task}_{idx:05d}",
        "input": input_text,
        "ground_truth": ground_truth,
        "metadata": {
            "benchmark": "chemllmbench",
            "chem_task": task,
            **(extra_meta or {}),
        },
    }


def convert_name_prediction(record: dict, idx: int, *, direction: str = "iupac_to_smiles") -> dict:
    """Two directions: iupac_to_smiles (default) or smiles_to_iupac."""
    if direction == "iupac_to_smiles":
        prompt = (
            "You are an expert chemist. Convert the following IUPAC name to its "
            "canonical SMILES string. Output only the SMILES on the last line, "
            "with no extra formatting or explanation.\n\n"
            f"IUPAC name: {record['iupac']}"
        )
        gt = record["smiles"]
    else:
        prompt = (
            "You are an expert chemist. Convert the following SMILES to its IUPAC name. "
            "Output only the IUPAC name on the last line.\n\n"
            f"SMILES: {record['smiles']}"
        )
        gt = record["iupac"]
    return _make_instance(
        "name_prediction", idx, prompt, gt,
        extra_meta={
            "direction": direction,
            "cid": record.get("cid", ""),
            "official_split": record.get("label", ""),
        },
    )


def convert_property_prediction(record: dict, idx: int) -> dict:
    subset = record["subset"]
    spec = _PROPERTY_SUBSETS[subset]
    gt = "Yes" if record["label"] == 1 else "No"
    prompt = (
        f"{spec['prompt_intro']}\n\n"
        f"SMILES: {record['smiles']}\n\n"
        "On the last line, write exactly one of: Yes, No"
    )
    return _make_instance(
        "property_prediction", idx, prompt, gt,
        extra_meta={"subset": subset, "smiles": record["smiles"]},
    )


def convert_reaction_prediction(record: dict, idx: int) -> dict:
    prompt = (
        "You are an expert chemist. Predict the product of the following reaction "
        "(reactants, reagents, and catalysts are given dot-separated). "
        "Output only the product SMILES on the last line, with no extra text.\n\n"
        f"Reactants: {record['reactants']}\n\n"
        "Product:"
    )
    return _make_instance(
        "reaction_prediction", idx, prompt, record["product"],
        extra_meta={"reactants": record["reactants"]},
    )


def convert_retrosynthesis(record: dict, idx: int) -> dict:
    prompt = (
        "You are an expert chemist. Given a target product, propose the reactants "
        "that would produce it in one step (retrosynthesis). If multiple reactants "
        "are needed, dot-separate them. Output only the reactant SMILES on the last line.\n\n"
        f"Product: {record['product']}\n\n"
        "Reactants:"
    )
    return _make_instance(
        "retrosynthesis", idx, prompt, record["reactants"],
        extra_meta={"product": record["product"]},
    )


def convert_yield_prediction(record: dict, idx: int) -> dict:
    subset = record["subset"]
    rxn_name = {"BH": "Buchwald-Hartwig", "SU": "Suzuki-Miyaura"}[subset]
    prompt = (
        f"You are an expert chemist. For the following {rxn_name} reaction "
        "(reactants/reagents/catalysts separated by '.', product separated by '>>'), "
        "predict whether the reaction will give a high yield (> 30%). "
        "Answer with only 'Yes' or 'No' on the last line.\n\n"
        f"Reaction: {record['reaction']}"
    )
    return _make_instance(
        "yield_prediction", idx, prompt, record["label"],
        extra_meta={"subset": subset, "reaction": record["reaction"]},
    )


def convert_molecule_design(record: dict, idx: int) -> dict:
    prompt = (
        "You are an expert chemist. Design a molecule (output its SMILES) matching "
        "the description below. The SMILES must be syntactically valid and represent "
        "a plausible molecule. Output only the SMILES on the last line.\n\n"
        f"Description: {record['description']}"
    )
    return _make_instance(
        "molecule_design", idx, prompt, record["smiles"],
        extra_meta={"description": record["description"]},
    )


def convert_molecule_captioning(record: dict, idx: int) -> dict:
    prompt = (
        "You are an expert chemist. Write a one-paragraph scientific description of "
        "the following molecule, following the style of the ChEBI database "
        "(mention the chemical class and any notable roles/functions). "
        "Do not include the SMILES in your answer.\n\n"
        f"SMILES: {record['smiles']}"
    )
    return _make_instance(
        "molecule_captioning", idx, prompt, record["description"],
        extra_meta={"smiles": record["smiles"]},
    )


def convert_reagent_selection(record: dict, idx: int) -> dict:
    # `task` already contains the full instruction + candidate list.
    prompt = (
        record["task"].rstrip()
        + "\n\nOutput only the chosen SMILES on the last line. No explanation."
    )
    # Ground truth: top-1 of candidate_rank (official optimal);
    # acceptance criterion will be "any candidate in candidate_over_30_yield".
    return _make_instance(
        "reagent_selection", idx, prompt, record["candidate_rank"][0] if record["candidate_rank"] else "",
        extra_meta={
            "subset": record["subset"],
            "candidate_rank": record["candidate_rank"],
            "candidate_over_30_yield": record["candidate_over_30_yield"],
        },
    )


CONVERTERS = {
    "name_prediction":      convert_name_prediction,
    "property_prediction":  convert_property_prediction,
    "reaction_prediction":  convert_reaction_prediction,
    "retrosynthesis":       convert_retrosynthesis,
    "yield_prediction":     convert_yield_prediction,
    "molecule_design":      convert_molecule_design,
    "molecule_captioning":  convert_molecule_captioning,
    "reagent_selection":    convert_reagent_selection,
}


# Task-type hints for prepare script

# All chem tasks are scored with a deterministic grader (RDKit canonicalize +
# exact/contains match + optional LLM judge for open-ended captioning), so
# "binary" is the right TaskType for most - the grader returns pass/fail.
# molecule_captioning is the only genuinely open-ended one.
TASK_TYPES = {
    "name_prediction":      "binary",
    "property_prediction":  "binary",
    "reaction_prediction":  "binary",
    "retrosynthesis":       "binary",
    "yield_prediction":     "binary",
    "molecule_design":      "binary",
    "molecule_captioning":  "open_ended",
    "reagent_selection":    "binary",
}
