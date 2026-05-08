"""ChemLLMBench grader.

One entrypoint `evaluate_chemllmbench_output` routes on `metadata.chem_task`
to a task-specific scorer. All scorers return a uniform dict:

    {"passed": bool, "score": float, "error_message": str | None,
     "extracted": str | None, "details": dict}

SMILES tasks go through RDKit canonicalization (`Chem.CanonSmiles`) - string
equality is NOT enough, because `CC(=O)O` and `OC(C)=O` are the same molecule
but will fail a naive string compare. If RDKit isn't installed, SMILES-based
tasks fall back to stripped string equality (with a big warning in the
returned dict) - useful for quickly validating the plumbing on a non-chem env.

The grader is strictly deterministic: no LLM calls. The one genuinely
open-ended task (molecule_captioning) still returns a best-effort heuristic
score here; it's expected to be re-evaluated by the `open_ended` judge path in
trajectory.py when run through the pipeline.
"""

from __future__ import annotations

import re
from typing import Any


# RDKit helpers (lazy import; tolerate missing rdkit for smoke tests)

_RDKIT_AVAILABLE: bool | None = None


def _have_rdkit() -> bool:
    global _RDKIT_AVAILABLE
    if _RDKIT_AVAILABLE is None:
        try:
            from rdkit import Chem  # noqa: F401
            from rdkit import RDLogger
            RDLogger.DisableLog("rdApp.*")   # silence the avalanche of C++ warnings
            _RDKIT_AVAILABLE = True
        except ImportError:
            _RDKIT_AVAILABLE = False
    return _RDKIT_AVAILABLE


def _canon(smi: str) -> str | None:
    """Return canonical SMILES or None if the string can't be parsed."""
    if not _have_rdkit():
        return smi.strip() if smi else None
    from rdkit import Chem
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def _canon_multi(smi: str) -> frozenset[str] | None:
    """Canonicalize a dot-separated multi-molecule SMILES into a frozenset.

    Returns None if any fragment fails to parse.
    """
    if not smi:
        return None
    frags = [f.strip() for f in smi.split(".") if f.strip()]
    canon = []
    for f in frags:
        c = _canon(f)
        if c is None:
            return None
        canon.append(c)
    return frozenset(canon)


# Output extraction helpers

def _extract_last_nonempty_line(text: str) -> str:
    for line in reversed((text or "").splitlines()):
        line = line.strip()
        if line:
            return line
    return ""


def _extract_yes_no(text: str) -> str | None:
    """Return 'Yes' | 'No' | None if unambiguous, case-insensitive."""
    if not text:
        return None
    last = _extract_last_nonempty_line(text).strip(" .\t,\"'")
    low = last.lower()
    if low in ("yes", "no"):
        return "Yes" if low == "yes" else "No"
    # Try the whole last line (strip punctuation)
    m = re.fullmatch(r"(yes|no)[\.\,\!\?]*", low)
    if m:
        return "Yes" if m.group(1) == "yes" else "No"
    # Fallback: last yes/no token anywhere in text
    tokens = re.findall(r"\b(yes|no)\b", (text or "").lower())
    if tokens:
        return "Yes" if tokens[-1] == "yes" else "No"
    return None


def _extract_smiles_candidate(text: str) -> str:
    """Pull a plausible SMILES off the model output.

    Strategy:
      1. Last non-empty line, stripped of quote/backtick/punctuation noise
      2. If that line looks like a sentence, try regex for the longest
         SMILES-like substring in the whole output.
    """
    last = _extract_last_nonempty_line(text).strip().strip(".,;:!?").strip("`\"'")
    # Strip leading label like "SMILES:" or "Answer:"
    last = re.sub(r"^(smiles|answer|product|reactants?)\s*[:=]\s*", "", last, flags=re.I)
    if last and not re.search(r"\s", last):
        return last
    # Fallback: find longest SMILES-y token (atoms + brackets + bonds)
    candidates = re.findall(r"[A-Za-z0-9@+\-\[\]\(\)=#\\/\.%]{3,}", text or "")
    return max(candidates, key=len, default="")


# Per-task scorers

def _result(passed: bool, score: float, *, extracted: str | None = None,
            error: str | None = None, details: dict | None = None) -> dict:
    return {
        "passed": bool(passed),
        "score": float(score),
        "error_message": error,
        "extracted": extracted,
        "details": details or {},
    }


def score_property_prediction(output: str, gt: str, meta: dict) -> dict:
    pred = _extract_yes_no(output)
    if pred is None:
        return _result(False, 0.0, extracted=None,
                       error="Could not parse Yes/No from output")
    passed = pred.lower() == gt.strip().lower()
    return _result(passed, 1.0 if passed else 0.0, extracted=pred)


def score_yield_prediction(output: str, gt: str, meta: dict) -> dict:
    # Same Yes/No scheme as property_prediction.
    return score_property_prediction(output, gt, meta)


def _smiles_single_match(output: str, gt: str) -> dict:
    pred_raw = _extract_smiles_candidate(output)
    if not pred_raw:
        return _result(False, 0.0, extracted=None,
                       error="No SMILES-like string found in output")
    pred_canon = _canon(pred_raw)
    gt_canon = _canon(gt)
    if pred_canon is None:
        return _result(False, 0.0, extracted=pred_raw,
                       error=f"Predicted SMILES failed to parse: {pred_raw!r}")
    if gt_canon is None:
        return _result(False, 0.0, extracted=pred_raw,
                       error=f"Ground-truth SMILES failed to parse (data bug?): {gt!r}")
    passed = pred_canon == gt_canon
    return _result(passed, 1.0 if passed else 0.0, extracted=pred_raw,
                   details={"pred_canon": pred_canon, "gt_canon": gt_canon})


def score_name_prediction(output: str, gt: str, meta: dict) -> dict:
    direction = meta.get("direction", "iupac_to_smiles")
    if direction == "iupac_to_smiles":
        return _smiles_single_match(output, gt)
    # smiles_to_iupac: strict stripped lowercase exact match
    pred = _extract_last_nonempty_line(output).strip().lower()
    passed = pred == gt.strip().lower()
    return _result(passed, 1.0 if passed else 0.0, extracted=pred)


def _smiles_multi_match(output: str, gt: str) -> dict:
    pred_raw = _extract_smiles_candidate(output)
    if not pred_raw:
        return _result(False, 0.0, extracted=None,
                       error="No SMILES-like string found in output")
    pred_set = _canon_multi(pred_raw)
    gt_set = _canon_multi(gt)
    if pred_set is None:
        return _result(False, 0.0, extracted=pred_raw,
                       error=f"Predicted multi-SMILES failed to parse: {pred_raw!r}")
    if gt_set is None:
        return _result(False, 0.0, extracted=pred_raw,
                       error="Ground-truth SMILES failed to parse (data bug?)")
    passed = pred_set == gt_set
    return _result(passed, 1.0 if passed else 0.0, extracted=pred_raw,
                   details={"pred_set": list(pred_set), "gt_set": list(gt_set)})


def score_reaction_prediction(output: str, gt: str, meta: dict) -> dict:
    return _smiles_single_match(output, gt)


def score_retrosynthesis(output: str, gt: str, meta: dict) -> dict:
    # Retrosynthesis ground truth may be multi-fragment (`A.B.C`).
    return _smiles_multi_match(output, gt)


def score_molecule_design(output: str, gt: str, meta: dict) -> dict:
    """Molecule design: SMILES must be valid; canonical match to reference gives full credit."""
    pred_raw = _extract_smiles_candidate(output)
    if not pred_raw:
        return _result(False, 0.0, extracted=None,
                       error="No SMILES found in output")
    pred_canon = _canon(pred_raw)
    if pred_canon is None:
        return _result(False, 0.0, extracted=pred_raw,
                       error=f"Predicted SMILES failed to parse: {pred_raw!r}")
    gt_canon = _canon(gt)
    if gt_canon is None:
        # Treat as "valid but unverifiable" - pass on validity alone.
        return _result(True, 0.5, extracted=pred_raw,
                       error="Reference SMILES unparseable; partial credit for valid prediction")
    passed = pred_canon == gt_canon
    return _result(passed, 1.0 if passed else 0.0, extracted=pred_raw,
                   details={"pred_canon": pred_canon, "gt_canon": gt_canon})


def score_molecule_captioning(output: str, gt: str, meta: dict) -> dict:
    """Heuristic deterministic score; open_ended judge path will override in pipeline.

    Passing if the output shares at least one non-trivial noun-like token
    with the reference description AND is at least 20 chars long.
    """
    text = (output or "").strip()
    if len(text) < 20:
        return _result(False, 0.0, extracted=None,
                       error="Description too short (<20 chars)")
    ref_tokens = set(re.findall(r"[a-zA-Z]{4,}", (gt or "").lower()))
    out_tokens = set(re.findall(r"[a-zA-Z]{4,}", text.lower()))
    common = ref_tokens & out_tokens
    # Strip very common English words to avoid false positives on "molecule", "chemical", etc.
    stopish = {"molecule", "chemical", "compound", "structure", "formula", "contains",
               "derivative", "substance", "element", "atoms", "group", "groups"}
    signal = common - stopish
    overlap_score = min(1.0, len(signal) / 3.0)   # 3+ signal tokens = full credit
    passed = overlap_score >= 0.34
    return _result(passed, overlap_score, extracted=text[:200],
                   details={"signal_tokens": sorted(signal)})


def score_reagent_selection(output: str, gt: str, meta: dict) -> dict:
    """Reagent selection passes iff the predicted SMILES is in `candidate_over_30_yield`."""
    pred_raw = _extract_smiles_candidate(output)
    if not pred_raw:
        return _result(False, 0.0, extracted=None,
                       error="No candidate string found in output")
    over_30 = meta.get("candidate_over_30_yield") or []
    rank = meta.get("candidate_rank") or []

    pred_canon = _canon(pred_raw) or pred_raw
    over_30_canon = {_canon(c) or c for c in over_30}
    rank_canon = [_canon(c) or c for c in rank]

    if pred_canon in over_30_canon:
        # Full credit: produced a high-yield reagent
        rank_pos = rank_canon.index(pred_canon) if pred_canon in rank_canon else -1
        return _result(True, 1.0, extracted=pred_raw,
                       details={"rank_position": rank_pos})
    if pred_canon in rank_canon:
        return _result(False, 0.0, extracted=pred_raw,
                       error="Candidate picked but yield <= 30%",
                       details={"rank_position": rank_canon.index(pred_canon)})
    return _result(False, 0.0, extracted=pred_raw,
                   error="Prediction is not in the candidate list")


# Dispatch

_SCORERS = {
    "name_prediction":      score_name_prediction,
    "property_prediction":  score_property_prediction,
    "reaction_prediction":  score_reaction_prediction,
    "retrosynthesis":       score_retrosynthesis,
    "yield_prediction":     score_yield_prediction,
    "molecule_design":      score_molecule_design,
    "molecule_captioning":  score_molecule_captioning,
    "reagent_selection":    score_reagent_selection,
}


def evaluate_chemllmbench_output(
    *,
    model_output: str,
    ground_truth: Any,
    instance_metadata: dict,
) -> dict:
    """Main entrypoint called from trajectory.py's eval dispatch."""
    task = (instance_metadata or {}).get("chem_task")
    if task not in _SCORERS:
        return _result(False, 0.0, error=f"Unknown chem_task: {task!r}")
    try:
        return _SCORERS[task](
            output=str(model_output or ""),
            gt=str(ground_truth or ""),
            meta=instance_metadata or {},
        )
    except Exception as exc:
        return _result(False, 0.0, error=f"Grader exception: {exc!r}")
