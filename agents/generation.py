"""Generation Agent: produce and refine a skill from the induction analysis."""

from __future__ import annotations

import json
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agents._resource_loader import make_load_reference_script
from models import (
    CandidateSkill,
    ContrastivePair,
    SkillAnalysis,
    TrajectoryCluster,
    VerificationFeedback,
)
from prompts import (
    COMPOSE_BODY_PROMPT,
    COMPOSE_BODY_SYSTEM,
    DECOMPOSE_PROMPT,
    DECOMPOSE_SYSTEM,
    GENERATE_EXEC_PROMPT,
    GENERATE_EXEC_SYSTEM,
    GENERATE_PLAN_PROMPT,
    GENERATE_PLAN_SYSTEM,
    GEN_REFERENCE_PROMPT,
    GEN_REFERENCE_SYSTEM,
    GEN_SCRIPT_PROMPT,
    GEN_SCRIPT_SYSTEM,
    REFINE_PROMPT,
    REFINE_SYSTEM,
)
import llm


# Analysis formatting helpers

def _format_clusters_for_prompt(
    clusters: list[TrajectoryCluster],
    *,
    max_clusters: int,
    max_summaries: int = 3,
) -> str:
    if not clusters:
        return "(none)"
    parts = []
    for idx, cluster in enumerate(clusters[:max_clusters]):
        examples = "\n".join(
            f"    * {s.strip()[:240]}" for s in cluster.member_summaries[:max_summaries]
        )
        parts.append(
            f"- Cluster {idx + 1} (size={cluster.size}):\n"
            f"  Pattern:\n{cluster.pattern or '(empty)'}\n"
            f"  Example member summaries:\n{examples or '    (none)'}"
        )
    return "\n\n".join(parts)


def _format_contrastive_pairs(
    pairs: list[ContrastivePair],
    *,
    max_pairs: int,
) -> str:
    kept = [p for p in pairs if p.same_type and p.analysis][:max_pairs]
    if not kept:
        return "(no same-type failure/success pairs were found)"
    parts = []
    for idx, pair in enumerate(kept):
        parts.append(
            f"- Pair {idx + 1} (similarity={pair.similarity:.2f})\n"
            f"  Same-type reason: {pair.same_type_reason}\n"
            f"  What success did vs failure missed: {pair.analysis.strip()}"
        )
    return "\n\n".join(parts)


def _format_plan_outline(outline: dict) -> str:
    if not isinstance(outline, dict) or not outline:
        return "(plan returned no outline)"
    sections = [
        ("Contextual abstract", outline.get("contextual_abstract", "")),
        ("Successful experiences", outline.get("successful_experiences", "")),
        ("Failure lessons", outline.get("failure_lessons", "")),
    ]
    return "\n".join(f"- {title}: {value.strip()}" for title, value in sections if value)


def _format_case_micro_recommendations(
    case_analyses, *, default: str = "(no case analyses available)"
) -> str:
    """Render per-case micro-recommendations, grouped by bucket.

    This is a condensed view of `VerificationFeedback.case_analyses` - it
    surfaces only the 1-sentence micro-recommendation the analyst produced
    for each case, together with the instance_id and skill_influence. The
    full narrative and all raw case text stay on disk (in the verification
    artefacts) and are intentionally withheld from the refiner prompt to
    keep the context window manageable.
    """
    if not case_analyses:
        return default
    groups: dict[str, list] = {"repair": [], "regression": [], "still_failing": []}
    for ca in case_analyses:
        groups.setdefault(ca.bucket, []).append(ca)

    lines: list[str] = []
    for bucket, header in (
        ("repair", "REPAIR (baseline FAILED -> skill PASSED)"),
        ("regression", "REGRESSION (baseline PASSED -> skill FAILED)"),
        ("still_failing", "STILL_FAILING (both failed)"),
    ):
        entries = groups.get(bucket, [])
        lines.append(f"### {header}  ({len(entries)} cases)")
        if not entries:
            lines.append("  (none)")
            continue
        for ca in entries:
            lines.append(
                f"- {ca.instance_id}  [influence={ca.skill_influence}]  "
                f"-> {ca.micro_recommendation or '(no rec)'}"
            )
    return "\n".join(lines)


def _save_candidate(candidate: CandidateSkill, candidate_dir: str | Path, tag: str = "") -> None:
    out = Path(candidate_dir)
    out.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    fname = out / f"{candidate.candidate_id}{suffix}.json"
    with fname.open("w", encoding="utf-8") as fh:
        json.dump(candidate.__dict__, fh, indent=2, default=str)


# Mode 1: Generation

def generate_skill(
    analysis: SkillAnalysis,
    *,
    plan_model: str = "openai/gpt-5.4-mini",
    execute_model: str = "openai/gpt-5.4-mini",
    max_failure_clusters: int = 6,
    max_success_clusters: int = 6,
    max_contrastive_pairs: int = 8,
    max_tokens_generation: int = 16384,
    candidate_dir: str | Path | None = "./candidates",
    # Accepted and ignored (kept for API compatibility with older callers)
    web_search_model: str | None = None,
    use_web_search: bool = False,
    max_search_queries: int = 0,
    generate_scripts: bool = False,
) -> CandidateSkill:
    """Plan -> execute. Produces the initial 3-section skill.

    The skill body contains exactly three sections: contextual abstract,
    successful experiences, and failure lessons. No URLs, references, or
    helper scripts are produced.
    """

    failure_patterns = _format_clusters_for_prompt(
        analysis.failure_clusters, max_clusters=max_failure_clusters,
    )
    success_patterns = _format_clusters_for_prompt(
        analysis.success_clusters, max_clusters=max_success_clusters,
    )
    contrastive_observations = _format_contrastive_pairs(
        analysis.contrastive_pairs, max_pairs=max_contrastive_pairs,
    )

    plan = llm.chat_json(
        GENERATE_PLAN_PROMPT.format(
            contextual_abstract=analysis.contextual_abstract or "(not available)",
            failure_patterns=failure_patterns,
            success_patterns=success_patterns,
            contrastive_observations=contrastive_observations,
            max_clusters=max(max_failure_clusters, max_success_clusters),
            max_pairs=max_contrastive_pairs,
        ),
        system=GENERATE_PLAN_SYSTEM, model=plan_model,
    )

    result = llm.chat_json(
        GENERATE_EXEC_PROMPT.format(
            contextual_abstract=analysis.contextual_abstract or "(not available)",
            failure_patterns=failure_patterns,
            success_patterns=success_patterns,
            contrastive_observations=contrastive_observations,
            plan_outline=_format_plan_outline(plan.get("outline") or {}),
            dedup_notes=(plan.get("dedup_notes") or "").strip() or "(none)",
            approach=plan.get("approach", ""),
        ),
        system=GENERATE_EXEC_SYSTEM, model=execute_model,
        max_tokens=max_tokens_generation,
    )

    body = (result.get("body") or "").strip()
    if not body:
        raise ValueError(
            "Generation LLM returned empty or missing 'body'. "
            f"Keys present: {sorted(result.keys())}"
        )

    candidate = CandidateSkill(
        candidate_id=str(uuid.uuid4()),
        analysis_id=analysis.analysis_id,
        body=body,
        contextual_abstract=(result.get("contextual_abstract") or analysis.contextual_abstract or "").strip(),
        scripts=[],
        requirements=[],
        references=[],
    )

    if candidate_dir:
        _save_candidate(candidate, candidate_dir, tag="gen")
    return candidate


# Mode 2: Refinement

_ANSWER_FORMAT_GUARD = (
    "## Answer format\n"
    "- Follow the original task prompt's output format EXACTLY. "
    "If the task says `Answer with only 'Yes' or 'No'`, the LAST "
    "non-empty line of your reply must be literally `Yes` or `No` "
    "with no trailing punctuation or prose.\n"
    "- If the task demands a single token (SMILES, number, class "
    "label), emit only that token on the final line.\n"
    "- Do NOT add 'reasoning', 'explanation', or qualifiers after "
    "the final-answer line - the grader parses that line literally.\n"
)


def _ensure_answer_format_section(body: str) -> str:
    """Guarantee the SKILL.md body has a `## Answer format` section.

    Long tool-calling chains dilute the original task's output-format
    instruction (empirical: Mistral on yield_prediction drifted into
    prose and the grader rejected every answer even when tools worked).
    The prompt-level fix is in `COMPOSE_BODY_SYSTEM` /
    `COMPOSE_BODY_PROMPT` / `REFINE_SYSTEM`, but LLMs periodically
    forget / strip the section - especially during refinement, where
    the model is laser-focused on repairing aggregate regressions and
    treats format preservation as afterthought. This helper is the
    runtime safety net applied both at first generation and after
    every refinement round.
    """
    if "## Answer format" in body:
        return body
    guard = "\n\n" + _ANSWER_FORMAT_GUARD
    ctx_match = re.search(
        r"(^## Contextual abstract.*?)(\n## )",
        body, flags=re.DOTALL | re.MULTILINE,
    )
    if ctx_match:
        return body[:ctx_match.end(1)] + guard + body[ctx_match.end(1):]
    return guard.lstrip() + "\n\n" + body


def refine_skill(
    candidate: CandidateSkill,
    analysis: SkillAnalysis,
    feedback: VerificationFeedback,
    *,
    model: str = "openai/gpt-5.4-mini",
    max_failure_clusters: int = 6,
    max_success_clusters: int = 6,
    max_tokens_generation: int = 16384,
    candidate_dir: str | Path | None = "./candidates",
    # Accepted and ignored (kept for API compatibility with older callers)
    web_search_model: str | None = None,
    use_web_search: bool = False,
    max_search_queries: int = 0,
    generate_scripts: bool = False,
) -> CandidateSkill:
    """Improve a candidate skill using verification feedback.

    Each round, the refinement prompt enforces a redundancy check: duplicate
    guidance between "Successful experiences" and "Failure lessons" is merged
    into a single home, filler bullets are dropped, and the body is kept in
    the 300-600 word budget.
    """

    eff_feedback = "N/A"
    if feedback.effectiveness:
        eff_feedback = feedback.effectiveness.diagnostic_summary

    revision_guidance = (
        feedback.revision_guidance
        or "(no revision guidance available - preserve current skill unless "
           "aggregate feedback clearly indicates a regression.)"
    )
    case_micro_recommendations = _format_case_micro_recommendations(
        feedback.case_analyses
    )

    failure_patterns = _format_clusters_for_prompt(
        analysis.failure_clusters, max_clusters=max_failure_clusters,
    )
    success_patterns = _format_clusters_for_prompt(
        analysis.success_clusters, max_clusters=max_success_clusters,
    )

    result = llm.chat_json(
        REFINE_PROMPT.format(
            body=candidate.body,
            contextual_abstract=candidate.contextual_abstract or analysis.contextual_abstract,
            failure_patterns=failure_patterns,
            success_patterns=success_patterns,
            eff_feedback=eff_feedback,
            revision_guidance=revision_guidance,
            case_micro_recommendations=case_micro_recommendations,
        ),
        system=REFINE_SYSTEM, model=model,
        max_tokens=max_tokens_generation,
    )

    candidate.body = _ensure_answer_format_section(
        (result.get("body") or candidate.body).strip()
    )
    candidate.contextual_abstract = (
        result.get("contextual_abstract") or candidate.contextual_abstract
    ).strip()
    # If the candidate shipped scripts / reference_docs (resource-bundle path),
    # preserve them across refinement: the body-only REFINE prompt has no
    # business rewriting deterministic RDKit / AST / regex helpers, and
    # dropping them would silently undo the bundle between rounds. A future
    # patch-op refinement can mutate the resource set; for now it is frozen.
    if not candidate.scripts:
        candidate.scripts = []
    if not candidate.requirements:
        candidate.requirements = []
    if not candidate.references:
        candidate.references = []
    if not getattr(candidate, "reference_docs", None):
        candidate.reference_docs = []

    if candidate_dir:
        _save_candidate(candidate, candidate_dir, tag="refine")
    return candidate


# Mode 3: Resource-bundle generation (scripts + references)


def _format_addressed_failures(failures: list[str]) -> str:
    """Render the `addresses_failures` list from the decomposition as bullets."""
    if not failures:
        return "  (none listed)"
    return "\n".join(f"  - {f.strip()}" for f in failures if f.strip())


def _select_focused_failure_text(
    analysis: SkillAnalysis,
    addressed: list[str],
    *,
    max_clusters: int = 4,
) -> str:
    """Pick the failure clusters most relevant to what this resource addresses.

    Heuristic: keyword overlap between the `addresses_failures` strings and
    the cluster pattern / example summaries. Falls back to the top N clusters
    by size if nothing matches.
    """
    if not analysis.failure_clusters:
        return "(no failure clusters available)"
    haystack_tokens: list[set[str]] = []
    for cluster in analysis.failure_clusters:
        text = " ".join(
            [cluster.pattern or ""] + [s[:400] for s in cluster.member_summaries[:3]]
        ).lower()
        tokens = set(t for t in text.split() if len(t) >= 3)
        haystack_tokens.append(tokens)

    if addressed:
        needle = set()
        for phrase in addressed:
            needle.update(t for t in phrase.lower().split() if len(t) >= 3)
        scored = [
            (len(needle & hay), idx) for idx, hay in enumerate(haystack_tokens)
        ]
        scored.sort(reverse=True)
        picked_idx = [idx for score, idx in scored if score > 0][:max_clusters]
    else:
        picked_idx = []

    if not picked_idx:
        # Fallback: first N clusters
        picked_idx = list(range(min(max_clusters, len(analysis.failure_clusters))))

    picks = [analysis.failure_clusters[i] for i in picked_idx]
    return _format_clusters_for_prompt(picks, max_clusters=max_clusters)


def _one_line_docstring(source: str) -> str:
    """Extract the first line of the first docstring inside a Python source.

    Used to build the compact script manifest for the COMPOSE_BODY prompt
    without re-sending the full script bodies.
    """
    src = source or ""
    for marker in ('"""', "'''"):
        start = src.find(marker)
        if start < 0:
            continue
        end = src.find(marker, start + 3)
        if end < 0:
            continue
        doc = src[start + 3: end].strip()
        return (doc.split("\n", 1)[0] or "")[:240]
    return "(no docstring)"


def _generate_one_script(
    spec: dict,
    analysis: SkillAnalysis,
    *,
    available_libraries: str,
    model: str,
    max_tokens: int,
) -> tuple[dict, str | None]:
    """Run one GEN_SCRIPT LLM call. Returns (spec, source) or (spec, None) on failure."""
    focused = _select_focused_failure_text(
        analysis, spec.get("addresses_failures") or [],
    )
    try:
        result = llm.chat_json(
            GEN_SCRIPT_PROMPT.format(
                name=spec.get("name", ""),
                signature=spec.get("signature", ""),
                purpose=spec.get("purpose", ""),
                addresses_failures=_format_addressed_failures(
                    spec.get("addresses_failures") or []
                ),
                focused_failures=focused,
                available_libraries=available_libraries,
            ),
            system=GEN_SCRIPT_SYSTEM, model=model, max_tokens=max_tokens,
        )
    except Exception:
        return spec, None
    source = (result.get("source") or "").strip()
    return spec, source or None


def _generate_one_reference(
    spec: dict,
    analysis: SkillAnalysis,
    *,
    model: str,
    max_tokens: int,
) -> tuple[dict, str | None]:
    """Run one GEN_REFERENCE LLM call. Returns (spec, markdown) or (spec, None)."""
    focused = _select_focused_failure_text(
        analysis, spec.get("addresses_failures") or [],
    )
    try:
        result = llm.chat_json(
            GEN_REFERENCE_PROMPT.format(
                name=spec.get("name", ""),
                summary=spec.get("summary", ""),
                addresses_failures=_format_addressed_failures(
                    spec.get("addresses_failures") or []
                ),
                focused_failures=focused,
            ),
            system=GEN_REFERENCE_SYSTEM, model=model, max_tokens=max_tokens,
        )
    except Exception:
        return spec, None
    content = (result.get("content") or "").strip()
    return spec, content or None


def generate_skill_with_resources(
    analysis: SkillAnalysis,
    *,
    decompose_model: str = "openai/gpt-5.4-mini",
    script_model: str = "openai/gpt-5.4-mini",
    reference_model: str = "openai/gpt-5.4-mini",
    compose_model: str = "openai/gpt-5.4-mini",
    available_libraries: list[str] | None = None,
    generate_references: bool = True,
    max_scripts: int = 5,
    max_references: int = 5,
    max_failure_clusters: int = 6,
    max_success_clusters: int = 6,
    max_contrastive_pairs: int = 8,
    max_tokens_generation: int = 16384,
    max_tokens_per_resource: int = 4096,
    max_workers: int = 8,
    candidate_dir: str | Path | None = "./candidates",
) -> CandidateSkill:
    """Generate a skill that ships executable scripts and reference docs.

    This is the "resource-bundle" generation path. Enable it per-dataset via
    `generation.generate_scripts: true` in the config. The output is still a
    `CandidateSkill` - same verification and eval paths - but with `scripts`
    and `reference_docs` populated.
    """
    available_libraries = available_libraries or []
    libs_str = (
        "\n".join(f"  - {lib}" for lib in available_libraries)
        if available_libraries else "  - (only the Python standard library)"
    )

    failure_patterns = _format_clusters_for_prompt(
        analysis.failure_clusters, max_clusters=max_failure_clusters,
    )
    success_patterns = _format_clusters_for_prompt(
        analysis.success_clusters, max_clusters=max_success_clusters,
    )
    contrastive_observations = _format_contrastive_pairs(
        analysis.contrastive_pairs, max_pairs=max_contrastive_pairs,
    )

    # Phase 1: decompose
    plan = llm.chat_json(
        DECOMPOSE_PROMPT.format(
            contextual_abstract=analysis.contextual_abstract or "(not available)",
            failure_patterns=failure_patterns,
            success_patterns=success_patterns,
            contrastive_observations=contrastive_observations,
            available_libraries=libs_str,
        ),
        system=DECOMPOSE_SYSTEM, model=decompose_model,
    )

    script_specs = [
        s for s in (plan.get("scripts") or [])
        if isinstance(s, dict) and s.get("name") and s.get("signature")
    ][:max_scripts]
    ref_specs = [
        r for r in (plan.get("references") or [])
        if isinstance(r, dict) and r.get("name")
    ][:max_references] if generate_references else []

    # Filter out any accidental `load_reference` spec - that script is
    # auto-generated later; letting the LLM write it would duplicate the
    # function name and shadow the correct inline-payload version.
    script_specs = [s for s in script_specs if s.get("name") != "load_reference"]

    # Phase 2 & 3: concurrent per-resource generation
    scripts_out: list[tuple[dict, str]] = []
    refs_out: list[dict] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        script_futures = [
            pool.submit(
                _generate_one_script, spec, analysis,
                available_libraries=libs_str,
                model=script_model, max_tokens=max_tokens_per_resource,
            )
            for spec in script_specs
        ]
        ref_futures = [
            pool.submit(
                _generate_one_reference, spec, analysis,
                model=reference_model, max_tokens=max_tokens_per_resource,
            )
            for spec in ref_specs
        ]

        for fut in script_futures:
            spec, source = fut.result()
            if source:
                scripts_out.append((spec, source))

        for fut in ref_futures:
            spec, content = fut.result()
            if content:
                refs_out.append({
                    "name": spec["name"],
                    "summary": spec.get("summary", ""),
                    "content": content,
                })

    # Phase 4: compose the body, referring to what we actually built
    scripts_manifest = "\n".join(
        f"- `skill_{spec['name']}({spec.get('signature', '').split('(', 1)[-1]}` - "
        f"{_one_line_docstring(src)}"
        for spec, src in scripts_out
    ) or "  (no scripts were generated for this skill)"

    references_manifest = "\n".join(
        f"- `{r['name']}` - {r.get('summary', '').strip()}"
        for r in refs_out
    ) or "  (no reference documents were generated for this skill)"

    body_result = llm.chat_json(
        COMPOSE_BODY_PROMPT.format(
            contextual_abstract=analysis.contextual_abstract or "(not available)",
            failure_patterns=failure_patterns,
            success_patterns=success_patterns,
            contrastive_observations=contrastive_observations,
            scripts_manifest=scripts_manifest,
            references_manifest=references_manifest,
        ),
        system=COMPOSE_BODY_SYSTEM, model=compose_model,
        max_tokens=max_tokens_generation,
    )

    body = (body_result.get("body") or "").strip()
    if not body:
        raise ValueError(
            "COMPOSE_BODY LLM returned empty body. "
            f"Keys present: {sorted(body_result.keys())}"
        )
    body = _ensure_answer_format_section(body)

    # Assemble the final scripts[] - prepend the auto-generated
    # `load_reference` script (if we have any refs) so `skill_load_reference`
    # is the first tool the agent sees.
    final_scripts: list[str] = [src for _, src in scripts_out]
    if refs_out:
        final_scripts = [make_load_reference_script(refs_out)] + final_scripts

    candidate = CandidateSkill(
        candidate_id=str(uuid.uuid4()),
        analysis_id=analysis.analysis_id,
        body=body,
        contextual_abstract=(
            body_result.get("contextual_abstract")
            or analysis.contextual_abstract
            or ""
        ).strip(),
        scripts=final_scripts,
        requirements=list(available_libraries),
        references=[],
        reference_docs=refs_out,
    )

    if candidate_dir:
        _save_candidate(candidate, candidate_dir, tag="gen_resources")
    return candidate
