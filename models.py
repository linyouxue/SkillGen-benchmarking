"""Data models: tasks, trajectories, skill analysis, and verification artifacts."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from enum import Enum


# Enums

class TaskType(str, Enum):
    BINARY = "binary"
    SCORED = "scored"
    OPEN_ENDED = "open_ended"


class SkillStatus(str, Enum):
    """Kept for on-disk back-compat. In single-skill mode we only emit ACTIVE."""
    CANDIDATE = "candidate"
    ACTIVE = "active"
    DEPRECATED = "deprecated"


# Task objects

@dataclass
class TaskInstance:
    instance_id: str
    input: Any
    ground_truth: Any = None
    metadata: dict = field(default_factory=dict)


@dataclass
class TaskDataset:
    dataset_id: str
    task_name: str
    task_type: TaskType
    instances: list[TaskInstance]
    metadata: dict = field(default_factory=dict)


# Trajectory

@dataclass
class Trajectory:
    trajectory_id: str
    instance_id: str
    agent_config: dict
    messages: list[dict]
    final_output: Any
    success: bool | None = None
    score: float | None = None
    error_summary: str | None = None
    token_usage: dict | None = None
    latency: float | None = None
    timestamp: str | None = None
    metadata: dict = field(default_factory=dict)


# Induction artifacts

@dataclass
class TrajectoryCluster:
    """A cluster of either failure or success trajectories, plus its pattern."""
    cluster_id: str
    kind: str  # "failure" | "success"
    member_trajectory_ids: list[str]
    member_instance_ids: list[str]
    member_summaries: list[str]  # per-member failure reason / solving sketch
    pattern: str                 # LLM-synthesized cluster-level pattern / technique
    size: int = 0

    def __post_init__(self):
        if not self.size:
            self.size = len(self.member_trajectory_ids)


@dataclass
class ContrastivePair:
    """A failure trajectory paired with the nearest success on the same task type.

    Only pairs where the LLM judge says `same_type=True` contribute to the
    "what made the difference" analysis that Generation consumes.
    """
    failure_instance_id: str
    failure_trajectory_id: str
    success_instance_id: str
    success_trajectory_id: str
    similarity: float
    same_type: bool
    same_type_reason: str
    analysis: str  # "what the success did that the failure missed" (empty if not same_type)


@dataclass
class SkillAnalysis:
    """Aggregated, multi-aspect induction output for a single dataset.

    This is the *single source of truth* handed to the Generation Agent.
    Persist it verbatim so a run can be audited or re-used without rerunning
    the (expensive) failure/success summarisation step.
    """
    analysis_id: str
    dataset_id: str | None
    task_name: str | None
    task_type: str
    contextual_abstract: str
    n_total: int
    n_failures: int
    n_successes: int
    failure_clusters: list[TrajectoryCluster] = field(default_factory=list)
    success_clusters: list[TrajectoryCluster] = field(default_factory=list)
    contrastive_pairs: list[ContrastivePair] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


# Skill objects

@dataclass
class CandidateSkill:
    """A draft skill produced by the Generation Agent (before verification)."""
    candidate_id: str
    analysis_id: str
    body: str                     # markdown with the 3 canonical sections
    contextual_abstract: str      # duplicated for quick retrieval / routing
    scripts: list[str] = field(default_factory=list)
    requirements: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    # NEW - progressive-disclosure reference documents. Each item is a dict
    # {"name": str, "summary": str, "content": str}. When non-empty, the agent
    # sees only the (name, summary) manifest in its system prompt and must
    # call the `load_reference(name=...)` tool to fetch `content`.
    reference_docs: list[dict] = field(default_factory=list)


@dataclass
class SkillItem:
    """A persisted skill on disk. Exactly one per (task, dataset)."""
    skill_id: str
    body: str
    contextual_abstract: str
    scripts: list[str] = field(default_factory=list)
    requirements: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    # NEW - see CandidateSkill.reference_docs for schema.
    reference_docs: list[dict] = field(default_factory=list)
    status: SkillStatus = SkillStatus.ACTIVE
    version: int = 1
    analysis_id: str | None = None
    dataset_id: str | None = None
    task_name: str | None = None
    verification_history: list[dict] = field(default_factory=list)
    token_count: int = 0

    # Legacy/optional fields retained so older JSON skill files still load
    behavioral_aim: str = ""
    semantic_aim: str = ""


# Verification artifacts

@dataclass
class EffectivenessResult:
    passed: bool
    n_target: int
    n_boundary: int
    paired_n: int
    baseline_acc: float
    skill_acc: float
    repair_count: int
    regression_count: int
    repair_rate: float
    regression_rate: float
    net_gain: int
    target_repair_count: int
    target_fail_count: int
    success_guard_regression_count: int
    success_guard_pass_count: int
    repaired_ids: list[str]
    regression_ids: list[str]
    failed_ids_after_skill: list[str]
    diagnostic_summary: str
    # ALL cases in the three diagnostic buckets (repair, regression,
    # still_failing), with full (un-truncated) task / ground_truth / baseline
    # and skill outputs. Used by the verification agent's per-case analysis.
    cases: list[dict] = field(default_factory=list)


@dataclass
class CaseAnalysis:
    """Verification agent's per-case narrative analysis.

    One `CaseAnalysis` is produced for every case in `EffectivenessResult.cases`.
    It records the analyst's reasoning about *why* the case is a repair, a
    regression, or a still-failing one, what role (if any) the current skill
    played, and a concrete micro-recommendation for the refiner.
    """
    instance_id: str
    bucket: str                  # "repair" | "regression" | "still_failing"
    analysis: str                # free-form narrative (NOT truncated)
    skill_influence: str         # "helped" | "hurt" | "neutral" | "unclear"
    micro_recommendation: str    # 1-2 sentence concrete suggestion


@dataclass
class VerificationFeedback:
    """Single-skill verification feedback. No overlap/minimality anymore.

    In addition to empirical effectiveness counts, the feedback carries the
    per-case narrative analysis and a synthesized revision guidance string
    that the generation agent consumes during refinement. Raw per-case
    examples are NOT sent to the refiner any more - only the distilled
    analyses + the synthesized guidance are.
    """
    effectiveness: EffectivenessResult | None = None
    round_idx: int = 0
    case_analyses: list[CaseAnalysis] = field(default_factory=list)
    revision_guidance: str = ""
