"""Schema 1.0 contracts. Model output is always validated as untrusted data."""

from datetime import UTC, datetime
from typing import Annotated, Any, Literal, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Id = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$")]
OperationKind = Literal["inherit", "recombine", "mutate", "innovate", "prune"]
TargetKind = Literal["skill", "memory", "tool"]
EvaluationStatus = Literal["not_available", "pending", "completed", "failed", "invalid"]
PolicyField = Literal[
    "diagnosis_policy",
    "selection_policy",
    "operator_policy",
    "generation_policy",
    "validation_guidance",
    "proposal_budget",
    "allowed_operations",
    "parent_selection",
]


def now() -> datetime:
    """Return a timezone-aware UTC timestamp for persistent evidence."""
    return datetime.now(UTC)


def uid(prefix: str) -> str:
    """Generate a collision-resistant record identity with a descriptive host prefix."""
    return f"{prefix}_{uuid4().hex}"


class Record(BaseModel, frozen=True):
    """Base contract for versioned, immutable records with explicit extensions."""

    @staticmethod
    def _preserve_wire_schema(schema: dict[str, Any]) -> None:
        # Python API documentation must not enlarge model-facing schemas or change
        # archived protocol fingerprints. Explicit field descriptions remain intact.
        schema.pop("description", None)

    model_config = ConfigDict(
        extra="forbid", frozen=True, allow_inf_nan=False, json_schema_extra=_preserve_wire_schema
    )
    schema_version: Literal["1.0", "1.1", "1.2"] = "1.0"
    extensions: dict[str, Any] = Field(default_factory=dict)

    @field_validator("*", mode="after")
    @classmethod
    def timezone_required(cls, value: Any) -> Any:
        """Reject naive timestamps whose chronological evidence would be ambiguous."""
        if isinstance(value, datetime) and value.tzinfo is None:
            raise ValueError("timezone required")
        return value


class ArtifactRef(Record, frozen=True):
    """Identify one immutable, versioned capability artifact by URI and content hash.

    The store verifies the referenced bytes before an agent can load them. A new
    artifact version receives a new reference rather than replacing inherited data.
    """

    artifact_id: Id
    version_id: Id
    content_sha256: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
    uri: str


class SkillRef(ArtifactRef, frozen=True):
    """Reference a Skill and its applicability, dependencies and conflict constraints.

    Selectors may use these declarations to choose context. Selection records show
    injection, which alone does not establish that the model used the Skill.
    """

    requires_tools: list[Id] = Field(default_factory=list)
    selection_tags: list[str] = Field(default_factory=list)
    applicability: str = ""
    conflicts_with: list[Id] = Field(default_factory=list)
    constraints: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def legacy_manifest(cls, value: Any) -> Any:
        """Normalize the legacy manifest reference for backward-compatible loading."""
        if isinstance(value, dict):
            value = dict(value)
            extensions = value.get("extensions", {})
            if not isinstance(extensions, dict):
                return value  # Normal field validation reports the invalid root object.
            legacy = extensions.get("heas")
            if legacy is None:
                legacy = {}
            if not isinstance(legacy, dict):
                raise ValueError("HEAS extension metadata must be an object when supplied")
            value.setdefault("requires_tools", legacy.get("requires_tools", []))
            value.setdefault("selection_tags", legacy.get("selection_tags", []))
        return value


class EvidenceRef(Record, frozen=True):
    """Point to a registered trace event, run, summary or evolution artifact.

    Evidence resolution checks existence and scope; a reference by itself does not
    turn a model-authored claim into independently verified evidence.
    """

    ref_type: Literal[
        "trace_event",
        "task_run",
        "feedback",
        "summary",
        "evolution_attempt",
        "policy_trial",
        "artifact",
        "evolution_decision",
        "evolution_outcome",
        "candidate_comparison",
        "memory_entry",
    ]
    ref_id: Id


class Claim(Record, frozen=True):
    """State an observation or hypothesis with evidence and explicit uncertainty.

    Observed claims require references. External verification additionally requires
    evaluator feedback, checked by the evidence service before accepting a report.
    """

    claim_id: Id
    text: str
    status: Literal["observed", "hypothesis", "externally_verified", "disputed"]
    evidence_refs: list[EvidenceRef]
    uncertainty: str = Field(min_length=1)

    @model_validator(mode="after")
    def evidence_required(self) -> Self:
        """Require traceable evidence for claims presented as observed or verified."""
        if self.status != "hypothesis" and not self.evidence_refs:
            raise ValueError("non-hypothesis claim needs evidence")
        if self.status == "externally_verified" and not any(
            e.ref_type == "feedback" for e in self.evidence_refs
        ):
            raise ValueError("external claim requires evaluator feedback")
        return self


class ModelPricing(Record, frozen=True):
    """Freeze the Host's versioned token prices and their stated source.

    These prices support cost accounting; they are not live provider quotations.
    """

    input_per_million: float = Field(ge=0)
    output_per_million: float = Field(ge=0)
    currency: str = "USD"
    source: str = Field(min_length=1)
    version: str = Field(min_length=1)


class Budget(Record, frozen=True):
    """Declare Host limits on calls, loop steps, elapsed time, tokens and cost.

    Optional monetary limits require a pricing contract. Policies can guide edit
    choices but cannot override the enclosing Host's resource limits.
    """

    max_model_calls: int = Field(default=8, ge=0)
    max_steps: int = Field(default=5, ge=0)
    max_tool_calls: int = Field(default=4, ge=0)
    max_wall_seconds: int = Field(default=30, ge=0)
    max_input_tokens: int | None = Field(default=None, ge=0)
    max_output_tokens: int | None = Field(default=None, ge=0)
    max_cost: float | None = Field(default=None, ge=0)
    pricing: ModelPricing | None = None

    @model_validator(mode="after")
    def unsupported_cost(self) -> Self:
        """Reject a monetary cap when no host pricing contract can enforce it."""
        if self.max_cost is not None and self.pricing is None:
            raise ValueError("monetary cost cap unsupported without a pricing adapter")
        return self


class Usage(Record, frozen=True):
    """Record consumed resources and whether reported token usage is available.

    Missing provider usage remains explicitly unknown; zero counters must not be
    interpreted as proof that an unreceipted request was free.
    """

    model_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    wall_seconds: float = Field(default=0, ge=0)
    token_usage_available: bool = True
    monetary_cost: float | None = Field(default=None, ge=0)


class Score(Record, frozen=True):
    """Represent an externally supplied metric with direction, unit and source.

    Comparisons require compatible evaluation contracts. HEAS does not normalize
    unrelated metrics into an invented cross-task fitness score.
    """

    name: str
    value: float
    direction: Literal["higher_is_better", "lower_is_better"]
    unit: str
    source: str


class EvaluationState(Record, frozen=True):
    """Separate evaluation availability from the outcome of task execution.

    Only completed evaluation can carry scores. Failed or invalid infrastructure
    receipts therefore remain distinct from legitimate low-scoring task answers.
    """

    evaluation_status: EvaluationStatus = "not_available"
    score: Score | None = None
    scores: list[Score] = Field(default_factory=list)

    @model_validator(mode="after")
    def no_fake_score(self) -> Self:
        """Prevent unavailable evaluation states from carrying objective scores."""
        if self.evaluation_status != "completed" and (self.score is not None or self.scores):
            raise ValueError("only completed external evaluation can have a score")
        keys = [(s.source, s.name) for s in self.scores]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate evaluation metric")
        if self.score and any(
            (s.source, s.name) == (self.score.source, self.score.name) and s != self.score
            for s in self.scores
        ):
            raise ValueError("primary score conflicts with metric")
        return self


class EvaluationResult(EvaluationState, frozen=True):
    """Bind an independent evaluator's receipt to a run and evaluator version.

    General feedback may contain private grading details. Only agent_feedback is
    included in the public evidence package; comparison eligibility is explicit.
    """

    evaluator_id: Id
    run_id: Id
    feedback: str
    agent_feedback: str | None = None
    evaluator_version: str = "unversioned"
    evaluator_versions: dict[str, str] = Field(default_factory=dict)
    comparison_eligible: bool | None = None

    @model_validator(mode="after")
    def score_source(self) -> Self:
        """Reject metrics whose declared source is outside the evaluator contract."""
        if any(
            s.source not in {self.evaluator_id, *self.evaluator_versions}
            for s in [*self.scores, *([self.score] if self.score else [])]
        ):
            raise ValueError("score must originate from evaluator")
        return self


class ToolBinding(Record, frozen=True):
    """Bind a genome to an approved, versioned Tool manifest and permissions.

    The trusted registry validates the manifest and implementation before execution;
    an agent cannot grant itself approval by constructing this record.
    """

    tool_id: Id
    version_id: Id
    manifest_ref: str
    permissions: list[str] = Field(default_factory=list)


class TaskGenome(Record, frozen=True):
    """Represent the heritable Skill, Memory and Tool capabilities of one Task Agent.

    Task execution reads a fixed version across a batch. Evolution creates a child
    with explicit parent lineage; runs and reflections are separate evidence and do
    not silently update inherited Memory. Models and evaluators remain Host choices.
    """

    agent_id: Id
    genome_id: Id
    generation: int = Field(default=0, ge=0)
    parent_genome_ids: list[Id] = Field(default_factory=list)
    skills: list[SkillRef] = Field(default_factory=list)
    memory: ArtifactRef
    tools: list[ToolBinding]
    created_from_proposal_id: Id | None = None
    created_at: datetime = Field(default_factory=now)
    artifact_status: Literal["draft", "static_validated", "quarantined", "rejected"]


class TaskSpec(Record, frozen=True):
    """Describe a provider-owned task with a public view and separately held private data.

    The runtime exposes public payloads to the Task Agent. Partition authorization
    and private evaluator inputs stay behind the provider and evaluator ports.
    """

    schema_version: Literal["1.0", "1.1"] = "1.1"
    task_id: Id
    provider_id: Id
    instruction: str
    payload_refs: list[ArtifactRef] = Field(default_factory=list)
    environment_spec: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)
    agent_input: dict[str, Any] = Field(default_factory=dict)
    public_payload_refs: list[ArtifactRef] = Field(default_factory=list)

    def public_view(self) -> dict[str, Any]:
        """Expose only the task instruction and agent input, withholding evaluator metadata."""
        return {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "agent_input": self.agent_input,
        }


class RunPlan(Record, frozen=True):
    """Freeze a batch's genomes, tasks, execution budgets and evaluation settings.

    Task sources and evaluators are Host inputs rather than inherited capabilities.
    One plan can run multiple independent tasks against the same genome snapshot.
    """

    run_plan_id: Id
    genome_ids: list[Id] = Field(min_length=1)
    task_provider_id: Id
    task_ids: list[Id] = Field(min_length=1)
    environment_adapter_id: Id
    task_evaluator_id: Id | None = None
    task_partition: Literal[
        "development", "evolution_train", "evolution_validation", "meta_validation", "final_test"
    ] = "development"
    attempts_per_task: int = Field(default=1, ge=1)
    random_seed: int = 23
    budget: Budget = Field(default_factory=Budget)
    model_config_ref: str = "configs/model_profile.json"
    memory_mode: Literal["frozen_inherited_memory"] = "frozen_inherited_memory"
    trace_level: Literal["redacted_structured"] = "redacted_structured"
    max_tasks: int | None = Field(default=None, ge=1)
    max_retries: int = Field(default=0, ge=0, le=10)
    retry_termination_reasons: list[str] = Field(default=["model_error", "environment_error"])
    retry_interrupted: bool = False
    task_suite_id: Id | None = None
    task_suite_version: str | None = None
    evaluation_suite_id: Id | None = None
    evaluator_version: str | None = None


class TaskResult(Record, frozen=True):
    """Carry an agent's text, structured output and content-addressed artifacts.

    The runtime attaches termination and evaluation state after execution. Produced
    artifacts must pass store verification before they enter archived run evidence.
    """

    text: str | None = None
    structured_output: dict[str, Any] | list[Any] | None = None
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    environment_state_ref: ArtifactRef | None = None
    termination_reason: str = "final_answer"
    errors: list[str] = Field(default_factory=list)
    evaluation: EvaluationState = Field(default_factory=EvaluationState)


class NativeToolCall(Record, frozen=True):
    """Represent one provider-native tool request with its correlation identity.

    Raw arguments may be retained for protocol receipts, but only validated argument
    objects can cross the trusted Tool Registry execution boundary.
    """

    call_id: str = Field(min_length=1, max_length=256)
    tool_id: str
    arguments: dict[str, Any] | None = None
    raw_arguments: str = "{}"


class NativeModelTurn(Record, frozen=True):
    """Validate one native model turn containing text, tool calls or a final result.

    The Task Loop checks call identities and permissions before dispatching tools.
    """

    protocol: Literal["native"] = "native"
    content: str | None = None
    tool_calls: list[NativeToolCall] = Field(default_factory=list)
    result: TaskResult | None = None


class TaskRunRecord(Record, frozen=True):
    """Archive one independent execution of a fixed genome on one task instance.

    Trace and public evidence references remain separately reloadable. Termination
    reason describes execution; evaluation availability does not imply success.
    """

    run_id: Id
    run_plan_id: Id
    genome_id: Id
    task_id: Id
    attempt_index: int = Field(ge=0)
    trace_ref: str
    start_time: datetime
    end_time: datetime
    termination_reason: Literal[
        "final_answer",
        "max_steps",
        "budget_exceeded",
        "tool_error",
        "model_error",
        "environment_error",
        "cancelled",
    ]
    objective_feedback_ref: str | None
    evaluation: EvaluationState
    usage: Usage
    task_partition: str = "development"
    evidence_ref: str | None = None
    evidence_sha256: str | None = None
    output: str = ""
    result: TaskResult | None = None
    retry_index: int = Field(default=0, ge=0)
    task_instance_sha256: str | None = None


class TraceEvent(Record, frozen=True):
    """Record an ordered, sanitized observation of one Task Agent execution.

    Capability injection and model-reported Skill use are distinct event evidence;
    neither is an independent measurement of capability improvement.
    """

    event_id: Id
    run_id: Id
    step: int = Field(ge=0)
    timestamp: datetime = Field(default_factory=now)
    event_type: Literal[
        "run_start",
        "capability_context",
        "model_call_start",
        "model_call_end",
        "tool_call_start",
        "tool_call_end",
        "tool_call_error",
        "feedback_received",
        "run_end",
        "security_event",
    ]
    actor: str = "runtime"
    skill_refs: list[str] = Field(default_factory=list)
    memory_refs: list[str] = Field(default_factory=list)
    tool_ids: list[str] = Field(default_factory=list)
    safe_payload: dict[str, Any] = Field(default_factory=dict)


class EditHypothesis(Record, frozen=True):
    """Suggest a capability edit with its rationale and supporting evidence.

    The Evolution Agent must still choose, validate and independently test edits;
    this report never modifies a genome directly.
    """

    target_kind: TargetKind
    operation: OperationKind
    rationale: str
    evidence_refs: list[EvidenceRef]
    verification_idea: str
    uncertainty: str = Field(min_length=1)


class PolicyEditHypothesis(Record, frozen=True):
    """Suggest a change to an allowlisted evolution-policy field.

    The Meta-Evolution Agent treats this as uncertain guidance from prior attempts,
    not authority to change evaluation, safety or promotion criteria.
    """

    target_policy_field: PolicyField
    proposed_change: Any
    rationale: str
    evidence_refs: list[EvidenceRef]
    verification_idea: str
    uncertainty: str = Field(min_length=1)


class RunReflection(Record, frozen=True):
    """Analyze one archived run without rewriting the Task Agent's capabilities.

    Observations, strengths, weaknesses and suggested edits retain their evidence
    and uncertainty. Model reflection is not independent evaluator feedback.
    """

    reflection_id: Id
    run_id: Id
    genome_id: Id
    observations: list[Claim]
    strength_hypotheses: list[Claim]
    weakness_hypotheses: list[Claim]
    failure_hypotheses: list[Claim] = Field(default_factory=list)
    suggested_capability_edits: list[EditHypothesis]
    feedback_status: EvaluationStatus
    source_trace_ref: str
    created_at: datetime = Field(default_factory=now)


class TaskExperienceSummary(Record, frozen=True):
    """Consolidate multiple runs of one genome into evidence for capability evolution.

    Program-computed statistics are stored separately from model-authored strengths,
    failure patterns and edit hypotheses. Summaries may cite previous windows.
    """

    summary_id: Id
    agent_id: Id
    genome_id: Id
    run_ids: list[Id] = Field(min_length=2)
    window_id: Id
    objective_stats_ref: str
    feedback_coverage: dict[str, int]
    strengths: list[Claim]
    weaknesses: list[Claim]
    recurring_patterns: list[Claim]
    capability_interactions: list[Claim] = Field(default_factory=list)
    evolution_hypotheses: list[EditHypothesis]
    limitations: list[str]
    created_at: datetime = Field(default_factory=now)
    aggregation_ref: str | None = None
    parent_summary_id: Id | None = None


class SkillContent(Record, frozen=True):
    """Supply proposed Skill text and its typed selection/dependency declarations.

    The materializer validates the resulting capability set before committing it.
    """

    content: str = Field(min_length=1)
    requires_tools: list[Id] = Field(default_factory=list)
    selection_tags: list[str] = Field(default_factory=list)
    applicability: str = ""
    conflicts_with: list[Id] = Field(default_factory=list)
    constraints: dict[str, str] = Field(default_factory=dict)


class MemoryContent(Record, frozen=True):
    """Supply an explicit proposed version of heritable Markdown Memory.

    Runtime traces and reflections enter this resource only through an approved
    evolution operation; executing a task does not implicitly write Memory.
    """

    content: str = Field(min_length=1)


class ToolContent(Record, frozen=True):
    """Propose an existing approved Tool binding or new source for Host review.

    Exactly one source is allowed. Generated code remains a proposal and cannot
    become executable merely by appearing in an evolution response.
    """

    binding: ToolBinding | None = None
    generated_source: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def one_source(self) -> Self:
        """Require exactly one executable tool source rather than an ambiguous proposal."""
        if (self.binding is None) == (self.generated_source is None):
            raise ValueError(
                "tool content requires exactly one approved binding or proposed source"
            )
        return self


class EvolutionOperation(Record, frozen=True):
    """Describe one typed inheritance, recombination, mutation, innovation or prune edit.

    Source lineage is explicit for inherited resources. New content is typed and
    must pass fixed capability and permission gates before materialization.
    """

    schema_version: Literal["1.1"] = "1.1"
    operation: OperationKind
    target_kind: TargetKind
    target_ref: Id | None = None
    change_request: str = Field(min_length=1)
    source_genome_id: Id | None = None
    source_artifact_id: Id | None = None
    skill: SkillContent | None = None
    memory: MemoryContent | None = None
    tool: ToolContent | None = None
    generate_content: bool = False

    @model_validator(mode="after")
    def executable_contract(self) -> Self:
        """Validate capability-operation payloads before trusted materialization."""
        if self.extensions:
            raise ValueError("operation extensions cannot carry executable input; use typed fields")
        content = {"skill": self.skill, "memory": self.memory, "tool": self.tool}
        if any(value is not None and key != self.target_kind for key, value in content.items()):
            raise ValueError("content type must match target_kind")
        if self.operation in {"mutate", "prune"} and not self.target_ref:
            raise ValueError("mutate/prune requires target_ref")
        if self.operation in {"inherit", "recombine"}:
            if not self.source_genome_id or not self.source_artifact_id:
                raise ValueError("inherit/recombine requires explicit parent and capability source")
        if self.operation in {"mutate", "innovate"}:
            if content[self.target_kind] is None and not self.generate_content:
                raise ValueError("edit requires typed content or explicit generate_content")
        elif any(value is not None for value in content.values()) or self.generate_content:
            raise ValueError("inherit/recombine/prune cannot carry new content")
        if self.generate_content and content[self.target_kind] is not None:
            raise ValueError("content and generate_content are mutually exclusive")
        return self


class EvolutionProposal(Record, frozen=True):
    """Propose evidence-linked capability edits under a specific EvolutionPolicy.

    Expected benefit and verification plan are hypotheses. Static validity and
    offspring creation do not establish independently measured improvement.
    """

    proposal_id: Id
    policy_id: Id
    parent_genome_ids: list[Id] = Field(min_length=1)
    operations: list[EvolutionOperation] = Field(min_length=1)
    evidence_refs: list[EvidenceRef] = Field(min_length=1)
    hypothesized_benefit: str
    verification_plan: str
    proposal_status: Literal["draft", "static_validated", "quarantined", "rejected"] = "draft"


class ValidationReport(Record, frozen=True):
    """Record static schema, dependency and permission checks on an evolution attempt.

    A passed security gate permits materialization; it is not a performance score.
    """

    schema_valid: bool
    dependencies_valid: bool
    permissions_valid: bool
    security_gate: Literal["passed", "rejected", "quarantined"]
    errors: list[str]


class EvolutionAttemptRecord(Record, frozen=True):
    """Archive one policy-guided attempt to produce Task Agent offspring.

    Parent/child lineage, validation and resources survive failed materialization.
    Independent execution outcomes are recorded separately by the evolution engine.
    """

    evolution_attempt_id: Id
    policy_id: Id
    proposal_id: Id
    parent_genome_ids: list[Id]
    offspring_genome_ids: list[Id]
    materialization_status: Literal["not_started", "materialized", "failed", "quarantined"]
    validation_report_ref: str
    evaluation: EvaluationState = Field(default_factory=EvaluationState)
    resource_usage: Usage
    started_at: datetime
    finished_at: datetime
    decision_ref: str | None = None


class EvolutionReflection(Record, frozen=True):
    """Analyze an attempt's materialization or independently observed outcome.

    New feedback creates a revision linked to earlier reflections. Strategy-change
    hypotheses remain evidence reports for the Meta-Evolution Agent.
    """

    evolution_reflection_id: Id
    evolution_attempt_id: Id
    observations: list[Claim]
    diagnosis_hypotheses: list[Claim]
    strategy_change_hypotheses: list[PolicyEditHypothesis]
    evidence_refs: list[EvidenceRef]
    created_at: datetime = Field(default_factory=now)
    reflection_phase: Literal["materialization", "outcome"] = "materialization"
    outcome_ref: str | None = None
    revision: int = Field(default=1, ge=1)
    supersedes_reflection_id: Id | None = None


class EvolutionOutcomeReflection(EvolutionReflection, frozen=True):
    """Specialize an attempt reflection for post-execution outcome evidence.

    This phase can discuss comparisons while preserving the original static report.
    """

    reflection_phase: Literal["outcome"] = "outcome"


class EvolutionDecision(Record, frozen=True):
    """Record L2 parent selection, objective and expected benefit before proposing edits.

    Selection rationale cites archived experience; its expectation remains uncertain
    until offspring are tested under a compatible independent evaluation contract.
    """

    decision_id: Id
    policy_id: Id
    parent_genome_ids: list[Id] = Field(min_length=1)
    objective: str
    selection_rationale: str
    expected_benefit: str
    evidence_refs: list[EvidenceRef]
    uncertainty: str = Field(min_length=1)
    parent_selection_ref: str | None = None


class EvolutionExperienceSummary(Record, frozen=True):
    """Consolidate multiple attempts under one policy into evidence for meta-evolution.

    Effective patterns, bottlenecks and strategy hypotheses are distinct from the
    objective statistics and evaluation coverage referenced by this report.
    """

    summary_id: Id
    evolution_agent_id: Id
    policy_id: Id
    evolution_attempt_ids: list[Id] = Field(min_length=2)
    objective_stats_ref: str
    evaluation_coverage: dict[str, int]
    effective_patterns: list[Claim]
    ineffective_patterns: list[Claim]
    stage_bottlenecks: list[Claim]
    strategy_hypotheses: list[PolicyEditHypothesis]
    limitations: list[str]
    created_at: datetime = Field(default_factory=now)
    aggregation_ref: str | None = None
    outcome_refs: list[EvidenceRef] = Field(default_factory=list)
    comparison_refs: list[EvidenceRef] = Field(default_factory=list)


class ParentSelectionConfig(Record, frozen=True):
    """Configure evidence-aware parent ranking and exploratory capability diversity.

    Weights guide a semantic heuristic rather than define universal objective
    fitness. Unknown or incomparable performance remains explicitly qualified.
    """

    max_parents: int = Field(default=1, ge=1, le=8)
    performance_weight: float = Field(default=1, ge=0)
    diversity_weight: float = Field(default=1, ge=0)
    historical_gain_weight: float = Field(default=1, ge=0)
    exploration_weight: float = Field(default=0.25, ge=0)


class EvolutionPolicy(Record, frozen=True):
    """Version L2 diagnosis, selection and operation guidance as an evolvable object.

    Meta-Evolution can create a descendant policy, separate from a TaskGenome.
    Host safety, hard resource ceilings and evaluator rules are outside its edit
    surface; activation requires independent evidence under fixed trial criteria.
    """

    policy_id: Id
    parent_policy_id: Id | None = None
    diagnosis_policy: str = "Inspect multi-run evidence and retain uncertainty."
    selection_policy: str = "Select evidence-covered parents; retain complementary skills."
    parent_selection: ParentSelectionConfig | None = None
    operator_policy: str = "Prefer mutate on repeated capability gaps."
    generation_policy: str = "Make one attributable change."
    validation_guidance: str = "Check dependencies and permissions before evaluation."
    allowed_operations: list[OperationKind] = Field(
        default=["inherit", "recombine", "mutate", "innovate", "prune"],
        min_length=1,
    )
    proposal_budget: int = Field(default=4, ge=1, le=8)
    policy_status: Literal[
        "draft", "static_validated", "trial_pending", "approved_baseline", "promoted", "rejected"
    ] = "approved_baseline"
    created_at: datetime = Field(default_factory=now)


class PolicyProposal(Record, frozen=True):
    """Archive the Meta-Evolution Agent's evidence-linked policy change set.

    The child is a candidate strategy version until an independent matched trial
    supports activation; proposing it alone does not replace the active policy.
    """

    proposal_id: Id
    parent_policy_id: Id
    proposed_policy_ref: ArtifactRef
    evidence_refs: list[EvidenceRef]
    expected_effect: str
    trial_design: dict[str, Any]
    change_set: dict[str, Any]


class MetaPolicyTrial(Record, frozen=True):
    """Archive an independent old/new evolution-policy comparison and fixed decision.

    Run plans, seeds and objective metrics remain reloadable evidence. An invalid,
    unevaluated or interrupted trial cannot justify policy adoption.
    """

    trial_id: Id
    old_policy_id: Id
    new_policy_id: Id
    run_plan_refs: list[str]
    start_genome_ids: list[Id]
    random_seeds: list[int]
    budget_rule: str
    comparison_status: Literal["not_available", "pending", "completed", "invalid"]
    objective_metrics_ref: str | None
    decision: Literal["undecided", "adopt", "reject"]
    decision_evidence_refs: list[EvidenceRef]

    @model_validator(mode="after")
    def no_unearned_promotion(self) -> Self:
        """Keep unfinished policy comparisons undecided and require objective evidence."""
        if self.comparison_status != "completed" and self.decision != "undecided":
            raise ValueError("uncompleted policy trial must remain undecided")
        if self.comparison_status == "completed" and not self.objective_metrics_ref:
            raise ValueError("completed comparison requires objective metrics")
        return self


def heas_ext(record: Record) -> dict[str, Any]:
    """Return producer-namespaced HEAS extensions from a versioned record."""
    metadata = record.extensions.get("heas")
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise ValueError("HEAS extension metadata must be an object when supplied")
    return dict(metadata)


class AgentObservation(Record, frozen=True):
    """Expose public environment state to one independent Task Agent.

    Private grading inputs belong to the external evaluator and must not appear
    in this observation or the model context assembled from it.
    """

    public_data: dict[str, Any]


class RunEvidencePackage(Record, frozen=True):
    """Freeze the public task, injected capabilities, trace and output for one run.

    Reflections consume this package rather than private evaluator payloads. The
    run record binds its stored bytes by hash for later evidence verification.
    """

    schema_version: Literal["1.1"] = "1.1"
    run_id: Id
    task_partition: str
    task_input: dict[str, Any]
    genome: TaskGenome
    loaded_capabilities: list[dict[str, Any]]
    events: list[TraceEvent]
    final_output: str
    final_artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    termination_reason: str
    evaluation: EvaluationState
    public_feedback: str | None
    evidence_refs: list[EvidenceRef]
    result: TaskResult | None = None
