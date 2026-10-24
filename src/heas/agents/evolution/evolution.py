"""L2 proposes; fixed materializer selects, copies, validates and commits offspring."""

import difflib
from dataclasses import dataclass
from typing import Any

from pydantic import Field

from heas.agents.evolution.parents import ArchiveParentSelection, ParentSelectionPolicy
from heas.agents.task.capabilities import CapabilityEditor
from heas.agents.task.memory import MemoryManager, explicit_conflicts
from heas.agents.task.tools import ToolRegistry
from heas.core.domain import (
    ArtifactRef,
    Budget,
    EvidenceRef,
    EvolutionAttemptRecord,
    EvolutionDecision,
    EvolutionOperation,
    EvolutionPolicy,
    EvolutionProposal,
    Record,
    SkillRef,
    TaskExperienceSummary,
    TaskGenome,
    ValidationReport,
    now,
    uid,
)
from heas.core.errors import BudgetExceeded, HEASError, RecoveryRequired, SecurityError
from heas.evidence.aggregation import bounded
from heas.evidence.experience import validate_evidence
from heas.evidence.history import HistoryContext
from heas.infrastructure.model import CallBudget, ModelGateway
from heas.infrastructure.storage import FileStore, digest, redact


class ProposalAnalysis(Record):
    """Validate a model's evidence-linked capability proposal before materialization."""

    parent_genome_ids: list[str] = Field(min_length=1)
    operations: list[EvolutionOperation] = Field(min_length=1)
    evidence_refs: list[EvidenceRef] = Field(min_length=1)
    hypothesized_benefit: str
    verification_plan: str


class DecisionAnalysis(Record):
    """Validate semantic parent selection and its explicitly uncertain rationale."""

    parent_genome_ids: list[str] = Field(min_length=1)
    objective: str
    selection_rationale: str
    expected_benefit: str
    evidence_refs: list[EvidenceRef] = Field(min_length=1)
    uncertainty: str = Field(min_length=1)


class CapabilityValidator:
    """Enforce dependency, conflict, and host permission constraints."""

    def __init__(self, store: FileStore, tools: ToolRegistry):
        self.store, self.tools = store, tools

    def validate(self, skills: list[SkillRef], tools: list[Any]) -> None:
        """Reject a capability configuration that violates its fixed host contract."""
        if len({s.artifact_id for s in skills}) != len(skills):
            raise HEASError("duplicate skill identity")
        if len({t.tool_id for t in tools}) != len(tools):
            raise HEASError("conflicting tool versions")
        visible = {t.tool_id for t in tools}
        self.tools.validate_dependencies(tools)
        constraints: dict[str, str] = {}
        instructions = []
        for tool in tools:
            self.tools.validate(tool)
        for skill in skills:
            instructions.append(self.store.verify(skill).decode("utf-8"))
            if set(skill.conflicts_with).intersection(s.artifact_id for s in skills):
                raise HEASError("explicit skill incompatibility")
            for key, value in skill.constraints.items():
                if key in constraints and constraints[key] != value:
                    raise HEASError("conflicting skill constraint: " + key)
                constraints[key] = value
            manifest = self.store.load(skill.uri.replace("SKILL.md", "manifest.json"), SkillRef)
            if manifest != skill:
                raise HEASError("skill manifest mismatch")
            if not set(skill.requires_tools).issubset(visible):
                raise HEASError("skill has missing tool dependencies")
        conflicts = explicit_conflicts(instructions)
        if conflicts:
            raise HEASError("; ".join(conflicts))


class GenomeMaterializer:
    """Create independent capability snapshots without rewriting parent genomes."""

    def __init__(self, store: FileStore, tools: ToolRegistry):
        self.store = store
        self.validator = CapabilityValidator(store, tools)

    def create_seed(self, skills: list[SkillRef], memory: str, tools: list[Any]) -> TaskGenome:
        """Commit a host-defined initial genome after static capability validation."""
        return self._commit(skills, memory, tools, [], None)

    def _commit(
        self,
        skills: list[SkillRef],
        memory: str,
        tools: list[Any],
        parents: list[TaskGenome],
        proposal: str | None,
        provenance: list[dict[str, Any]] | None = None,
    ) -> TaskGenome:
        self.validator.validate(skills, tools)
        MemoryManager.merge([memory])  # Validate explicit contradictions without rewriting a seed.
        agent_id, genome_id = uid("task_agent"), uid("genome")
        memory_ref = ArtifactRef(
            artifact_id=uid("memory"),
            version_id="v1",
            content_sha256=digest(memory.encode()),
            uri=f"task_agents/{agent_id}/{genome_id}/MEMORY.md",
        )
        genome = TaskGenome(
            agent_id=agent_id,
            genome_id=genome_id,
            generation=max((p.generation for p in parents), default=-1) + 1,
            parent_genome_ids=[p.genome_id for p in parents],
            skills=skills,
            memory=memory_ref,
            tools=tools,
            created_from_proposal_id=proposal,
            artifact_status="static_validated",
            extensions={
                "heas": {"asset_provenance": provenance or [], "quality_improvement": "unknown"}
            },
        )
        self.store.commit_genome(genome, memory)
        return genome

    def materialize(self, proposal: EvolutionProposal) -> TaskGenome:
        """Apply typed capability edits and commit a separate, lineage-linked genome.

        Parent resources are read by version and hash. Approved Tool manifests are
        rechecked; proposed new source is quarantined rather than executed. Static
        validation of the child makes no claim about task-quality improvement.
        """
        parents = [self.store.genome(p) for p in proposal.parent_genome_ids]
        first = parents[0]
        skills, tools = list(first.skills), list(first.tools)
        memory = self.store.verify(first.memory).decode("utf-8")
        provenance = [
            {
                "source_genome_id": first.genome_id,
                "source_artifact_id": ref.artifact_id,
                "content_sha256": ref.content_sha256,
            }
            for ref in [*first.skills, first.memory]
        ]
        diffs: list[str] = []
        for op in proposal.operations:
            old = ""
            if op.operation in {"inherit", "recombine"}:
                source = next((p for p in parents if p.genome_id == op.source_genome_id), None)
                if source is None:
                    raise HEASError("capability source must belong to a selected parent")
                if op.target_kind == "skill":
                    ref = next(
                        (s for s in source.skills if s.artifact_id == op.source_artifact_id), None
                    )
                    if ref is None:
                        raise HEASError("source skill not found")
                    self.store.verify(ref)
                    if ref not in skills:
                        skills.append(ref)
                    provenance.append(
                        {
                            "source_genome_id": source.genome_id,
                            "source_artifact_id": ref.artifact_id,
                            "content_sha256": ref.content_sha256,
                        }
                    )
                elif op.target_kind == "memory":
                    if source.memory.artifact_id != op.source_artifact_id:
                        raise HEASError("source memory not found")
                    text = self.store.verify(source.memory).decode("utf-8")
                    old = memory
                    memory = (
                        text if op.operation == "inherit" else MemoryManager.merge([memory, text])
                    )
                    diffs.extend(difflib.unified_diff(old.splitlines(), memory.splitlines()))
                    provenance.append(
                        {
                            "source_genome_id": source.genome_id,
                            "source_artifact_id": source.memory.artifact_id,
                            "content_sha256": source.memory.content_sha256,
                        }
                    )
                else:
                    binding = next(
                        (t for t in source.tools if t.tool_id == op.source_artifact_id), None
                    )
                    if binding is None:
                        raise HEASError("source tool not found")
                    self.validator.tools.validate(binding)
                    if binding not in tools:
                        tools.append(binding)
                    provenance.append(
                        {
                            "source_genome_id": source.genome_id,
                            "source_tool_id": binding.tool_id,
                            "manifest_ref": binding.manifest_ref,
                        }
                    )
                continue
            if op.target_kind == "tool":
                if op.tool and op.tool.generated_source:
                    quarantine_id = uid("tool_proposal")
                    self.store.put_bytes(
                        f"quarantine/{quarantine_id}/proposed.py",
                        redact(op.tool.generated_source).encode("utf-8"),
                    )
                    self.store.put(
                        f"quarantine/{quarantine_id}/review.json",
                        {
                            "schema_version": "1.0",
                            "executable": False,
                            "status": "pending_trusted_review",
                            "proposal_id": proposal.proposal_id,
                        },
                    )
                    raise SecurityError("generated tool quarantined; trusted approval required")
                if op.operation in {"prune", "mutate"} and not any(
                    t.tool_id == op.target_ref for t in tools
                ):
                    raise HEASError("target tool not found")
                if op.operation == "prune":
                    tools = [t for t in tools if t.tool_id != op.target_ref]
                else:
                    if op.tool is None or op.tool.binding is None:
                        raise HEASError("typed tool binding required")
                    binding = op.tool.binding
                    self.validator.tools.validate(binding)
                    tools = [
                        t
                        for t in tools
                        if t.tool_id != op.target_ref and t.tool_id != binding.tool_id
                    ] + [binding]
                continue
            if op.target_kind == "memory":
                if (
                    op.operation in {"mutate", "prune"}
                    and op.target_ref != first.memory.artifact_id
                ):
                    raise HEASError("target memory not found")
                old = memory
                if op.operation == "prune":
                    memory = "# Memory\nNo retained knowledge.\n"
                else:
                    if op.memory is None:
                        raise HEASError("typed memory content required")
                    memory = (
                        op.memory.content
                        if op.operation == "mutate"
                        else (memory + "\n" + op.memory.content)
                    )
                new = memory
            else:
                ref = next((s for s in skills if s.artifact_id == op.target_ref), None)
                if op.operation in {"mutate", "prune"} and ref is None:
                    raise HEASError("target skill not found")
                old = self.store.verify(ref).decode("utf-8") if ref else ""
                if ref and op.operation in {"prune", "mutate"}:
                    skills.remove(ref)
                new = ""
                if op.operation != "prune":
                    if op.skill is None:
                        raise HEASError("typed skill content required")
                    new = op.skill.content
                    if redact(new) != new:
                        raise SecurityError("credential-like content in capability")
                    skills.append(
                        self.store.skill(
                            new,
                            op.skill.requires_tools,
                            op.skill.selection_tags,
                            applicability=op.skill.applicability,
                            conflicts_with=op.skill.conflicts_with,
                            constraints=op.skill.constraints,
                        )
                    )
            if redact(new) != new:
                raise SecurityError("credential-like content in capability")
            diffs.extend(
                difflib.unified_diff(
                    old.splitlines(),
                    new.splitlines(),
                    fromfile="parent",
                    tofile="offspring",
                    lineterm="",
                )
            )
        self.store.put(
            f"evolution/diffs/{proposal.proposal_id}.json",
            {"schema_version": "1.0", "diff": diffs, "asset_provenance": provenance},
        )
        return self._commit(skills, memory, tools, parents, proposal.proposal_id, provenance)


@dataclass
class _AttemptProgress:
    """Retain diagnostic state even when a generation or validation phase fails."""

    directory: str
    proposal_id: str
    parent_ids: list[str]
    schema_valid: bool = False
    decision_uri: str | None = None
    phase: str = "generation"


@dataclass
class _EvolutionContext:
    """Carry the bounded evidence chosen for one diagnosis and proposal."""

    summaries: list[TaskExperienceSummary]
    refs: list[EvidenceRef]
    supporting_evidence: list[dict[str, Any]]
    decision: EvolutionDecision


class EvolutionService:
    """Translate archived task experience into statically validated child genomes.

    Diagnosis and edit generation use the active policy and shared call budget.
    Materialization enforces fixed dependency and permission gates. The engine,
    not this service, executes offspring and obtains independent comparisons.
    """

    def __init__(
        self,
        store: FileStore,
        gateway: ModelGateway,
        materializer: GenomeMaterializer,
        budget: Budget | None = None,
    ):
        self.store, self.gateway, self.materializer = store, gateway, materializer
        self.budget = budget or Budget(max_wall_seconds=60)
        self.parent_selection: ParentSelectionPolicy = ArchiveParentSelection()
        self.history = HistoryContext(store, gateway)

    def attempt(
        self,
        summaries: list[TaskExperienceSummary],
        policy: EvolutionPolicy,
        *,
        proposal_override: EvolutionProposal | None = None,
        attempt_id: str | None = None,
        resume: bool = False,
    ) -> EvolutionAttemptRecord:
        """Create one offspring attempt with archived diagnosis and static validation.

        Parent selection, proposal generation, and capability materialization share
        one call budget. Failures retain their generation/validation phase and do
        not imply a task-quality score. Independent evaluation occurs in the engine.

        Args:
            summaries: Archived multi-run evidence eligible for parent selection.
            policy: Archived, statically admitted guidance; Host gates remain fixed.
            proposal_override: Explicit Host edit proposal, bypassing model generation.
            attempt_id: Optional stable identity reserved by the experiment journal.
            resume: Reuse a completed identical attempt without replaying effects.

        Returns:
            The committed attempt and its static outcome; improvement stays unknown.

        Raises:
            RecoveryRequired: If an existing request changed or remains uncommitted.
        """
        attempt_id, proposal_id = attempt_id or uid("attempt"), uid("proposal")
        directory = f"evolution/attempts/{attempt_id}"
        request = {
            "policy_id": policy.policy_id,
            "summary_ids": [s.summary_id for s in summaries],
            "proposal_override": proposal_override.model_dump(mode="json")
            if proposal_override
            else None,
        }
        request_uri = f"{directory}/request.json"
        if self.store.path(request_uri).exists():
            if not resume or self.store.read(request_uri) != request:
                raise RecoveryRequired("attempt already started or request changed")
            if self.store.path(f"{directory}/attempt.json").exists():
                return self.store.load(f"{directory}/attempt.json", EvolutionAttemptRecord)
            raise RecoveryRequired(
                "uncommitted materialization requires host inspection; not replayed"
            )
        self.store.put(request_uri, request)
        progress = _AttemptProgress(
            directory, proposal_id, list(dict.fromkeys(s.genome_id for s in summaries))
        )
        started = now()
        meter = CallBudget(self.budget)
        offspring: list[str] = []
        errors: list[str] = []
        status: Any = "failed"
        try:
            context = self._diagnose(summaries, policy, proposal_override, meter, progress)
            proposal = self._propose(context, policy, proposal_override, meter, progress)
            progress.schema_valid = True
            progress.phase = "validation"
            child = self._materialize(
                proposal, policy, proposal_override is not None, context.refs, meter, progress
            )
            offspring = [child.genome_id]
            status = "materialized"
        except Exception as exc:
            errors = [f"{progress.phase}:{type(exc).__name__}"]
            self.store.put(
                f"{directory}/failure.json",
                {
                    "schema_version": "1.0",
                    "phase": progress.phase,
                    "error_type": type(exc).__name__,
                    "message": redact(str(exc))
                    if isinstance(exc, HEASError)
                    else "model/schema failure",
                },
            )
            if isinstance(exc, SecurityError) and "quarantined" in str(exc):
                status = "quarantined"
        valid = status == "materialized"
        report_uri = f"{directory}/validation.json"
        report = ValidationReport(
            schema_valid=progress.schema_valid,
            dependencies_valid=valid,
            permissions_valid=valid,
            security_gate="passed"
            if valid
            else ("quarantined" if status == "quarantined" else "rejected"),
            errors=errors,
        )
        self.store.put(report_uri, report)
        attempt = EvolutionAttemptRecord(
            evolution_attempt_id=attempt_id,
            policy_id=policy.policy_id,
            proposal_id=progress.proposal_id,
            parent_genome_ids=progress.parent_ids,
            offspring_genome_ids=offspring,
            materialization_status=status,
            validation_report_ref=report_uri,
            resource_usage=meter.usage(),
            started_at=started,
            finished_at=now(),
            decision_ref=progress.decision_uri,
            extensions={"heas": {"quality_improvement": "unknown", "engineering_approved": valid}},
        )
        self.store.save_record(
            f"{directory}/attempt.json", attempt, "evolution_attempt", attempt_id
        )
        return attempt

    def _diagnose(
        self,
        summaries: list[TaskExperienceSummary],
        policy: EvolutionPolicy,
        proposal_override: EvolutionProposal | None,
        meter: CallBudget,
        progress: _AttemptProgress,
    ) -> _EvolutionContext:
        """Validate archived inputs, select evidence-covered parents, and save the L2 decision."""
        directory = progress.directory
        if not summaries:
            raise HEASError("evolution requires task experience")
        persisted_policy = EvolutionPolicy.model_validate(
            self.store.resolve(EvidenceRef(ref_type="artifact", ref_id=policy.policy_id))
        )
        if persisted_policy != policy or policy.policy_status in {"draft", "rejected"}:
            raise HEASError("policy must be an archived, statically admitted version")
        for summary in summaries:
            saved = self.store.resolve(EvidenceRef(ref_type="summary", ref_id=summary.summary_id))
            if TaskExperienceSummary.model_validate(saved) != summary:
                raise HEASError("evolution summary differs from archived evidence")
        selection_ref = None
        if proposal_override is None:
            summaries, selection = self.parent_selection.select(self.store, summaries, policy)
            selection_ref = f"{directory}/parent_selection.json"
            self.store.put(selection_ref, selection)
            progress.parent_ids = [s.genome_id for s in summaries]
        refs = [EvidenceRef(ref_type="summary", ref_id=s.summary_id) for s in summaries]
        supporting_evidence = []
        for summary in summaries:
            for run_id in summary.run_ids:
                detail = self.history.retriever.run_detail(run_id)
                detail["failure"] = detail["termination_reason"] != "final_answer"
                supporting_evidence.append(detail)
        supporting_evidence.sort(key=lambda row: not row["failure"])
        decision_context = {
            "policy": policy.model_dump(mode="json"),
            "summaries": [bounded(s.model_dump(mode="json"), 256, 2) for s in summaries],
            "candidate_genome_ids": progress.parent_ids,
            "evidence_refs": [r.model_dump(mode="json") for r in refs],
            "response_schema": DecisionAnalysis.model_json_schema(),
            "parent_selection_ref": selection_ref,
            "parent_requirement": "Use all configured candidate_genome_ids.",
        }
        if proposal_override is None:
            decision_context = self.history.prepare(
                "evolution_decision", decision_context, [], meter
            )
            diagnosis = self.gateway.structured(
                "evolution_decision",
                decision_context,
                meter,
                DecisionAnalysis,
                lambda report: validate_evidence(
                    self.store, report, {(r.ref_type, r.ref_id) for r in refs}
                ),
            )
        else:
            diagnosis = DecisionAnalysis(
                parent_genome_ids=proposal_override.parent_genome_ids,
                objective="Host-provided capability experiment",
                selection_rationale="Explicit host proposal with archived source evidence",
                expected_benefit=proposal_override.hypothesized_benefit,
                evidence_refs=proposal_override.evidence_refs,
                uncertainty="Benefit is a hypothesis until independently compared",
            )
        validate_evidence(self.store, diagnosis, {(r.ref_type, r.ref_id) for r in refs})
        if not set(diagnosis.parent_genome_ids).issubset(progress.parent_ids):
            raise HEASError("decision selected a parent without archived experience")
        if (
            policy.parent_selection
            and proposal_override is None
            and set(diagnosis.parent_genome_ids) != set(progress.parent_ids)
        ):
            raise HEASError("decision must honor the configured parent selection")
        decision = EvolutionDecision(
            decision_id=uid("decision"),
            parent_selection_ref=selection_ref,
            policy_id=policy.policy_id,
            **diagnosis.model_dump(exclude={"schema_version", "extensions"}),
        )
        progress.decision_uri = f"{directory}/decision.json"
        self.store.save_record(
            progress.decision_uri, decision, "evolution_decision", decision.decision_id
        )
        progress.parent_ids = decision.parent_genome_ids
        return _EvolutionContext(summaries, refs, supporting_evidence, decision)

    def _propose(
        self,
        context: _EvolutionContext,
        policy: EvolutionPolicy,
        proposal_override: EvolutionProposal | None,
        meter: CallBudget,
        progress: _AttemptProgress,
    ) -> EvolutionProposal:
        """Generate or accept the explicit Host proposal and persist it before validation."""
        directory = progress.directory
        summaries, refs = context.summaries, context.refs
        decision, supporting_evidence = context.decision, context.supporting_evidence
        if proposal_override is None:
            parents = [self.store.genome(p) for p in progress.parent_ids]
            proposal_context = {
                "policy": policy.model_dump(mode="json"),
                "decision": decision.model_dump(mode="json"),
                "parents": [bounded(p.model_dump(mode="json"), 512, 8) for p in parents],
                "parent_capabilities": [
                    {
                        "genome_id": p.genome_id,
                        "skills": [
                            {
                                "ref": s.model_dump(mode="json"),
                                "excerpt": self.store.verify(s).decode("utf-8")[:512],
                            }
                            for s in p.skills[:8]
                        ],
                        "memory": {
                            "ref": p.memory.model_dump(mode="json"),
                            "excerpt": self.store.verify(p.memory).decode("utf-8")[:512],
                        },
                    }
                    for p in parents
                ],
                "summaries": [bounded(s.model_dump(mode="json"), 256, 2) for s in summaries],
                "evidence_refs": [r.model_dump(mode="json") for r in refs],
                "response_schema": ProposalAnalysis.model_json_schema(),
            }
            proposal_context = self.history.prepare(
                "evolution_proposal", proposal_context, supporting_evidence, meter
            )
            analysis = self.gateway.structured(
                "evolution_proposal",
                proposal_context,
                meter,
                ProposalAnalysis,
                lambda report: validate_evidence(
                    self.store, report, {(r.ref_type, r.ref_id) for r in refs}
                ),
            )
            proposal = EvolutionProposal(
                proposal_id=progress.proposal_id,
                policy_id=policy.policy_id,
                **analysis.model_dump(exclude={"schema_version", "extensions"}),
            )
        else:
            proposal = proposal_override
            progress.proposal_id = proposal.proposal_id
        self.store.put(f"{directory}/proposal.json", proposal)
        return proposal

    def _materialize(
        self,
        proposal: EvolutionProposal,
        policy: EvolutionPolicy,
        host_override: bool,
        refs: list[EvidenceRef],
        meter: CallBudget,
        progress: _AttemptProgress,
    ) -> TaskGenome:
        """Enforce evidence, operation, budget and permission gates before creating a child."""
        directory = progress.directory
        validate_evidence(self.store, proposal, {(r.ref_type, r.ref_id) for r in refs})
        if proposal.policy_id != policy.policy_id:
            raise HEASError("proposal policy mismatch")
        if not set(proposal.parent_genome_ids).issubset(progress.parent_ids):
            raise HEASError("parent selection lacks summary evidence")
        if policy.parent_selection and not host_override:
            if set(proposal.parent_genome_ids) != set(progress.parent_ids):
                raise HEASError("proposal must honor configured multi-parent selection")
        if len(proposal.operations) > policy.proposal_budget:
            raise BudgetExceeded("proposal edit budget exceeded")
        if any(op.operation not in policy.allowed_operations for op in proposal.operations):
            raise SecurityError("operation outside active policy allowlist")
        progress.parent_ids = proposal.parent_genome_ids
        meter.check()
        resolved = EvolutionProposal.model_validate(
            {
                **proposal.model_dump(),
                "operations": [
                    CapabilityEditor(self.gateway)
                    .resolve(op, [self.store.genome(p) for p in progress.parent_ids], meter)
                    .model_dump()
                    for op in proposal.operations
                ],
            }
        )
        self.store.put(f"{directory}/resolved_proposal.json", resolved)
        child = self.materializer.materialize(resolved)
        return child
