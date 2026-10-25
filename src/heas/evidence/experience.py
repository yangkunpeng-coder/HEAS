"""Program-computed statistics followed by separate, evidence-checked model reports."""

from collections import Counter
from typing import Any

from pydantic import Field

from heas.core.domain import (
    Budget,
    Claim,
    EditHypothesis,
    EvidenceRef,
    EvolutionAttemptRecord,
    EvolutionExperienceSummary,
    EvolutionOutcomeReflection,
    EvolutionReflection,
    PolicyEditHypothesis,
    Record,
    RunEvidencePackage,
    RunReflection,
    TaskExperienceSummary,
    TaskRunRecord,
    uid,
)
from heas.core.errors import HEASError, IntegrityError, MissingEvidence
from heas.evidence.aggregation import AggregationConfig, HierarchicalExperienceAggregator, bounded
from heas.infrastructure.context import EvidenceRetriever
from heas.infrastructure.model import CallBudget, ModelGateway
from heas.infrastructure.storage import FileStore, digest


class RunAnalysis(Record, frozen=True):
    """Validate model reflection claims about one independently recorded execution."""

    observations: list[Claim]
    strength_hypotheses: list[Claim]
    weakness_hypotheses: list[Claim]
    failure_hypotheses: list[Claim] = Field(default_factory=list)
    suggested_capability_edits: list[EditHypothesis]


class TaskAnalysis(Record, frozen=True):
    """Validate capability diagnoses and hypotheses over a multi-run task window."""

    strengths: list[Claim]
    weaknesses: list[Claim]
    recurring_patterns: list[Claim]
    capability_interactions: list[Claim] = Field(default_factory=list)
    evolution_hypotheses: list[EditHypothesis]


class EvolutionAnalysis(Record, frozen=True):
    """Validate an attempt reflection without accepting self-assessed objective gains."""

    observations: list[Claim]
    diagnosis_hypotheses: list[Claim]
    strategy_change_hypotheses: list[PolicyEditHypothesis]


class EvolutionWindowAnalysis(Record, frozen=True):
    """Validate evidence-linked conclusions about multiple evolution attempts."""

    effective_patterns: list[Claim]
    ineffective_patterns: list[Claim]
    stage_bottlenecks: list[Claim]
    strategy_hypotheses: list[PolicyEditHypothesis]


def validate_evidence(
    store: FileStore, record: Record, allowed: set[tuple[str, str]] | None = None
) -> None:
    """Resolve allowed evidence identities before accepting model-authored claims."""

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            if "ref_type" in item and "ref_id" in item:
                ref = EvidenceRef.model_validate(item)
                if allowed is not None and (ref.ref_type, ref.ref_id) not in allowed:
                    raise MissingEvidence("evidence is outside analysis window")
                store.resolve(ref)
            if item.get("status") == "externally_verified":
                feedback = [
                    store.resolve(EvidenceRef.model_validate(e))
                    for e in item["evidence_refs"]
                    if e["ref_type"] == "feedback"
                ]
                if not feedback or any(f["evaluation_status"] != "completed" for f in feedback):
                    raise MissingEvidence("external claim lacks completed evaluator feedback")
            for value in item.values():
                visit(value)
        elif isinstance(item, list):
            for value in item:
                visit(value)

    visit(record.model_dump(mode="json"))


def latest_evolution_reflection(store: FileStore, attempt_id: str) -> str:
    """Read the newest attempt reflection revision while retaining older revisions."""
    directory = f"evolution/attempts/{attempt_id}"
    revisions = sorted(store.path(f"{directory}/feedback_reflections").glob("*.json"))
    return (
        revisions[-1].relative_to(store.root).as_posix()
        if revisions
        else f"{directory}/reflection.json"
    )


class ExperienceService:
    """Build evidence reports for the next evolution layer, never evaluator truth.

    Run reflections and task summaries inform L2; attempt reflections and evolution
    summaries inform L3. Statistics, claims and original evidence stay separately
    archived. Final-test evidence cannot enter this learning service.
    """

    def __init__(
        self,
        store: FileStore,
        gateway: ModelGateway,
        budget: Budget | None = None,
        aggregation_config: AggregationConfig | None = None,
    ):
        self.store, self.gateway = store, gateway
        self.budget = budget or Budget(max_wall_seconds=60)
        self.aggregator = HierarchicalExperienceAggregator(store, gateway, aggregation_config)
        self.retriever = EvidenceRetriever(store)
        self.archive_scope: str | None = None

    def _analyze(
        self,
        purpose: str,
        context: dict[str, Any],
        cls: type[Record],
        allowed: set[tuple[str, str]] | None = None,
    ) -> Record:
        if purpose == "evolution_reflection":
            context = bounded(context, self.aggregator.config.max_excerpt_chars, 4)
        if purpose == "task_reflection":
            context = bounded(context, self.aggregator.config.max_excerpt_chars, 4)
            # Keep the actual terminal event, even when middle trace events are sampled.
            if context.get("events"):
                context["events"] = context["events"][:3] + context.get("terminal_events", [])
        request = {**context, "response_schema": cls.model_json_schema()}
        from heas.infrastructure.storage import encode

        if len(encode(request)) > self.aggregator.config.max_context_bytes:
            from heas.core.errors import BudgetExceeded

            raise BudgetExceeded("single-analysis context exceeds configured cap")
        return self.gateway.structured(
            purpose,
            request,
            CallBudget(self.budget),
            cls,
            lambda record: validate_evidence(self.store, record, allowed),
        )

    def reflect_run(self, run: TaskRunRecord, *, enabled: bool = True) -> RunReflection | None:
        """Archive a model reflection over a hash-verified public run package.

        The optional model analysis does not alter the run, Genome or inherited Memory.
        Claims must resolve within this run's evidence scope; final_test is rejected.

        Args:
            run: Archived execution whose public evidence is analyzed.
            enabled: Return None without model analysis when False.

        Returns:
            The stored evidence-linked reflection, or None when disabled.
        """
        if not enabled:
            return None
        if run.task_partition == "final_test":
            raise HEASError("final_test evidence cannot enter learning/reflection contexts")
        persisted = self.retriever.get(EvidenceRef(ref_type="task_run", ref_id=run.run_id))
        if TaskRunRecord.model_validate(persisted) != run:
            raise HEASError("run is not identical to persisted evidence")
        events = self.store.trace(run.trace_ref)
        evidence_uri = run.evidence_ref
        if not evidence_uri:
            raise HEASError("reflection requires a complete RunEvidencePackage")
        if digest(self.store.path(evidence_uri).read_bytes()) != run.evidence_sha256:
            raise HEASError("run evidence package hash mismatch")
        package = self.store.load(evidence_uri, RunEvidencePackage)
        if package.run_id != run.run_id or package.genome.genome_id != run.genome_id:
            raise HEASError("run evidence package identity mismatch")
        allowed = {("trace_event", e.event_id) for e in events} | {("task_run", run.run_id)}
        allowed.update(
            ("memory_entry", entry["entry_id"])
            for capability in package.loaded_capabilities
            for entry in capability.get("memory", {}).get("entries", [])
        )
        if run.objective_feedback_ref:
            allowed.add(("feedback", run.run_id))
        semantic = self._analyze(
            "task_reflection",
            {
                "events": [e.model_dump(mode="json") for e in events],
                "terminal_events": [e.model_dump(mode="json") for e in events[-1:]],
                "analysis_rules": "Final answer is a stop condition, not task success. "
                "Distinguish injected skills, model-reported adoption, observable actions, "
                "and independently verified contributions. Explain failure hypotheses "
                "with event evidence and uncertainty; never invent facts or scores.",
                "termination_reason": run.termination_reason,
                "evaluation": run.evaluation.model_dump(mode="json"),
                "run_evidence": package.model_dump(mode="json"),
            },
            RunAnalysis,
            allowed,
        )
        validate_evidence(self.store, semantic, allowed)
        report = RunReflection(
            reflection_id=uid("reflection"),
            run_id=run.run_id,
            genome_id=run.genome_id,
            feedback_status=run.evaluation.evaluation_status,
            source_trace_ref=run.trace_ref,
            **semantic.model_dump(exclude={"schema_version", "extensions"}),
        )
        self.store.put(f"runs/{run.run_id}/reflection.json", report)
        return report

    def task_summary(
        self,
        runs: list[TaskRunRecord],
        window_id: str,
        previous_summary: TaskExperienceSummary | None = None,
    ) -> TaskExperienceSummary:
        """Summarize a fixed-genome run window into evidence for capability evolution.

        Program statistics precede bounded model aggregation. Recurring patterns and
        edit suggestions remain qualified hypotheses. A previous summary supports
        incremental windows without replacing their original run evidence.
        """
        incoming_ids = {r.run_id for r in runs}
        if previous_summary:
            saved_previous = self.store.resolve(
                EvidenceRef(ref_type="summary", ref_id=previous_summary.summary_id)
            )
            if TaskExperienceSummary.model_validate(saved_previous) != previous_summary:
                raise HEASError("previous summary differs from archived version")
            if incoming_ids.intersection(previous_summary.run_ids):
                raise HEASError("incremental summary received already-covered runs")
            prior = [
                TaskRunRecord.model_validate(
                    self.store.resolve(EvidenceRef(ref_type="task_run", ref_id=rid))
                )
                for rid in previous_summary.run_ids
            ]
            runs = prior + runs
        if len(runs) < 2 or len({r.genome_id for r in runs}) != 1:
            raise HEASError("summary requires multiple runs of a single frozen genome")
        if len({r.run_id for r in runs}) != len(runs):
            raise HEASError("duplicate run in summary window")
        if any(r.task_partition == "final_test" for r in runs):
            raise HEASError("final_test cannot enter evolution-oriented summaries")
        for run in runs:
            saved = self.store.resolve(EvidenceRef(ref_type="task_run", ref_id=run.run_id))
            if TaskRunRecord.model_validate(saved) != run:
                raise HEASError("summary input differs from immutable run")
            evidence_uri = run.evidence_ref
            if not evidence_uri or digest(self.store.path(evidence_uri).read_bytes()) != (
                run.evidence_sha256
            ):
                raise HEASError("summary requires hash-verified complete run evidence")
        genome = self.store.genome(runs[0].genome_id)
        traces = [e for r in runs for e in self.store.trace(r.trace_ref)]
        stats: dict[str, Any] = {
            "schema_version": "1.0",
            "total_runs": len(runs),
            "termination_counts": dict(Counter(r.termination_reason for r in runs)),
            "tool_execution_counts": dict(
                Counter(
                    e.safe_payload.get("tool_id") for e in traces if e.event_type == "tool_call_end"
                )
            ),
            "skill_injection_counts": dict(
                Counter(
                    s for e in traces if e.event_type == "capability_context" for s in e.skill_refs
                )
            ),
            "skill_model_report_counts": dict(
                Counter(
                    s
                    for e in traces
                    if e.event_type == "model_call_end"
                    for s in e.safe_payload.get("model_reported_skill_ids", [])
                )
            ),
            "skill_linked_tool_action_counts": dict(
                Counter(s for e in traces if e.event_type == "tool_call_end" for s in e.skill_refs)
            ),
            "verified_positive_capability_contributions": None,
            "observable_error_counts": dict(
                Counter(
                    e.safe_payload.get("error_type")
                    for e in traces
                    if e.event_type == "tool_call_error"
                )
            ),
            "externally_evaluated_runs": sum(
                r.evaluation.evaluation_status == "completed" for r in runs
            ),
            "usage": {
                field: sum(getattr(r.usage, field) for r in runs)
                for field in (
                    "model_calls",
                    "tool_calls",
                    "input_tokens",
                    "output_tokens",
                    "wall_seconds",
                )
            },
        }
        refs = [EvidenceRef(ref_type="task_run", ref_id=r.run_id) for r in runs]
        stats["usage"]["token_usage_available"] = all(r.usage.token_usage_available for r in runs)
        if not stats["usage"]["token_usage_available"]:
            stats["usage"]["input_tokens"] = None
            stats["usage"]["output_tokens"] = None
        allowed = {(r.ref_type, r.ref_id) for r in refs} | {
            ("trace_event", e.event_id) for e in traces
        }
        allowed |= {("feedback", r.run_id) for r in runs if r.objective_feedback_ref}
        rows = []
        if previous_summary:
            previous_ref = EvidenceRef(ref_type="summary", ref_id=previous_summary.summary_id)
            allowed.add((previous_ref.ref_type, previous_ref.ref_id))
            rows.append(
                {
                    "source_ref": previous_ref.model_dump(mode="json"),
                    "prior_hypotheses": bounded(previous_summary.model_dump(mode="json")),
                }
            )
        for run in runs:
            if previous_summary and run.run_id not in incoming_ids:
                continue
            evidence_uri = run.evidence_ref
            if not evidence_uri:
                raise HEASError("summary requires complete run evidence")
            package = self.store.load(evidence_uri, RunEvidencePackage)
            events = [
                e
                for e in package.events
                if e.event_type in {"tool_call_error", "run_end", "tool_call_end"}
            ]
            for capability in package.loaded_capabilities:
                for entry in capability.get("memory", {}).get("entries", []):
                    allowed.add(("memory_entry", entry["entry_id"]))
            reflection_uri = f"runs/{run.run_id}/reflection.json"
            reflection = (
                self.store.read(reflection_uri)
                if self.store.path(reflection_uri).exists()
                else None
            )
            rows.append(
                {
                    "source_ref": EvidenceRef(ref_type="task_run", ref_id=run.run_id).model_dump(
                        mode="json"
                    ),
                    "task": bounded(package.task_input),
                    "result": bounded(
                        run.result.model_dump(mode="json") if run.result else {"text": run.output}
                    ),
                    "termination_reason": run.termination_reason,
                    "reflection": bounded(reflection),
                    "representative_events": [
                        bounded(e.model_dump(mode="json"))
                        for e in events[: self.aggregator.config.representative_events]
                    ],
                }
            )
        semantic, aggregation_ref = self.aggregator.aggregate(
            "task_summary",
            rows,
            stats,
            TaskAnalysis,
            self.budget,
            lambda report, scope: validate_evidence(self.store, report, scope),
            allowed,
        )
        sid = uid("task_summary")
        stats_uri = f"analysis/{sid}.stats.json"
        self.store.put(stats_uri, stats)
        summary = TaskExperienceSummary(
            extensions={"heas": {"scope": self.archive_scope}} if self.archive_scope else {},
            summary_id=sid,
            agent_id=genome.agent_id,
            genome_id=genome.genome_id,
            run_ids=[r.run_id for r in runs],
            window_id=window_id,
            aggregation_ref=aggregation_ref,
            parent_summary_id=previous_summary.summary_id if previous_summary else None,
            objective_stats_ref=stats_uri,
            feedback_coverage={
                "total_runs": len(runs),
                "externally_evaluated_runs": sum(
                    r.evaluation.evaluation_status == "completed" for r in runs
                ),
            },
            limitations=["No quality conclusion from execution or synthetic model output."],
            **semantic.model_dump(exclude={"schema_version", "extensions"}),
        )
        self.store.save_record(
            f"task_agents/{genome.agent_id}/{genome.genome_id}/summaries/{sid}.json",
            summary,
            "summary",
            sid,
        )
        return summary

    def _outcome_comparison_refs(self, outcome: dict[str, Any]) -> list[EvidenceRef]:
        """Collect present comparison URIs, rejecting malformed required outcome objects."""
        children = outcome.get("offspring")
        if not isinstance(children, list):
            raise IntegrityError("evolution outcome requires an offspring list")
        refs = []
        for child in children:
            if not isinstance(child, dict) or not isinstance(child.get("candidate"), dict):
                raise IntegrityError("evolution offspring requires a candidate object")
            candidate = child["candidate"]
            comparison_uri = candidate.get("comparison_ref")
            # An unassessed candidate legitimately has no comparison evidence.
            if comparison_uri is not None:
                refs.append(self.store.json_record_ref(comparison_uri, "candidate_comparison"))
        return refs

    def reflect_attempt(self, attempt: EvolutionAttemptRecord) -> EvolutionReflection:
        """Append a materialization or outcome reflection for one archived attempt.

        The latest available outcome determines the phase. Earlier revisions remain
        intact, and unavailable objective evaluation never becomes an invented gain.
        """
        refs = [EvidenceRef(ref_type="evolution_attempt", ref_id=attempt.evolution_attempt_id)]
        outcome_uri = f"evolution/attempts/{attempt.evolution_attempt_id}/outcome.json"
        outcome = (
            self.store.read_object(outcome_uri) if self.store.path(outcome_uri).exists() else None
        )
        execution_evidence = []
        directory = f"evolution/attempts/{attempt.evolution_attempt_id}"
        validation_uri = f"{directory}/validation_outcome.json"
        validation = (
            self.store.read_object(validation_uri)
            if self.store.path(validation_uri).exists()
            else None
        )
        if outcome and validation:
            outcome = {**outcome, "validation_outcome": validation}
        if outcome:
            refs.append(
                EvidenceRef(ref_type="evolution_outcome", ref_id=attempt.evolution_attempt_id)
            )
            refs.extend(self._outcome_comparison_refs(outcome))
            run_groups = [child["run_ids"] for child in outcome["offspring"]]
            if validation:
                run_groups.append(validation["run_ids"])
            for run_ids in run_groups:
                for run_id in run_ids:
                    run = TaskRunRecord.model_validate(
                        self.store.resolve(EvidenceRef(ref_type="task_run", ref_id=run_id))
                    )
                    if run.task_partition == "final_test":
                        raise HEASError("final_test cannot enter evolution outcome reflection")
                    refs.append(EvidenceRef(ref_type="task_run", ref_id=run_id))
                    evidence = {
                        "run_id": run_id,
                        "output": run.output,
                        "termination_reason": run.termination_reason,
                        "evaluation": run.evaluation.model_dump(mode="json"),
                        "events": [
                            e.model_dump(mode="json")
                            for e in self.store.trace(run.trace_ref)
                            if e.event_type in {"run_end", "tool_call_error"}
                        ],
                    }
                    if run.objective_feedback_ref:
                        refs.append(EvidenceRef(ref_type="feedback", ref_id=run_id))
                        evidence["public_feedback"] = self.store.read(
                            run.objective_feedback_ref
                        ).get("agent_feedback")
                    execution_evidence.append(evidence)
                    refs.extend(
                        EvidenceRef(ref_type="trace_event", ref_id=e["event_id"])
                        for e in evidence["events"]
                    )
        semantic = self._analyze(
            "evolution_reflection",
            {
                "attempt": attempt.model_dump(mode="json"),
                "validation_report": self.store.read(attempt.validation_report_ref),
                "decision": self.store.read(attempt.decision_ref) if attempt.decision_ref else None,
                "outcome": outcome,
                "execution_evidence": execution_evidence,
                "analysis_rules": "Distinguish legality, execution, external evaluation, matched "
                "improvement and reuse hypotheses. Outcome is unknown without independent "
                "comparison; a score difference alone does not establish causal attribution.",
                "evidence_refs": [r.model_dump(mode="json") for r in refs],
            },
            EvolutionAnalysis,
            {(r.ref_type, r.ref_id) for r in refs},
        )
        validate_evidence(self.store, semantic, {(r.ref_type, r.ref_id) for r in refs})
        cls = EvolutionOutcomeReflection if outcome else EvolutionReflection
        previous_uri = latest_evolution_reflection(self.store, attempt.evolution_attempt_id)
        previous = self.store.read(previous_uri) if self.store.path(previous_uri).exists() else None
        revision = previous.get("revision", 1) + 1 if previous else 1
        report = cls(
            revision=revision,
            supersedes_reflection_id=previous["evolution_reflection_id"] if previous else None,
            evolution_reflection_id=uid("evolution_reflection"),
            evolution_attempt_id=attempt.evolution_attempt_id,
            evidence_refs=refs,
            outcome_ref=outcome_uri if outcome else None,
            **semantic.model_dump(exclude={"schema_version", "extensions"}),
        )
        uri = f"{directory}/feedback_reflections/{revision:06d}.json"
        self.store.save_record(uri, report, "artifact", report.evolution_reflection_id)
        if not self.store.path(f"{directory}/reflection.json").exists():
            self.store.put(f"{directory}/reflection.json", report)
        if outcome and not self.store.path(f"{directory}/outcome_reflection.json").exists():
            self.store.put(f"{directory}/outcome_reflection.json", report)
        return report

    def evolution_summary(
        self, attempts: list[EvolutionAttemptRecord]
    ) -> EvolutionExperienceSummary:
        """Summarize multiple same-policy attempts into evidence for meta-evolution.

        Success/failure counts and available comparisons are computed independently of
        model hypotheses. Bounded aggregation retains source references for policy
        diagnosis without asserting that a self-reported strategy is superior.
        """
        if len(attempts) < 2 or len({a.policy_id for a in attempts}) != 1:
            raise HEASError("evolution summary requires multiple attempts under one policy")
        if len({a.evolution_attempt_id for a in attempts}) != len(attempts):
            raise HEASError("duplicate attempt")
        refs = [
            EvidenceRef(ref_type="evolution_attempt", ref_id=a.evolution_attempt_id)
            for a in attempts
        ]
        for ref, attempt in zip(refs, attempts, strict=True):
            if EvolutionAttemptRecord.model_validate(self.store.resolve(ref)) != attempt:
                raise HEASError("attempt differs from persisted evidence")
        reports = [self.store.read(a.validation_report_ref) for a in attempts]
        valid = sum(a.materialization_status == "materialized" for a in attempts)
        outcomes = [
            self.store.read(f"evolution/attempts/{a.evolution_attempt_id}/outcome.json")
            if self.store.path(f"evolution/attempts/{a.evolution_attempt_id}/outcome.json").exists()
            else None
            for a in attempts
        ]
        evaluated = sum(
            (
                self.store.read(
                    f"evolution/attempts/{a.evolution_attempt_id}/validation_outcome.json"
                )["evaluation_status"]
                if self.store.path(
                    f"evolution/attempts/{a.evolution_attempt_id}/validation_outcome.json"
                ).exists()
                else attempt_outcome["evaluation_status"]
                if attempt_outcome
                else a.evaluation.evaluation_status
            )
            == "completed"
            for a, attempt_outcome in zip(attempts, outcomes, strict=True)
        )
        stats: dict[str, Any] = {
            "schema_version": "1.0",
            "attempts": len(attempts),
            "valid_candidates": valid,
            "static_validity_rate": valid / len(attempts),
            "validation_error_counts": dict(
                Counter(error for report in reports for error in report["errors"])
            ),
            "quality_evaluated_attempts": evaluated,
            "parent_selection_counts": dict(
                Counter(parent for a in attempts for parent in a.parent_genome_ids)
            ),
            "resource_usage": {
                field: sum(
                    attempt_outcome["resource_usage"][field]
                    if attempt_outcome
                    else getattr(a.resource_usage, field)
                    for a, attempt_outcome in zip(attempts, outcomes, strict=True)
                )
                for field in (
                    "model_calls",
                    "input_tokens",
                    "output_tokens",
                    "tool_calls",
                    "wall_seconds",
                )
            },
        }
        reflections = []
        for attempt in attempts:
            uri = latest_evolution_reflection(self.store, attempt.evolution_attempt_id)
            if self.store.path(uri).exists():
                reflections.append(
                    self.store.load(uri, EvolutionReflection).model_dump(mode="json")
                )
        operation_counts: Counter[str] = Counter()
        comparison_refs = []
        rows = []
        for attempt, outcome in zip(attempts, outcomes, strict=True):
            directory = f"evolution/attempts/{attempt.evolution_attempt_id}"
            proposal_uri = f"{directory}/resolved_proposal.json"
            proposal = (
                self.store.read(proposal_uri) if self.store.path(proposal_uri).exists() else None
            )
            operations = [op["operation"] for op in proposal["operations"]] if proposal else []
            operation_counts.update(operations)
            completion_uri = f"{directory}/feedback_completion.json"
            trial_completion_uri = f"{directory}/trial_feedback_completion.json"
            if self.store.path(trial_completion_uri).exists():
                completion_uri = trial_completion_uri
            if self.store.path(completion_uri).exists():
                costs = self.store.read(completion_uri)["resource_usage"]
            else:
                costs = (
                    outcome["resource_usage"] if outcome else attempt.resource_usage.model_dump()
                )
            reflection_uri = latest_evolution_reflection(self.store, attempt.evolution_attempt_id)
            reflection = (
                self.store.read(reflection_uri)
                if self.store.path(reflection_uri).exists()
                else None
            )
            refs_for_row = []
            if outcome:
                refs_for_row.append(
                    EvidenceRef(ref_type="evolution_outcome", ref_id=attempt.evolution_attempt_id)
                )
                comparison_refs.extend(self._outcome_comparison_refs(outcome))
            rows.append(
                {
                    "source_ref": EvidenceRef(
                        ref_type="evolution_attempt", ref_id=attempt.evolution_attempt_id
                    ).model_dump(mode="json"),
                    "decision": bounded(self.store.read(attempt.decision_ref))
                    if attempt.decision_ref
                    else None,
                    "operations": operations,
                    "outcome": bounded(outcome),
                    "reflection": bounded(reflection),
                    "costs": costs,
                    "outcome_refs": [
                        outcome_reference.model_dump(mode="json")
                        for outcome_reference in refs_for_row
                    ],
                }
            )
        resource_totals: dict[str, int | float | bool | None] = {
            field: sum(row["costs"].get(field, 0) for row in rows)
            for field in (
                "model_calls",
                "input_tokens",
                "output_tokens",
                "tool_calls",
                "wall_seconds",
            )
        }
        resource_totals["token_usage_available"] = all(
            row["costs"].get("token_usage_available", True) for row in rows
        )
        if not resource_totals["token_usage_available"]:
            resource_totals["input_tokens"] = None
            resource_totals["output_tokens"] = None
        stats["resource_usage"] = resource_totals
        stats["operation_counts"] = dict(operation_counts)
        stats["outcome_coverage"] = sum(o is not None for o in outcomes)
        stats["comparison_status_counts"] = dict(
            Counter(
                child["candidate"]["comparison_status"]
                for attempt_outcome in outcomes
                if attempt_outcome
                for child in attempt_outcome["offspring"]
            )
        )
        scope: set[tuple[str, str]] = {(r.ref_type, r.ref_id) for r in refs}
        scope.update(
            ("evolution_outcome", a.evolution_attempt_id)
            for a, attempt_outcome in zip(attempts, outcomes, strict=True)
            if attempt_outcome
        )
        scope.update((r.ref_type, r.ref_id) for r in comparison_refs)
        semantic, aggregation_ref = self.aggregator.aggregate(
            "evolution_summary",
            rows,
            stats,
            EvolutionWindowAnalysis,
            self.budget,
            lambda report, allowed: validate_evidence(self.store, report, allowed),
            scope,
        )
        sid = uid("evolution_summary")
        stats_uri = f"analysis/{sid}.stats.json"
        self.store.put(stats_uri, stats)
        summary = EvolutionExperienceSummary(
            summary_id=sid,
            evolution_agent_id="evolution_agent_main",
            aggregation_ref=aggregation_ref,
            outcome_refs=[
                EvidenceRef(ref_type="evolution_outcome", ref_id=a.evolution_attempt_id)
                for a, attempt_outcome in zip(attempts, outcomes, strict=True)
                if attempt_outcome
            ],
            comparison_refs=comparison_refs,
            policy_id=attempts[0].policy_id,
            evolution_attempt_ids=[a.evolution_attempt_id for a in attempts],
            objective_stats_ref=stats_uri,
            evaluation_coverage={
                "attempts": len(attempts),
                "quality_evaluated_attempts": evaluated,
            },
            limitations=["Static validity does not establish task quality or causal gain."],
            **semantic.model_dump(exclude={"schema_version", "extensions"}),
        )
        self.store.save_record(f"evolution/summaries/{sid}.json", summary, "summary", sid)
        return summary
