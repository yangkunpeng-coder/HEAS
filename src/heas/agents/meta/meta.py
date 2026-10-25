"""L3 policy proposals, immutable versions and conservative matched-trial states."""

from typing import Any, Literal

from heas.core.domain import (
    Budget,
    EvidenceRef,
    EvolutionExperienceSummary,
    EvolutionPolicy,
    MetaPolicyTrial,
    PolicyProposal,
    Record,
    RunPlan,
    now,
    uid,
)
from heas.core.errors import HEASError, SecurityError
from heas.evidence.aggregation import bounded
from heas.evidence.experience import latest_evolution_reflection, validate_evidence
from heas.evidence.history import HistoryContext
from heas.infrastructure.model import CallBudget, ModelGateway
from heas.infrastructure.storage import FileStore

POLICY_FIELDS = frozenset(
    {
        "diagnosis_policy",
        "selection_policy",
        "parent_selection",
        "operator_policy",
        "generation_policy",
        "validation_guidance",
        "proposal_budget",
        "allowed_operations",
    }
)


class PolicyAnalysis(Record, frozen=True):
    """Validate a model's admissible policy edits and supporting hypotheses."""

    change_set: dict[str, Any]
    evidence_refs: list[EvidenceRef]
    expected_effect: str


def save_policy(store: FileStore, policy: EvolutionPolicy) -> None:
    """Persist an immutable policy version and its evidence index."""
    store.save_record(
        f"policies/{policy.policy_id}/policy.json", policy, "artifact", policy.policy_id
    )


def load_policy(store: FileStore, policy_id: str) -> EvolutionPolicy:
    """Load an archived policy through its validated artifact reference."""
    return EvolutionPolicy.model_validate(
        store.resolve(EvidenceRef(ref_type="artifact", ref_id=policy_id))
    )


def edit_policy(parent: EvolutionPolicy, changes: dict[str, Any]) -> EvolutionPolicy:
    """Build a child policy from the permitted host-owned edit surface."""
    if not changes or not set(changes).issubset(POLICY_FIELDS):
        raise SecurityError("Meta may only edit allowlisted policy fields")
    return EvolutionPolicy.model_validate(
        {
            **parent.model_dump(),
            **changes,
            "policy_id": uid("policy"),
            "parent_policy_id": parent.policy_id,
            "policy_status": "static_validated",
            "created_at": now(),
        }
    )


def matching_settings(
    old: RunPlan, new: RunPlan, old_starts: list[str], new_starts: list[str]
) -> bool:
    """Check whether two plans share the immutable comparison conditions."""
    fields = {
        "task_ids",
        "task_provider_id",
        "environment_adapter_id",
        "task_evaluator_id",
        "task_partition",
        "attempts_per_task",
        "random_seed",
        "budget",
        "model_config_ref",
        "memory_mode",
        "trace_level",
        "task_suite_id",
        "task_suite_version",
        "evaluation_suite_id",
        "evaluator_version",
        "max_tasks",
        "max_retries",
        "retry_termination_reasons",
        "retry_interrupted",
    }
    return old_starts == new_starts and all(getattr(old, f) == getattr(new, f) for f in fields)


class MetaEvolutionService:
    """Propose evidence-guided versions of L2 policy without changing fixed Host rules.

    Only allowlisted EvolutionPolicy fields can change. This service neither
    modifies itself nor activates a candidate without an independent policy trial.
    """

    def __init__(self, store: FileStore, gateway: ModelGateway, budget: Budget | None = None):
        self.store, self.gateway = store, gateway
        self.budget = budget or Budget(max_wall_seconds=60)
        self.history = HistoryContext(store, gateway)

    def propose(
        self,
        summary: EvolutionExperienceSummary,
        parent: EvolutionPolicy,
        trial_design: dict[str, Any],
    ) -> tuple[EvolutionPolicy, PolicyProposal]:
        """Create an archived policy candidate and evidence-linked change proposal.

        Analysis uses same-policy evolution experience and bounded attempt history.
        Policy changes pass the fixed edit allowlist and receive a new lineage-linked
        identity; independent trial and adoption remain separate Host-controlled steps.
        """
        if summary.policy_id != parent.policy_id:
            raise HEASError("meta summary and parent policy mismatch")
        stored = self.store.resolve(EvidenceRef(ref_type="summary", ref_id=summary.summary_id))
        if EvolutionExperienceSummary.model_validate(stored) != summary:
            raise HEASError("meta input differs from archived evolution summary")
        refs = [EvidenceRef(ref_type="summary", ref_id=summary.summary_id)]
        attempts = []
        for attempt_id in summary.evolution_attempt_ids:
            directory = f"evolution/attempts/{attempt_id}"
            proposal_uri = f"{directory}/proposal.json"
            if self.store.path(proposal_uri).exists():
                proposal_data = self.store.read(proposal_uri)
                for ref in proposal_data["evidence_refs"]:
                    source = self.store.resolve(EvidenceRef.model_validate(ref))
                    for run_id in source.get("run_ids", []):
                        run = self.store.resolve(EvidenceRef(ref_type="task_run", ref_id=run_id))
                        if run.get("task_partition") == "final_test":
                            raise SecurityError("final_test evidence cannot enter Meta-Evolution")
            attempt = self.store.resolve(
                EvidenceRef(ref_type="evolution_attempt", ref_id=attempt_id)
            )
            reflection_uri = latest_evolution_reflection(self.store, attempt_id)
            attempts.append(
                {
                    "source_ref": EvidenceRef(
                        ref_type="evolution_attempt", ref_id=attempt_id
                    ).model_dump(mode="json"),
                    "failure": attempt["materialization_status"] != "materialized",
                    "attempt": attempt,
                    "reflection": self.store.read(reflection_uri)
                    if self.store.path(reflection_uri).exists()
                    else None,
                    "validation": self.store.read(attempt["validation_report_ref"]),
                }
            )
        meter = CallBudget(self.budget)
        attempts.sort(key=lambda row: not row["failure"])
        context = self.history.prepare(
            "meta_policy",
            {
                "summary": bounded(summary.model_dump(mode="json"), 768, 4),
                "objective_stats": self.store.read(summary.objective_stats_ref),
                "policy": parent.model_dump(mode="json"),
                "evidence_refs": [r.model_dump(mode="json") for r in refs],
                "response_schema": PolicyAnalysis.model_json_schema(),
            },
            attempts,
            meter,
        )

        def validate_analysis(candidate_analysis: PolicyAnalysis) -> None:
            edit_policy(parent, candidate_analysis.change_set)
            validate_evidence(
                self.store, candidate_analysis, {(r.ref_type, r.ref_id) for r in refs}
            )

        analysis = self.gateway.structured(
            "meta_policy",
            context,
            meter,
            PolicyAnalysis,
            validate_analysis,
        )
        validate_evidence(self.store, analysis, {(r.ref_type, r.ref_id) for r in refs})
        child = edit_policy(parent, analysis.change_set)
        save_policy(self.store, child)
        proposal = PolicyProposal(
            proposal_id=uid("policy_proposal"),
            parent_policy_id=parent.policy_id,
            proposed_policy_ref=self.store.artifact(
                f"policies/{child.policy_id}/policy.json", child.policy_id
            ),
            evidence_refs=analysis.evidence_refs,
            expected_effect=analysis.expected_effect,
            trial_design=trial_design,
            change_set=analysis.change_set,
        )
        self.store.put(f"meta/proposals/{proposal.proposal_id}.json", proposal)
        self.store.put(
            f"meta/policy_diffs/{proposal.proposal_id}.json",
            {
                "schema_version": "1.1",
                "parent_policy_id": parent.policy_id,
                "new_policy_id": child.policy_id,
                "changes": {
                    field: {"before": getattr(parent, field), "after": getattr(child, field)}
                    for field in analysis.change_set
                },
            },
        )
        return child, proposal

    def trial(
        self,
        old: EvolutionPolicy,
        new: EvolutionPolicy,
        old_plan: RunPlan,
        new_plan: RunPlan,
        old_starts: list[str],
        new_starts: list[str],
    ) -> MetaPolicyTrial:
        """Persist supplied trial evidence without granting unearned policy activation."""
        matched = matching_settings(old_plan, new_plan, old_starts, new_starts)
        status: Literal["invalid", "pending", "not_available"] = (
            "invalid"
            if not matched
            else ("pending" if old_plan.task_evaluator_id else "not_available")
        )
        trial = MetaPolicyTrial(
            trial_id=uid("trial"),
            old_policy_id=old.policy_id,
            new_policy_id=new.policy_id,
            run_plan_refs=[
                f"plans/{old_plan.run_plan_id}.json",
                f"plans/{new_plan.run_plan_id}.json",
            ],
            start_genome_ids=old_starts,
            random_seeds=[old_plan.random_seed],
            budget_rule="Matched per-run budgets and fixed reporting/L2/L3 budgets.",
            comparison_status=status,
            objective_metrics_ref=None,
            decision="undecided",
            decision_evidence_refs=[],
            extensions={
                "heas": {
                    "execution_mode": "exploratory_candidate",
                    "settings_matched": matched,
                    "old_settings": old_plan.model_dump(mode="json"),
                    "new_settings": new_plan.model_dump(mode="json"),
                    "quality_improvement": "unknown",
                }
            },
        )
        self.store.save_record(
            f"meta/trials/{trial.trial_id}/trial.json", trial, "policy_trial", trial.trial_id
        )
        return trial
