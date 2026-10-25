"""Generic matched ablation execution; differences remain non-causal observations."""

from typing import Any

from pydantic import Field

from heas.agents.evolution.candidates import compare_windows
from heas.core.domain import Budget, EvidenceRef, Id, Record, RunPlan, uid
from heas.core.errors import HEASError
from heas.experimentation.ports import ExperimentServices
from heas.infrastructure.model import CallBudget
from heas.infrastructure.resources import ResourceLedger


class AblationCondition(Record, frozen=True):
    """Describe an immutable host-created variant and its capability-change evidence."""

    label: Id
    genome_id: Id
    removed_capability_ids: list[Id] = Field(default_factory=list)
    operation_evidence_refs: list[EvidenceRef] = Field(default_factory=list)


class AttributionSpec(Record, frozen=True):
    """Freeze matched ablation conditions without asserting causal attribution."""

    experiment_id: Id
    baseline_genome_id: Id
    conditions: list[AblationCondition] = Field(min_length=1)
    run_plan: RunPlan
    random_seeds: list[int] = Field(default=[23], min_length=1)
    total_budget: Budget | None = None
    uncertainty: str = "Matched differences alone do not identify a causal capability contribution."


class AttributionExperimentRunner:
    """Execute host-defined matched capability variants without assuming causality."""

    def __init__(self, services: ExperimentServices):
        self.services = services
        self.store = services.store

    def execute(self, spec: AttributionSpec) -> dict[str, Any]:
        """Execute Host-defined capability variants under matched task conditions.

        Variant differences are explicit and use the existing materializer. Reported
        outcomes preserve missing evaluation rather than assigning synthetic fitness.
        """
        if spec.run_plan.task_partition == "final_test":
            raise HEASError("ablation/exploration cannot consume final_test")
        store, runtime = self.store, self.services.runtime
        directory = f"experiments/attribution/{spec.experiment_id}"
        store.put(f"{directory}/spec.json", spec)
        baseline = store.genome(spec.baseline_genome_id)
        rows = []
        checkpoint = ResourceLedger(store).checkpoint()
        scope = CallBudget(spec.total_budget) if spec.total_budget else None
        if scope:
            self.services.gateway.budget_scopes.append(scope)
        try:
            for seed in spec.random_seeds:
                baseline_plan = RunPlan.model_validate(
                    {
                        **spec.run_plan.model_dump(),
                        "run_plan_id": uid("plan"),
                        "genome_ids": [baseline.genome_id],
                        "random_seed": seed,
                    }
                )
                baseline_runs = runtime.batch(baseline_plan)
                for condition in spec.conditions:
                    for ref in condition.operation_evidence_refs:
                        store.resolve(ref)
                    variant = store.genome(condition.genome_id)
                    old_skills = {s.content_sha256 for s in baseline.skills}
                    new_skills = {s.content_sha256 for s in variant.skills}
                    plan = RunPlan.model_validate(
                        {
                            **baseline_plan.model_dump(),
                            "run_plan_id": uid("plan"),
                            "genome_ids": [variant.genome_id],
                        }
                    )
                    runs = runtime.batch(plan)
                    comparison = compare_windows(
                        store, [r.run_id for r in baseline_runs], [r.run_id for r in runs]
                    )
                    rows.append(
                        {
                            "condition": condition.model_dump(mode="json"),
                            "seed": seed,
                            "comparison": comparison,
                            "changes": {
                                "added_skill_hashes": sorted(new_skills - old_skills),
                                "removed_skill_hashes": sorted(old_skills - new_skills),
                                "memory_changed": baseline.memory.content_sha256
                                != variant.memory.content_sha256,
                                "old_tools": [(t.tool_id, t.version_id) for t in baseline.tools],
                                "new_tools": [(t.tool_id, t.version_id) for t in variant.tools],
                            },
                            "proposal_id": variant.created_from_proposal_id,
                            "causal_attribution": "not_established",
                            "uncertainty": spec.uncertainty,
                        }
                    )
        finally:
            if scope:
                self.services.gateway.budget_scopes.remove(scope)
        report = {
            "schema_version": "1.2",
            "experiment_id": spec.experiment_id,
            "conditions": rows,
            "resource_usage": ResourceLedger(store).report(checkpoint),
            "causal_attribution": "not_established",
        }
        store.put(f"{directory}/report.json", report)
        return report
