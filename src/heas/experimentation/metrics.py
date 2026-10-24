"""Evidence-derived research metrics and validation-only candidate freezing.

These are HEAS metrics, not substitutes for an external benchmark's definitions.
Unknown measurements remain unknown, including unmetered external solver costs.
"""

from typing import Any, Literal

from pydantic import Field

from heas.agents.evolution.candidates import (
    archived_runs,
    compare_windows,
    comparison_protocol,
    external_score,
)
from heas.core.domain import EvaluationResult, EvidenceRef, Id, Record, now
from heas.core.errors import HEASError, SecurityError
from heas.infrastructure.resources import ResourceLedger
from heas.infrastructure.storage import FileStore, digest, encode


class MetricValue(Record):
    """Represent a derived value together with its inputs and missing-evidence reason."""

    status: Literal["completed", "not_available"]
    value: float | None = None
    unit: str
    reason: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    origin: Literal["heas_derived"] = "heas_derived"


def unknown(unit: str, reason: str) -> MetricValue:
    """Return an explicit unavailable measurement instead of a synthetic zero."""
    return MetricValue(status="not_available", unit=unit, reason=reason)


def measured(value: float, unit: str, evidence: list[str]) -> MetricValue:
    """Attach immutable record identities to an observed derived value."""
    return MetricValue(status="completed", value=value, unit=unit, evidence_ids=evidence)


class FrozenSelection(Record):
    """Record a validation-selected genome before any final-test execution."""

    selection_id: Id
    genome_id: Id
    baseline_genome_id: Id
    policy_id: Id
    validation_run_ids: list[Id]
    candidate_genome_ids: list[Id]
    k: int = Field(ge=1)
    validation_protocol_sha256: str
    created_at: str
    selection_rule: str = "best_mean_of_one_comparable_validation_metric"


class EvolutionMetrics:
    """Compute conservative reports without mutating candidates, policies, or genomes."""

    def __init__(self, store: FileStore):
        self.store = store

    def success_rate(self, run_ids: list[str]) -> MetricValue:
        """Use explicit independent success booleans, never model self-assessment."""
        runs = archived_runs(self.store, run_ids)
        outcomes = []
        for run in runs:
            if run.evaluation.evaluation_status != "completed" or not run.objective_feedback_ref:
                return unknown("ratio", "independent success evidence missing")
            feedback = EvaluationResult.model_validate(
                self.store.resolve(EvidenceRef(ref_type="feedback", ref_id=run.run_id))
            )
            success = feedback.extensions.get("heas", {}).get("success")
            if not isinstance(success, bool):
                return unknown("ratio", "evaluator did not define binary success")
            outcomes.append(success)
        return measured(sum(outcomes) / len(outcomes), "ratio", run_ids)

    def gain(
        self, before: list[str], after: list[str], *, report_only: bool = False
    ) -> MetricValue:
        """Compute an oriented difference only for one complete comparable objective."""
        result = compare_windows(self.store, before, after, learning=not report_only)
        if result["evaluation_status"] != "completed" or result.get("improvement") is None:
            return unknown("score_delta", result.get("reason", "no comparable scalar objective"))
        return measured(result["improvement"], "score_delta", before + after)

    def freeze(
        self, selection_id: str, windows: list[list[str]], baseline_genome_id: str, policy_id: str
    ) -> FrozenSelection:
        """Select among validation windows and persist a decision before final tests.

        Args:
            selection_id: New host-owned immutable decision identity.
            windows: Complete validation run windows, one frozen genome per window.
            baseline_genome_id: Fixed reference for the later final report.
            policy_id: Policy frozen together with the selected genome.

        Raises:
            SecurityError: A final test has already been run for a candidate.
            HEASError: Validation evidence is missing or incomparable.
        """
        if not windows:
            raise HEASError("validation candidate windows are required")
        runs = [archived_runs(self.store, ids) for ids in windows]
        if any(r.task_partition != "evolution_validation" for group in runs for r in group):
            raise SecurityError("candidate selection requires evolution_validation only")
        gids = [group[0].genome_id for group in runs]
        if len(gids) != len(set(gids)) or any(
            len({r.genome_id for r in group}) != 1 for group in runs
        ):
            raise HEASError("each candidate must have one distinct frozen validation window")
        self.store.genome(baseline_genome_id)
        from heas.agents.meta.meta import load_policy

        load_policy(self.store, policy_id)
        for path in self.store.path("runs").glob("*/run.json"):
            row = self.store.read(path.relative_to(self.store.root).as_posix())
            if row["task_partition"] == "final_test" and row["genome_id"] in {
                baseline_genome_id,
                *gids,
            }:
                raise SecurityError("freeze must precede final-test exposure")
        values = []
        for ids, group in zip(windows, runs, strict=True):
            comparison = compare_windows(self.store, windows[0], ids)
            if comparison["evaluation_status"] != "completed":
                raise HEASError("validation windows are not objectively comparable")
            score = external_score(self.store, group)
            if score is None:
                raise HEASError("validation requires one explicit scalar selection objective")
            values.append(score.value * (1 if score.direction == "higher_is_better" else -1))
        selected = FrozenSelection(
            selection_id=selection_id,
            genome_id=gids[max(range(len(values)), key=values.__getitem__)],
            baseline_genome_id=baseline_genome_id,
            policy_id=policy_id,
            validation_run_ids=[r.run_id for group in runs for r in group],
            candidate_genome_ids=gids,
            k=len(gids),
            validation_protocol_sha256=comparison_protocol(self.store, runs[0]),
            created_at=now().isoformat(),
        )
        self.store.put(f"metrics/selections/{selection_id}.json", selected)
        return selected

    def improvement_at_k(
        self, selection_id: str, baseline_runs: list[str], selected_runs: list[str]
    ) -> MetricValue:
        """Report held-out gain for a previously validation-frozen candidate."""
        selection = self.store.load(f"metrics/selections/{selection_id}.json", FrozenSelection)
        from datetime import datetime

        windows = [archived_runs(self.store, ids) for ids in (baseline_runs, selected_runs)]
        for group, gid in zip(
            windows, (selection.baseline_genome_id, selection.genome_id), strict=True
        ):
            if any(
                r.genome_id != gid
                or r.task_partition != "final_test"
                or r.start_time < datetime.fromisoformat(selection.created_at)
                for r in group
            ):
                raise SecurityError("held-out report violates the frozen selection boundary")
        return self.gain(baseline_runs, selected_runs, report_only=True)

    def report(
        self,
        before: list[str],
        after: list[str],
        *,
        attempts: list[str] | None = None,
        learning_run_ids: list[str] | None = None,
        retention_reference: list[str] | None = None,
        interference_run_ids: list[str] | None = None,
        selection_id: str | None = None,
        policy_trial_id: str | None = None,
        policy_selection_id: str | None = None,
    ) -> dict[str, Any]:
        """Report nine requested metrics, preserving missing protocol evidence explicitly."""
        before_runs, after_runs = (
            archived_runs(self.store, before),
            archived_runs(self.store, after),
        )
        final = any(r.task_partition == "final_test" for r in before_runs + after_runs)
        metrics = {
            "task_success_rate": self.success_rate(after),
            "evolution_gain": self.gain(before, after, report_only=final),
            "improvement_at_k": self.improvement_at_k(selection_id, before, after)
            if selection_id
            else unknown("score_delta", "validation selection has not been frozen"),
            "transfer_gain": unknown("score_delta", "disjoint learning evidence is required"),
            "retention": unknown("score_delta", "matched pre-interference evidence is required"),
            "valid_offspring_rate": unknown("ratio", "no evolution attempt window"),
            "policy_improvement": unknown("score_delta", "no independently matched policy trial"),
            "meta_improvement_at_k": unknown(
                "score_delta", "no validation-selected policy final report"
            ),
        }
        if learning_run_ids:
            learning = archived_runs(self.store, learning_run_ids)
            if any(r.task_partition == "final_test" for r in learning):
                raise SecurityError("final tests cannot be transfer learning evidence")
            if {r.task_id for r in learning}.isdisjoint(r.task_id for r in after_runs) and (
                max(r.end_time for r in learning) < min(r.start_time for r in after_runs)
            ):
                metrics["transfer_gain"] = metrics["evolution_gain"]
        if retention_reference and interference_run_ids:
            reference = archived_runs(self.store, retention_reference)
            interference = archived_runs(self.store, interference_run_ids)
            if max(r.end_time for r in reference) < min(r.start_time for r in interference) and (
                max(r.end_time for r in interference) < min(r.start_time for r in after_runs)
            ):
                metrics["retention"] = self.gain(retention_reference, after, report_only=final)
        if attempts:
            from heas.core.domain import EvolutionAttemptRecord

            records = [
                EvolutionAttemptRecord.model_validate(
                    self.store.resolve(EvidenceRef(ref_type="evolution_attempt", ref_id=identity))
                )
                for identity in dict.fromkeys(attempts)
            ]
            metrics["valid_offspring_rate"] = measured(
                sum(bool(a.offspring_genome_ids) for a in records) / len(records), "ratio", attempts
            )
        if policy_trial_id:
            from heas.agents.meta.policy_metrics import PolicyMetrics

            trial, _, status = PolicyMetrics(self.store).assess(policy_trial_id)
            # A policy proposal or an undecided trial alone is never an improvement.
            if status.get("decision") == "adopt" and trial.comparison_status == "completed":
                if status["mean_difference"] is not None:
                    metrics["policy_improvement"] = measured(
                        status["mean_difference"], "score_delta", [policy_trial_id]
                    )
        if policy_selection_id:
            self.store.read(f"metrics/policy_selections/{policy_selection_id}.json")
            final_report = self.store.read(
                f"metrics/policy_final/{policy_selection_id}/report.json"
            )
            metrics["meta_improvement_at_k"] = MetricValue.model_validate(
                final_report["meta_improvement_at_k"]
            )
        resources = ResourceLedger(self.store).report()
        metrics["evolution_cost"] = (
            measured(resources["monetary_cost"], "currency", attempts or [])
            if resources["monetary_cost"] is not None
            else unknown("currency", "complete priced external and HEAS usage is unavailable")
        )
        report = {
            "origin": "heas_derived",
            "metrics": {k: v.model_dump(mode="json") for k, v in metrics.items()},
            "resources": resources,
            "measurement_target": "see independent evaluator receipts",
        }
        self.store.put("metrics/reports/report_" + digest(encode(report))[7:31] + ".json", report)
        return report
