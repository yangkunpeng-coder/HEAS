"""Append-only candidate history and fixed, independently evaluated comparisons."""

from collections import Counter
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field

from heas.agents.meta.meta import matching_settings
from heas.core.domain import (
    EvaluationResult,
    EvaluationState,
    EvidenceRef,
    Id,
    Record,
    RunPlan,
    Score,
    TaskGenome,
    TaskRunRecord,
    now,
    uid,
)
from heas.core.errors import HEASError, SecurityError
from heas.infrastructure.storage import FileStore, digest, encode

if TYPE_CHECKING:
    from heas.agents.evolution.parents import ParentSelectionPolicy
    from heas.core.domain import EvolutionPolicy


class CandidateRecord(Record):
    """Archive one immutable revision of a candidate's execution and comparison state."""

    schema_version: Literal["1.1"] = "1.1"
    genome_id: Id
    parent_genome_ids: list[Id]
    evolution_attempt_id: Id | None = None
    revision: int = Field(ge=1)
    static_validated: bool
    execution_status: Literal["not_run", "completed", "failed"] = "not_run"
    evaluation: EvaluationState = Field(default_factory=EvaluationState)
    comparison_status: Literal[
        "pending_evaluation", "not_comparable", "improved", "not_improved"
    ] = "pending_evaluation"
    promotion_status: Literal["archived", "eligible", "promoted"] = "archived"
    run_ids: list[Id] = Field(default_factory=list)
    comparison_ref: str | None = None
    branch_id: str | None = None
    capability_fingerprint: str | None = None


def archived_runs(store: FileStore, run_ids: list[str]) -> list[TaskRunRecord]:
    """Load a unique nonempty run window through resolvable immutable evidence."""
    if not run_ids or len(run_ids) != len(set(run_ids)):
        raise HEASError("a unique nonempty run window is required")
    return [
        TaskRunRecord.model_validate(store.resolve(EvidenceRef(ref_type="task_run", ref_id=r)))
        for r in run_ids
    ]


def external_score(store: FileStore, runs: list[TaskRunRecord]) -> Score | None:
    """Average one compatible independent objective, returning None for missing evidence."""
    scores = []
    for run in runs:
        if run.evaluation.evaluation_status != "completed" or not run.objective_feedback_ref:
            return None
        result = EvaluationResult.model_validate(
            store.resolve(EvidenceRef(ref_type="feedback", ref_id=run.run_id))
        )
        if (
            result.run_id != run.run_id
            or result.score is None
            or result.score != run.evaluation.score
            or result.evaluation_status != "completed"
        ):
            return None
        scores.append(result.score)
    if not scores:
        return None
    if len({(s.name, s.unit, s.direction, s.source) for s in scores}) != 1:
        raise HEASError(
            "scores have incompatible evaluator/scales; explicit normalization required"
        )
    return Score(**{**scores[0].model_dump(), "value": sum(s.value for s in scores) / len(scores)})


def evaluated_metrics(
    store: FileStore, run: TaskRunRecord
) -> tuple[dict[str, Score], dict[str, str], bool | None] | None:
    """Validate feedback identity and extract versioned independent metric contracts."""
    if run.evaluation.evaluation_status != "completed" or not run.objective_feedback_ref:
        return None
    feedback = EvaluationResult.model_validate(
        store.resolve(EvidenceRef(ref_type="feedback", ref_id=run.run_id))
    )
    if (
        feedback.run_id != run.run_id
        or feedback.evaluation_status != "completed"
        or feedback.score != run.evaluation.score
        or feedback.scores != run.evaluation.scores
    ):
        return None
    metrics = {f"{s.source}:{s.name}": s for s in feedback.scores}
    if feedback.score:
        metrics[f"{feedback.score.source}:{feedback.score.name}"] = feedback.score
    if not metrics:
        return None
    versions = {
        key: feedback.evaluator_versions.get(score.source, feedback.evaluator_version)
        for key, score in metrics.items()
    }
    return metrics, versions, feedback.comparison_eligible


def comparison_protocol(store: FileStore, runs: list[TaskRunRecord]) -> str:
    """Fingerprint the versioned Host comparison conditions independently of scores."""
    rows = []
    fields = (
        "task_provider_id",
        "environment_adapter_id",
        "task_evaluator_id",
        "evaluator_version",
        "task_suite_id",
        "task_suite_version",
        "evaluation_suite_id",
        "task_partition",
        "budget",
        "model_config_ref",
        "random_seed",
        "attempts_per_task",
        "max_retries",
        "retry_termination_reasons",
        "retry_interrupted",
        "memory_mode",
        "trace_level",
    )
    for run in runs:
        plan = store.load(f"plans/{run.run_plan_id}.json", RunPlan)
        measured = evaluated_metrics(store, run)
        rows.append(
            {
                "task_id": run.task_id,
                "attempt": run.attempt_index,
                "task_instance_sha256": run.task_instance_sha256,
                "environment_input_sha256": run.extensions.get("heas", {}).get(
                    "environment_input_sha256"
                ),
                "plan": {key: plan.model_dump(mode="json")[key] for key in fields},
                "model_identity": run.extensions.get("heas", {}).get("model_identity"),
                "model_parameters_sha256": run.extensions.get("heas", {}).get(
                    "model_parameters_sha256"
                ),
                "metrics": {
                    key: score.model_dump(exclude={"value"}) for key, score in measured[0].items()
                }
                if measured
                else None,
                "evaluator_versions": measured[1] if measured else None,
            }
        )
    return digest(encode(sorted(rows, key=lambda row: encode(row))))


def compare_windows(
    store: FileStore,
    old_ids: list[str],
    new_ids: list[str],
    *,
    min_improvement: float = 0,
    learning: bool = True,
) -> dict[str, Any]:
    """Compare complete matched run windows without inventing missing evaluation."""
    old, new = archived_runs(store, old_ids), archived_runs(store, new_ids)
    if learning and any(r.task_partition == "final_test" for r in old + new):
        raise SecurityError("final_test results cannot enter candidate or policy selection")
    if min_improvement < 0:
        raise HEASError("min_improvement must be nonnegative")
    if len({r.run_plan_id for r in old + new}) > 2 or any(
        len({r.run_plan_id for r in window}) > 1 for window in (old, new)
    ):
        return _compare_suite_windows(store, old, new, min_improvement, learning=learning)

    def settings(runs: list[TaskRunRecord]) -> tuple[RunPlan, Counter[Any], set[tuple[str, str]]]:
        if len({r.run_plan_id for r in runs}) != 1 or len({r.genome_id for r in runs}) != 1:
            raise HEASError("comparison windows must each use one plan and frozen genome")
        plan = store.load(f"plans/{runs[0].run_plan_id}.json", RunPlan)
        expected = Counter(
            (task, attempt) for task in plan.task_ids for attempt in range(plan.attempts_per_task)
        )
        actual = Counter((r.task_id, r.attempt_index) for r in runs)
        if expected != actual:
            raise HEASError("comparison window does not cover the complete configured task set")
        identities = {
            (
                r.extensions.get("heas", {}).get("model_identity", ""),
                r.extensions.get("heas", {}).get("model_parameters_sha256", ""),
            )
            for r in runs
        }
        return plan, actual, identities

    old_plan, old_tasks, old_models = settings(old)
    new_plan, new_tasks, new_models = settings(new)
    base = {
        "schema_version": "1.1",
        "old_run_ids": old_ids,
        "new_run_ids": new_ids,
        "evaluation_status": "not_available",
        "improvement": None,
        "quality_improvement": "unknown",
        "decision": "undecided",
    }
    if (
        not matching_settings(old_plan, new_plan, [], [])
        or old_tasks != new_tasks
        or old_models != new_models
        or any(not identity or not parameters for identity, parameters in old_models)
    ):
        return {**base, "comparison_status": "not_comparable", "reason": "settings_mismatch"}
    if any(not r.task_instance_sha256 for r in old + new):
        return {**base, "comparison_status": "not_comparable", "reason": "missing_task_snapshot"}
    old = sorted(old, key=lambda r: (r.task_id, r.attempt_index))
    new = sorted(new, key=lambda r: (r.task_id, r.attempt_index))
    if any(a.task_instance_sha256 != b.task_instance_sha256 for a, b in zip(old, new, strict=True)):
        return {**base, "comparison_status": "not_comparable", "reason": "task_instance_mismatch"}
    if any(
        a.extensions.get("heas", {}).get("environment_input_sha256")
        != b.extensions.get("heas", {}).get("environment_input_sha256")
        for a, b in zip(old, new, strict=True)
    ):
        return {
            **base,
            "comparison_status": "not_comparable",
            "reason": "environment_input_mismatch",
        }
    measured = [evaluated_metrics(store, r) for r in old + new]
    if any(item is None for item in measured):
        return {**base, "comparison_status": "pending_evaluation", "reason": "missing_evaluation"}
    complete = [item for item in measured if item is not None]
    for run, (_, _, eligible) in zip(old + new, complete, strict=True):
        if eligible is False or (
            run.termination_reason in {"model_error", "environment_error", "cancelled"}
            and eligible is not True
        ):
            return {**base, "comparison_status": "not_comparable", "reason": "infrastructure_error"}
    contracts = [
        {key: (s.model_dump(exclude={"value"}), versions[key]) for key, s in metrics.items()}
        for metrics, versions, _ in complete
    ]
    if any(c != contracts[0] for c in contracts):
        return {**base, "comparison_status": "not_comparable", "reason": "metric_protocol_mismatch"}
    old_metrics = complete[: len(old)]
    new_metrics = complete[len(old) :]
    changes = {}
    for key in contracts[0]:
        before = sum(m[0][key].value for m in old_metrics) / len(old)
        after = sum(m[0][key].value for m in new_metrics) / len(new)
        sign = 1 if complete[0][0][key].direction == "higher_is_better" else -1
        changes[key] = {"old": before, "new": after, "improvement": sign * (after - before)}
    primary_keys = {
        f"{r.evaluation.score.source}:{r.evaluation.score.name}" if r.evaluation.score else None
        for r in old + new
    }
    if len(primary_keys) != 1:
        return {**base, "comparison_status": "not_comparable", "reason": "primary_metric_mismatch"}
    primary = next(iter(primary_keys))
    diffs = [m["improvement"] for m in changes.values()]
    if any(d < 0 for d in diffs) and any(d > 0 for d in diffs):
        return {
            **base,
            "comparison_status": "not_comparable",
            "reason": "metric_tradeoff",
            "metric_changes": changes,
        }
    improvement = changes[primary]["improvement"] if primary else None
    improved = (
        improvement > min_improvement
        if improvement is not None
        else any(d > min_improvement for d in diffs)
    ) and all(d >= 0 for d in diffs)
    paired = []
    if primary:
        sign = 1 if complete[0][0][primary].direction == "higher_is_better" else -1
        paired = [
            sign * (b[0][primary].value - a[0][primary].value)
            for a, b in zip(old_metrics, new_metrics, strict=True)
        ]
    return {
        **base,
        "evaluation_status": "completed",
        "improvement": improvement,
        "quality_improvement": "improved" if improved else "not_improved",
        "comparison_status": "improved" if improved else "not_improved",
        "decision": "adopt" if improved else "reject",
        "metric_changes": changes,
        "paired_differences": paired,
        "primary_metric": primary,
        "failure_counts": dict(Counter(r.termination_reason for r in old + new)),
        "evidence_refs": [
            EvidenceRef(ref_type="feedback", ref_id=r.run_id).model_dump(mode="json")
            for r in old + new
        ],
    }


def _compare_suite_windows(
    store: FileStore,
    old: list[TaskRunRecord],
    new: list[TaskRunRecord],
    minimum: float,
    *,
    learning: bool = True,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "schema_version": "1.2",
        "old_run_ids": [r.run_id for r in old],
        "new_run_ids": [r.run_id for r in new],
        "evaluation_status": "not_available",
        "comparison_status": "not_comparable",
        "improvement": None,
        "quality_improvement": "unknown",
        "decision": "undecided",
    }

    def groups(runs: list[TaskRunRecord]) -> dict[str, list[str]]:
        if len({r.genome_id for r in runs}) != 1:
            raise HEASError("suite comparison requires one frozen genome per side")
        rows: dict[str, list[str]] = {}
        seen_plans: dict[str, str] = {}
        for run in runs:
            plan = store.load(f"plans/{run.run_plan_id}.json", RunPlan)
            key = digest(
                encode(plan.model_dump(mode="json", exclude={"run_plan_id", "genome_ids"}))
            )
            if key in seen_plans and seen_plans[key] != run.run_plan_id:
                raise HEASError("ambiguous duplicate suite group; compare explicit windows")
            seen_plans[key] = run.run_plan_id
            rows.setdefault(key, []).append(run.run_id)
        return rows

    before, after = groups(old), groups(new)
    if before.keys() != after.keys():
        return {**base, "reason": "suite_group_contract_mismatch"}
    parts = [
        compare_windows(store, before[key], after[key], min_improvement=minimum, learning=learning)
        for key in sorted(before)
    ]
    base["group_comparisons"] = parts
    if any(c["comparison_status"] == "not_comparable" for c in parts):
        return {**base, "reason": "incomparable_suite_group"}
    if any(c["evaluation_status"] != "completed" for c in parts):
        return {**base, "comparison_status": "pending_evaluation", "reason": "missing_evaluation"}
    # Pareto comparison across heterogeneous metrics does not invent a scalar scale.
    improved = any(c["decision"] == "adopt" for c in parts) and all(
        all(m["improvement"] >= 0 for m in c.get("metric_changes", {}).values()) for c in parts
    )
    primary = {c.get("primary_metric") for c in parts}
    paired = [v for c in parts for v in c.get("paired_differences", [])]
    contracts = [evaluated_metrics(store, r) for r in old + new]
    scales = [
        {key: (score.model_dump(exclude={"value"}), row[1][key]) for key, score in row[0].items()}
        for row in contracts
        if row
    ]
    scalar = (
        len(primary) == 1
        and None not in primary
        and paired
        and all(scale == scales[0] for scale in scales)
    )
    return {
        **base,
        "evaluation_status": "completed",
        "comparison_status": "improved" if improved else "not_improved",
        "quality_improvement": "improved" if improved else "not_improved",
        "decision": "adopt" if improved else "reject",
        "improvement": sum(paired) / len(paired) if scalar else None,
        "paired_differences": paired if scalar else [],
        "primary_metric": next(iter(primary)) if scalar else None,
        "evidence_refs": [ref for c in parts for ref in c.get("evidence_refs", [])],
    }


class CandidateRegistry:
    """Maintain append-only candidate revisions and evidence-gated activation."""

    def __init__(self, store: FileStore):
        self.store = store

    def get(self, genome_id: str) -> CandidateRecord:
        """Load the latest immutable revision of a registered candidate."""
        genome = self.store.genome(genome_id)
        directory = self.store.path(f"candidates/{genome.genome_id}/revisions")
        paths = sorted(directory.glob("*.json"))
        if not paths:
            raise HEASError("candidate is not registered")
        return self.store.load(paths[-1].relative_to(self.store.root).as_posix(), CandidateRecord)

    def save(self, record: CandidateRecord) -> CandidateRecord:
        """Append a candidate revision without overwriting its previous evidence."""
        self.store.put(
            f"candidates/{record.genome_id}/revisions/{record.revision:06d}.json", record
        )
        return record

    def register(self, genome: TaskGenome, attempt_id: str | None = None) -> CandidateRecord:
        """Register a new host-owned artifact or approved binding, preserving prior identities."""
        if self.store.genome(genome.genome_id) != genome:
            raise HEASError("candidate differs from archived genome")
        return self.save(
            CandidateRecord(
                genome_id=genome.genome_id,
                parent_genome_ids=genome.parent_genome_ids,
                evolution_attempt_id=attempt_id,
                revision=1,
                branch_id=attempt_id or genome.genome_id,
                capability_fingerprint=digest(
                    encode(
                        {
                            "skills": [s.content_sha256 for s in genome.skills],
                            "memory": genome.memory.content_sha256,
                            "tools": [(t.tool_id, t.version_id) for t in genome.tools],
                        }
                    )
                ),
                static_validated=genome.artifact_status == "static_validated",
            )
        )

    def record_execution(self, genome_id: str, run_ids: list[str]) -> CandidateRecord:
        """Archive a candidate's run window without inferring success from termination."""
        current = self.get(genome_id)
        runs = archived_runs(self.store, run_ids)
        if any(r.task_partition == "final_test" for r in runs):
            raise SecurityError("final_test results cannot replace candidate learning windows")
        if any(r.genome_id != genome_id for r in runs):
            raise HEASError("candidate run genome mismatch")
        score = external_score(self.store, runs)
        evaluation = (
            EvaluationState(evaluation_status="completed", score=score)
            if score
            else (EvaluationState())
        )
        measured = [evaluated_metrics(self.store, r) for r in runs]
        complete = [m for m in measured if m is not None]
        if len(complete) == len(runs) and all(
            m[0].keys() == complete[0][0].keys() for m in complete
        ):
            metrics = []
            for key, sample in complete[0][0].items():
                if any(
                    m[0][key].model_dump(exclude={"value"}) != sample.model_dump(exclude={"value"})
                    or m[1][key] != complete[0][1][key]
                    for m in complete
                ):
                    break
                metrics.append(
                    sample.model_copy(
                        update={"value": sum(m[0][key].value for m in complete) / len(complete)}
                    )
                )
            else:
                evaluation = EvaluationState(
                    evaluation_status="completed", score=score, scores=metrics
                )
        return self.save(
            CandidateRecord.model_validate(
                {
                    **current.model_dump(),
                    "revision": current.revision + 1,
                    "run_ids": run_ids,
                    "evaluation": evaluation,
                    "execution_status": "completed"
                    if all(r.termination_reason == "final_answer" for r in runs)
                    else "failed",
                    "promotion_status": "archived",
                    "comparison_status": "pending_evaluation",
                    "comparison_ref": None,
                }
            )
        )

    def incumbent(self) -> str | None:
        """Read the latest explicitly activated candidate identity."""
        paths = sorted(self.store.path("candidates/activations").glob("*.json"))
        return (
            self.store.read(paths[-1].relative_to(self.store.root).as_posix())["genome_id"]
            if paths
            else None
        )

    def list_candidates(self, *, evaluated: bool | None = None) -> list[CandidateRecord]:
        """Load the latest immutable revision of each archived candidate."""
        if not self.store.path("candidates").is_dir():
            return []
        records = [
            self.get(path.name)
            for path in self.store.path("candidates").iterdir()
            if path.is_dir() and (path / "revisions").is_dir()
        ]
        return [
            r
            for r in records
            if evaluated is None or (r.evaluation.evaluation_status == "completed") == evaluated
        ]

    def diversity(self) -> dict[str, Any]:
        """Report retained capability and lineage diversity without a quality claim."""
        records = self.list_candidates()
        return {
            "candidates": len(records),
            "branches": len({r.branch_id for r in records}),
            "capability_fingerprints": len({r.capability_fingerprint for r in records}),
            "parent_genomes": sorted({p for r in records for p in r.parent_genome_ids}),
            "quality_conclusion": "unknown_without_independent_comparisons",
        }

    def experience_pool(
        self,
        limit: int = 16,
        *,
        policy: "EvolutionPolicy | None" = None,
        selector: "ParentSelectionPolicy | None" = None,
    ) -> list[Any]:
        """Return retained learning summaries while excluding policy-trial artifacts."""
        from heas.core.domain import TaskExperienceSummary

        pool = []
        for record in self.list_candidates():
            genome = self.store.genome(record.genome_id)
            directory = self.store.path(
                f"task_agents/{genome.agent_id}/{genome.genome_id}/summaries"
            )
            summaries = [
                self.store.load(p.relative_to(self.store.root).as_posix(), TaskExperienceSummary)
                for p in directory.glob("*.json")
            ]
            summaries = [
                s for s in summaries if s.extensions.get("heas", {}).get("scope") != "policy_trial"
            ]
            if summaries:
                pool.append(max(summaries, key=lambda summary: summary.created_at))
        if policy and selector and pool:
            pool, selection = selector.select(self.store, pool, policy)
            self.store.put(f"candidates/selections/{uid('archive_selection')}.json", selection)
            return pool[:limit]
        return sorted(pool, key=lambda summary: summary.created_at, reverse=True)[:limit]

    def initialize_incumbent(self, genome_id: str) -> None:
        """Record the initial host-selected baseline, without asserting superiority."""
        self.get(genome_id)
        if self.incumbent() is not None:
            raise HEASError("incumbent already initialized")
        self.store.put(
            "candidates/activations/000001.json",
            {"schema_version": "1.1", "genome_id": genome_id, "reason": "host_baseline"},
        )

    def compare(self, genome_id: str, reference_id: str | None = None) -> CandidateRecord:
        """Append a candidate comparison using matched independent execution evidence.

        Missing scores or incompatible task/model/evaluator contracts keep the
        candidate ineligible. Generating a child or passing static validation does
        not by itself qualify the child for promotion.
        """
        current = self.get(genome_id)
        reference_id = reference_id or self.incumbent()
        if not reference_id or reference_id == genome_id:
            raise HEASError("comparison requires a distinct archived reference candidate")
        reference = self.get(reference_id)
        comparison = compare_windows(self.store, reference.run_ids, current.run_ids)
        uri = f"candidates/comparisons/{uid('comparison')}.json"
        self.store.put(
            uri,
            {**comparison, "reference_genome_id": reference_id, "candidate_genome_id": genome_id},
        )
        self.store.index("candidate_comparison", uri.rsplit("/", 1)[-1][:-5], uri)
        return self.save(
            CandidateRecord.model_validate(
                {
                    **current.model_dump(),
                    "revision": current.revision + 1,
                    "comparison_ref": uri,
                    "comparison_status": comparison["comparison_status"],
                    "promotion_status": "eligible"
                    if comparison["decision"] == "adopt"
                    else "archived",
                }
            )
        )

    def promote(self, genome_id: str) -> CandidateRecord:
        """Promote only a candidate admitted by the fixed comparison gate.

        The archive records a new revision and incumbent history. Model reflection
        and policy guidance cannot override eligibility or supply missing scores.
        """
        current = self.get(genome_id)
        if current.promotion_status != "eligible" or not current.comparison_ref:
            raise HEASError("candidate lacks independently confirmed improvement")
        comparison = self.store.read(current.comparison_ref)
        if comparison["reference_genome_id"] != self.incumbent():
            raise HEASError("comparison reference is no longer incumbent")
        fresh = compare_windows(self.store, comparison["old_run_ids"], comparison["new_run_ids"])
        if fresh["decision"] != "adopt":
            raise HEASError("comparison evidence no longer supports promotion")
        revision = len(list(self.store.path("candidates/activations").glob("*.json"))) + 1
        self.store.put(
            f"candidates/activations/{revision:06d}.json",
            {
                "schema_version": "1.1",
                "genome_id": genome_id,
                "reason": "objective_comparison",
                "comparison_ref": current.comparison_ref,
                "created_at": now().isoformat(),
            },
        )
        return self.save(
            CandidateRecord.model_validate(
                {
                    **current.model_dump(),
                    "revision": current.revision + 1,
                    "promotion_status": "promoted",
                }
            )
        )
