"""Task-suite plan organization and host-side evaluation composition without domain rules."""

from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from heas.core.domain import (
    Budget,
    EvaluationResult,
    Id,
    Record,
    RunPlan,
    TaskResult,
    TaskRunRecord,
    TaskSpec,
    uid,
)
from heas.core.errors import HEASError
from heas.infrastructure.adapters import AdapterRegistry
from heas.infrastructure.storage import FileStore


class TaskSuiteGroup(Record):
    """Bind authorized task instances to one provider/environment pair."""

    provider_id: Id
    environment_id: Id
    task_ids: list[Id] = Field(min_length=1)


class TaskSuite(Record):
    """Group independent task runs under a versioned, explicit partition."""

    suite_id: Id
    version: Id
    partition: str = "development"
    groups: list[TaskSuiteGroup] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_instances(self) -> "TaskSuite":
        """Reject duplicate provider/task pairs in a versioned suite."""
        pairs = [(g.provider_id, task) for g in self.groups for task in g.task_ids]
        if len(pairs) != len(set(pairs)):
            raise ValueError("suite has duplicate provider/task instances")
        return self


class EvaluationSuiteSpec(Record):
    """Freeze evaluator versions and an optional scalar selection objective."""

    suite_id: Id
    version: Id
    evaluator_ids: list[Id] = Field(min_length=1)
    evaluator_versions: dict[str, str] = Field(default_factory=dict)
    primary_metric: str | None = None  # source:name; no implicit cross-metric normalization

    @model_validator(mode="after")
    def unique_evaluators(self) -> "EvaluationSuiteSpec":
        """Reject duplicate evaluators or versions for unregistered sources."""
        if len(set(self.evaluator_ids)) != len(self.evaluator_ids):
            raise ValueError("duplicate suite evaluator")
        if not set(self.evaluator_versions).issubset(self.evaluator_ids):
            raise ValueError("version references unknown evaluator")
        return self


class EvaluationSuite:
    """Combine independent evaluator receipts without silently overriding metrics."""

    def __init__(self, spec: EvaluationSuiteSpec, registry: AdapterRegistry):
        self.spec, self.registry = spec, registry
        self.evaluator_id, self.version = spec.suite_id, spec.version
        if self.evaluator_id in spec.evaluator_ids:
            raise HEASError("evaluation suite cannot include itself")
        if any(e not in registry.evaluators for e in spec.evaluator_ids):
            raise HEASError("evaluation suite requires registered host evaluators")

    def evaluate_result(
        self,
        task: TaskSpec,
        run: TaskRunRecord,
        result: TaskResult,
        workspace: Path,
    ) -> EvaluationResult:
        """Evaluate structured task output independently of the solving model's claims."""
        feedbacks = []
        versions: dict[str, str] = {}
        for evaluator_id in self.spec.evaluator_ids:
            adapter = self.registry.evaluators[evaluator_id]
            raw = (
                adapter.evaluate_result(task, run, result, workspace)
                if hasattr(adapter, "evaluate_result")
                else adapter.evaluate(task, run, result.text or "", workspace)
            )
            feedback = EvaluationResult.model_validate(raw)
            if feedback.run_id != run.run_id or feedback.evaluator_id != evaluator_id:
                raise HEASError("suite evaluator receipt identity mismatch")
            version = (
                getattr(adapter, "version", "unversioned")
                if feedback.evaluator_version == "unversioned"
                else feedback.evaluator_version
            )
            if self.spec.evaluator_versions.get(evaluator_id, version) != version:
                raise HEASError("suite evaluator version mismatch")
            for source, source_version in {
                evaluator_id: version,
                **feedback.evaluator_versions,
            }.items():
                if source in versions and versions[source] != source_version:
                    raise HEASError("evaluation suite source version conflict")
                versions[source] = source_version
            feedbacks.append(feedback)
        complete = all(f.evaluation_status == "completed" for f in feedbacks)
        metrics = {}
        if complete:
            for feedback in feedbacks:
                own = {f"{s.source}:{s.name}": s for s in feedback.scores}
                if feedback.score:
                    own[f"{feedback.score.source}:{feedback.score.name}"] = feedback.score
                for key, score in own.items():
                    if key in metrics:
                        raise HEASError("duplicate evaluation suite metric namespace: " + key)
                    metrics[key] = score
        if complete and self.spec.primary_metric and self.spec.primary_metric not in metrics:
            raise HEASError("suite primary metric is missing")
        flags = [f.comparison_eligible for f in feedbacks]
        return EvaluationResult(
            evaluator_id=self.evaluator_id,
            evaluator_version=self.version,
            run_id=run.run_id,
            evaluator_versions=versions,
            evaluation_status="completed" if complete else "not_available",
            score=metrics.get(self.spec.primary_metric or ""),
            scores=list(metrics.values()),
            feedback="\n".join(f.feedback for f in feedbacks),
            agent_feedback="\n".join(f.agent_feedback for f in feedbacks if f.agent_feedback)
            or None,
            comparison_eligible=False
            if False in flags
            else True
            if all(f is True for f in flags)
            else None,
        )


def compile_suite(
    store: FileStore,
    suite: TaskSuite,
    genome_ids: list[str],
    evaluation: EvaluationSuiteSpec | None = None,
    budget: Budget | None = None,
    random_seeds: list[int] | None = None,
) -> list[RunPlan]:
    """Compile versioned task groups into explicit serial run plans."""
    uri = f"suites/tasks/{suite.suite_id}/{suite.version}.json"
    if store.path(uri).exists():
        if store.load(uri, TaskSuite) != suite:
            raise HEASError("task suite version is immutable")
    else:
        store.put(uri, suite)
    if evaluation:
        uri = f"suites/evaluations/{evaluation.suite_id}/{evaluation.version}.json"
        if store.path(uri).exists():
            if store.load(uri, EvaluationSuiteSpec) != evaluation:
                raise HEASError("evaluation suite version is immutable")
        else:
            store.put(uri, evaluation)
    return [
        RunPlan.model_validate(
            {
                "run_plan_id": uid("plan"),
                "genome_ids": genome_ids,
                "task_provider_id": group.provider_id,
                "task_ids": group.task_ids,
                "environment_adapter_id": group.environment_id,
                "task_partition": suite.partition,
                "task_evaluator_id": evaluation.suite_id if evaluation else None,
                "evaluator_version": evaluation.version if evaluation else None,
                "task_suite_id": suite.suite_id,
                "task_suite_version": suite.version,
                "evaluation_suite_id": evaluation.suite_id if evaluation else None,
                "random_seed": seed,
                "budget": budget or Budget(),
            }
        )
        for seed in (random_seeds or [23])
        for group in suite.groups
    ]


def run_suite(
    services: Any,
    suite: TaskSuite,
    genome_ids: list[str],
    evaluation: EvaluationSuiteSpec | None = None,
    random_seeds: list[int] | None = None,
) -> list[TaskRunRecord]:
    """Execute compiled task groups with an optional independent evaluation suite."""
    if evaluation and evaluation.suite_id not in services.adapters.evaluators:
        services.adapters.register_evaluator(EvaluationSuite(evaluation, services.adapters))
    return [
        run
        for plan in compile_suite(
            services.store, suite, genome_ids, evaluation, random_seeds=random_seeds
        )
        for run in services.runtime.batch(plan)
    ]
