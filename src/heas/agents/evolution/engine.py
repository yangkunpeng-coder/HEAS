"""General L2 lifecycle: archive offspring, run, evaluate, compare, retain evidence."""

from typing import Any

from heas.agents.evolution.candidates import CandidateRegistry
from heas.agents.evolution.evolution import EvolutionService
from heas.agents.task.runtime import TaskRunService
from heas.core.domain import EvolutionAttemptRecord, EvolutionPolicy, RunPlan, TaskExperienceSummary
from heas.core.errors import BudgetExceeded, HEASError, RecoveryRequired
from heas.evidence.experience import ExperienceService
from heas.infrastructure.resources import ResourceLedger
from heas.infrastructure.storage import FileStore, digest


class EvolutionEngine:
    """Coordinate L2 materialization, execution, comparison, and outcome reflection."""

    def __init__(
        self,
        store: FileStore,
        evolution: EvolutionService,
        runtime: TaskRunService,
        experience: ExperienceService,
        candidates: CandidateRegistry,
    ):
        self.store, self.evolution, self.runtime = store, evolution, runtime
        self.experience, self.candidates = experience, candidates

    def explore(
        self,
        policy: EvolutionPolicy,
        execution_plan: RunPlan | list[RunPlan] | None = None,
        *,
        archive_limit: int = 16,
    ) -> dict[str, Any]:
        """Select archived learning evidence before executing one bounded evolution step."""
        pool = self.candidates.experience_pool(
            archive_limit, policy=policy, selector=self.evolution.parent_selection
        )
        if not pool:
            raise HEASError("candidate archive has no learning evidence")
        return self.step(pool, policy, execution_plan)

    def step(
        self,
        summaries: list[TaskExperienceSummary],
        policy: EvolutionPolicy,
        execution_plan: RunPlan | list[RunPlan] | None = None,
        *,
        compare: bool = True,
        attempt_id: str | None = None,
        resume: bool = False,
    ) -> dict[str, Any]:
        """Execute offspring; retain unknown improvement without independent scores."""
        plans = (
            execution_plan
            if isinstance(execution_plan, list)
            else ([execution_plan] if execution_plan else [])
        )
        if any(p.task_partition == "final_test" for p in plans):
            raise HEASError("evolution cannot consume final_test execution data")
        ledger = ResourceLedger(self.store)
        checkpoint = ledger.checkpoint()
        attempt = self.evolution.attempt(summaries, policy, attempt_id=attempt_id, resume=resume)
        directory = f"evolution/attempts/{attempt.evolution_attempt_id}"
        request_uri = f"{directory}/execution_request.json"
        execution = {"plans": [p.model_dump(mode="json") for p in plans], "compare": compare}
        if self.store.path(request_uri).exists():
            if self.store.read(request_uri) != execution:
                raise RecoveryRequired("attempt execution contract changed")
        else:
            self.store.put(request_uri, execution)
        if self.store.path(f"{directory}/outcome.json").exists():
            outcome = self.store.read(f"{directory}/outcome.json")
            completion = self._feedback(attempt, directory, checkpoint)
            return {"attempt": attempt, **outcome, "feedback_completion": completion}
        results = []
        for genome_id in attempt.offspring_genome_ids:
            genome = self.store.genome(genome_id)
            if not self.store.path(f"candidates/{genome_id}/revisions").exists():
                self.candidates.register(genome, attempt.evolution_attempt_id)
            candidate = self.candidates.get(genome_id)
            result: dict[str, Any] = {"genome_id": genome_id, "run_ids": [], "summary_id": None}
            try:
                if plans:
                    runs = []
                    for template in plans:
                        plan_id = (
                            "plan_"
                            + digest(
                                f"{attempt.evolution_attempt_id}|{genome_id}|{template.run_plan_id}".encode()
                            )[7:39]
                        )
                        plan = RunPlan.model_validate(
                            {
                                **template.model_dump(),
                                "run_plan_id": plan_id,
                                "genome_ids": [genome_id],
                            }
                        )
                        runs.extend(self.runtime.batch(plan, resume=resume))
                    result["run_ids"] = [r.run_id for r in runs]
                    if candidate.run_ids != result["run_ids"]:
                        candidate = self.candidates.record_execution(genome_id, result["run_ids"])
                    for run in runs:
                        if not self.store.path(f"runs/{run.run_id}/reflection.json").exists():
                            self.experience.reflect_run(run)
                    if len(runs) >= 2:
                        window = attempt.evolution_attempt_id
                        archived = [
                            self.store.load(
                                p.relative_to(self.store.root).as_posix(), TaskExperienceSummary
                            )
                            for p in self.store.path(
                                f"task_agents/{genome.agent_id}/{genome_id}/summaries"
                            ).glob("*.json")
                        ]
                        summary = next((s for s in archived if s.window_id == window), None)
                        if summary is None:
                            summary = self.experience.task_summary(runs, window)
                        result["summary_id"] = summary.summary_id
                    reference_id = self.candidates.incumbent()
                    if (
                        compare
                        and reference_id
                        and self.candidates.get(reference_id).run_ids
                        and candidate.comparison_ref is None
                    ):
                        candidate = self.candidates.compare(genome_id)
            except (BudgetExceeded, RecoveryRequired):
                raise
            except Exception as exc:
                result["execution_analysis_error"] = type(exc).__name__
                candidate = self.candidates.get(genome_id)
            result["candidate"] = candidate.model_dump(mode="json")
            results.append(result)
        outcome = {
            "schema_version": "1.1",
            "evolution_attempt_id": attempt.evolution_attempt_id,
            "offspring": results,
            "resource_usage": ledger.usage(checkpoint).model_dump(),
            "evaluation_status": "completed"
            if results
            and all(
                r["candidate"]["evaluation"]["evaluation_status"] == "completed" for r in results
            )
            else "not_available",
        }
        self.store.put(f"{directory}/outcome.json", outcome)
        self.store.index(
            "evolution_outcome", attempt.evolution_attempt_id, f"{directory}/outcome.json"
        )
        completion = self._feedback(attempt, directory, checkpoint)
        return {"attempt": attempt, **outcome, "feedback_completion": completion}

    def _feedback(
        self, attempt: EvolutionAttemptRecord, directory: str, checkpoint: Any
    ) -> dict[str, Any]:
        uri = f"{directory}/feedback_completion.json"
        if self.store.path(uri).exists():
            return self.store.read(uri)
        status, error_type = "completed", None
        try:
            self.experience.reflect_attempt(attempt)
        except Exception as exc:
            status, error_type = "failed", type(exc).__name__
        completion = {
            "schema_version": "1.2",
            "reflection_status": status,
            "error_type": error_type,
            "resource_usage": ResourceLedger(self.store).usage(checkpoint).model_dump(mode="json"),
        }
        self.store.put(uri, completion)
        return completion
