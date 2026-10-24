"""Serial paired SEABench runs with explicit HEAS capability evolution boundaries."""

from collections.abc import Callable
from typing import Any

from benchmark_adapters.common.config import Partition, SEABenchConfig
from benchmark_adapters.common.runtime import BenchmarkEnvironment
from benchmark_adapters.seabench.provider import SEABenchProvider
from heas.agents.meta.meta import load_policy
from heas.core.domain import Budget, EvolutionAttemptRecord, RunPlan, TaskRunRecord, now
from heas.core.errors import HEASError, RecoveryRequired
from heas.experimentation.isolation import IsolatedPolicyTrialRunner
from heas.infrastructure.model import CallBudget
from heas.infrastructure.resources import ResourceLedger
from heas.infrastructure.storage import digest


class SEABenchSequenceRunner:
    """Run matched evolving/control arms without learning from downstream outcomes.

    The host controls ordering and when evolution is permitted. All capability
    decisions still pass through the existing HEAS EvolutionEngine. This fixed
    manifest protocol does not reproduce the official adaptive failure discovery.
    """

    def __init__(self, services: Any, config: SEABenchConfig):
        self.services, self.config, self.store = services, config, services.store

    def execute(
        self,
        experiment_id: str,
        genome_id: str,
        policy_id: str,
        *,
        window_size: int = 2,
        run_budget: Budget | None = None,
        total_budget: Budget | None = None,
        factory: Callable[[Any], Any] | None = None,
    ) -> dict[str, Any]:
        """Execute each manifest position once per independent arm.

        Args:
            experiment_id: New immutable experiment identity.
            genome_id: Identical initial genome in both arms.
            policy_id: Identical initial evolution policy in both arms.
            window_size: Upstream runs per frozen genome before an L2 boundary.
            run_budget: Per-task limit shared by both arms.
            total_budget: Equal per-arm limit covering task and evolution calls.
            factory: Trusted independent service constructor for opaque transports.

        Raises:
            RecoveryRequired: An interrupted sequence has uncertain external effects.
        """
        if window_size < 2:
            raise HEASError("multi-run evolution windows require at least two tasks")
        directory = "experiments/seabench/" + experiment_id
        # FileStore.path validates the host identifier before interpolated writes.
        self.store.path(directory)
        request = {
            "genome_id": genome_id,
            "policy_id": policy_id,
            "window_size": window_size,
            "sequence_sha256": digest(self.config.sequence_file.read_bytes()),
            "run_budget": (run_budget or Budget()).model_dump(mode="json"),
            "total_budget": (total_budget or Budget(max_model_calls=500)).model_dump(mode="json"),
        }
        if self.store.path(f"{directory}/request.json").exists():
            if self.store.read(f"{directory}/request.json") != request:
                raise HEASError("sequence request changed; use a new experiment ID")
            if self.store.path(f"{directory}/report.json").exists():
                return self.store.read(f"{directory}/report.json")
            raise RecoveryRequired("sequence interrupted; inspect sealed records before a new run")
        self.store.put(f"{directory}/request.json", request)
        isolation = IsolatedPolicyTrialRunner(self.services, directory, factory)
        arms = []
        for label, evolving in (("evolving", True), ("control", False)):
            branch = isolation.create(label, 0, 23)
            arm = self._arm(
                branch,
                genome_id,
                policy_id,
                evolving,
                window_size,
                run_budget or Budget(),
                total_budget or Budget(max_model_calls=500),
            )
            arm["label"], arm["workspace"] = label, str(branch.store.root)
            arms.append(arm)
        report = {
            "benchmark": "seabench",
            "revision": self.config.revision,
            "protocol": "heas_fixed_manifest_paired_v1",
            "origin": self.config.data_origin,
            "arms": arms,
            "officially_verified": False,
            "attribution_status": "not_available",
            "attribution_reason": "paired completion/safety observations do not prove attribution",
            "safety_delta": self._safety_delta(arms),
        }
        self.store.put(f"{directory}/report.json", report)
        return report

    def _arm(
        self,
        services: Any,
        initial: str,
        policy_id: str,
        evolving: bool,
        window_size: int,
        run_budget: Budget,
        total_budget: Budget,
    ) -> dict[str, Any]:
        provider = next(
            p for p in services.adapters.providers.values() if isinstance(p, SEABenchProvider)
        )
        environment = services.adapters.environments[provider.provider_id + "_environment"]
        if not isinstance(environment, BenchmarkEnvironment):
            raise HEASError("SEABench requires the registered sequence environment")
        environment.sequence_context = {"session_id": "fixed_manifest_sequence"}
        policy = load_policy(services.store, policy_id)
        genome_id = initial
        window: list[TaskRunRecord] = []
        attempts: list[EvolutionAttemptRecord] = []
        records: list[dict[str, Any]] = []
        pending: EvolutionAttemptRecord | None = None
        frozen = None
        summary_id, proposed_policy = None, None
        meter = CallBudget(total_budget)
        services.gateway.budget_scopes.append(meter)
        try:
            for position, row in enumerate(provider.rows):
                partition: Partition = (
                    "final_test" if row.phase == "downstream" else "evolution_train"
                )
                task = next(
                    t
                    for t in provider.list_tasks(partition)
                    if provider.native_id(t.task_id) == row.official_task_id
                )
                if partition == "final_test" and frozen is None:
                    if evolving and len(attempts) >= 2:
                        summary = services.experience.evolution_summary(attempts)
                        summary_id = summary.summary_id
                        child_policy, _ = services.meta.propose(
                            summary, policy, {"status": "pending_independent_meta_validation"}
                        )
                        proposed_policy = child_policy.policy_id
                    frozen = {
                        "genome_id": genome_id,
                        "policy_id": policy_id,
                        "frozen_at": now().isoformat(),
                        "position": position,
                    }
                    services.store.put("sequence/frozen.json", frozen)
                evaluator = services.adapters.evaluators[provider.provider_id + "_evaluator"]
                plan = RunPlan(
                    run_plan_id=f"sequence_{position:04d}",
                    genome_ids=[genome_id],
                    task_provider_id=provider.provider_id,
                    task_ids=[task.task_id],
                    environment_adapter_id=environment.environment_id,
                    task_evaluator_id=evaluator.evaluator_id,
                    evaluator_version=evaluator.version,
                    task_partition=partition,
                    budget=run_budget,
                    random_seed=23,
                )
                run = services.runtime.batch(plan)[0]
                records.append(
                    {
                        "position": position,
                        "phase": row.phase,
                        "official_task_id": row.official_task_id,
                        "run_id": run.run_id,
                        "genome_id": genome_id,
                        "policy_id": policy_id,
                        "evaluation": run.evaluation.model_dump(mode="json"),
                        "usage": run.usage.model_dump(mode="json"),
                    }
                )
                if partition == "final_test" or not evolving:
                    continue
                services.experience.reflect_run(run)
                window.append(run)
                if len(window) < window_size:
                    continue
                if pending:
                    services.store.put(
                        f"evolution/attempts/{pending.evolution_attempt_id}/validation_outcome.json",
                        {
                            "run_ids": [r.run_id for r in window],
                            "partition": "evolution_train",
                            "evaluation_status": "completed"
                            if all(r.evaluation.evaluation_status == "completed" for r in window)
                            else "not_available",
                        },
                    )
                    services.experience.reflect_attempt(pending)
                    pending = None
                summary = services.experience.task_summary(window, f"sequence_window_{position}")
                # The final upstream boundary freezes the evaluated genome rather
                # than inventing an untested child just before the held-out suite.
                remaining = sum(r.phase == "upstream" for r in provider.rows[position + 1 :])
                if remaining >= window_size:
                    outcome = services.engine.step([summary], policy, compare=False)
                    pending = outcome["attempt"]
                    attempts.append(pending)
                    if pending.offspring_genome_ids:
                        genome_id = pending.offspring_genome_ids[0]
                window = []
            return {
                "runs": records,
                "frozen": frozen,
                "attempt_ids": [a.evolution_attempt_id for a in attempts],
                "evolution_summary_id": summary_id,
                "proposed_policy_id": proposed_policy,
                "resources": ResourceLedger(services.store).report(),
            }
        finally:
            services.gateway.budget_scopes.remove(meter)
            environment.bridge.close()

    @staticmethod
    def _safety_delta(arms: list[dict[str, Any]]) -> dict[str, Any]:
        values = []
        identities = []
        for arm in arms:
            rows = [r for r in arm["runs"] if r["phase"] == "downstream"]
            identities.append([r["official_task_id"] for r in rows])
            metrics = [
                next(
                    (s["value"] for s in r["evaluation"]["scores"] if s["name"] == "safety_pass"),
                    None,
                )
                for r in rows
            ]
            if not metrics or any(value is None for value in metrics):
                return {"status": "not_available", "value": None}
            values.append(
                sum(float(value) for value in metrics if value is not None) / len(metrics)
            )
        if identities[0] != identities[1]:
            raise HEASError("paired downstream task identities differ")
        return {
            "status": "completed",
            "value": values[0] - values[1],
            "unit": "ratio_delta",
            "origin": "heas_derived",
        }
