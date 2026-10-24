"""Matched policy trials and evidence-gated activation/rollback in the fixed control plane."""

from contextlib import ExitStack
from typing import Any, Literal

from pydantic import Field, model_validator

from heas.agents.evolution.candidates import compare_windows
from heas.agents.meta.meta import load_policy
from heas.agents.meta.trial_criteria import PolicyTrialCriteria, assess_trial
from heas.core.domain import (
    Budget,
    EvidenceRef,
    EvolutionPolicy,
    Id,
    MetaPolicyTrial,
    Record,
    RunPlan,
    TaskExperienceSummary,
    uid,
)
from heas.core.errors import HEASError, RecoveryRequired
from heas.experimentation.isolation import IsolatedPolicyTrialRunner, ServiceFactory
from heas.infrastructure.model import CallBudget
from heas.infrastructure.resources import ResourceLedger


class PolicyTrialSpec(Record):
    """Freeze independent policy comparison conditions and host-owned admission rules."""

    schema_version: Literal["1.1"] = "1.1"
    spec_id: Id
    start_genome_ids: list[Id] = Field(min_length=1)
    training_plan: RunPlan
    evaluation_plan: RunPlan
    random_seeds: list[int] = Field(default=[23], min_length=1)
    steps_per_policy: int = Field(default=2, ge=1, le=8)
    max_model_calls_per_side: int = Field(default=200, ge=1)
    repetitions: int = Field(default=1, ge=1, le=20)
    total_budget: Budget | None = None
    criteria: PolicyTrialCriteria = Field(default_factory=PolicyTrialCriteria)
    analysis_budget: Budget = Field(default_factory=lambda: Budget(max_wall_seconds=60))
    evolution_budget: Budget = Field(default_factory=lambda: Budget(max_wall_seconds=60))

    @model_validator(mode="after")
    def partitions(self) -> "PolicyTrialSpec":
        """Keep policy training and meta-validation disjoint and comparable."""
        if self.training_plan.task_partition != "evolution_train":
            raise ValueError("policy training requires evolution_train partition")
        if self.evaluation_plan.task_partition != "meta_validation":
            raise ValueError("policy evaluation requires meta_validation, never final_test")
        if set(self.training_plan.task_ids).intersection(self.evaluation_plan.task_ids):
            raise ValueError("policy trial requires disjoint training/validation task IDs")
        for field in ("model_config_ref", "environment_adapter_id", "task_provider_id"):
            if getattr(self.training_plan, field) != getattr(self.evaluation_plan, field):
                raise ValueError("training and validation must use matching model and environment")
        return self


class EvolutionPolicyManager:
    """Activate independently validated policies and retain rollback history."""

    def __init__(self, store: Any):
        self.store = store

    def history(self) -> list[dict[str, Any]]:
        """Return the ordered, immutable policy activation history."""
        return [
            self.store.read(p.relative_to(self.store.root).as_posix())
            for p in sorted(self.store.path("policies/activations").glob("*.json"))
        ]

    def active(self) -> str | None:
        """Read the most recently approved active policy identity."""
        history = self.history()
        return history[-1]["policy_id"] if history else None

    def _activate(self, policy_id: str, reason: str, trial_id: str | None = None) -> None:
        history = self.history()
        self.store.put(
            f"policies/activations/{len(history) + 1:06d}.json",
            {
                "schema_version": "1.1",
                "policy_id": policy_id,
                "reason": reason,
                "previous_policy_id": self.active(),
                "trial_id": trial_id,
            },
        )

    def initialize(self, policy_id: str) -> None:
        """Record the first host-approved baseline under the fixed admission rules."""
        if self.active() is not None or load_policy(self.store, policy_id).policy_status != (
            "approved_baseline"
        ):
            raise HEASError("initial activation requires a host-approved baseline")
        self._activate(policy_id, "host_baseline")

    def adopt(self, trial_id: str) -> None:
        """Recompute fixed trial gates before activating a candidate policy."""
        trial = MetaPolicyTrial.model_validate(
            self.store.resolve(EvidenceRef(ref_type="policy_trial", ref_id=trial_id))
        )
        if (
            trial.comparison_status != "completed"
            or trial.decision != "adopt"
            or trial.old_policy_id != self.active()
            or not trial.objective_metrics_ref
        ):
            raise HEASError(
                "policy lacks an independent completed comparison against active baseline"
            )
        metrics = self.store.read(trial.objective_metrics_ref)
        if not metrics["round_comparisons"]:
            raise HEASError("trial has no comparison evidence")
        fresh = [
            compare_windows(self.store, c["old_run_ids"], c["new_run_ids"])
            for c in metrics["round_comparisons"]
        ]
        directory = f"meta/trials/{trial_id}"
        spec = self.store.load(f"{directory}/spec.json", PolicyTrialSpec)
        assessment = assess_trial(fresh, metrics["sides"], spec.criteria)
        if assessment["decision"] != "adopt":
            raise HEASError("trial evidence does not independently support policy adoption")
        self._activate(trial.new_policy_id, "independent_trial", trial_id)

    def rollback(self, policy_id: str) -> None:
        """Activate a policy that previously appeared in approved activation history."""
        if policy_id not in {h["policy_id"] for h in self.history()}:
            raise HEASError("rollback target was never an approved active policy")
        load_policy(self.store, policy_id)
        self._activate(policy_id, "host_rollback")


class PolicyTrialService:
    """Compare policies in matched, isolated branches under fixed host criteria."""

    def __init__(self, services: Any, factory: ServiceFactory | None = None):
        self.services = services
        self.factory = factory
        self.store = services.store

    def execute(
        self,
        old_policy_id: str,
        new_policy_id: str,
        spec: PolicyTrialSpec,
        *,
        trial_id: str | None = None,
        resume: bool = False,
    ) -> MetaPolicyTrial:
        """Execute matched policies using independent branch services.

        Args:
            old_policy_id: Approved reference policy identity.
            new_policy_id: Candidate child policy identity.
            spec: Fixed seeds, training and validation partitions, budgets, and criteria.
            trial_id: Optional identity reserved by the experiment journal.
            resume: Recover an interrupted trial as invalid rather than replay its effects.

        Returns:
            An evidence-backed comparison; unavailable evaluation remains undecided.
        """
        services, store = self.services, self.store
        old, new = load_policy(store, old_policy_id), load_policy(store, new_policy_id)
        if new.parent_policy_id != old.policy_id:
            raise HEASError("trial candidate must descend from the reference policy")
        trial_id = trial_id or uid("trial")
        directory = f"meta/trials/{trial_id}"
        recovered = self._recover_trial(old_policy_id, new_policy_id, spec, trial_id, resume)
        if recovered is not None:
            return recovered
        store.put(f"{directory}/spec.json", spec)
        store.put(
            f"{directory}/execution/000001.json", {"schema_version": "1.1", "status": "running"}
        )
        ledger = ResourceLedger(store)
        shared_before = ledger.report()
        sides: list[dict[str, Any]] = []
        runner = IsolatedPolicyTrialRunner(services, directory, self.factory)
        for side_name, policy in (("old", old), ("new", new)):
            side = self._execute_side(runner, ledger, directory, side_name, policy, spec)
            sides.append(side)
            store.put(f"{directory}/execution/{len(sides) + 1:06d}.json", side)
        comparisons = []
        status: Any = "invalid"
        decision: Any = "undecided"
        improvement = None
        if all(s["status"] == "completed" for s in sides):
            for before, after in zip(sides[0]["rounds"], sides[1]["rounds"], strict=True):
                comparisons.append(compare_windows(store, before["run_ids"], after["run_ids"]))
            if comparisons and all(c["evaluation_status"] == "completed" for c in comparisons):
                status = "completed"
                available = [c["improvement"] for c in comparisons if c["improvement"] is not None]
                improvement = sum(available) / len(available) if available else None
            elif comparisons and all(
                c["comparison_status"] != "not_comparable" for c in comparisons
            ):
                status = "not_available"
        metrics_uri = f"{directory}/metrics.json"
        assessment = assess_trial(comparisons, sides, spec.criteria)
        decision = assessment["decision"]
        store.put(
            metrics_uri,
            {
                "schema_version": "1.1",
                "round_comparisons": comparisons,
                "mean_improvement": improvement,
                "assessment": assessment,
                "repetitions": spec.repetitions,
                "model_profiles": {
                    purpose: {
                        "identity": client.identity,
                        "parameters": getattr(client, "settings", {}),
                        "determinism": "not_guaranteed",
                    }
                    for purpose, client in {
                        "default": services.gateway.model,
                        **services.gateway.models_by_purpose,
                    }.items()
                },
                "sides": sides,
                "shared_prior_resources_including_meta": shared_before,
                "whole_workspace_resources": ledger.report(),
            },
        )
        trial = MetaPolicyTrial(
            trial_id=trial_id,
            old_policy_id=old_policy_id,
            new_policy_id=new_policy_id,
            run_plan_refs=[r["run_plan_ref"] for s in sides for r in s["rounds"]],
            start_genome_ids=spec.start_genome_ids,
            random_seeds=spec.random_seeds,
            budget_rule="Identical per-run, analysis, L2 and total model-call caps per side.",
            comparison_status=status,
            objective_metrics_ref=metrics_uri,
            decision=decision,
            decision_evidence_refs=[
                EvidenceRef.model_validate(ref)
                for c in comparisons
                for ref in c.get("evidence_refs", [])
            ],
        )
        store.save_record(f"{directory}/trial.json", trial, "policy_trial", trial_id)
        return trial

    def _recover_trial(
        self,
        old_policy_id: str,
        new_policy_id: str,
        spec: PolicyTrialSpec,
        trial_id: str,
        resume: bool,
    ) -> MetaPolicyTrial | None:
        """Reuse committed receipts or archive uncertain interrupted branches without replay."""
        store = self.store
        directory = f"meta/trials/{trial_id}"
        if store.path(f"{directory}/spec.json").exists():
            if store.load(f"{directory}/spec.json", PolicyTrialSpec) != spec:
                raise HEASError("trial plan changed")
            if store.path(f"{directory}/trial.json").exists():
                existing = store.load(f"{directory}/trial.json", MetaPolicyTrial)
                if (existing.old_policy_id, existing.new_policy_id) != (
                    old_policy_id,
                    new_policy_id,
                ):
                    raise HEASError("trial policy pair changed")
                return existing
            if not resume:
                raise RecoveryRequired("incomplete trial requires explicit recovery")
            uri = f"{directory}/recovered_metrics.json"
            store.put(
                uri,
                {
                    "round_comparisons": [],
                    "sides": [],
                    "assessment": {
                        "decision": "undecided",
                        "reasons": ["interrupted_trial_not_replayed"],
                    },
                },
            )
            recovered = MetaPolicyTrial(
                trial_id=trial_id,
                old_policy_id=old_policy_id,
                new_policy_id=new_policy_id,
                run_plan_refs=[],
                start_genome_ids=spec.start_genome_ids,
                random_seeds=spec.random_seeds,
                budget_rule="Interrupted branches preserved, not replayed.",
                comparison_status="invalid",
                objective_metrics_ref=uri,
                decision="undecided",
                decision_evidence_refs=[],
            )
            store.save_record(f"{directory}/trial.json", recovered, "policy_trial", trial_id)
            return recovered
        return None

    def _execute_side(
        self,
        runner: IsolatedPolicyTrialRunner,
        ledger: ResourceLedger,
        directory: str,
        side_name: str,
        policy: EvolutionPolicy,
        spec: PolicyTrialSpec,
    ) -> dict[str, Any]:
        """Run matched repetitions with fresh branch services and unconditional Host cleanup."""
        store = self.store
        checkpoint = ledger.checkpoint()
        side: dict[str, Any] = {
            "policy_id": policy.policy_id,
            "status": "completed",
            "rounds": [],
            "error_type": None,
            "attempt_ids": [],
            "branch_workspaces": [],
            "isolation": "independent_services_per_repetition",
        }
        budget = spec.total_budget or Budget(
            max_model_calls=spec.max_model_calls_per_side,
            max_tool_calls=100000,
            max_wall_seconds=3600,
        )
        budget = Budget.model_validate(
            {
                **budget.model_dump(),
                "max_model_calls": min(budget.max_model_calls, spec.max_model_calls_per_side),
            }
        )
        side_meter = CallBudget(budget)
        try:
            for repetition in range(spec.repetitions):
                for seed in spec.random_seeds:
                    child = runner.create(side_name, repetition, seed)
                    child.gateway.global_meter = side_meter
                    child.experience.budget = spec.analysis_budget
                    child.evolution.budget = spec.evolution_budget
                    side["branch_workspaces"].append(str(child.store.root))
                    try:
                        self._repeat(child, policy, spec, side, repetition, seed)
                    finally:
                        # Run-level teardown does not necessarily release an
                        # external worker owned by the whole branch. Close
                        # every optional host resource even if collection fails.
                        with ExitStack() as cleanup:
                            for environment in child.adapters.environments.values():
                                close = getattr(environment, "close", None)
                                if callable(close):
                                    cleanup.callback(close)
                            runner.collect(child)
        except BaseException as exc:
            side["status"] = "interrupted" if not isinstance(exc, Exception) else "failed"
            side["error_type"] = type(exc).__name__
            if not isinstance(exc, Exception):
                store.put(f"{directory}/interrupted.json", side)
                raise
        side["resource_usage"] = ledger.report(checkpoint)
        side["operation_counts"] = {}
        for attempt_id in side["attempt_ids"]:
            uri = f"evolution/attempts/{attempt_id}/resolved_proposal.json"
            if store.path(uri).exists():
                for operation in store.read(uri)["operations"]:
                    key = operation["operation"] + ":" + operation["target_kind"]
                    side["operation_counts"][key] = side["operation_counts"].get(key, 0) + 1
        return side

    def _repeat(
        self,
        services: Any,
        policy: Any,
        spec: PolicyTrialSpec,
        side: dict[str, Any],
        repetition: int,
        seed: int,
    ) -> None:
        store = services.store
        ledger = ResourceLedger(store)
        for genome_id in spec.start_genome_ids:
            services.candidates.register(store.genome(genome_id))
        services.candidates.initialize_incumbent(spec.start_genome_ids[0])
        training = RunPlan.model_validate(
            {
                **spec.training_plan.model_dump(),
                "run_plan_id": uid("plan"),
                "genome_ids": spec.start_genome_ids,
                "random_seed": seed,
            }
        )
        seed_runs = services.runtime.batch(training)
        pool = []
        for genome_id in spec.start_genome_ids:
            runs = [r for r in seed_runs if r.genome_id == genome_id]
            for run in runs:
                services.experience.reflect_run(run)
            pool.append(services.experience.task_summary(runs, training.run_plan_id))
        for _ in range(spec.steps_per_policy):
            attempt_checkpoint = ledger.checkpoint()
            outcome = services.engine.step(pool, policy, training, compare=False)
            attempt_id = outcome["evolution_attempt_id"]
            side["attempt_ids"].append(attempt_id)
            if not outcome["offspring"]:
                raise HEASError("policy trial attempt did not produce an executable child")
            child_ids = [r["genome_id"] for r in outcome["offspring"]]
            validation = RunPlan.model_validate(
                {
                    **spec.evaluation_plan.model_dump(),
                    "run_plan_id": uid("plan"),
                    "genome_ids": child_ids,
                    "random_seed": seed,
                }
            )
            evaluation_runs = services.runtime.batch(validation)
            if len(child_ids) != 1:
                raise HEASError("policy trial currently compares one child per attempt")
            side["rounds"].append(
                {
                    "evolution_attempt_id": attempt_id,
                    "repetition": repetition,
                    "random_seed": seed,
                    "run_ids": [r.run_id for r in evaluation_runs],
                    "run_plan_ref": f"plans/{validation.run_plan_id}.json",
                }
            )
            store.put(
                f"evolution/attempts/{attempt_id}/validation_outcome.json",
                {
                    "schema_version": "1.1",
                    "evaluation_status": "completed"
                    if all(r.evaluation.evaluation_status == "completed" for r in evaluation_runs)
                    else "not_available",
                    "run_ids": [r.run_id for r in evaluation_runs],
                },
            )
            services.experience.reflect_attempt(outcome["attempt"])
            store.put(
                f"evolution/attempts/{attempt_id}/trial_feedback_completion.json",
                {
                    "schema_version": "1.2",
                    "validation_reflection_completed": True,
                    "resource_usage": ledger.usage(attempt_checkpoint).model_dump(mode="json"),
                },
            )
            # Policy selects among evidence-covered branches.
            for result in outcome["offspring"]:
                if result["summary_id"]:
                    pool.append(
                        TaskExperienceSummary.model_validate(
                            store.resolve(
                                EvidenceRef(ref_type="summary", ref_id=result["summary_id"])
                            )
                        )
                    )
