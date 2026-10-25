"""Validation-selected policy freezing and independent final reporting, without activation."""

from contextlib import ExitStack
from typing import Any

from heas.agents.evolution.candidates import compare_windows
from heas.agents.meta.meta import load_policy
from heas.agents.meta.policy_trials import PolicyTrialSpec
from heas.agents.meta.trial_criteria import assess_trial
from heas.core.domain import EvidenceRef, MetaPolicyTrial, RunPlan, now, uid
from heas.core.errors import HEASError, RecoveryRequired, SecurityError
from heas.experimentation.isolation import IsolatedPolicyTrialRunner
from heas.experimentation.metrics import measured, unknown
from heas.experimentation.ports import ExperimentServices
from heas.infrastructure.model import CallBudget
from heas.infrastructure.resources import ResourceLedger
from heas.infrastructure.storage import FileStore, digest, encode


class PolicyMetrics:
    """Select policies on matched meta validation, then freeze and report held-out gain."""

    def __init__(self, store: FileStore):
        self.store = store

    def assess(self, trial_id: str) -> tuple[MetaPolicyTrial, PolicyTrialSpec, dict[str, Any]]:
        """Recompute fixed objective gates instead of trusting an archived recommendation."""
        trial = MetaPolicyTrial.model_validate(
            self.store.resolve(EvidenceRef(ref_type="policy_trial", ref_id=trial_id))
        )
        spec = self.store.load(f"meta/trials/{trial_id}/spec.json", PolicyTrialSpec)
        metrics_uri = trial.objective_metrics_ref
        data = self.store.read_object(metrics_uri) if metrics_uri is not None else {}
        fresh = [
            compare_windows(self.store, row["old_run_ids"], row["new_run_ids"])
            for row in data.get("round_comparisons", [])
        ]
        return trial, spec, assess_trial(fresh, data.get("sides", []), spec.criteria)

    def freeze(self, selection_id: str, trial_ids: list[str]) -> dict[str, Any]:
        """Choose the best admitted validation policy before any final-test exposure."""
        if not trial_ids or len(set(trial_ids)) != len(trial_ids):
            raise HEASError("distinct policy-validation trials are required")
        rows = [self.assess(identity) for identity in trial_ids]
        signatures = []
        for trial, spec, assessment in rows:
            if trial.comparison_status != "completed" or assessment["decision"] != "adopt":
                raise HEASError("policy selection requires fixed-gate independent validation")
            contract = spec.model_dump(mode="json", exclude={"spec_id"})
            for plan in ("training_plan", "evaluation_plan"):
                contract[plan].pop("run_plan_id")
            signatures.append(digest(encode({"old_policy": trial.old_policy_id, "spec": contract})))
        if len(set(signatures)) != 1:
            raise HEASError("policy trials do not share the same initial conditions and budgets")
        if any(
            self.store.read(p.relative_to(self.store.root).as_posix())["task_partition"]
            == "final_test"
            for p in self.store.path("runs").glob("*/run.json")
        ):
            raise SecurityError("policy selection must precede all final tests in its workspace")
        trial, spec, assessment = max(rows, key=lambda row: row[2]["mean_difference"])
        selection = {
            "schema_version": "1.2",
            "selection_id": selection_id,
            "policy_id": trial.new_policy_id,
            "baseline_policy_id": trial.old_policy_id,
            "trial_ids": trial_ids,
            "selected_trial_id": trial.trial_id,
            "created_at": now().isoformat(),
            "protocol_sha256": signatures[0],
            "spec": spec.model_dump(mode="json"),
        }
        self.store.put(f"metrics/policy_selections/{selection_id}.json", selection)
        return selection

    def final_report(
        self, services: ExperimentServices, selection_id: str, final_plan: RunPlan
    ) -> dict[str, Any]:
        """Train paired frozen policies independently, then evaluate their last child once.

        Final scores are report-only: they never enter an attempt reflection,
        experience summary, candidate registry, or policy activation decision.
        """
        selection = self.store.read(f"metrics/policy_selections/{selection_id}.json")
        spec = PolicyTrialSpec.model_validate(selection["spec"])
        if final_plan.task_partition != "final_test":
            raise SecurityError("policy held-out reporting requires final_test")
        if set(final_plan.task_ids) & (
            set(spec.training_plan.task_ids) | set(spec.evaluation_plan.task_ids)
        ):
            raise SecurityError("final policy tasks overlap learning or validation")
        directory = f"metrics/policy_final/{selection_id}"
        request = final_plan.model_dump(mode="json")
        if self.store.path(f"{directory}/report.json").exists():
            if self.store.read(f"{directory}/request.json") != request:
                raise HEASError("frozen final report request changed")
            return self.store.read_object(f"{directory}/report.json")
        if self.store.path(f"{directory}/request.json").exists():
            raise RecoveryRequired("interrupted policy final report cannot be blindly replayed")
        self.store.put(f"{directory}/request.json", request)
        isolation = IsolatedPolicyTrialRunner(services, directory)
        sides = []
        for label, pid in (
            ("old", selection["baseline_policy_id"]),
            ("new", selection["policy_id"]),
        ):
            all_runs = []
            resources = []
            for repetition in range(spec.repetitions):
                for seed in spec.random_seeds:
                    branch = isolation.create(label, repetition, seed)
                    branch.experience.budget = spec.analysis_budget
                    branch.evolution.budget = spec.evolution_budget
                    meter = (
                        CallBudget(spec.total_budget)
                        if spec.total_budget
                        else CallBudget(
                            spec.training_plan.budget.model_copy(
                                update={
                                    "max_model_calls": spec.max_model_calls_per_side,
                                }
                            )
                        )
                    )
                    branch.gateway.budget_scopes.append(meter)
                    try:
                        policy = load_policy(branch.store, pid)
                        training = spec.training_plan.model_copy(
                            update={
                                "run_plan_id": uid("plan"),
                                "genome_ids": spec.start_genome_ids,
                                "random_seed": seed,
                            }
                        )
                        runs = branch.runtime.batch(training)
                        pool = []
                        for gid in spec.start_genome_ids:
                            group = [r for r in runs if r.genome_id == gid]
                            for run in group:
                                branch.experience.reflect_run(run)
                            pool.append(branch.experience.task_summary(group, training.run_plan_id))
                        child_ids: list[str] = []
                        for _ in range(spec.steps_per_policy):
                            outcome = branch.engine.step(pool, policy, training, compare=False)
                            child_ids = [
                                offspring_record["genome_id"]
                                for offspring_record in outcome["offspring"]
                            ]
                            for child in outcome["offspring"]:
                                if child["summary_id"]:
                                    from heas.core.domain import TaskExperienceSummary

                                    pool.append(
                                        TaskExperienceSummary.model_validate(
                                            branch.store.resolve(
                                                EvidenceRef(
                                                    ref_type="summary", ref_id=child["summary_id"]
                                                )
                                            )
                                        )
                                    )
                        if len(child_ids) != 1:
                            raise HEASError("paired policy report requires one final child per arm")
                        branch.store.put(
                            "metrics/frozen_before_final.json",
                            {
                                "genome_id": child_ids[0],
                                "policy_id": pid,
                                "frozen_at": now().isoformat(),
                            },
                        )
                        plan = final_plan.model_copy(
                            update={
                                "run_plan_id": uid("plan"),
                                "genome_ids": child_ids,
                                "random_seed": seed,
                            }
                        )
                        all_runs.append([r.run_id for r in branch.runtime.batch(plan)])
                        resources.append(ResourceLedger(branch.store).report())
                        isolation.collect(branch)
                    finally:
                        branch.gateway.budget_scopes.remove(meter)
                        with ExitStack() as cleanup:
                            for environment in branch.adapters.environments.values():
                                close = getattr(environment, "close", None)
                                if callable(close):
                                    cleanup.callback(close)
            sides.append({"policy_id": pid, "windows": all_runs, "resources": resources})
        comparisons = [
            compare_windows(self.store, a, b, learning=False)
            for a, b in zip(sides[0]["windows"], sides[1]["windows"], strict=True)
        ]
        values = [c.get("improvement") for c in comparisons]
        metric = unknown("score_delta", "incomplete or incomparable independent policy outcomes")
        if values and all(value is not None for value in values):
            metric = measured(
                sum(float(value) for value in values if value is not None) / len(values),
                "score_delta",
                selection["trial_ids"],
            )
        report = {
            "selection_id": selection_id,
            "k": len(selection["trial_ids"]),
            "meta_improvement_at_k": metric.model_dump(mode="json"),
            "sides": sides,
            "comparisons": comparisons,
            "report_only": True,
        }
        self.store.put(f"{directory}/report.json", report)
        return report
