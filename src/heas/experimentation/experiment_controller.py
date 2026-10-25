"""Durable serial L1/L2/L3 scheduling. Model decisions stay in the existing services."""

import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from heas.agents.evolution.candidates import CandidateRegistry, archived_runs
from heas.agents.meta.meta import load_policy
from heas.agents.meta.policy_trials import EvolutionPolicyManager, PolicyTrialService
from heas.core.domain import (
    EvidenceRef,
    EvolutionAttemptRecord,
    EvolutionExperienceSummary,
    EvolutionPolicy,
    RunPlan,
    TaskExperienceSummary,
    now,
)
from heas.core.errors import BudgetExceeded, HEASError, RecoveryRequired
from heas.experimentation.experiment_plan import ExperimentPlan
from heas.experimentation.ports import ExperimentServices
from heas.experimentation.suites import EvaluationSuite, TaskSuite, compile_suite
from heas.infrastructure.model import CallBudget
from heas.infrastructure.reproducibility import export_manifest
from heas.infrastructure.resources import ResourceCheckpoint, ResourceLedger
from heas.infrastructure.storage import FileStore, digest


class ExperimentControl:
    """Inspect durable experiment state and queue Host requests without model services."""

    def __init__(self, store: FileStore):
        self.store = store

    @property
    def candidates(self) -> CandidateRegistry:
        """Read candidate state from the same workspace without constructing execution services."""
        return CandidateRegistry(self.store)

    @staticmethod
    def directory(experiment_id: str) -> str:
        # Validate before interpolating a user-controlled identifier.
        """Validate an experiment identity before deriving its workspace-relative directory."""
        from pydantic import TypeAdapter

        from heas.core.domain import Id

        return f"experiments/evolution/{TypeAdapter(Id).validate_python(experiment_id)}"

    def status(self, eid: str) -> dict[str, Any]:
        """Read the last committed lifecycle event and completed cycle count."""
        directory = self.directory(eid)
        self.store.load(f"{directory}/plan.json", ExperimentPlan)
        events = sorted(self.store.path(f"{directory}/events").glob("*.json"))
        if not events:
            raise RecoveryRequired("experiment plan has no committed lifecycle event")
        event = self.store.read_object(events[-1].relative_to(self.store.root).as_posix())
        cycles = sorted(self.store.path(f"{directory}/cycles").glob("*/completed.json"))
        return {"experiment_id": eid, **event, "completed_cycles": len(cycles)}

    def request(self, eid: str, action: str) -> dict[str, Any]:
        """Queue a host pause or cancellation request for the next stage boundary."""
        if action not in {"pause", "cancel"}:
            raise HEASError("unsupported lifecycle action")
        if self.status(eid)["status"] in {"completed", "cancelled", "failed"}:
            raise HEASError("experiment is already terminal")
        directory = self.directory(eid)
        n = len(list(self.store.path(f"{directory}/requests").glob("*.json"))) + 1
        self.store.put(f"{directory}/requests/{n:06d}.json", {"action": action})
        # Consumed at a stage boundary; CLI remains truthful about pending requests.
        return {**self.status(eid), "requested_action": action}

    def _resources(self, eid: str) -> dict[str, Any]:
        origin = self.store.read(f"{self.directory(eid)}/resource_origin.json")
        return ResourceLedger(self.store).report(
            ResourceCheckpoint(set(origin["calls"]), set(origin["runs"]), set(origin["tools"])),
            scope_id=origin.get("scope_id"),
        )

    def report(self, eid: str) -> dict[str, Any]:
        """Read evidence and resource accounting without making model calls."""
        directory = self.directory(eid)
        reports = sorted(self.store.path(f"{directory}/reports").glob("*.json"))
        stages = [
            {
                "stage": p.parent.name,
                "cycle": p.parent.parent.parent.name,
                "record_ref": p.relative_to(self.store.root).as_posix(),
                **self.store.read(p.relative_to(self.store.root).as_posix())["result"],
            }
            for p in sorted(
                self.store.path(f"{directory}/cycles").glob("*/stages/*/completed.json")
            )
        ]
        return {
            **self.status(eid),
            "workspace": str(self.store.root),
            "plan_ref": f"{directory}/plan.json",
            "report_ref": reports[-1].relative_to(self.store.root).as_posix() if reports else None,
            "reproducibility_ref": self.store.read(f"{directory}/reproducibility.json")[
                "manifest_ref"
            ]
            if self.store.path(f"{directory}/reproducibility.json").exists()
            else None,
            "stages": stages,
            "active_policy_id": EvolutionPolicyManager(self.store).active(),
            "incumbent_genome_id": self.candidates.incumbent(),
            "resources": self._resources(eid),
            "limitations": [
                "Synthetic runs establish engineering behavior only.",
                "Unavailable independent evaluation implies unknown improvement.",
            ],
            "quality_improvement": "unknown"
            if not any(
                c.get("comparison_status") == "improved"
                for stage in stages
                for c in stage.get("candidates", [])
            )
            else "observed_in_configured_evaluation",
        }


class EvolutionExperimentController(ExperimentControl):
    """Schedule bounded L1/L2/L3 cycles with durable recovery checkpoints."""

    def __init__(
        self, services: ExperimentServices, on_checkpoint: Callable[[str], None] | None = None
    ):
        super().__init__(services.store)
        self.services = services
        self.on_checkpoint = on_checkpoint

    @property
    def candidates(self) -> CandidateRegistry:
        """Use the injected execution registry for live experiment reports."""
        return self.services.candidates

    def create(self, plan: ExperimentPlan) -> None:
        """Initialize the immutable host configuration or an independent execution branch."""
        store, services = self.store, self.services
        directory = self.directory(plan.experiment_id)
        if Path(plan.output_directory).resolve() != store.root:
            raise HEASError("experiment output_directory must equal the host workspace")
        if store.path(f"{directory}/plan.json").exists():
            if store.load(f"{directory}/plan.json", ExperimentPlan) != plan:
                raise HEASError("experiment plan is immutable")
            return
        policy = load_policy(store, plan.initial_policy_id)
        if policy.policy_status != "approved_baseline":
            raise HEASError("experiment starts from a host-approved baseline policy")
        if plan.parent_selection and plan.parent_selection != policy.parent_selection:
            raise HEASError("plan parent selection must match the archived initial policy")
        for gid in plan.initial_genome_ids:
            services.materializer.validator.validate(
                store.genome(gid).skills, store.genome(gid).tools
            )
            if not store.path(f"candidates/{gid}/revisions").exists():
                services.candidates.register(store.genome(gid))
        if services.candidates.incumbent() is None:
            services.candidates.initialize_incumbent(plan.initial_genome_ids[0])
        manager = EvolutionPolicyManager(store)
        if manager.active() is None:
            manager.initialize(policy.policy_id)
        if manager.active() != policy.policy_id:
            raise HEASError("workspace active policy differs from the experiment baseline")
        checkpoint = ResourceLedger(store).checkpoint()
        store.put(f"{directory}/plan.json", plan)
        store.put(
            f"{directory}/resource_origin.json",
            {
                "calls": sorted(checkpoint),
                "runs": sorted(checkpoint.runs),
                "tools": sorted(checkpoint.tools),
                "scope_id": f"experiment:{plan.experiment_id}",
            },
        )
        self._event(plan.experiment_id, "created", "host_plan")

    def _event(self, eid: str, status: str, reason: str) -> None:
        directory = self.directory(eid)
        n = len(list(self.store.path(f"{directory}/events").glob("*.json"))) + 1
        self.store.put(
            f"{directory}/events/{n:06d}.json",
            {
                "status": status,
                "reason": reason,
                "recorded_at": now().isoformat(),
            },
        )

    def _requested(self, eid: str) -> str | None:
        directory = self.directory(eid)
        paths = sorted(self.store.path(f"{directory}/requests").glob("*.json"))
        for path in paths:
            marker = f"{directory}/consumed_requests/{path.name}"
            if not self.store.path(marker).exists():
                action = self.store.read_object(path.relative_to(self.store.root).as_posix()).get(
                    "action"
                )
                if not isinstance(action, str) or action not in {"pause", "cancel"}:
                    raise RecoveryRequired("invalid lifecycle request; inspect before consuming")
                self.store.put(marker, {"consumed": True})
                return action
        return None

    def _stage(
        self,
        eid: str,
        cycle: int,
        name: str,
        operation: Callable[[], dict[str, Any]],
        *,
        recoverable: bool = False,
    ) -> dict[str, Any]:
        directory = f"{self.directory(eid)}/cycles/{cycle:04d}/stages/{name}"
        uri = f"{directory}/completed.json"
        if self.store.path(uri).exists():
            result = self.store.read_object(uri).get("result")
            if not isinstance(result, dict):
                raise RecoveryRequired("completed stage has no valid result; do not replay")
            return result
        action = self._requested(eid)
        if action:
            self._event(eid, "paused" if action == "pause" else "cancelled", "host_request")
            raise _StopExperiment()
        started = f"{directory}/started.json"
        if self.store.path(started).exists() and not recoverable:
            raise RecoveryRequired(f"uncertain stage {cycle}/{name}; inspect before retry")
        if not self.store.path(started).exists():
            self.store.put(
                started, {"stage_id": f"{eid}.{cycle}.{name}", "recorded_at": now().isoformat()}
            )
        result = operation()
        self.store.put(uri, {"result": result, "resources": self._resources(eid)})
        if self.on_checkpoint:
            self.on_checkpoint(f"{cycle}/{name}")
        return result

    def _plans(
        self, plan: ExperimentPlan, cycle: int, label: str, suite: TaskSuite, gids: list[str]
    ) -> list[RunPlan]:
        uri = f"{self.directory(plan.experiment_id)}/cycles/{cycle:04d}/{label}_plans.json"
        if self.store.path(uri).exists():
            return [RunPlan.model_validate(p) for p in self.store.read(uri)["plans"]]
        plans = compile_suite(
            self.store, suite, gids, plan.evaluation_suite, plan.run_budget, plan.random_seeds
        )
        self.store.put(uri, {"plans": [p.model_dump(mode="json") for p in plans]})
        return plans

    def _execute(self, plans: list[RunPlan]) -> dict[str, Any]:
        runs = [r for p in plans for r in self.services.runtime.batch(p, resume=True)]
        for gid in dict.fromkeys(r.genome_id for r in runs):
            ids = [r.run_id for r in runs if r.genome_id == gid]
            if self.services.candidates.get(gid).run_ids != ids:
                self.services.candidates.record_execution(gid, ids)
        return {"run_ids": [r.run_id for r in runs]}

    def _analyze(self, run_ids: list[str], window: str) -> dict[str, Any]:
        runs = archived_runs(self.store, run_ids)
        summaries = []
        for gid in dict.fromkeys(r.genome_id for r in runs):
            items = [r for r in runs if r.genome_id == gid]
            for run in items:
                if not self.store.path(f"runs/{run.run_id}/reflection.json").exists():
                    self.services.experience.reflect_run(run)
            genome = self.store.genome(gid)
            existing = [
                self.store.load(p.relative_to(self.store.root).as_posix(), TaskExperienceSummary)
                for p in self.store.path(f"task_agents/{genome.agent_id}/{gid}/summaries").glob(
                    "*.json"
                )
            ]
            summary = next((s for s in existing if s.window_id == window), None)
            if summary is None:
                summary = self.services.experience.task_summary(items, window)
            summaries.append(summary.summary_id)
        return {"summary_ids": summaries}

    def _summary(self, sid: str) -> TaskExperienceSummary:
        return TaskExperienceSummary.model_validate(
            self.store.resolve(EvidenceRef(ref_type="summary", ref_id=sid))
        )

    def run(self, plan: ExperimentPlan, *, resume: bool = False) -> dict[str, Any]:
        """Run bounded evolution cycles or resume a previously created experiment.

        Args:
            plan: Immutable suites, initial genomes, policy, and hard resource limits.
            resume: Reuse durable stages; ambiguous side effects pause for host recovery.

        Returns:
            The archived lifecycle report with evidence and resource references.

        Raises:
            HEASError: The requested plan conflicts with the archived host contract.
        """
        self.create(plan)
        eid, services = plan.experiment_id, self.services
        prior = self.status(eid)
        if prior["status"] in {"completed", "cancelled"}:
            return self.report(eid)
        if prior["status"] != "created" and not resume:
            raise RecoveryRequired("existing experiment requires explicit resume")
        meter = CallBudget(plan.total_budget)
        meter.scope_id = f"experiment:{eid}"
        resources = self._resources(eid)
        meter.model_calls, meter.tool_calls = resources["model_calls"], resources["tool_calls"]
        meter.input_tokens = resources["known_input_tokens"]
        meter.output_tokens = resources["known_output_tokens"]
        meter.usage_available = (
            resources["unknown_usage_calls"] == 0 and not resources["external_usage_unknown"]
        )
        if plan.total_budget.pricing:
            price = plan.total_budget.pricing
            meter.cost = (
                meter.input_tokens * price.input_per_million
                + meter.output_tokens * price.output_per_million
            ) / 1_000_000
        sessions = list(self.store.path(f"{self.directory(eid)}/sessions").glob("*.json"))
        active_seconds = sum(
            self.store.read(p.relative_to(self.store.root).as_posix())["seconds"] for p in sessions
        )
        # A crashed running session has uncertain elapsed host time: conservatively charge
        # from its durable start to now, including downtime, before allowing another call.
        events = [
            self.store.read(p.relative_to(self.store.root).as_posix())
            for p in sorted(self.store.path(f"{self.directory(eid)}/events").glob("*.json"))
        ]
        if prior["status"] == "running":
            from datetime import datetime

            active_seconds += (
                now() - datetime.fromisoformat(events[-1]["recorded_at"])
            ).total_seconds()
        meter.started -= active_seconds
        saved = services.experience.budget, services.evolution.budget, services.meta.budget
        services.experience.budget, services.evolution.budget, services.meta.budget = (
            plan.analysis_budget,
            plan.evolution_budget,
            plan.meta_budget,
        )
        services.gateway.budget_scopes.append(meter)
        started = time.monotonic()
        self._event(eid, "running", "resume" if resume else "start")
        try:
            if plan.evaluation_suite:
                spec = plan.evaluation_suite
                existing = services.adapters.evaluators.get(spec.suite_id)
                if existing and getattr(existing, "spec", None) != spec:
                    raise HEASError("registered evaluation suite contract differs")
                if not existing:
                    services.adapters.register_evaluator(EvaluationSuite(spec, services.adapters))
            gids = list(plan.initial_genome_ids)
            exploratory: str | None = None
            for cycle in range(1, plan.max_cycles + 1):
                done_uri = f"{self.directory(eid)}/cycles/{cycle:04d}/completed.json"
                if self.store.path(done_uri).exists():
                    previous = self.store.read(done_uri)
                    gids, exploratory = (
                        previous["next_genome_ids"],
                        previous["exploratory_policy_id"],
                    )
                    continue
                gids, exploratory = self._run_cycle(plan, cycle, gids, exploratory, meter)
            self._event(eid, "completed", "max_cycles_reached")
        except _StopExperiment:
            pass
        except RecoveryRequired:
            self._event(eid, "paused", "recovery_required")
        except BudgetExceeded:
            self._event(eid, "completed", "budget_exhausted")
        except (KeyboardInterrupt, SystemExit):
            self._event(eid, "paused", "interrupted")
        except Exception as exc:
            self._event(eid, "failed", type(exc).__name__)
        finally:
            services.gateway.budget_scopes.remove(meter)
            services.experience.budget, services.evolution.budget, services.meta.budget = saved
            self.store.put(
                f"{self.directory(eid)}/sessions/{len(sessions) + 1:06d}.json",
                {"seconds": time.monotonic() - started},
            )
        if self.status(eid)["status"] == "completed":
            manifest_ref = export_manifest(self.store)
            self.store.put(
                f"{self.directory(eid)}/reproducibility.json", {"manifest_ref": manifest_ref}
            )
        result = self.report(eid)
        reports = self.store.path(f"{self.directory(eid)}/reports")
        uri = f"{self.directory(eid)}/reports/{len(list(reports.glob('*.json'))) + 1:06d}.json"
        result["report_ref"] = uri
        self.store.put(uri, result)
        return result

    def _run_cycle(
        self,
        plan: ExperimentPlan,
        cycle: int,
        gids: list[str],
        exploratory: str | None,
        meter: CallBudget,
    ) -> tuple[list[str], str | None]:
        """Run durable L1 evidence, L2 offspring and optional L3 policy stages.

        Stage names, stable attempt IDs and commit order are the recovery protocol.
        Finalize population state only after validation and any independent policy
        trial have completed; a generated child is never an improvement claim.
        """
        eid = plan.experiment_id
        done_uri = f"{self.directory(eid)}/cycles/{cycle:04d}/completed.json"
        meter.check()
        if meter.model_calls >= meter.budget.max_model_calls:
            raise BudgetExceeded("experiment call budget exhausted")
        active = EvolutionPolicyManager(self.store).active()
        policy_id = exploratory or active
        if not policy_id:
            raise RecoveryRequired("experiment has no selected policy; inspect activations")
        selection = self._stage(
            eid,
            cycle,
            "policy",
            partial(
                dict,
                policy_id=policy_id,
                active_policy_id=active,
                usage="exploratory" if exploratory else "approved",
            ),
        )
        selected_policy_id = selection.get("policy_id")
        if not isinstance(selected_policy_id, str) or not selected_policy_id:
            raise RecoveryRequired("policy checkpoint has no valid policy identity")
        policy = load_policy(self.store, selected_policy_id)
        templates = self._plans(plan, cycle, "training", plan.training_suite, gids)
        batch = self._stage(
            eid,
            cycle,
            "tasks",
            partial(self._execute, templates),
            recoverable=True,
        )
        analysis = self._stage(
            eid,
            cycle,
            "experience",
            partial(self._analyze, batch["run_ids"], f"{eid}.{cycle}"),
            recoverable=True,
        )
        pool = [self._summary(sid) for sid in analysis["summary_ids"]]
        attempts, children = self._evolve_candidates(plan, cycle, templates, pool, policy)
        self._validate_candidates(plan, cycle, attempts, children)
        exploratory = self._meta_cycle(plan, cycle, attempts, policy)
        gids = list(dict.fromkeys([*gids, *children]))[-plan.population_limit :]
        self.store.put(
            done_uri,
            {
                "cycle_id": f"{eid}.{cycle}",
                "next_genome_ids": gids,
                "exploratory_policy_id": exploratory,
                "attempt_ids": attempts,
                "policy_id": policy.policy_id,
            },
        )
        if self.on_checkpoint:
            self.on_checkpoint(f"{cycle}/completed")
        return gids, exploratory

    def _evolve_candidates(
        self,
        plan: ExperimentPlan,
        cycle: int,
        templates: list[RunPlan],
        pool: list[TaskExperienceSummary],
        policy: EvolutionPolicy,
    ) -> tuple[list[str], list[str]]:
        """Create journaled attempts, extending the evidence pool with committed children."""
        eid, services = plan.experiment_id, self.services
        attempts, children = [], []
        for number in range(plan.attempts_per_cycle):
            aid = "attempt_" + digest(f"{eid}|{cycle}|{number}".encode())[7:39]

            def evolve(
                experience_pool: list[TaskExperienceSummary] = pool,
                evolution_policy: EvolutionPolicy = policy,
                execution_templates: list[RunPlan] = templates,
                attempt_id: str = aid,
            ) -> dict[str, Any]:
                outcome = services.engine.step(
                    experience_pool,
                    evolution_policy,
                    execution_templates,
                    compare=plan.validation_suite is None,
                    attempt_id=attempt_id,
                    resume=True,
                )
                return {
                    "attempt_id": attempt_id,
                    "children": [c["genome_id"] for c in outcome["offspring"]],
                    "summary_ids": [
                        c["summary_id"] for c in outcome["offspring"] if c["summary_id"]
                    ],
                }

            result = self._stage(eid, cycle, f"evolve_{number:03d}", evolve, recoverable=True)
            attempts.append(result["attempt_id"])
            children.extend(result["children"])
            pool.extend(self._summary(sid) for sid in result["summary_ids"])
        return attempts, children

    def _validate_candidates(
        self,
        plan: ExperimentPlan,
        cycle: int,
        attempts: list[str],
        children: list[str],
    ) -> None:
        """Run independent candidate checks before recording comparison feedback."""
        eid, services = plan.experiment_id, self.services
        validation_suite = plan.validation_suite
        if validation_suite and children:
            reference = services.candidates.incumbent()
            validation_genomes = children.copy()
            if reference:
                validation_genomes.insert(0, reference)
            validate_gids = list(dict.fromkeys(validation_genomes))
            validation = self._plans(plan, cycle, "validation", validation_suite, validate_gids)
            self._stage(
                eid,
                cycle,
                "validation",
                partial(self._execute, validation),
                recoverable=True,
            )

        def compare(offspring_ids: list[str] = children) -> dict[str, Any]:
            """Compare a candidate against its reference using independently matched evidence."""
            comparisons = []
            for gid in offspring_ids:
                current = services.candidates.get(gid)
                if current.run_ids and services.candidates.incumbent() != gid:
                    current = services.candidates.compare(gid)
                if plan.auto_promote_candidates and current.promotion_status == "eligible":
                    current = services.candidates.promote(gid)
                comparisons.append(current.model_dump(mode="json"))
            return {"candidates": comparisons}

        self._stage(eid, cycle, "comparison", compare, recoverable=True)
        if plan.validation_suite:

            def feedback(attempt_ids: list[str] = attempts) -> dict[str, Any]:
                for aid in attempt_ids:
                    attempt = self.store.load(
                        f"evolution/attempts/{aid}/attempt.json", EvolutionAttemptRecord
                    )
                    ids = [
                        rid
                        for gid in attempt.offspring_genome_ids
                        for rid in services.candidates.get(gid).run_ids
                    ]
                    uri = f"evolution/attempts/{aid}/validation_outcome.json"
                    if not self.store.path(uri).exists():
                        self.store.put(
                            uri,
                            {
                                "run_ids": ids,
                                "evaluation_status": "completed"
                                if ids
                                and all(
                                    r.evaluation.evaluation_status == "completed"
                                    for r in archived_runs(self.store, ids)
                                )
                                else "not_available",
                            },
                        )
                        services.experience.reflect_attempt(attempt)
                return {"attempt_ids": attempt_ids}

            self._stage(eid, cycle, "validation_feedback", feedback, recoverable=True)

    def _meta_cycle(
        self,
        plan: ExperimentPlan,
        cycle: int,
        attempts: list[str],
        policy: EvolutionPolicy,
    ) -> str | None:
        """Summarize owned attempts and trial new policy guidance under fixed Host criteria."""
        eid, services = plan.experiment_id, self.services
        exploratory = None
        if plan.meta_every and cycle % plan.meta_every == 0:
            rows = [
                self.store.load(p.relative_to(self.store.root).as_posix(), EvolutionAttemptRecord)
                for p in self.store.path("evolution/attempts").glob("*/attempt.json")
            ]
            owned = self._attempt_ids(eid, cycle, attempts)
            rows = [
                a
                for a in rows
                if a.evolution_attempt_id in owned and a.policy_id == policy.policy_id
            ]
            if len(rows) >= 2:
                es = self._stage(
                    eid,
                    cycle,
                    "evolution_experience",
                    partial(self._evolution_summary, rows),
                )

                def propose(
                    experience_checkpoint: dict[str, Any] = es,
                    source_policy: EvolutionPolicy = policy,
                ) -> dict[str, Any]:
                    """Propose a child policy that still requires independent evaluation."""
                    summary = EvolutionExperienceSummary.model_validate(
                        self.store.resolve(
                            EvidenceRef(
                                ref_type="summary", ref_id=experience_checkpoint["summary_id"]
                            )
                        )
                    )
                    candidate, policy_proposal = services.meta.propose(
                        summary, source_policy, {"requires_matched_independent_trial": True}
                    )
                    return {
                        "policy_id": candidate.policy_id,
                        "proposal_id": policy_proposal.proposal_id,
                    }

                proposal = self._stage(eid, cycle, "meta", propose)

                def trial(
                    cycle_number: int = cycle,
                    reference_policy: EvolutionPolicy = policy,
                    proposal_checkpoint: dict[str, Any] = proposal,
                ) -> dict[str, Any]:
                    trial_spec = plan.policy_trial
                    if trial_spec is None:
                        return {
                            "status": "pending",
                            "decision": "undecided",
                            "reason": "host_trial_spec_not_configured",
                        }
                    tid = "trial_" + digest(f"{eid}|{cycle_number}".encode())[7:39]
                    record = PolicyTrialService(services).execute(
                        reference_policy.policy_id,
                        proposal_checkpoint["policy_id"],
                        trial_spec,
                        trial_id=tid,
                        resume=True,
                    )
                    if record.decision == "adopt":
                        manager = EvolutionPolicyManager(self.store)
                        if manager.active() == record.old_policy_id:
                            manager.adopt(record.trial_id)
                    return {
                        "trial_id": tid,
                        "status": record.comparison_status,
                        "decision": record.decision,
                    }

                trial_result = self._stage(eid, cycle, "policy_trial", trial, recoverable=True)
                if plan.allow_exploratory_policy and trial_result["decision"] != "adopt":
                    exploratory = proposal["policy_id"]
        return exploratory

    def _attempt_ids(self, eid: str, cycle: int, current: list[str]) -> set[str]:
        result = set(current)
        for n in range(1, cycle):
            uri = f"{self.directory(eid)}/cycles/{n:04d}/completed.json"
            if self.store.path(uri).exists():
                result.update(self.store.read(uri)["attempt_ids"])
        return result

    def resume(self, eid: str) -> dict[str, Any]:
        """Reload the immutable plan and continue without replaying committed work."""
        plan = self.store.load(f"{self.directory(eid)}/plan.json", ExperimentPlan)
        return self.run(plan, resume=True)

    def _evolution_summary(self, rows: list[EvolutionAttemptRecord]) -> dict[str, Any]:
        return {"summary_id": self.services.experience.evolution_summary(rows).summary_id}


class _StopExperiment(Exception):
    pass
