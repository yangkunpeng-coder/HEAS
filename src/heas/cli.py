"""Production workspace management CLI."""

import argparse
import json
from pathlib import Path
from typing import Any

from heas.agents.evolution.candidates import CandidateRegistry, archived_runs
from heas.agents.evolution.evolution import GenomeMaterializer
from heas.agents.meta.meta import load_policy, save_policy
from heas.agents.meta.policy_trials import (
    EvolutionPolicyManager,
    PolicyTrialService,
    PolicyTrialSpec,
)
from heas.agents.task.tools import ToolRegistry
from heas.core.domain import (
    EvidenceRef,
    EvolutionAttemptRecord,
    EvolutionExperienceSummary,
    EvolutionPolicy,
    Record,
    RunPlan,
    TaskExperienceSummary,
    uid,
)
from heas.experimentation.coordinator import ApplicationServices
from heas.experimentation.experiment_controller import (
    EvolutionExperimentController,
    ExperimentControl,
)
from heas.experimentation.experiment_plan import ExperimentPlan
from heas.experimentation.experiments import AttributionExperimentRunner, AttributionSpec
from heas.experimentation.suites import EvaluationSuiteSpec, TaskSuite, run_suite
from heas.infrastructure.application import build_services, host_components
from heas.infrastructure.config import ApplicationConfig
from heas.infrastructure.reproducibility import export_manifest
from heas.infrastructure.resources import ResourceLedger
from heas.infrastructure.storage import FileStore


def main() -> None:
    """Parse a trusted Host request, dispatch it, and print a sanitized result.

    CLI argument names and service injection globals remain stable for Host
    plugins. Provider exception messages never enter terminal diagnostics.
    """
    parser = _build_parser()
    args = parser.parse_args()
    config = (
        ApplicationConfig.model_validate(json.loads(args.config.read_text(encoding="utf-8")))
        if args.config
        else ApplicationConfig(workspace="workspace")
    )
    experiment_plan = (
        ExperimentPlan.model_validate_json(args.experiment_config.read_text(encoding="utf-8"))
        if args.command == "experiment" and args.experiment_action == "run"
        else None
    )
    store = FileStore(
        args.workspace
        or (experiment_plan.output_directory if experiment_plan else config.workspace)
    )
    try:
        result = _dispatch(args, config, store, experiment_plan)
        if isinstance(result, Record):
            result = result.model_dump(mode="json")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        # Provider exceptions may contain private task data or credentials.
        parser.exit(1, f"HEAS {args.command} failed: {type(exc).__name__}\n")


def _build_parser() -> argparse.ArgumentParser:
    """Declare the existing CLI commands and flags without initializing runtime services."""
    parser = argparse.ArgumentParser(prog="heas")
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--config", type=Path, help="Trusted host and model configuration")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Create a new immutable Task Agent and baseline policy")
    init.add_argument("--skill", type=Path, nargs="+", required=True)
    init.add_argument("--memory", type=Path, required=True)
    init.add_argument("--tool", action="append", default=[], help="Host-approved tool_id@version")
    run = sub.add_parser("run", help="Execute a configured task batch")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--genome-id", nargs="+")
    run.add_argument("--no-reflection", action="store_true")
    run.add_argument("--resume", action="store_true", help="Reuse the archived plan ID and journal")
    run.add_argument("--max-tasks", type=int)
    suite = sub.add_parser("run-suite", help="Compile and run cross-provider task suite plans")
    suite.add_argument("--suite", type=Path, required=True)
    suite.add_argument("--evaluation-suite", type=Path)
    suite.add_argument("--genome-id", nargs="+", required=True)
    suite.add_argument("--seed", nargs="+", type=int, default=[23])
    summarize = sub.add_parser("summarize", help="Summarize multiple runs of one Genome")
    summarize.add_argument("--run-ids", nargs="+", required=True)
    summarize.add_argument("--previous-summary-id")
    evolve = sub.add_parser("evolve", help="Propose, materialize, run and archive offspring")
    evolve.add_argument("--summary-ids", nargs="+", required=True)
    evolve.add_argument("--policy-id", required=True)
    evolve.add_argument("--plan", type=Path)
    explore = sub.add_parser("explore", help="Select evidence-covered parents from the archive")
    explore.add_argument("--policy-id", required=True)
    explore.add_argument("--plan", type=Path)
    explore.add_argument("--archive-limit", type=int, default=16)
    ablate = sub.add_parser("ablate", help="Execute matched, host-defined capability variants")
    ablate.add_argument("--spec", type=Path, required=True)
    es = sub.add_parser("summarize-evolution", help="Analyze multiple EvolutionAttempts")
    es.add_argument("--attempt-ids", nargs="+", required=True)
    meta = sub.add_parser("meta-evolve", help="Propose an unevaluated EvolutionPolicy version")
    meta.add_argument("--summary-id", required=True)
    meta.add_argument("--policy-id", required=True)
    trial = sub.add_parser("policy-trial", help="Execute matched old/new policy trials")
    trial.add_argument("--old-policy-id", required=True)
    trial.add_argument("--new-policy-id", required=True)
    trial.add_argument("--spec", type=Path, required=True)
    adopt = sub.add_parser("policy-adopt", help="Adopt only from independent completed trial")
    adopt.add_argument("--trial-id", required=True)
    rollback = sub.add_parser("policy-rollback", help="Return to a previously approved policy")
    rollback.add_argument("--policy-id", required=True)
    for name in ("compare", "promote"):
        cmd = sub.add_parser(name, help=f"{name.capitalize()} an evaluated candidate")
        cmd.add_argument("--genome-id", required=True)
    show = sub.add_parser("show", help="Show Agent, Genome, Candidate or Policy")
    show.add_argument("kind", choices=["agent", "genome", "candidate", "policy"])
    show.add_argument("id")
    sub.add_parser("resources", help="Report whole-workspace model resource use")
    sub.add_parser("population", help="List retained candidates and capability diversity")
    export = sub.add_parser("export-manifest", help="Export hashes and reproducibility metadata")
    export.add_argument("--code-root", type=Path)
    inspect = sub.add_parser("inspect", help="Read an archived workspace record")
    inspect.add_argument("uri")
    experiment = sub.add_parser("experiment", help="Run and resume bounded L1/L2/L3 experiments")
    operations = experiment.add_subparsers(dest="experiment_action", required=True)
    start = operations.add_parser("run")
    start.add_argument("--config", dest="experiment_config", type=Path, required=True)
    for action in ("status", "resume", "report", "pause", "cancel"):
        operation = operations.add_parser(action)
        operation.add_argument("--id", required=True)
    return parser


def _dispatch(
    args: argparse.Namespace,
    config: ApplicationConfig,
    store: FileStore,
    experiment_plan: ExperimentPlan | None,
) -> Any:
    """Route workspace queries or explicit mutations to the appropriate service boundary."""
    if args.command == "experiment":
        result = _dispatch_experiment(args, config, store, experiment_plan)
    elif args.command == "inspect":
        result = store.read(args.uri)
    elif args.command == "resources":
        result = ResourceLedger(store).report()
    elif args.command == "population":
        registry = CandidateRegistry(store)
        result = {
            "candidates": [c.model_dump(mode="json") for c in registry.list_candidates()],
            "diversity": registry.diversity(),
        }
    elif args.command == "export-manifest":
        result = {"manifest_ref": export_manifest(store, args.code_root)}
    elif args.command == "show":
        if args.kind == "policy":
            result = load_policy(store, args.id)
        elif args.kind == "candidate":
            result = CandidateRegistry(store).get(args.id)
        elif args.kind == "agent":
            paths = list(store.path(f"task_agents/{args.id}").glob("*/genome.json"))
            result = [store.read(p.relative_to(store.root).as_posix()) for p in paths]
        else:
            result = store.genome(args.id)
    elif args.command in {"compare", "promote"}:
        registry = CandidateRegistry(store)
        result = getattr(registry, args.command)(args.genome_id)
    elif args.command in {"policy-adopt", "policy-rollback"}:
        manager = EvolutionPolicyManager(store)
        if args.command == "policy-adopt":
            manager.adopt(args.trial_id)
        else:
            manager.rollback(args.policy_id)
        result = {"active_policy_id": manager.active()}
    else:
        tools, adapters = host_components(store, config)
        if args.command == "init":
            result = _initialize_agent(args, store, tools)
        else:
            services = build_services(store, config, tools, adapters)
            result = _dispatch_execution(args, store, services)
    return result


def _dispatch_experiment(
    args: argparse.Namespace,
    config: ApplicationConfig,
    store: FileStore,
    experiment_plan: ExperimentPlan | None,
) -> Any:
    """Read experiment state without model setup, or assemble services for run and resume."""
    controller = ExperimentControl(store)
    if args.experiment_action == "resume" and controller.status(args.id)["status"] in {
        "completed",
        "cancelled",
    }:
        result = controller.report(args.id)
    elif args.experiment_action in {"run", "resume"}:
        tools, adapters = host_components(store, config)
        services = build_services(store, config, tools, adapters)
        controller = EvolutionExperimentController(services)
        result = controller.run(experiment_plan) if experiment_plan else controller.resume(args.id)
    else:
        result = (
            getattr(controller, args.experiment_action)(args.id)
            if args.experiment_action in {"status", "report"}
            else controller.request(args.id, args.experiment_action)
        )
    return result


def _initialize_agent(args: argparse.Namespace, store: FileStore, tools: ToolRegistry) -> Any:
    """Create an immutable seed and baseline policy from Host-approved resources."""
    bindings = []
    for value in args.tool:
        tool_id, _, version = value.partition("@")
        uri = f"artifacts/tools/{tool_id}/{version}/manifest.json"
        from heas.core.domain import ToolBinding
        from heas.infrastructure.storage import digest

        manifest = tools.entries[(tool_id, version)][0]
        bindings.append(
            ToolBinding(
                tool_id=tool_id,
                version_id=version,
                manifest_ref=uri,
                permissions=manifest.permissions,
                extensions={"heas": {"manifest_sha256": digest(store.path(uri).read_bytes())}},
            )
        )
    genome = GenomeMaterializer(store, tools).create_seed(
        [store.skill(path.read_text(encoding="utf-8")) for path in args.skill],
        args.memory.read_text(encoding="utf-8"),
        bindings,
    )
    registry = CandidateRegistry(store)
    registry.register(genome)
    if registry.incumbent() is None:
        registry.initialize_incumbent(genome.genome_id)
    policy = EvolutionPolicy(policy_id=uid("policy"))
    save_policy(store, policy)
    manager = EvolutionPolicyManager(store)
    if manager.active() is None:
        manager.initialize(policy.policy_id)
    result = {
        "agent_id": genome.agent_id,
        "genome_id": genome.genome_id,
        "policy_id": policy.policy_id,
    }
    return result


def _dispatch_execution(
    args: argparse.Namespace, store: FileStore, services: ApplicationServices
) -> Any:
    """Execute configured task, evidence, capability and policy commands."""
    result: Any
    if args.command == "run-suite":
        task_suite = TaskSuite.model_validate_json(args.suite.read_text(encoding="utf-8"))
        evaluation_suite = (
            EvaluationSuiteSpec.model_validate_json(
                args.evaluation_suite.read_text(encoding="utf-8")
            )
            if args.evaluation_suite
            else None
        )
        suite_runs = run_suite(services, task_suite, args.genome_id, evaluation_suite, args.seed)
        result = {
            "suite_id": task_suite.suite_id,
            "run_ids": [r.run_id for r in suite_runs],
        }
    elif args.command == "run":
        plan_data = json.loads(args.plan.read_text(encoding="utf-8"))
        if not args.resume:
            plan_data["run_plan_id"] = uid("plan")
        if args.max_tasks is not None:
            plan_data["max_tasks"] = args.max_tasks
        if args.genome_id:
            plan_data["genome_ids"] = args.genome_id
        plan = RunPlan.model_validate(plan_data)
        runs = services.runtime.batch(plan, resume=args.resume)
        for run_record in runs:
            if (
                not args.no_reflection
                and plan.task_partition != "final_test"
                and not store.path(f"runs/{run_record.run_id}/reflection.json").exists()
            ):
                services.experience.reflect_run(run_record)
        for genome_id in plan.genome_ids if plan.task_partition != "final_test" else []:
            directory = store.path(f"candidates/{genome_id}/revisions")
            if not directory.exists():
                services.candidates.register(store.genome(genome_id))
            services.candidates.record_execution(
                genome_id, [r.run_id for r in runs if r.genome_id == genome_id]
            )
        result = {"run_ids": [r.run_id for r in runs], "run_plan_id": plan.run_plan_id}
    elif args.command == "summarize":
        previous = (
            TaskExperienceSummary.model_validate(
                store.resolve(EvidenceRef(ref_type="summary", ref_id=args.previous_summary_id))
            )
            if args.previous_summary_id
            else None
        )
        result = services.experience.task_summary(
            archived_runs(store, args.run_ids), uid("window"), previous_summary=previous
        )
    elif args.command in {"evolve", "explore"}:
        summaries = [
            TaskExperienceSummary.model_validate(
                store.resolve(EvidenceRef(ref_type="summary", ref_id=sid))
            )
            for sid in (args.summary_ids if args.command == "evolve" else [])
        ]
        execution_plan = (
            RunPlan.model_validate_json(args.plan.read_text(encoding="utf-8"))
            if args.plan
            else None
        )
        policy = load_policy(store, args.policy_id)
        if args.command == "explore":
            outcome = services.engine.explore(
                policy, execution_plan, archive_limit=args.archive_limit
            )
        else:
            outcome = services.engine.step(summaries, policy, execution_plan)
        result = {**outcome, "attempt": outcome["attempt"].model_dump(mode="json")}
    elif args.command == "ablate":
        attribution_spec = AttributionSpec.model_validate_json(
            args.spec.read_text(encoding="utf-8")
        )
        result = AttributionExperimentRunner(services).execute(attribution_spec)
    elif args.command == "summarize-evolution":
        attempts = [
            EvolutionAttemptRecord.model_validate(
                store.resolve(EvidenceRef(ref_type="evolution_attempt", ref_id=aid))
            )
            for aid in args.attempt_ids
        ]
        result = services.experience.evolution_summary(attempts)
    elif args.command == "meta-evolve":
        summary = EvolutionExperienceSummary.model_validate(
            store.resolve(EvidenceRef(ref_type="summary", ref_id=args.summary_id))
        )
        policy, proposal = services.meta.propose(
            summary,
            load_policy(store, args.policy_id),
            {"requires_matched_trial": True},
        )
        result = {"policy_id": policy.policy_id, "proposal_id": proposal.proposal_id}
    else:
        spec = PolicyTrialSpec.model_validate_json(args.spec.read_text(encoding="utf-8"))
        result = PolicyTrialService(services).execute(args.old_policy_id, args.new_policy_id, spec)
    return result
