"""Safe catalog/preflight commands and opt-in real HEAS benchmark execution."""

import argparse
import json
from pathlib import Path
from typing import Any

from benchmark_adapters.common.config import SEABenchConfig, load_config
from benchmark_adapters.common.preflight import preflight
from benchmark_adapters.common.runtime import BridgedEnvironment
from benchmark_adapters.integration import provider_for, register_config
from benchmark_adapters.seabench.sequence import SEABenchSequenceRunner
from benchmark_adapters.skilllearnbench.exporter import SkillExporter
from heas.agents.evolution.evolution import GenomeMaterializer
from heas.agents.meta.meta import save_policy
from heas.agents.meta.policy_metrics import PolicyMetrics
from heas.agents.task.tools import ToolRegistry
from heas.core.domain import EvolutionPolicy, RunPlan, uid
from heas.core.errors import HEASError
from heas.experimentation.coordinator import ApplicationServices
from heas.experimentation.experiment_controller import EvolutionExperimentController
from heas.experimentation.experiment_plan import ExperimentPlan
from heas.experimentation.metrics import EvolutionMetrics
from heas.experimentation.suites import EvaluationSuiteSpec, TaskSuite, TaskSuiteGroup
from heas.infrastructure.adapters import AdapterRegistry
from heas.infrastructure.application import build_services, host_components
from heas.infrastructure.config import ApplicationConfig
from heas.infrastructure.storage import FileStore


def main() -> None:
    """Parse a host request; help and inspection never start official processes."""
    parser = argparse.ArgumentParser(prog="heas-bench")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in (
        "preflight",
        "catalog",
        "init",
        "run",
        "plan",
        "experiment",
        "sequence",
        "policy-final",
    ):
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", type=Path, required=True, help="Benchmark JSON configuration")
        if name in {"catalog", "run"}:
            cmd.add_argument(
                "--partition",
                default="evolution_train",
                choices=[
                    "development",
                    "evolution_train",
                    "evolution_validation",
                    "meta_validation",
                    "final_test",
                ],
            )
        if name not in {"preflight", "catalog"}:
            cmd.add_argument("--workspace", type=Path, required=True)
        if name == "init":
            cmd.add_argument("--skill", type=Path, required=True)
            cmd.add_argument("--memory", type=Path, required=True)
        if name in {"run", "plan", "sequence"}:
            cmd.add_argument("--genome-id", required=True)
        if name in {"plan", "sequence"}:
            cmd.add_argument("--policy-id", required=True)
            cmd.add_argument("--id", required=True)
        if name in {"run", "experiment", "sequence", "policy-final"}:
            cmd.add_argument("--host-config", type=Path, required=True)
        if name == "run":
            cmd.add_argument("--task-id", nargs="+", help="Official IDs, not hashed HEAS IDs")
        if name == "plan":
            cmd.add_argument("--output", type=Path, required=True)
            cmd.add_argument("--cycles", type=int, default=2)
        if name == "experiment":
            cmd.add_argument("--plan", type=Path, required=True)
            cmd.add_argument("--resume", action="store_true")
        if name == "policy-final":
            cmd.add_argument("--selection-id", required=True)
            cmd.add_argument("--plan", type=Path, required=True)
    export = sub.add_parser("export-skills")
    export.add_argument("--workspace", type=Path, required=True)
    export.add_argument("--genome-id", required=True)
    export.add_argument("--task-id", nargs="+", required=True)
    export.add_argument("--output", type=Path, required=True)
    metrics = sub.add_parser("metrics")
    metrics.add_argument("--workspace", type=Path, required=True)
    metrics.add_argument(
        "--request",
        type=Path,
        required=True,
        help="JSON keyword arguments for EvolutionMetrics.report",
    )
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--workspace", type=Path, required=True)
    freeze.add_argument("--kind", choices=["genome", "policy"], default="genome")
    freeze.add_argument(
        "--request",
        type=Path,
        required=True,
        help="JSON keyword arguments for EvolutionMetrics.freeze",
    )
    args = parser.parse_args()
    try:
        result = dispatch(args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (HEASError, ValueError, OSError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")


def dispatch(args: argparse.Namespace) -> Any:
    """Route an implemented command through the existing production services."""
    if args.command == "export-skills":
        return SkillExporter().export(
            FileStore(args.workspace), args.genome_id, args.task_id, args.output
        )
    if args.command in {"metrics", "freeze"}:
        metric_service = (
            PolicyMetrics(FileStore(args.workspace))
            if args.command == "freeze" and args.kind == "policy"
            else EvolutionMetrics(FileStore(args.workspace))
        )
        operation = getattr(
            metric_service,
            "report" if args.command == "metrics" else "freeze",
        )
        result = operation(**json.loads(args.request.read_text(encoding="utf-8-sig")))
        return result.model_dump(mode="json") if hasattr(result, "model_dump") else result
    config = load_config(args.config)
    if args.command == "preflight":
        return preflight(config)
    provider = provider_for(config)
    if args.command == "catalog":
        return {
            "provider_id": provider.provider_id,
            "origin": config.data_origin,
            "tasks": [task.model_dump(mode="json") for task in provider.list_tasks(args.partition)],
        }
    store = FileStore(args.workspace)
    if args.command == "init":
        tools, adapters = ToolRegistry(store), AdapterRegistry()
        bundle = register_config(store, tools, adapters, config)
        genome = GenomeMaterializer(store, tools).create_seed(
            [store.skill(args.skill.read_text(encoding="utf-8-sig"))],
            args.memory.read_text(encoding="utf-8-sig"),
            [bundle.tool_binding],
        )
        policy = EvolutionPolicy(policy_id=uid("policy"))
        save_policy(store, policy)
        return {
            "agent_id": genome.agent_id,
            "genome_id": genome.genome_id,
            "policy_id": policy.policy_id,
            "tool_id": bundle.tool_binding.tool_id,
        }
    if args.command == "plan":

        def suite(partition: str) -> TaskSuite:
            return TaskSuite(
                suite_id=partition,
                version="v1",
                partition=partition,
                groups=[
                    TaskSuiteGroup(
                        provider_id=provider.provider_id,
                        environment_id=provider.provider_id + "_environment",
                        task_ids=[t.task_id for t in provider.list_tasks(partition)],
                    )
                ],
            )

        plan = ExperimentPlan(
            experiment_id=args.id,
            output_directory=str(store.root),
            initial_genome_ids=[args.genome_id],
            initial_policy_id=args.policy_id,
            max_cycles=args.cycles,
            training_suite=suite("evolution_train"),
            validation_suite=suite("evolution_validation")
            if provider.list_tasks("evolution_validation")
            else None,
            evaluation_suite=EvaluationSuiteSpec(
                suite_id="official_evaluation",
                version="v1",
                evaluator_ids=[provider.provider_id + "_evaluator"],
            ),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            plan.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        return {"plan": str(args.output.resolve()), "experiment_id": plan.experiment_id}
    if not config.allow_execution:
        raise HEASError("official execution disabled; review preflight and enable it explicitly")
    host = ApplicationConfig.model_validate_json(args.host_config.read_text(encoding="utf-8-sig"))

    def factory(target: FileStore) -> ApplicationServices:
        branch_tools, branch_adapters = host_components(target, host)
        if provider.provider_id not in branch_adapters.providers:
            register_config(target, branch_tools, branch_adapters, config)
        branch_services = build_services(target, host, branch_tools, branch_adapters)
        branch_services.isolation_factory = factory
        return branch_services

    services = factory(store)
    try:
        if args.command == "policy-final":
            final_plan = RunPlan.model_validate_json(args.plan.read_text(encoding="utf-8-sig"))
            return PolicyMetrics(store).final_report(services, args.selection_id, final_plan)
        if args.command == "experiment":
            plan = ExperimentPlan.model_validate_json(args.plan.read_text(encoding="utf-8-sig"))
            return EvolutionExperimentController(services).run(plan, resume=args.resume)
        if args.command == "sequence":
            if not isinstance(config, SEABenchConfig):
                raise HEASError("sequence execution requires a SEABench configuration")
            return SEABenchSequenceRunner(services, config).execute(
                args.id, args.genome_id, args.policy_id, factory=factory
            )
        tasks = provider.list_tasks(args.partition)
        if args.task_id:
            allowed = {provider.native_id(t.task_id): t for t in tasks}
            if not set(args.task_id).issubset(allowed):
                raise HEASError("requested official task is absent from the selected partition")
            tasks = [allowed[native] for native in args.task_id]
        if not tasks:
            raise HEASError("selected partition has no tasks")
        evaluator = services.adapters.evaluators[provider.provider_id + "_evaluator"]
        evaluator_version = getattr(evaluator, "version", None)
        if not isinstance(evaluator_version, str) or not evaluator_version:
            raise HEASError("official evaluator requires an explicit version")
        run_plan = RunPlan(
            run_plan_id=uid("plan"),
            genome_ids=[args.genome_id],
            task_provider_id=provider.provider_id,
            task_ids=[t.task_id for t in tasks],
            environment_adapter_id=provider.provider_id + "_environment",
            task_evaluator_id=evaluator.evaluator_id,
            evaluator_version=evaluator_version,
            task_partition=args.partition,
        )
        runs = services.runtime.batch(run_plan)
        if args.partition != "final_test":
            for run in runs:
                services.experience.reflect_run(run)
            if len(runs) >= 2:
                services.experience.task_summary(runs, run_plan.run_plan_id)
        return {
            "run_plan_id": run_plan.run_plan_id,
            "runs": [r.model_dump(mode="json") for r in runs],
        }
    finally:
        for environment in services.adapters.environments.values():
            if isinstance(environment, BridgedEnvironment):
                environment.bridge.close()
