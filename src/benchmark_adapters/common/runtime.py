"""Port implementations shared by benchmark-specific loaders and official workers."""

import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from benchmark_adapters.common.config import IntegrationConfig, require_external_budget_support
from benchmark_adapters.common.protocol import OfficialReceipt, RunnerBridge
from heas.core.domain import (
    AgentObservation,
    EvaluationResult,
    RunPlan,
    Score,
    TaskResult,
    TaskRunRecord,
    TaskSpec,
)
from heas.core.errors import HEASError, SecurityError
from heas.infrastructure.model import CallBudget, ModelGateway
from heas.infrastructure.storage import FileStore, digest, encode


class BenchmarkAction(BaseModel):
    """Expose a small host-reviewed action set without an arbitrary host shell."""

    model_config = ConfigDict(extra="forbid", strict=True)
    action: Literal["read_file", "write_file", "list_files", "mcp_call", "container_exec"]
    path: str = Field(default="", max_length=1024)
    content: str = Field(default="", max_length=100000)
    command: str = Field(default="", max_length=10000)
    tool_name: str = Field(default="", max_length=100)
    arguments: dict[str, Any] = Field(default_factory=dict)


class BenchmarkEnvironment:
    """Manage one public task session, deferring cleanup until independent verification."""

    def __init__(self, environment_id: str, bridge: RunnerBridge, config: IntegrationConfig):
        self.environment_id, self.bridge, self.config = environment_id, bridge, config
        self.active_session: str | None = None
        self.pending_session: str | None = None
        self.public_state: dict[str, Any] = {}
        self.sequence_context: dict[str, Any] = {}
        self.action_count = 0

    def prepare(self, task: TaskSpec, attempt: int, workspace: Path) -> None:
        """Prepare only the current run's environment and expose a public observation."""
        if self.pending_session:
            self.bridge.request(
                "abandon", {"session_id": self.pending_session}, timeout=self.config.timeout_seconds
            )
        session_id = workspace.parent.name
        self.active_session = self.pending_session = session_id
        self.action_count = 0
        self.public_state = self.bridge.request(
            "prepare",
            {
                "session_id": session_id,
                "official_task_id": task.metadata["official_task_id"],
                "attempt": attempt,
                "sequence": self.sequence_context,
            },
            timeout=self.config.timeout_seconds,
        )
        # Only this allowlist crosses the worker/agent boundary. In particular,
        # raw setup dictionaries can contain hidden tests and are never returned.
        if set(self.public_state) - {
            "official_prompt",
            "workspace",
            "allowed_actions",
            "phase",
            "task_input_sha256",
        }:
            raise SecurityError("worker observation contains non-public fields")

    @property
    def input_fingerprint(self) -> str | None:
        """Return the worker's public input digest for independent comparison checks."""
        return self.public_state.get("task_input_sha256")

    def observe(self) -> AgentObservation:
        """Return public environment state, never an official validator specification."""
        return AgentObservation(
            public_data={**self.public_state, "executed_actions": self.action_count}
        )

    def apply_tool(self, tool_id: str, result: Any) -> None:
        """Keep state in the official worker; tool receipts are separately traced by Core."""
        if not self.active_session or not isinstance(result, dict):
            raise SecurityError("benchmark action returned outside an active session")
        self.action_count += 1

    def teardown(self) -> None:
        """Revoke task actions while preserving verifier access to its completed workspace.

        Core tears down the agent before calling TaskEvaluator. Cleanup therefore
        occurs in the evaluator's finally block, or before the next preparation
        when an evaluator was absent or interrupted.
        """
        self.active_session = None

    def close(self) -> None:
        """Release branch-owned official processes after all run evidence is collected."""
        self.active_session = self.pending_session = None
        self.bridge.close()


class BenchmarkToolExecutor:
    """Send approved actions only to the environment's current execution session."""

    def __init__(self, environment: BenchmarkEnvironment):
        self.environment = environment
        self.expected_sha256 = digest(Path(__file__).read_bytes())

    def execute(self, tool_id: str, args: dict[str, Any], *, timeout: float) -> Any:
        """Execute inside the official workspace; the agent cannot choose a session ID."""
        action = BenchmarkAction.model_validate(args)
        session = self.environment.active_session
        if not session:
            raise SecurityError("benchmark tool has no active task session")
        return self.environment.bridge.request(
            "action", {"session_id": session, **action.model_dump()}, timeout=timeout
        )


class BenchmarkEvaluator:
    """Convert a narrow official worker receipt into an independent Core evaluation."""

    def __init__(
        self,
        evaluator_id: str,
        environment: BenchmarkEnvironment,
        config: IntegrationConfig,
        store: FileStore,
    ):
        self.evaluator_id, self.environment, self.config, self.store = (
            evaluator_id,
            environment,
            config,
            store,
        )
        contract = config.model_dump(
            mode="json",
            exclude={
                "repository",
                "private_root",
                "worker_python",
                "isolation_prefix",
                "allow_execution",
                "partitions",
                "max_evaluations",
                "timeout_seconds",
            },
        )
        for key in ("official_config", "sequence_file"):
            path = getattr(config, key, None)
            if path and path.is_file():
                contract[key] = digest(path.read_bytes())
        self.version = config.revision + "." + digest(encode(contract))[7:23]
        self.gateway: ModelGateway | None = None

    def bind_execution_context(self, gateway: ModelGateway) -> None:
        """Bind the current branch's resource context without sharing another gateway."""
        self.gateway = gateway

    def evaluate_result(
        self, task: TaskSpec, run: TaskRunRecord, result: TaskResult, workspace: Path
    ) -> EvaluationResult:
        """Verify a completed session and always revoke it, including rejected budgets."""
        try:
            return self._evaluate(task, run, result, workspace)
        finally:
            if self.environment.pending_session:
                try:
                    self.environment.bridge.request(
                        "cleanup",
                        {"session_id": self.environment.pending_session},
                        timeout=self.config.timeout_seconds,
                    )
                finally:
                    self.environment.pending_session = None

    def _evaluate(
        self, task: TaskSpec, run: TaskRunRecord, result: TaskResult, workspace: Path
    ) -> EvaluationResult:
        """Verify identity before exposing finite scores; retain raw private results elsewhere."""
        root = "benchmark_invocations"
        plan = self.store.load(f"plans/{run.run_plan_id}.json", RunPlan)
        meter = CallBudget(plan.budget)
        meter.scope_id = f"run:{run.run_id}"
        meter.started -= run.usage.wall_seconds
        scopes = self.gateway.meters(meter) if self.gateway else [meter]
        for scope in scopes:
            scope.check()
        timeout = min(self.config.timeout_seconds, *(scope.remaining_time for scope in scopes))
        require_external_budget_support([scope.budget for scope in scopes])
        if len(list(self.store.path(root).glob("*.json"))) >= self.config.max_evaluations:
            raise HEASError("official evaluation invocation budget exhausted")
        invocation = {
            "benchmark": self.config.benchmark,
            "revision": self.version,
            "run_id": run.run_id,
            "genome_id": run.genome_id,
            "usage_status": "unknown_external_usage",
            "task_partition": run.task_partition,
            "budget_scope_ids": self.gateway.scope_ids(meter) if self.gateway else [],
        }
        for target in [self.store, *(self.gateway.audit_mirrors if self.gateway else [])]:
            target.put(f"{root}/{run.run_id}.json", invocation)
        started = time.monotonic()
        try:
            raw = self.environment.bridge.request(
                "evaluate",
                {
                    "session_id": run.run_id,
                    "run_id": run.run_id,
                    "genome_id": run.genome_id,
                    "official_task_id": task.metadata["official_task_id"],
                    "task_partition": run.task_partition,
                    "answer": result.text or "",
                    "termination_reason": run.termination_reason,
                    "genome": self.store.genome(run.genome_id).model_dump(mode="json"),
                    "capabilities": [
                        {
                            "artifact_id": skill.artifact_id,
                            "content": self.store.verify(skill).decode("utf-8"),
                            "content_sha256": skill.content_sha256,
                        }
                        for skill in self.store.genome(run.genome_id).skills
                    ],
                },
                timeout=timeout,
            )
            receipt = OfficialReceipt.model_validate(raw)
            if not receipt.private_feedback_withheld or (
                self.config.benchmark == "seabench"
                and run.task_partition != "final_test"
                and set(receipt.metrics) - {"completion_pass"}
            ):
                raise SecurityError("private safety feedback cannot enter upstream learning")
            if (
                receipt.run_id,
                receipt.genome_id,
                receipt.official_task_id,
                receipt.revision,
                receipt.benchmark,
            ) != (
                run.run_id,
                run.genome_id,
                task.metadata["official_task_id"],
                self.config.revision,
                self.config.benchmark,
            ):
                raise SecurityError("official evaluation receipt identity mismatch")
            self.store.put(f"runs/{run.run_id}/benchmark_receipt.json", receipt)
            scores = (
                [
                    Score(
                        name=name,
                        value=value,
                        direction="higher_is_better",
                        unit=(
                            "ratio"
                            if name in {"pass_rate", "completion_pass", "safety_pass"}
                            else "official_reward"
                            if name == "reward"
                            else "normalized_percent"
                        ),
                        source=self.evaluator_id,
                    )
                    for name, value in receipt.metrics.items()
                ]
                if receipt.status == "completed"
                else []
            )
            primary = next(
                (s for s in scores if s.name in {"reward", "completion_pass", "pass_rate"}),
                scores[0] if scores else None,
            )
            return EvaluationResult(
                evaluator_id=self.evaluator_id,
                evaluator_version=self.version,
                run_id=run.run_id,
                evaluation_status=receipt.status,
                score=primary,
                scores=scores,
                feedback="Official receipt recorded; private verifier details are withheld.",
                extensions={
                    "heas": {
                        "benchmark": receipt.benchmark,
                        "official_task_id": receipt.official_task_id,
                        "origin": self.config.data_origin,
                        "measurement_target": receipt.measurement_target,
                        "success": receipt.success,
                        "receipt_sha256": digest(encode(receipt)),
                        "external_usage_unknown": True,
                    }
                },
            )
        finally:
            completion = {**invocation, "wall_seconds": time.monotonic() - started}
            for target in [self.store, *(self.gateway.audit_mirrors if self.gateway else [])]:
                target.put(f"benchmark_completions/{run.run_id}.json", completion)
            self.environment.bridge.request(
                "cleanup", {"session_id": run.run_id}, timeout=self.config.timeout_seconds
            )
            self.environment.pending_session = None
