"""Core depends only on these ports, never on task-specific adapters."""

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from heas.core.domain import AgentObservation, EvaluationResult, TaskResult, TaskRunRecord, TaskSpec


class ModelClient(Protocol):
    """Define a replaceable provider transport returning public data and usage receipts."""

    identity: str

    def complete(
        self,
        purpose: str,
        context: dict[str, Any],
        *,
        timeout: float,
        max_output_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Perform one provider transport request and return its response with usage evidence."""
        ...


@runtime_checkable
class TaskProvider(Protocol):
    """Supply public task instances with explicit partition access checks."""

    provider_id: str

    def list_tasks(self, partition: str) -> list[TaskSpec]:
        """List only task instances authorized for the requested partition."""
        ...

    def get_task(self, task_id: str) -> TaskSpec:
        """Return an explicitly catalogued task or fail on an unknown identity."""
        ...


@runtime_checkable
class TaskEnvironment(Protocol):
    """Own the lifecycle and public observations of one independent task execution."""

    environment_id: str

    def prepare(self, task: TaskSpec, attempt: int, workspace: Path) -> None:
        """Prepare one execution workspace without exposing private evaluator inputs."""
        ...

    def observe(self) -> AgentObservation:
        """Return public environment state while withholding private evaluation inputs."""
        ...

    def apply_tool(self, tool_id: str, result: Any) -> None:
        """Apply a completed approved tool result to the active task environment."""
        ...

    def teardown(self) -> None:
        """Revoke agent access and release the task environment's owned resources."""
        ...


@runtime_checkable
class TaskEvaluator(Protocol):
    """Evaluate a completed text result independently from the solving model."""

    evaluator_id: str

    def evaluate(
        self, task: TaskSpec, run: TaskRunRecord, output: str, workspace: Path
    ) -> EvaluationResult:
        """Return independent task feedback with explicit identity, scale, and availability."""
        ...


@runtime_checkable
class StructuredTaskEvaluator(Protocol):
    """Evaluate structured results and artifacts using an independent host protocol."""

    evaluator_id: str

    def evaluate_result(
        self, task: TaskSpec, run: TaskRunRecord, result: TaskResult, workspace: Path
    ) -> EvaluationResult:
        """Evaluate structured task output independently of the solving model's claims."""
        ...


@runtime_checkable
class ToolExecutor(Protocol):
    """Invoke a host-approved implementation within a bounded execution timeout."""

    def execute(self, tool_id: str, args: dict[str, Any], *, timeout: float) -> Any:
        """Invoke the Host-approved tool with validated arguments and a time limit."""
        ...


@runtime_checkable
class FingerprintedEnvironment(Protocol):
    """Optionally expose the public input digest without extending every environment."""

    @property
    def input_fingerprint(self) -> str | None:
        """Return a Host input digest, or None when the adapter has no digest."""
        ...
