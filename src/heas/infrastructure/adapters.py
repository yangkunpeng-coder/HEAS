"""Host plugin registration and a task-neutral file catalog adapter."""

import json
from pathlib import Path

from heas.core.domain import TaskSpec
from heas.core.errors import HEASError
from heas.core.ports import StructuredTaskEvaluator, TaskEnvironment, TaskEvaluator, TaskProvider


def _check_adapter(adapter: object, identity_field: str, methods: tuple[str, ...]) -> str:
    """Validate dynamic plugin objects before publishing them through typed ports."""
    identifier: object = getattr(adapter, identity_field, None)
    if not isinstance(identifier, str) or not identifier:
        raise HEASError(f"adapter requires a nonempty {identity_field}")
    if any(not callable(getattr(adapter, method, None)) for method in methods):
        raise HEASError(f"adapter {identifier} requires callable methods: {', '.join(methods)}")
    return identifier


class AdapterRegistry:
    """Register host-supplied task ports without introducing benchmark logic into Core."""

    def __init__(self) -> None:
        self.providers: dict[str, TaskProvider] = {}
        self.environments: dict[str, TaskEnvironment] = {}
        self.evaluators: dict[str, TaskEvaluator | StructuredTaskEvaluator] = {}

    def register_provider(self, adapter: TaskProvider) -> None:
        """Register a task provider with a unique host-owned identifier."""
        identifier = _check_adapter(adapter, "provider_id", ("list_tasks", "get_task"))
        if identifier in self.providers:
            raise HEASError("duplicate provider")
        self.providers[identifier] = adapter

    def register_environment(self, adapter: TaskEnvironment) -> None:
        """Register an environment without replacing an existing binding."""
        identifier = _check_adapter(
            adapter, "environment_id", ("prepare", "observe", "apply_tool", "teardown")
        )
        if identifier in self.environments:
            raise HEASError("duplicate environment")
        self.environments[identifier] = adapter

    def register_evaluator(self, adapter: TaskEvaluator | StructuredTaskEvaluator) -> None:
        """Register an independent evaluator with a unique host-owned identifier."""
        identifier = _check_adapter(adapter, "evaluator_id", ())
        # Structured output keeps priority when an adapter implements both ports.
        method = "evaluate_result" if hasattr(adapter, "evaluate_result") else "evaluate"
        _check_adapter(adapter, "evaluator_id", (method,))
        if identifier in self.evaluators:
            raise HEASError("duplicate evaluator")
        self.evaluators[identifier] = adapter


class FileTaskProvider:
    """Generic host-side task catalog with explicit partition access lists."""

    def __init__(self, provider_id: str, source: Path):
        self.provider_id = provider_id
        self.tasks: dict[str, TaskSpec] = {}
        self.partitions: dict[str, set[str]] = {}
        for entry in json.loads(source.read_text(encoding="utf-8"))["tasks"]:
            task = TaskSpec.model_validate(entry["task"])
            if task.provider_id != provider_id or task.task_id in self.tasks:
                raise HEASError("task catalog provider/identity conflict")
            self.tasks[task.task_id] = task
            for partition in entry["partitions"]:
                self.partitions.setdefault(partition, set()).add(task.task_id)

    def list_tasks(self, partition: str) -> list[TaskSpec]:
        """List only task instances authorized for the requested partition."""
        return [self.tasks[task_id] for task_id in sorted(self.partitions.get(partition, set()))]

    def get_task(self, task_id: str) -> TaskSpec:
        """Return an explicitly catalogued task or fail on an unknown identity."""
        if task_id not in self.tasks:
            raise HEASError("task not present in provider catalog")
        return self.tasks[task_id]
