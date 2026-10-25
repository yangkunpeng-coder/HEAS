"""Independent host services and immutable trial snapshots; no domain or test dependencies."""

import copy
from collections.abc import Callable as Callable
from typing import Any

from heas.agents.task.selectors import LLMSkillSelector
from heas.agents.task.tools import ToolRegistry
from heas.core.errors import HEASError, IntegrityError
from heas.experimentation.ports import ExperimentServices
from heas.experimentation.ports import ServiceFactory as ServiceFactory
from heas.infrastructure.storage import FileStore, digest


def independent(value: Any) -> Any:
    """Opaque live clients must expose fork() or use a host service factory."""
    if hasattr(value, "fork"):
        clone = value.fork()
    else:
        for cls in type(value).__mro__:
            for name, item in vars(cls).items():
                if not name.startswith("_") and isinstance(item, (list, dict, set)):
                    if name not in vars(value):
                        raise HEASError(
                            "mutable class state requires an explicit isolation factory"
                        )
        try:
            clone = copy.deepcopy(value)
        except Exception as exc:
            raise HEASError("host isolation factory required for opaque client/adapter") from exc
    if clone is value:
        raise HEASError("isolation factory returned shared mutable state")
    return clone


class ServiceBlueprint:
    """Freeze source state once, before either side is run, then clone per repetition."""

    def __init__(self, services: ExperimentServices):
        self.model = independent(services.gateway.model)
        self.models = {p: independent(m) for p, m in services.gateway.models_by_purpose.items()}
        self.adapters = independent(services.adapters)
        self.entries = {
            key: (manifest, arguments, independent(executor))
            for key, (manifest, arguments, executor) in services.tools.entries.items()
        }
        self.permissions = services.tools.allowed_permissions
        self.context_limits = services.runtime.context_limits
        self.aggregation = services.experience.aggregator.config
        self.analysis_budget = services.experience.budget
        self.evolution_budget = services.evolution.budget
        self.meta_budget = services.meta.budget
        self.repair_limit = services.gateway.structured_repair_limit
        self.context_limit = services.gateway.context_budget.max_tokens
        self.parent_selection = independent(services.evolution.parent_selection)
        self.tool_selector = independent(services.runtime.tool_selector)
        selector = services.runtime.skills.selector
        self.selector_inventory = (
            selector.max_inventory if isinstance(selector, LLMSkillSelector) else None
        )
        self.selector = None if self.selector_inventory else independent(selector)

    def create(self, store: FileStore) -> ExperimentServices:
        """Initialize the immutable host configuration or an independent execution branch."""
        from heas.experimentation.coordinator import ApplicationServices

        tools = ToolRegistry(store, self.permissions)
        tools.entries = {
            key: (manifest, arguments, independent(executor))
            for key, (manifest, arguments, executor) in self.entries.items()
        }
        services = ApplicationServices(
            store,
            independent(self.model),
            tools,
            independent(self.adapters),
            {p: independent(m) for p, m in self.models.items()},
            context_limits=self.context_limits,
            aggregation_config=self.aggregation,
            analysis_budget=self.analysis_budget,
        )
        services.evolution.budget, services.meta.budget = self.evolution_budget, self.meta_budget
        services.evolution.parent_selection = independent(self.parent_selection)
        services.runtime.tool_selector = independent(self.tool_selector)
        services.runtime.skills.selector = (
            LLMSkillSelector(services.gateway, self.selector_inventory)
            if self.selector_inventory
            else independent(self.selector)
        )
        services.gateway.context_budget.max_tokens = self.context_limit
        services.gateway.structured_repair_limit = self.repair_limit
        return services


def copy_files(source: FileStore, target: FileStore, directories: tuple[str, ...]) -> None:
    """Copy immutable evidence without overwriting conflicts or following symlinks."""
    for directory in directories:
        for path in source.path(directory).rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            uri = path.relative_to(source.root).as_posix()
            data = path.read_bytes()
            if target.path(uri).exists():
                if target.path(uri).read_bytes() != data:
                    raise IntegrityError("isolated result conflicts with immutable parent artifact")
            else:
                target.put_bytes(uri, data)


class IsolatedPolicyTrialRunner:
    """Create matched snapshots and collect evidence without importing activations."""

    def __init__(
        self, services: ExperimentServices, directory: str, factory: ServiceFactory | None = None
    ):
        self.source = services
        self.store = services.store
        self.directory = directory
        supplied = factory or getattr(services, "isolation_factory", None)
        self.factory = supplied or ServiceBlueprint(services).create
        self.snapshot = FileStore(self.store.path(f"{directory}/snapshot"))
        copy_files(
            self.store, self.snapshot, ("artifacts", "policies", "configs", "index/artifact")
        )
        for path in self.store.path("task_agents").glob("*/*/genome.json"):
            uri = path.relative_to(self.store.root).as_posix()
            genome = self.store.read(uri)
            for asset_uri in (uri, genome["memory"]["uri"]):
                if not self.snapshot.path(asset_uri).exists():
                    self.snapshot.put_bytes(asset_uri, self.store.path(asset_uri).read_bytes())
        self.snapshot_files = [
            {"uri": p.relative_to(self.snapshot.root).as_posix(), "sha256": digest(p.read_bytes())}
            for p in self.snapshot.root.rglob("*")
            if p.is_file()
        ]
        self.store.put(f"{directory}/snapshot_manifest.json", {"files": self.snapshot_files})
        self.instances: list[Any] = []

    def create(self, side: str, repetition: int, seed: int) -> ExperimentServices:
        """Initialize the immutable host configuration or an independent execution branch."""
        branch = FileStore(
            self.store.path(f"{self.directory}/branches/{side}/r{repetition}_s{seed}")
        )
        if any(branch.root.iterdir()):
            raise HEASError("branch directory already exists; do not replay an uncertain trial")
        for item in self.snapshot_files:
            data = self.snapshot.path(item["uri"]).read_bytes()
            if digest(data) != item["sha256"]:
                raise IntegrityError("trial initial snapshot changed")
            branch.put_bytes(item["uri"], data)
        child = self.factory(branch)
        if child.store.root != branch.root:
            raise HEASError("isolation factory returned the wrong workspace")
        mutable = [
            child.gateway.model,
            *child.gateway.models_by_purpose.values(),
            *child.adapters.providers.values(),
            *child.adapters.environments.values(),
            *child.adapters.evaluators.values(),
            *(entry[2] for entry in child.tools.entries.values()),
        ]
        original = [
            self.source.gateway.model,
            *self.source.gateway.models_by_purpose.values(),
            *self.source.adapters.providers.values(),
            *self.source.adapters.environments.values(),
            *self.source.adapters.evaluators.values(),
            *(entry[2] for entry in self.source.tools.entries.values()),
            *self.instances,
        ]
        if any(a is b for a in mutable for b in original):
            raise HEASError("trial branches share a mutable client/adapter")
        self.instances.extend(mutable)
        child.gateway.audit_mirrors.append(self.store)
        child.gateway.budget_scopes.extend(
            [
                *self.source.gateway.budget_scopes,
                *([self.source.gateway.global_meter] if self.source.gateway.global_meter else []),
            ]
        )
        child.experience.archive_scope = "policy_trial"
        return child

    def collect(self, services: ExperimentServices) -> None:
        # Candidate revisions / activations intentionally stay inside the branch.
        """Import immutable branch evidence while keeping candidate activations isolated."""
        copy_files(
            services.store,
            self.store,
            (
                "artifacts",
                "task_agents",
                "runs",
                "plans",
                "configs",
                "analysis",
                "contexts",
                "structured_calls",
                "evolution",
                "model_calls",
                "model_requests",
                "benchmark_invocations",
                "benchmark_completions",
                "tool_calls",
                "index",
                "batches",
            ),
        )
