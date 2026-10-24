"""Register benchmark ports through the existing trusted host plugin boundary."""

import os
from dataclasses import dataclass
from pathlib import Path

from benchmark_adapters.common.catalog import CatalogProvider
from benchmark_adapters.common.config import IntegrationConfig, load_config
from benchmark_adapters.common.process import ProcessBridge
from benchmark_adapters.common.protocol import RunnerBridge
from benchmark_adapters.common.runtime import (
    BenchmarkAction,
    BenchmarkEnvironment,
    BenchmarkEvaluator,
    BenchmarkToolExecutor,
)
from benchmark_adapters.evoagentbench.provider import EvoAgentBenchProvider
from benchmark_adapters.seabench.provider import SEABenchProvider
from benchmark_adapters.skilllearnbench.provider import SkillLearnBenchProvider
from heas.agents.task.tools import ToolRegistry
from heas.core.domain import ToolBinding
from heas.core.errors import HEASError
from heas.infrastructure.adapters import AdapterRegistry
from heas.infrastructure.storage import FileStore, digest


@dataclass
class BenchmarkBundle:
    """Group one independently owned provider, environment, evaluator, and approved tool."""

    config: IntegrationConfig
    provider: CatalogProvider
    environment: BenchmarkEnvironment
    evaluator: BenchmarkEvaluator
    tool_binding: ToolBinding


def provider_for(config: IntegrationConfig) -> CatalogProvider:
    """Load only public manifests; do not import official runtime dependencies."""
    if config.benchmark == "evoagentbench":
        return EvoAgentBenchProvider(config)
    if config.benchmark == "seabench":
        return SEABenchProvider(config)
    return SkillLearnBenchProvider(config)


def register_config(
    store: FileStore,
    tools: ToolRegistry,
    adapters: AdapterRegistry,
    config: IntegrationConfig,
    bridge: RunnerBridge | None = None,
) -> BenchmarkBundle:
    """Register real ports, optionally injecting a test transport from outside production code."""
    provider = provider_for(config)
    # A trial branch has a different FileStore root and therefore a different
    # worker directory even when both policies use the same host configuration.
    private = config.private_root / digest(str(store.root).encode())[7:31]
    client = bridge or ProcessBridge(config, private)
    environment = BenchmarkEnvironment(provider.provider_id + "_environment", client, config)
    evaluator = BenchmarkEvaluator(provider.provider_id + "_evaluator", environment, config, store)
    executor = BenchmarkToolExecutor(environment)
    sources = sorted(Path(__file__).parent.rglob("*.py"))
    implementation = digest(b"".join(path.read_bytes() for path in sources))
    binding = tools.register(
        provider.provider_id + "_action",
        "v1",
        "Perform host-approved actions in this benchmark's public task workspace",
        BenchmarkAction,
        executor,
        approval_id="host-benchmark-registration",
        implementation_sha256=implementation,
        selection_tags=(config.benchmark,),
    )
    adapters.register_provider(provider)
    adapters.register_environment(environment)
    adapters.register_evaluator(evaluator)
    return BenchmarkBundle(config, provider, environment, evaluator, binding)


def register(store: FileStore, tools: ToolRegistry, adapters: AdapterRegistry) -> None:
    """Host plugin entry point selected through ApplicationConfig.plugin_factory.

    HEAS_BENCH_CONFIG contains one configuration path or multiple paths separated
    by the platform's path separator. No benchmark-specific field enters Core.
    """
    value = os.getenv("HEAS_BENCH_CONFIG", "")
    if not value:
        raise HEASError("set HEAS_BENCH_CONFIG to the trusted benchmark configuration path")
    for path in value.split(os.pathsep):
        register_config(store, tools, adapters, load_config(Path(path)))
