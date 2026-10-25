"""Trusted host plugin loading and production ModelClient construction."""

import importlib
import inspect
import os
from typing import Protocol

from heas.agents.task.selectors import LLMSkillSelector
from heas.agents.task.tools import ToolRegistry
from heas.core.errors import HEASError
from heas.core.ports import ModelClient
from heas.experimentation.coordinator import ApplicationServices
from heas.infrastructure.adapters import AdapterRegistry
from heas.infrastructure.config import ApplicationConfig
from heas.infrastructure.model import DeepSeekClient
from heas.infrastructure.storage import FileStore


class HostPluginFactory(Protocol):
    """Register trusted Host components without constructing models or executing tasks."""

    def __call__(self, store: FileStore, tools: ToolRegistry, adapters: AdapterRegistry) -> None:
        """Register reviewed providers, environments, evaluators and approved executors."""
        ...


def _load_plugin_factory(reference: str) -> HostPluginFactory:
    """Contain unknown module attributes at the single trusted plugin boundary."""
    module_name, separator, factory_name = reference.partition(":")
    if not separator or not module_name or not factory_name:
        raise HEASError("plugin_factory must be a host-reviewed module:callable")
    module = importlib.import_module(module_name)
    factory: object = getattr(module, factory_name, None)
    if not callable(factory):
        raise HEASError("plugin_factory must identify a callable")
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        # Opaque native callables retain support; their call is still contained in
        # the wrapper and every published adapter crosses registry validation.
        signature = None

    def register(store: FileStore, tools: ToolRegistry, adapters: AdapterRegistry) -> None:
        if signature is not None:
            try:
                signature.bind(store, tools, adapters)
            except TypeError as exc:
                raise HEASError("plugin_factory must accept store, tools and adapters") from exc
        factory(store, tools, adapters)

    return register


def host_components(
    store: FileStore, config: ApplicationConfig
) -> tuple[ToolRegistry, AdapterRegistry]:
    """Load only the host-approved plugin and register independent task ports."""
    tools = ToolRegistry(store, frozenset(config.allowed_tool_permissions))
    adapters = AdapterRegistry()
    if config.plugin_factory:
        factory = _load_plugin_factory(config.plugin_factory)
        factory(store, tools, adapters)
    return tools, adapters


def build_services(
    store: FileStore, config: ApplicationConfig, tools: ToolRegistry, adapters: AdapterRegistry
) -> ApplicationServices:
    """Construct production role clients and a factory that rebuilds isolated branches."""
    profile = config.model_profile
    if profile and not os.getenv(profile.api_key_env):
        raise HEASError("configured model credential environment variable is missing")
    model = (
        DeepSeekClient(
            model=profile.model,
            base_url=profile.base_url,
            api_key=os.getenv(profile.api_key_env, ""),
            temperature=profile.temperature,
            tool_protocol=profile.tool_protocol,
            supports_seed=profile.supports_seed,
            request_parameters=profile.request_parameters,
        )
        if profile
        else DeepSeekClient(tool_protocol="native")
    )
    models: dict[str, ModelClient] = {}
    for purpose, purpose_profile in config.models_by_purpose.items():
        key = os.getenv(purpose_profile.api_key_env)
        if not key:
            raise HEASError("role model credential environment variable is missing")
        models[purpose] = DeepSeekClient(
            model=purpose_profile.model,
            base_url=purpose_profile.base_url,
            api_key=key,
            temperature=purpose_profile.temperature,
            tool_protocol=purpose_profile.tool_protocol,
            supports_seed=purpose_profile.supports_seed,
            request_parameters=purpose_profile.request_parameters,
        )
    services = ApplicationServices(
        store,
        model,
        tools,
        adapters,
        models,
        config.global_budget,
        config.context_limits,
        config.aggregation,
        analysis_budget=config.analysis_budget,
    )
    if config.skill_selector == "llm":
        services.runtime.skills.selector = LLMSkillSelector(services.gateway)
    services.gateway.context_budget.max_tokens = config.max_context_tokens
    services.gateway.structured_repair_limit = config.structured_repair_limit

    def factory(branch_store: FileStore) -> ApplicationServices:
        branch_tools, branch_adapters = host_components(branch_store, config)
        return build_services(branch_store, config, branch_tools, branch_adapters)

    services.isolation_factory = factory
    return services
