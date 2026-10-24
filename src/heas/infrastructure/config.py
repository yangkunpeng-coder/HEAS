"""Production model and trusted host integration configuration."""

from typing import Any, Literal

from pydantic import Field

from heas.agents.task.selectors import ContextLimits
from heas.core.domain import Budget, Record
from heas.evidence.aggregation import AggregationConfig


class ModelProfile(Record):
    """Configure a host-selected provider without inheriting model settings in a Genome."""

    model: str = Field(min_length=1)
    base_url: str = "https://api.deepseek.com"
    api_key_env: str = "HEAS_API_KEY"
    temperature: float = Field(default=0, ge=0, le=2)
    tool_protocol: Literal["json", "native"] = "native"
    supports_seed: bool = False
    request_parameters: dict[str, Any] = Field(default_factory=dict)


class ApplicationConfig(Record):
    """Bind trusted ports, role models, context limits, and shared resource budgets."""

    workspace: str
    model_profile: ModelProfile | None = None
    plugin_factory: str | None = None
    allowed_tool_permissions: list[str] = Field(default_factory=list)
    models_by_purpose: dict[str, ModelProfile] = Field(default_factory=dict)
    global_budget: Budget | None = None
    context_limits: ContextLimits = Field(default_factory=ContextLimits)
    aggregation: AggregationConfig = Field(default_factory=AggregationConfig)
    skill_selector: Literal["rule", "llm"] = "rule"
    analysis_budget: Budget | None = None
    max_context_tokens: int = Field(default=32000, ge=8192)
    structured_repair_limit: int = Field(default=1, ge=0, le=3)
