"""Pinned, opt-in integration configuration without benchmark dependencies in Core."""

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, model_validator

from heas.core.domain import Budget, Record
from heas.core.errors import HEASError

EVO_REVISION = "948a17288782d5120778da16b4cf1cad9305d8b4"
SEA_REVISION = "8e0a108cddf069bf640d565edbd712891f230384"
SKILL_REVISION = "726e1f79b96b0cb3d729d0e55a1c847e0b5353ae"
DOMAINS = ("information_retrieval", "code_implementation", "software_engineering", "knowledge_work")
Partition = Literal[
    "development", "evolution_train", "evolution_validation", "meta_validation", "final_test"
]


class BenchmarkConfig(Record, frozen=True):
    """Declare trusted source locations and explicit execution boundaries.

    Paths are host configuration, never fields chosen by a Task Agent. Execution
    remains disabled until the host explicitly enables the integration.
    """

    repository: Path
    private_root: Path
    allow_execution: bool = False
    allow_paid_judges: bool = False
    offline_data: bool = True
    worker_python: str | None = None
    isolation_prefix: list[str] = Field(default_factory=list)
    timeout_seconds: float = Field(default=120, gt=0, le=86400)
    max_evaluations: int = Field(default=100, ge=1)
    partitions: dict[Partition, list[str]] = Field(default_factory=dict)
    data_origin: Literal["official", "sample/synthetic"] = "official"

    @model_validator(mode="after")
    def private_storage(self) -> "BenchmarkConfig":
        """Keep private runtime artifacts outside the upstream source checkout."""
        if self.private_root.resolve().is_relative_to(self.repository.resolve()):
            raise ValueError("private runtime data must stay outside the official source checkout")
        return self


class EvoAgentBenchConfig(BenchmarkConfig, frozen=True):
    """Configure one of the four official EvoAgentBench domain adapters."""

    benchmark: Literal["evoagentbench"] = "evoagentbench"
    revision: str = EVO_REVISION
    domain: Literal[
        "information_retrieval", "code_implementation", "software_engineering", "knowledge_work"
    ]
    official_config: Path
    mcp_tool_allowlist: list[str] = Field(default_factory=lambda: ["search", "fetch"])


class SEABenchConfig(BenchmarkConfig, frozen=True):
    """Configure an ordered upstream/frozen-downstream SEABench sequence."""

    benchmark: Literal["seabench"] = "seabench"
    revision: str = SEA_REVISION
    sequence_file: Path
    judge_model: str | None = None
    judge_base_url: str = "https://openrouter.ai/api/v1"
    judge_key_env: str = "OPENROUTER_API_KEY"


class SkillLearnBenchConfig(BenchmarkConfig, frozen=True):
    """Configure exported-genome-skill evaluation by the official solving agent."""

    benchmark: Literal["skilllearnbench"] = "skilllearnbench"
    revision: str = SKILL_REVISION
    task_ids: list[str] = Field(min_length=1)
    solver_agent: Literal["claude-code", "codex", "gemini-code"] = "claude-code"
    solver_model: str = "claude-sonnet-4-6"
    judge_model: str = "gpt-5-mini"
    repeats: int = Field(default=1, ge=1, le=20)
    max_steps: int = Field(default=100, ge=1, le=1000)
    compute_quality_metrics: bool = False


IntegrationConfig = Annotated[
    EvoAgentBenchConfig | SEABenchConfig | SkillLearnBenchConfig,
    Field(discriminator="benchmark"),
]


def load_config(path: Path) -> IntegrationConfig:
    """Load a typed configuration and resolve paths relative to that file."""
    config = TypeAdapter(IntegrationConfig).validate_json(path.read_text(encoding="utf-8-sig"))
    values = config.model_dump()
    for field in ("repository", "private_root", "official_config", "sequence_file"):
        value = values.get(field)
        if isinstance(value, Path) and not value.is_absolute():
            values[field] = (path.resolve().parent / value).resolve()
    return TypeAdapter(IntegrationConfig).validate_python(values)


def require_partition_map(partitions: Mapping[str, list[str]]) -> None:
    """Reject train/validation/test overlap instead of inventing an official split."""
    learning = set(partitions.get("development", [])) | set(partitions.get("evolution_train", []))
    reserved = [
        set(partitions.get(p, []))
        for p in ("evolution_validation", "meta_validation", "final_test")
    ]
    for index, group in enumerate(reserved):
        if learning & group or any(group & previous for previous in reserved[:index]):
            raise HEASError("learning, validation, and final-test task instances must be disjoint")
    if any(len(ids) != len(set(ids)) for ids in partitions.values()):
        raise HEASError("duplicate task identity in partition")


def require_external_budget_support(budgets: list[Budget]) -> None:
    """Fail closed when an external runner cannot honor token or currency limits."""
    if any(
        b.max_cost is not None or b.max_input_tokens is not None or b.max_output_tokens is not None
        for b in budgets
    ):
        raise HEASError("external judge/solver usage is not bounded by the HEAS token/cost meter")
