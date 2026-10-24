"""Replaceable capability selectors operating only on Agent-visible input and state."""

import re
from typing import Any, Protocol

from pydantic import Field

from heas.agents.task.tools import ToolRegistry
from heas.core.domain import Record, SkillRef, TaskSpec, ToolBinding
from heas.core.errors import BudgetExceeded, HEASError
from heas.infrastructure.model import CallBudget, ModelGateway


class ContextLimits(Record):
    """Bound skill and memory material loaded into a task's public context."""

    max_skills: int = Field(default=2, ge=0)
    max_skill_tokens: int = Field(default=4096, ge=0)
    max_memory_entries: int = Field(default=4, ge=0)
    max_memory_tokens: int = Field(default=2048, ge=0)
    max_tools: int = Field(default=8, ge=0)


class SelectionChoice(Record):
    """Attach a bounded selection rationale to an artifact identity."""

    artifact_id: str
    reason: str


class SkillSelection(Record):
    """Return selected skills and their host-verifiable selection evidence."""

    choices: list[SelectionChoice]


class SkillSelector(Protocol):
    """Define a replaceable selector over the current immutable skill inventory."""

    def select(
        self,
        skills: list[SkillRef],
        task: TaskSpec,
        state: dict[str, Any],
        limits: ContextLimits,
        meter: CallBudget | None = None,
    ) -> SkillSelection:
        """Select admissible capabilities or parents and retain the reasons for the choice."""
        ...


def words(value: str) -> set[str]:
    """Extract normalized words for the deterministic host selection heuristic."""
    return set(re.findall(r"\w+", value.lower()))


class RuleSkillSelector:
    """Select skills by explicit applicability and bounded lexical matching."""

    def select(
        self,
        skills: list[SkillRef],
        task: TaskSpec,
        state: dict[str, Any],
        limits: ContextLimits,
        meter: CallBudget | None = None,
    ) -> SkillSelection:
        """Select admissible capabilities or parents and retain the reasons for the choice."""
        tokens = words(task.instruction + " " + str(state))
        ranked = []
        for index, skill in enumerate(skills):
            tags = set(tag.lower() for tag in skill.selection_tags)
            score = len(tokens.intersection(tags))
            if score or not tags:
                ranked.append((-score, index, skill))
        if not ranked and skills:
            ranked = [(0, 0, skills[0])]
        return SkillSelection(
            choices=[
                SelectionChoice(
                    artifact_id=s.artifact_id,
                    reason=f"rule overlap={-score}; applicability={s.applicability or 'general'}",
                )
                for score, _, s in sorted(ranked)[: limits.max_skills]
            ]
        )


class LLMSkillSelector:
    """Select from a bounded skill inventory through the audited model gateway."""

    def __init__(self, gateway: ModelGateway, max_inventory: int = 32):
        self.gateway, self.max_inventory = gateway, max_inventory

    def select(
        self,
        skills: list[SkillRef],
        task: TaskSpec,
        state: dict[str, Any],
        limits: ContextLimits,
        meter: CallBudget | None = None,
    ) -> SkillSelection:
        """Select admissible capabilities or parents and retain the reasons for the choice."""
        if meter is None:
            raise HEASError("LLM selection must share the runtime call budget")
        inventory = skills[: self.max_inventory]
        data = self.gateway.call(
            "skill_selection",
            {
                "task": task.public_view(),
                "state": state,
                "inventory": [
                    {
                        "artifact_id": s.artifact_id,
                        "tags": s.selection_tags[:12],
                        "applicability": s.applicability[:256],
                    }
                    for s in inventory
                ],
                "max_skills": limits.max_skills,
                "response_schema": SkillSelection.model_json_schema(),
            },
            meter,
            validator=SkillSelection.model_validate,
        )
        selection = SkillSelection.model_validate(data)
        if (
            len(selection.choices) > limits.max_skills
            or len({c.artifact_id for c in selection.choices}) != len(selection.choices)
            or not {c.artifact_id for c in selection.choices}.issubset(
                {s.artifact_id for s in inventory}
            )
        ):
            raise HEASError("selector returned duplicate, excessive or unknown skills")
        return selection


class ToolSelector(Protocol):
    """Define host selection over already approved tool bindings."""

    def select(
        self,
        tools: list[ToolBinding],
        skills: list[SkillRef],
        task: TaskSpec,
        state: dict[str, Any],
        registry: ToolRegistry,
        limits: ContextLimits,
    ) -> list[ToolBinding]:
        """Select admissible capabilities or parents and retain the reasons for the choice."""
        ...


class RuleToolSelector:
    """Select approved tools and their required dependency closure."""

    def select(
        self,
        tools: list[ToolBinding],
        skills: list[SkillRef],
        task: TaskSpec,
        state: dict[str, Any],
        registry: ToolRegistry,
        limits: ContextLimits,
    ) -> list[ToolBinding]:
        """Select admissible capabilities or parents and retain the reasons for the choice."""
        available = {t.tool_id: t for t in tools}
        required = {tool_id for skill in skills for tool_id in skill.requires_tools}
        tokens = words(task.instruction + " " + str(state))
        selected = set(required)
        for binding in tools:
            registry.validate(binding)
            manifest = registry.entries[(binding.tool_id, binding.version_id)][0]
            if not manifest.selection_tags or tokens.intersection(manifest.selection_tags):
                if len(selected) < limits.max_tools:
                    selected.add(binding.tool_id)

        def include(tool_id: str, visiting: set[str]) -> None:
            if tool_id in visiting:
                raise HEASError("cyclic tool dependency")
            if tool_id not in available:
                raise HEASError("selected capability has unavailable tool dependency")
            binding = available[tool_id]
            registry.validate(binding)
            manifest = registry.entries[(binding.tool_id, binding.version_id)][0]
            for dependency in manifest.requires_tools:
                selected.add(dependency)
                include(dependency, visiting | {tool_id})

        for tool_id in list(selected):
            include(tool_id, set())
        if len(selected) > limits.max_tools:
            raise BudgetExceeded("required tool dependency closure exceeds context limit")
        return [tool for tool in tools if tool.tool_id in selected]
