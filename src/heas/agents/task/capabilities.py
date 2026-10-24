"""Read-only skill selection and bounded, typed capability content generation."""

import re
from typing import Any

from heas.agents.task.selectors import ContextLimits, RuleSkillSelector, SkillSelector
from heas.core.domain import (
    EvolutionOperation,
    MemoryContent,
    Record,
    SkillContent,
    TaskGenome,
    TaskSpec,
    ToolContent,
)
from heas.infrastructure.model import CallBudget, ModelGateway
from heas.infrastructure.storage import FileStore


class SkillManager:
    """Load selected immutable skills within the host context limits."""

    def __init__(
        self,
        store: FileStore,
        max_skills: int = 2,
        selector: SkillSelector | None = None,
        limits: ContextLimits | None = None,
    ):
        if max_skills < 1:
            raise ValueError("max_skills must be positive")
        self.store, self.max_skills = store, max_skills
        self.selector = selector or RuleSkillSelector()
        self.limits = limits or ContextLimits(max_skills=max_skills)
        self.last_selection: list[dict[str, Any]] = []

    def select(self, genome: TaskGenome, task: TaskSpec) -> list[Any]:
        """Select relevant skills, including the supported empty-skill ablation baseline."""
        if not genome.skills:
            return []
        words = set(re.findall(r"\w+", task.instruction.lower()))
        scored = []
        for index, skill in enumerate(genome.skills):
            tags = skill.selection_tags
            score = len(words.intersection(str(tag).lower() for tag in tags))
            if score or not tags:
                scored.append((-score, index, skill))
        if not scored:
            return [genome.skills[0]]
        return [item[2] for item in sorted(scored)[: self.max_skills]]

    def load(
        self,
        genome: TaskGenome,
        task: TaskSpec,
        state: dict[str, Any] | None = None,
        meter: CallBudget | None = None,
    ) -> tuple[str, list[Any]]:
        """Load and validate persisted content without changing its inherited version."""
        selection = self.selector.select(genome.skills, task, state or {}, self.limits, meter)
        selected: list[Any] = []
        used = 0
        self.last_selection = []
        for choice in selection.choices:
            skill = next(s for s in genome.skills if s.artifact_id == choice.artifact_id)
            size = len(self.store.verify(skill))
            fits = (
                len(selected) < self.limits.max_skills
                and used + size <= self.limits.max_skill_tokens
            )
            self.last_selection.append(
                {
                    "artifact_id": skill.artifact_id,
                    "reason": choice.reason,
                    "loaded": fits,
                    "token_upper_bound": size,
                }
            )
            if fits:
                selected.append(skill)
                used += size
        return "\n".join(self.store.verify(s).decode("utf-8") for s in selected), selected


class CapabilityEditor:
    """Resolve typed capability edits while preserving proposal provenance."""

    def __init__(self, gateway: ModelGateway):
        self.gateway = gateway

    def resolve(
        self, operation: EvolutionOperation, parents: list[TaskGenome], meter: CallBudget
    ) -> EvolutionOperation:
        """Resolve generated capability content through the audited, bounded model gateway."""
        if not operation.generate_content:
            return operation
        content_types: dict[str, type[Record]] = {
            "skill": SkillContent,
            "memory": MemoryContent,
            "tool": ToolContent,
        }
        cls = content_types[operation.target_kind]
        store = self.gateway.store
        source: list[dict[str, Any]] = []
        for parent in parents:
            refs = [*parent.skills, parent.memory]
            source.extend(
                {"artifact_id": ref.artifact_id, "content": store.verify(ref).decode("utf-8")}
                for ref in refs
                if ref.artifact_id == operation.target_ref
            )
        content = self.gateway.structured(
            "capability_edit",
            {
                "operation": operation.model_dump(mode="json"),
                "source_capabilities": source,
                "response_schema": cls.model_json_schema(),
            },
            meter,
            cls,
        )
        return EvolutionOperation.model_validate(
            {
                **operation.model_dump(),
                "generate_content": False,
                operation.target_kind: content.model_dump(mode="json"),
            }
        )
