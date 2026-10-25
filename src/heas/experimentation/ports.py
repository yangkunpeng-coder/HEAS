"""Structural Host services for experiments; no construction or persistence policy."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from heas.agents.evolution.candidates import CandidateRegistry
    from heas.agents.evolution.engine import EvolutionEngine
    from heas.agents.evolution.evolution import EvolutionService, GenomeMaterializer
    from heas.agents.meta.meta import MetaEvolutionService
    from heas.agents.task.runtime import TaskRunService
    from heas.agents.task.tools import ToolRegistry
    from heas.evidence.experience import ExperienceService
    from heas.infrastructure.adapters import AdapterRegistry
    from heas.infrastructure.model import ModelGateway
    from heas.infrastructure.storage import FileStore


class ExperimentServices(Protocol):
    """Describe independently owned services consumed by experiment orchestration.

    Hosts may supply their own composition root. A trial factory must construct
    fresh mutable clients and adapters for the supplied store; structural typing
    does not replace the runner's runtime isolation checks.
    """

    store: FileStore
    gateway: ModelGateway
    tools: ToolRegistry
    adapters: AdapterRegistry
    materializer: GenomeMaterializer
    runtime: TaskRunService
    experience: ExperienceService
    evolution: EvolutionService
    meta: MetaEvolutionService
    candidates: CandidateRegistry
    engine: EvolutionEngine
    isolation_factory: ServiceFactory | None


ServiceFactory = Callable[["FileStore"], ExperimentServices]
