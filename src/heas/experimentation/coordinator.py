"""Production composition root; model and external adapters are explicitly supplied."""

from heas.agents.evolution.candidates import CandidateRegistry
from heas.agents.evolution.engine import EvolutionEngine
from heas.agents.evolution.evolution import EvolutionService, GenomeMaterializer
from heas.agents.meta.meta import MetaEvolutionService
from heas.agents.task.runtime import TaskRunService
from heas.agents.task.selectors import ContextLimits
from heas.agents.task.tools import ToolRegistry
from heas.core.domain import Budget
from heas.core.ports import ModelClient
from heas.evidence.aggregation import AggregationConfig
from heas.evidence.experience import ExperienceService
from heas.experimentation.isolation import ServiceFactory
from heas.infrastructure.adapters import AdapterRegistry
from heas.infrastructure.model import ModelGateway
from heas.infrastructure.storage import FileStore


class ApplicationServices:
    """Compose the existing role services around one store and independently owned ports."""

    def __init__(
        self,
        store: FileStore,
        model: ModelClient,
        tools: ToolRegistry,
        adapters: AdapterRegistry,
        models_by_purpose: dict[str, ModelClient] | None = None,
        global_budget: Budget | None = None,
        context_limits: ContextLimits | None = None,
        aggregation_config: AggregationConfig | None = None,
        analysis_budget: Budget | None = None,
    ):
        self.store, self.tools, self.adapters = store, tools, adapters
        self.isolation_factory: ServiceFactory | None = None
        self.gateway = ModelGateway(model, store, models_by_purpose, global_budget)
        self.materializer = GenomeMaterializer(store, tools)
        self.runtime = TaskRunService(store, self.gateway, adapters, tools, context_limits)
        self.experience = ExperienceService(
            store, self.gateway, budget=analysis_budget, aggregation_config=aggregation_config
        )
        self.evolution = EvolutionService(store, self.gateway, self.materializer)
        self.meta = MetaEvolutionService(store, self.gateway)
        self.candidates = CandidateRegistry(store)
        self.engine = EvolutionEngine(
            store, self.evolution, self.runtime, self.experience, self.candidates
        )
