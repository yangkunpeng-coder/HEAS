"""Host-owned, task-neutral bounded experiment contract."""

from typing import Literal, Self

from pydantic import Field, model_validator

from heas.agents.meta.policy_trials import PolicyTrialSpec
from heas.core.domain import Budget, Id, ParentSelectionConfig, Record
from heas.experimentation.suites import EvaluationSuiteSpec, TaskSuite


class ExperimentPlan(Record):
    """Freeze task suites, initial candidates, lifecycle limits, and resource budgets."""

    schema_version: Literal["1.2"] = "1.2"
    experiment_id: Id
    version: Id = "v1"
    output_directory: str = Field(min_length=1)
    initial_genome_ids: list[Id] = Field(min_length=1)
    initial_policy_id: Id
    training_suite: TaskSuite
    validation_suite: TaskSuite | None = None
    evaluation_suite: EvaluationSuiteSpec | None = None
    parent_selection: ParentSelectionConfig | None = None
    max_cycles: int = Field(default=2, ge=1, le=1000)
    attempts_per_cycle: int = Field(default=2, ge=1, le=32)
    population_limit: int = Field(default=8, ge=1, le=128)
    random_seeds: list[int] = Field(default=[23], min_length=1, max_length=32)
    meta_every: int | None = Field(default=2, ge=1)
    policy_trial: PolicyTrialSpec | None = None
    allow_exploratory_policy: bool = False
    auto_promote_candidates: bool = True
    total_budget: Budget = Field(
        default_factory=lambda: Budget(
            max_model_calls=1000, max_tool_calls=1000, max_wall_seconds=3600
        )
    )
    run_budget: Budget = Field(default_factory=Budget)
    analysis_budget: Budget = Field(default_factory=lambda: Budget(max_wall_seconds=60))
    evolution_budget: Budget = Field(default_factory=lambda: Budget(max_wall_seconds=60))
    meta_budget: Budget = Field(default_factory=lambda: Budget(max_wall_seconds=60))

    @model_validator(mode="after")
    def learning_contract(self) -> Self:
        """Reject duplicate starts and overlapping or invalid learning/validation partitions."""
        if len(self.initial_genome_ids) != len(set(self.initial_genome_ids)):
            raise ValueError("duplicate initial genome")
        if len(set(self.random_seeds)) != len(self.random_seeds):
            raise ValueError("duplicate random seed")
        if self.training_suite.partition not in {"development", "evolution_train"}:
            raise ValueError("training suite requires a learning partition")
        if sum(len(g.task_ids) for g in self.training_suite.groups) < 2:
            raise ValueError("multi-run experience requires at least two training tasks")
        if self.validation_suite:
            if self.validation_suite.partition != "evolution_validation":
                raise ValueError("candidate validation requires evolution_validation")
            train = {(g.provider_id, t) for g in self.training_suite.groups for t in g.task_ids}
            validate = {
                (g.provider_id, t) for g in self.validation_suite.groups for t in g.task_ids
            }
            if train & validate:
                raise ValueError("training and validation task instances must be disjoint")
        if self.policy_trial and not set(self.policy_trial.start_genome_ids).issubset(
            self.initial_genome_ids
        ):
            raise ValueError("trial starts must be immutable initial experiment genomes")
        return self
