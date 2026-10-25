"""Bounded transparent single-agent loop and serial frozen-genome task batches."""

from collections.abc import Callable as Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import Field, ValidationError, model_validator

from heas.agents.task.capabilities import SkillManager
from heas.agents.task.memory import MemoryManager
from heas.agents.task.selectors import ContextLimits, RuleToolSelector, ToolSelector
from heas.agents.task.tools import ToolRegistry
from heas.core.domain import (
    AgentObservation,
    EvaluationResult,
    EvaluationState,
    EvidenceRef,
    NativeModelTurn,
    NativeToolCall,
    Record,
    RunEvidencePackage,
    RunPlan,
    SkillRef,
    TaskGenome,
    TaskResult,
    TaskRunRecord,
    TaskSpec,
    ToolBinding,
    TraceEvent,
    now,
    uid,
)
from heas.core.errors import BudgetExceeded, HEASError, SecurityError
from heas.core.ports import (
    FingerprintedEnvironment,
    ModelClient,
    StructuredTaskEvaluator,
    TaskEnvironment,
    TaskEvaluator,
)
from heas.experimentation.scheduler import SerialBatchScheduler
from heas.infrastructure.adapters import AdapterRegistry
from heas.infrastructure.model import CallBudget, ModelGateway
from heas.infrastructure.storage import FileStore, digest, encode, redact

_TerminationReason = Literal[
    "final_answer",
    "max_steps",
    "budget_exceeded",
    "tool_error",
    "model_error",
    "environment_error",
    "cancelled",
]

_EventKind = Literal[
    "run_start",
    "capability_context",
    "model_call_start",
    "model_call_end",
    "tool_call_start",
    "tool_call_end",
    "tool_call_error",
    "feedback_received",
    "run_end",
    "security_event",
]


class _EventEmitter(Protocol):
    """Accept only trace event kinds and capability references owned by the runtime."""

    def __call__(
        self,
        kind: _EventKind,
        step: int,
        payload: dict[str, Any] | None = None,
        *,
        skill_refs: list[str] | None = None,
        memory_refs: list[str] | None = None,
        tool_ids: list[str] | None = None,
    ) -> None: ...


class TaskAction(Record, frozen=True):
    """Validate the JSON loop's mutually exclusive final response or approved tool request."""

    action: Literal["final", "tool"]
    answer: str | None = None
    tool_id: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    adopted_skill_ids: list[str] = Field(default_factory=list)
    result: TaskResult | None = None

    @model_validator(mode="after")
    def consistent_action(self) -> "TaskAction":
        """Reject ambiguous final/tool action payloads before executing either branch."""
        if self.action == "final" and self.answer is None and self.result is None:
            raise ValueError("final action requires an answer")
        if self.action == "tool" and not self.tool_id:
            raise ValueError("tool action requires tool_id")
        return self


@dataclass
class _TaskExecution:
    """Keep transient loop state separate from persisted, versioned evidence."""

    answer: str
    result: TaskResult
    reason: _TerminationReason
    errors: list[str]
    capabilities: list[dict[str, Any]]
    client: ModelClient
    environment: TaskEnvironment


@dataclass
class _DecodedTurn:
    """Normalize transport actions without conflating requests with successful effects."""

    calls: list[NativeToolCall]
    result: TaskResult
    reported_skill_ids: list[str]
    payload: dict[str, Any]


class TaskRunService:
    """Execute independent task runs against a frozen capability genome."""

    def __init__(
        self,
        store: FileStore,
        gateway: ModelGateway,
        adapters: AdapterRegistry,
        tools: ToolRegistry,
        context_limits: ContextLimits | None = None,
        tool_selector: ToolSelector | None = None,
    ):
        self.store, self.gateway, self.adapters, self.tools = store, gateway, adapters, tools
        self.skills = SkillManager(store)
        self.context_limits = context_limits or ContextLimits()
        self.skills.limits = self.context_limits
        self.memory = MemoryManager(store)
        self.tool_selector = tool_selector or RuleToolSelector()
        for evaluator in adapters.evaluators.values():
            bind = getattr(evaluator, "bind_execution_context", None)
            if callable(bind):
                # Resource-aware adapters reject unsupported external token/cost
                # limits before invoking an independently billed official runner.
                bind(self.gateway)

    def batch(self, plan: RunPlan, *, resume: bool = False) -> list[TaskRunRecord]:
        """Execute serial independent tasks using each plan's frozen Genome versions.

        Args:
            plan: Host-approved tasks, ports, resource limits and Genome identities.
            resume: Reuse committed units after checking the archived plan identity.

        Returns:
            Separate archived runs, including failed tasks and their measured usage.

        Raises:
            HEASError: Plan identity or an injected execution port is invalid.
        """
        if plan.task_provider_id not in self.adapters.providers:
            raise HEASError("task provider not registered")
        if plan.environment_adapter_id not in self.adapters.environments:
            raise HEASError("task environment not registered")
        if plan.task_evaluator_id and plan.task_evaluator_id not in self.adapters.evaluators:
            raise HEASError("requested evaluator not registered; use null for unavailable")
        provider = self.adapters.providers[plan.task_provider_id]
        permitted_ids = {t.task_id for t in provider.list_tasks(plan.task_partition)}
        if not set(plan.task_ids).issubset(permitted_ids):
            raise SecurityError("requested tasks are outside the authorized partition")
        if plan.max_tasks is not None:
            plan = RunPlan.model_validate(
                {**plan.model_dump(), "task_ids": plan.task_ids[: plan.max_tasks]}
            )
        tasks = [provider.get_task(task_id) for task_id in plan.task_ids]
        if any(t.provider_id != plan.task_provider_id for t in tasks):
            raise HEASError("provider returned task with mismatched identity")
        model_profile = {
            "schema_version": "1.0",
            "model_identity": self.gateway.client_for("task").identity,
            "parameters": redact(getattr(self.gateway.client_for("task"), "settings", {})),
        }
        profile_uri = f"configs/model_profiles/{digest(encode(model_profile))[7:]}.json"
        if not self.store.path(profile_uri).exists():
            self.store.put(profile_uri, model_profile)
        elif self.store.read(profile_uri) != model_profile:
            raise HEASError("model profile snapshot integrity mismatch")
        plan = RunPlan.model_validate({**plan.model_dump(), "model_config_ref": profile_uri})
        plan_uri = f"plans/{plan.run_plan_id}.json"
        if self.store.path(plan_uri).exists():
            if not resume or self.store.load(plan_uri, RunPlan) != plan:
                raise HEASError("existing RunPlan requires identical settings and resume=True")
        else:
            self.store.put(plan_uri, plan)
        for genome_id in plan.genome_ids:
            for tool in self.store.genome(genome_id).tools:
                self.tools.validate(tool)
        return SerialBatchScheduler(self.store).execute(plan, tasks, self.run, resume=resume)

    def run(
        self,
        genome: TaskGenome,
        task: TaskSpec,
        plan: RunPlan,
        attempt: int,
        *,
        run_id: str | None = None,
        retry_index: int = 0,
    ) -> TaskRunRecord:
        """Execute one task with a fixed genome and independently evaluate its result.

        Args:
            genome: Immutable capability snapshot shared by this run's decisions.
            task: Public task specification supplied through the provider port.
            plan: Host execution, partition, model, and budget constraints.
            attempt: Logical repeat index, independent of transport retries.
            run_id: Optional durable identity reserved by the batch journal.
            retry_index: Retry index retained for resource and recovery audits.

        Returns:
            An archived run with its trace and explicit evaluation availability.
        """
        run_id = run_id or uid("run")
        started = now()
        meter = CallBudget(plan.budget)
        meter.scope_id = f"run:{run_id}"
        trace_uri = f"runs/{run_id}/trace.jsonl"

        def emit(
            kind: _EventKind,
            step: int,
            payload: dict[str, Any] | None = None,
            *,
            skill_refs: list[str] | None = None,
            memory_refs: list[str] | None = None,
            tool_ids: list[str] | None = None,
        ) -> None:
            self.store.append_event(
                TraceEvent(
                    event_id=uid("event"),
                    run_id=run_id,
                    step=step,
                    event_type=kind,
                    safe_payload=redact(payload or {}),
                    skill_refs=skill_refs or [],
                    memory_refs=memory_refs or [],
                    tool_ids=tool_ids or [],
                )
            )

        execution = self._execute_task(genome, task, plan, attempt, run_id, meter, emit)
        answer, task_result = execution.answer, execution.result
        reason, errors = execution.reason, execution.errors
        task_client, environment = execution.client, execution.environment
        evaluation = EvaluationState()
        task_result = TaskResult.model_validate(
            redact(
                {
                    **task_result.model_dump(mode="json"),
                    "termination_reason": reason,
                    "errors": errors,
                    "evaluation": evaluation.model_dump(),
                }
            )
        )
        public_feedback = None
        feedback_uri = None
        environment_metadata: dict[str, str | None] = {}
        if isinstance(environment, FingerprintedEnvironment):
            fingerprint = environment.input_fingerprint
            if fingerprint is not None and not isinstance(fingerprint, str):
                raise HEASError("environment input fingerprint must be a string or None")
            environment_metadata["environment_input_sha256"] = fingerprint
        run = TaskRunRecord(
            run_id=run_id,
            run_plan_id=plan.run_plan_id,
            genome_id=genome.genome_id,
            task_id=task.task_id,
            attempt_index=attempt,
            trace_ref=trace_uri,
            start_time=started,
            end_time=now(),
            termination_reason=reason,
            objective_feedback_ref=None,
            evaluation=evaluation,
            usage=meter.usage(),
            task_partition=plan.task_partition,
            retry_index=retry_index,
            task_instance_sha256=digest(encode(task.model_dump(mode="json"))),
            evidence_ref=f"runs/{run_id}/evidence.json",
            output=redact(answer),
            result=task_result,
            extensions={
                "heas": {
                    "output": redact(answer),
                    "errors": errors,
                    "budget_scope_ids": self.gateway.scope_ids(meter),
                    "model_identity": task_client.identity,
                    "model_parameters_sha256": digest(encode(getattr(task_client, "settings", {}))),
                    **environment_metadata,
                }
            },
        )
        feedback = self._evaluate_run(task, plan, run, task_result, answer)
        if feedback is not None:
            feedback_uri = f"runs/{run_id}/feedback.json"
            self.store.save_record(feedback_uri, feedback, "feedback", run_id)
            evaluation = EvaluationState(
                evaluation_status=feedback.evaluation_status,
                score=feedback.score,
                scores=feedback.scores,
            )
            public_feedback = feedback.agent_feedback
            emit("feedback_received", 0, {"evaluation_status": evaluation.evaluation_status})
        emit(
            "run_end",
            0,
            {"termination_reason": reason, "evaluation_status": evaluation.evaluation_status},
        )
        task_result = TaskResult.model_validate(
            {
                **task_result.model_dump(),
                "termination_reason": reason,
                "errors": errors,
                "evaluation": evaluation,
            }
        )
        run = TaskRunRecord.model_validate(
            {
                **run.model_dump(),
                "result": task_result,
                "evaluation": evaluation,
                "objective_feedback_ref": feedback_uri,
                "end_time": now(),
                "usage": meter.usage(),
            }
        )
        return self._archive_run(run, genome, task, execution, public_feedback)

    def _evaluate_run(
        self,
        task: TaskSpec,
        plan: RunPlan,
        run: TaskRunRecord,
        task_result: TaskResult,
        answer: str,
    ) -> EvaluationResult | None:
        """Separate evaluator execution faults from rejected evaluation contracts."""
        evaluator_id = plan.task_evaluator_id
        if evaluator_id is None:
            return None
        adapter = self.adapters.evaluators[evaluator_id]
        try:
            workspace = self.store.path(f"runs/{run.run_id}")
            if isinstance(adapter, StructuredTaskEvaluator):
                raw_feedback = adapter.evaluate_result(task, run, task_result, workspace)
            elif isinstance(adapter, TaskEvaluator):
                raw_feedback = adapter.evaluate(task, run, answer, workspace)
            else:
                raise HEASError("registered evaluator does not implement a task evaluation port")
        except ValidationError as exc:
            return self._reject_evaluation(run, plan, "evaluation_schema_error", type(exc).__name__)
        except Exception as exc:
            return self._reject_evaluation(
                run, plan, "evaluator_execution_error", type(exc).__name__
            )
        try:
            # Revalidate model instances as data too: a Host adapter may construct
            # a receipt without validation, or copy it with unchecked updates.
            feedback_data: object = raw_feedback
            if isinstance(feedback_data, EvaluationResult):
                feedback_data = feedback_data.model_dump()
            feedback = EvaluationResult.model_validate(feedback_data)
            if feedback.evaluator_version == "unversioned":
                feedback = EvaluationResult.model_validate(
                    {
                        **feedback.model_dump(),
                        "evaluator_version": getattr(adapter, "version", "unversioned"),
                    }
                )
        except ValidationError as exc:
            return self._reject_evaluation(run, plan, "evaluation_schema_error", type(exc).__name__)
        category = None
        if feedback.evaluator_id != evaluator_id:
            category = "evaluator_identity_mismatch"
        elif feedback.run_id != run.run_id:
            category = "run_identity_mismatch"
        elif plan.evaluator_version and feedback.evaluator_version != plan.evaluator_version:
            category = "evaluator_version_mismatch"
        if category:
            return self._reject_evaluation(run, plan, category, "EvaluationContractMismatch")
        return feedback

    def _reject_evaluation(
        self,
        run: TaskRunRecord,
        plan: RunPlan,
        category: str,
        error_type: str,
    ) -> EvaluationResult:
        # Exception messages and rejected payloads can contain private grading data.
        # Persist only a fixed category and exception class, never a fabricated score.
        evaluator_id = plan.task_evaluator_id
        if evaluator_id is None:
            raise HEASError("evaluation rejection requires a configured evaluator")
        diagnostic_ref = f"runs/{run.run_id}/evaluation_diagnostic.json"
        self.store.put(
            diagnostic_ref,
            {
                "schema_version": "1.0",
                "run_id": run.run_id,
                "category": category,
                "error_type": error_type,
            },
        )
        return EvaluationResult(
            evaluator_id=evaluator_id,
            run_id=run.run_id,
            evaluator_version=plan.evaluator_version or "unversioned",
            evaluation_status="failed" if category == "evaluator_execution_error" else "invalid",
            feedback=category,
            comparison_eligible=False,
            extensions={"heas": {"diagnostic_ref": diagnostic_ref}},
        )

    def _execute_task(
        self,
        genome: TaskGenome,
        task: TaskSpec,
        plan: RunPlan,
        attempt: int,
        run_id: str,
        meter: CallBudget,
        emit: _EventEmitter,
    ) -> _TaskExecution:
        """Run the bounded loop and always revoke environment access before evaluation."""
        environment = self.adapters.environments[plan.environment_adapter_id]
        reason: _TerminationReason = "max_steps"
        answer = ""
        task_result = TaskResult()
        task_client = self.gateway.client_for("task")
        native = getattr(task_client, "tool_protocol", "json") == "native"
        messages: list[dict[str, Any]] = []
        call_ids: set[str] = set()
        errors: list[str] = []
        emit(
            "run_start",
            0,
            {
                "genome_id": genome.genome_id,
                "task_id": task.task_id,
                "model_identity": task_client.identity,
            },
        )
        results: list[dict[str, Any]] = []
        loaded_capabilities: list[dict[str, Any]] = []
        visible: list[str] = []
        stage = "environment"
        try:
            environment.prepare(task, attempt, self.store.path(f"runs/{run_id}/environment"))
            for step in range(1, plan.budget.max_steps + 1):
                meter.check()
                stage = "environment"
                observation = AgentObservation.model_validate(environment.observe())
                context, selected, selected_tools = self._prepare_turn(
                    genome, task, plan, observation, meter, results, loaded_capabilities, step, emit
                )
                visible = [t.tool_id for t in selected_tools]
                stage = "model"
                if native:
                    public_context = {
                        k: v
                        for k, v in context.items()
                        if k not in {"tool_results", "response_schema"}
                    }
                    messages.append(
                        {"role": "user", "content": encode(redact(public_context)).decode("utf-8")}
                    )
                    context["messages"] = messages
                emit(
                    "model_call_start",
                    step,
                    {"call_purpose": "task", "protocol": "native" if native else "json"},
                )
                response = self.gateway.call(
                    "task",
                    context,
                    meter,
                    validator=(NativeModelTurn if native else TaskAction).model_validate,
                )
                turn = self._decode_turn(response, native, selected, messages, call_ids)
                calls, final_result, adopted = turn.calls, turn.result, turn.reported_skill_ids
                emit(
                    "model_call_end",
                    step,
                    {
                        "action": turn.payload,
                        "model_reported_skill_ids": adopted,
                        "usage": meter.usage().model_dump(mode="json"),
                    },
                )
                if not calls:
                    for ref in final_result.artifact_refs:
                        self.store.verify(ref)
                    if final_result.environment_state_ref:
                        self.store.verify(final_result.environment_state_ref)
                    task_result = final_result
                    answer = final_result.text or (
                        encode(final_result.structured_output).decode("utf-8")
                        if final_result.structured_output is not None
                        else ""
                    )
                    reason = "final_answer"
                    break
                for call in calls:
                    stage = "tool"
                    tool_id = call.tool_id
                    emit(
                        "tool_call_start",
                        step,
                        {"tool_id": tool_id, "call_id": call.call_id, "arguments": call.arguments},
                    )
                    self.gateway.tool(meter)
                    try:
                        if call.arguments is None:
                            raise HEASError("invalid native tool arguments")
                        if tool_id not in visible:
                            raise SecurityError("requested tool not exposed by genome")
                        binding = next(t for t in selected_tools if t.tool_id == tool_id)
                        output = self.tools.execute_binding(
                            binding, call.arguments, timeout=meter.remaining_time
                        )
                        meter.check()
                        stage = "environment"
                        environment.apply_tool(tool_id, output)
                        receipt = {"tool_id": tool_id, "call_id": call.call_id, "result": output}
                        results.append(receipt)
                        emit("tool_call_end", step, receipt, skill_refs=adopted)
                    except BudgetExceeded:
                        raise
                    except Exception as exc:
                        if stage == "environment":
                            raise
                        error = type(exc).__name__
                        receipt = {"tool_id": tool_id, "call_id": call.call_id, "error": error}
                        results.append(receipt)
                        errors.append(error)
                        emit("tool_call_error", step, {**receipt, "error_type": error})
                        reason = "tool_error"
                    if native:
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.call_id,
                                "content": encode(redact(receipt)).decode("utf-8"),
                            }
                        )
        except BudgetExceeded:
            reason = "budget_exceeded"
        except Exception as exc:
            reason = "environment_error" if stage == "environment" else "model_error"
            errors.append(type(exc).__name__)
        finally:
            try:
                environment.teardown()
            except Exception as exc:
                reason = "environment_error"
                errors.append(type(exc).__name__)
        return _TaskExecution(
            answer, task_result, reason, errors, loaded_capabilities, task_client, environment
        )

    @staticmethod
    def _decode_turn(
        response: dict[str, Any],
        native: bool,
        selected_skills: list[SkillRef],
        messages: list[dict[str, Any]],
        call_ids: set[str],
    ) -> _DecodedTurn:
        """Normalize model actions and retain native history before any tool runs.

        The JSON protocol may report adoption only of injected Skills. Native
        call IDs must be unique across this run so subsequent receipts correlate
        with exactly one requested effect. Neither protocol establishes success.
        """
        if native:
            turn = NativeModelTurn.model_validate(response)
            calls = turn.tool_calls
            if len({call.call_id for call in calls}) != len(calls) or any(
                call.call_id in call_ids for call in calls
            ):
                raise HEASError("duplicate native tool call ID")
            call_ids.update(call.call_id for call in calls)
            assistant: dict[str, Any] = {"role": "assistant", "content": turn.content}
            if calls:
                assistant["tool_calls"] = [
                    {
                        "id": call.call_id,
                        "type": "function",
                        "function": {"name": call.tool_id, "arguments": call.raw_arguments},
                    }
                    for call in calls
                ]
            messages.append(assistant)
            return _DecodedTurn(
                calls,
                turn.result or TaskResult(text=turn.content),
                [],
                turn.model_dump(mode="json"),
            )
        action = TaskAction.model_validate(response)
        adopted = action.adopted_skill_ids
        if not set(adopted).issubset({skill.artifact_id for skill in selected_skills}):
            raise SecurityError("model claimed a skill outside injected context")
        calls = []
        if action.action == "tool":
            tool_id = action.tool_id
            if not tool_id:
                raise HEASError("tool action requires tool_id")
            calls.append(
                NativeToolCall(
                    call_id=uid("tool_call"), tool_id=tool_id, arguments=action.arguments
                )
            )
        return _DecodedTurn(
            calls,
            action.result or TaskResult(text=action.answer),
            adopted,
            action.model_dump(mode="json"),
        )

    def _prepare_turn(
        self,
        genome: TaskGenome,
        task: TaskSpec,
        plan: RunPlan,
        observation: AgentObservation,
        meter: CallBudget,
        results: list[dict[str, Any]],
        loaded_capabilities: list[dict[str, Any]],
        step: int,
        emit: _EventEmitter,
    ) -> tuple[dict[str, Any], list[SkillRef], list[ToolBinding]]:
        """Select frozen capabilities, assemble public context, and record injection evidence."""
        skills, selected = self.skills.load(genome, task, observation.public_data, meter)
        memory_entries = self.memory.select(
            genome.memory,
            task,
            observation.public_data,
            max_entries=self.context_limits.max_memory_entries,
            max_tokens=self.context_limits.max_memory_tokens,
        )
        memory = "\n\n".join(e.text for e in memory_entries)
        selected_tools = self.tool_selector.select(
            genome.tools,
            selected,
            task,
            observation.public_data,
            self.tools,
            self.context_limits,
        )
        visible = [t.tool_id for t in selected_tools]
        loaded_capabilities.append(
            {
                "step": step,
                "skills": [
                    {
                        "ref": s.model_dump(mode="json"),
                        "content": self.store.verify(s).decode("utf-8"),
                    }
                    for s in selected
                ],
                "memory": {
                    "ref": genome.memory.model_dump(mode="json"),
                    "content": memory,
                    "entries": [e.model_dump(mode="json") for e in memory_entries],
                },
                "tools": self.tools.schemas_for(selected_tools),
            }
        )
        loaded_capabilities[-1]["public_observation"] = observation.public_data
        context = {
            "task": task.public_view(),
            "public_payloads": [
                {
                    "artifact_id": ref.artifact_id,
                    "content": self.store.verify(ref).decode("utf-8"),
                }
                for ref in task.public_payload_refs
            ],
            "skills": skills,
            "memory": memory,
            "visible_tools": visible,
            "tool_schemas": self.tools.schemas_for(selected_tools),
            "tool_descriptions": {
                tool.tool_id: self.tools.entries[(tool.tool_id, tool.version_id)][0].description
                for tool in selected_tools
            },
            "tool_results": results,
            "observation": observation.public_data,
            "random_seed": plan.random_seed,
            "response_schema": TaskAction.model_json_schema(),
        }
        emit(
            "capability_context",
            step,
            {
                "available_skills": [s.artifact_id for s in genome.skills],
                "selected_skills": [s.artifact_id for s in selected],
                "injected_skills": [s.artifact_id for s in selected],
                "prompt_sha256": digest(encode(redact(context))),
                "usage_note": "injection does not prove semantic skill use",
                "selection_reasons": self.skills.last_selection,
                "memory_entry_ids": [e.entry_id for e in memory_entries],
            },
            skill_refs=[s.artifact_id for s in selected],
            memory_refs=[genome.memory.artifact_id],
            tool_ids=visible,
        )
        return context, selected, selected_tools

    def _archive_run(
        self,
        run: TaskRunRecord,
        genome: TaskGenome,
        task: TaskSpec,
        execution: _TaskExecution,
        public_feedback: str | None,
    ) -> TaskRunRecord:
        """Commit the public evidence package before binding its hash to the run record."""
        run_id, trace_uri = run.run_id, run.trace_ref
        answer, task_result = execution.answer, run.result
        if task_result is None:
            raise HEASError("archiving a run requires its completed task result")
        reason, loaded_capabilities = execution.reason, execution.capabilities
        evaluation = run.evaluation
        plan_partition = run.task_partition
        package = RunEvidencePackage(
            run_id=run_id,
            task_partition=plan_partition,
            task_input=redact(
                {
                    **task.public_view(),
                    "public_payloads": [
                        {
                            "artifact_id": ref.artifact_id,
                            "content": self.store.verify(ref).decode("utf-8"),
                        }
                        for ref in task.public_payload_refs
                    ],
                }
            ),
            genome=genome,
            loaded_capabilities=redact(loaded_capabilities),
            events=self.store.trace(trace_uri),
            final_output=redact(answer),
            result=TaskResult.model_validate(redact(task_result.model_dump(mode="json"))),
            final_artifact_refs=task_result.artifact_refs,
            termination_reason=reason,
            evaluation=evaluation,
            public_feedback=redact(public_feedback),
            evidence_refs=[
                EvidenceRef(ref_type="trace_event", ref_id=e.event_id)
                for e in self.store.trace(trace_uri)
            ],
        )
        self.store.put(f"runs/{run_id}/evidence.json", package)
        run = TaskRunRecord.model_validate(
            {
                **run.model_dump(),
                "evidence_sha256": digest(
                    self.store.path(f"runs/{run_id}/evidence.json").read_bytes()
                ),
            }
        )
        self.store.save_record(f"runs/{run_id}/run.json", run, "task_run", run_id)
        return run
