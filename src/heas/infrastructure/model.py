"""Bounded ModelClient gateway and configurable OpenAI-compatible transport."""

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from openai.types.chat import ChatCompletionMessage
from pydantic import ValidationError

from heas.core.domain import Budget, NativeModelTurn, NativeToolCall, Record, Usage, uid
from heas.core.errors import BudgetExceeded, HEASError, ModelProtocolError
from heas.core.ports import ModelClient
from heas.infrastructure.context import ContextBudgetManager, ContextBuilder
from heas.infrastructure.storage import FileStore, digest, encode, redact

R = TypeVar("R", bound=Record)


@dataclass
class _StructuredOutput(Generic[R]):
    """Retain raw repair evidence separately from the validated record type."""

    data: dict[str, Any] | None = None
    record: R | None = None


class CallBudget:
    """Track a bounded resource scope shared by calls and approved tools."""

    def __init__(self, budget: Budget):
        self.budget = budget
        self.started = time.monotonic()
        self.model_calls = 0
        self.tool_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost = 0.0
        self.usage_available = True
        self.scope_id: str | None = None

    def check(self) -> None:
        """Reject work that cannot fit within the fixed budget or context boundary."""
        if self.remaining_time <= 0:
            raise BudgetExceeded("wall time exceeded")
        if not self.usage_available and any(
            limit is not None
            for limit in (
                self.budget.max_input_tokens,
                self.budget.max_output_tokens,
                self.budget.max_cost,
            )
        ):
            raise BudgetExceeded("cannot enforce token/cost limits without provider usage")
        input_limit = self.budget.max_input_tokens
        output_limit = self.budget.max_output_tokens
        cost_limit = self.budget.max_cost
        if input_limit is not None and self.input_tokens > input_limit:
            raise BudgetExceeded("input token budget exceeded")
        if output_limit is not None and self.output_tokens > output_limit:
            raise BudgetExceeded("output token budget exceeded")
        if cost_limit is not None and self.cost > cost_limit:
            raise BudgetExceeded("monetary budget exceeded")

    @property
    def remaining_time(self) -> float:
        """Return the remaining wall-clock allowance; callers enforce exhaustion."""
        return self.budget.max_wall_seconds - (time.monotonic() - self.started)

    def tool(self) -> None:
        """Reserve an approved tool invocation against every active resource scope."""
        self.check()
        if self.tool_calls >= self.budget.max_tool_calls:
            raise BudgetExceeded("tool call budget exceeded")
        self.tool_calls += 1

    def usage(self) -> Usage:
        """Return measured usage while preserving unknown token or monetary accounting."""
        return Usage(
            model_calls=self.model_calls,
            tool_calls=self.tool_calls,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            wall_seconds=time.monotonic() - self.started,
            token_usage_available=self.usage_available,
            monetary_cost=self.cost if self.budget.pricing and self.usage_available else None,
        )


class ModelGateway:
    """All model invocations, including summaries and failures, use this ledger."""

    def __init__(
        self,
        model: ModelClient,
        store: FileStore,
        models_by_purpose: dict[str, ModelClient] | None = None,
        global_budget: Budget | None = None,
    ):
        self.model, self.store = model, store
        self.models_by_purpose = models_by_purpose or {}
        self.global_meter = CallBudget(global_budget) if global_budget else None
        self.budget_scopes: list[CallBudget] = []
        self.total_call_limit: int | None = None
        self.context_budget = ContextBudgetManager()
        self.structured_repair_limit = 1
        self.audit_mirrors: list[FileStore] = []

    def client_for(self, purpose: str) -> ModelClient:
        """Resolve the host-configured client for a role-specific call purpose."""
        return self.models_by_purpose.get(purpose, self.model)

    def meters(self, local: CallBudget) -> list[CallBudget]:
        """Return distinct nested resource meters so one call is charged once per scope."""
        return list(
            dict.fromkeys(
                [local, *([self.global_meter] if self.global_meter else []), *self.budget_scopes]
            )
        )

    def tool(self, local: CallBudget) -> None:
        """Reserve an approved tool invocation against every active resource scope."""
        for meter in self.meters(local):
            meter.check()
            if meter.tool_calls >= meter.budget.max_tool_calls:
                raise BudgetExceeded("shared tool call budget exceeded")
        for meter in self.meters(local):
            meter.tool()
        # Commit the reservation before host code runs; interrupted calls remain charged.
        uri = f"tool_calls/{uid('tool_receipt')}.json"
        receipt = {
            "schema_version": "1.2",
            "status": "reserved",
            "budget_scope_ids": self.scope_ids(local),
        }
        for store in [self.store, *self.audit_mirrors]:
            store.put(uri, receipt)

    def scope_ids(self, local: CallBudget) -> list[str]:
        """Return stable identities of the active resource scopes for durable accounting."""
        return list(dict.fromkeys(m.scope_id for m in self.meters(local) if m.scope_id))

    def call(
        self,
        purpose: str,
        context: dict[str, Any],
        meter: CallBudget,
        *,
        validator: Callable[[dict[str, Any]], Any] | None = None,
        request_group_id: str | None = None,
        retry_index: int = 0,
    ) -> dict[str, Any]:
        """Validate, meter, and audit one provider request.

        Args:
            purpose: Role-specific model purpose used for client selection.
            context: Public, host-bounded inputs to the provider.
            meter: Local budget charged alongside enclosing scopes.
            validator: Optional fixed validator applied before completion is recorded.
            request_group_id: Identity shared by bounded repair requests.
            retry_index: Zero-based attempt number within the request group.

        Returns:
            Validated public response data.

        Raises:
            BudgetExceeded: A resource limit is exhausted or cannot be enforced.
            ModelProtocolError: The provider returned an invalid envelope or JSON value.
        """
        self.context_budget.check(context)
        if self.total_call_limit is not None:
            if len(list(self.store.path("model_calls").glob("*.json"))) >= self.total_call_limit:
                raise BudgetExceeded("whole-trial model call limit exhausted")
        client = self.client_for(purpose)
        meters = self.meters(meter)
        payload = redact(context)
        remaining = self._output_allowance(meters, payload)
        call_id = uid("call")
        call_started = time.monotonic()
        for current in meters:
            current.model_calls += 1
        receipt: dict[str, Any] = {
            "schema_version": "1.0",
            "purpose": purpose,
            "model_identity": client.identity,
            "model_parameters": redact(getattr(client, "settings", {})),
            "context_sha256": digest(encode(payload)),
            "call_id": call_id,
            "status": "failed",
            "request_group_id": request_group_id or call_id,
            "retry_index": retry_index,
            "budget_scope_ids": self.scope_ids(meter),
        }
        # A process crash must not erase a possibly billed provider invocation.
        for target in [self.store, *self.audit_mirrors]:
            target.put(f"model_requests/{call_id}.json", {**receipt, "status": "reserved"})
        try:
            result = client.complete(
                purpose,
                payload,
                timeout=min(m.remaining_time for m in meters),
                max_output_tokens=remaining,
            )
            if not isinstance(result, dict):
                raise ModelProtocolError("model envelope must be an object")
            self._account_response(result, receipt, meters)
            data = result.get("data")
            if not isinstance(data, dict):
                raise ModelProtocolError("model response data must be a JSON object", data)
            if set(data).intersection({"invalid_json_text", "invalid_json_value"}):
                raise ModelProtocolError("model response contains invalid JSON", data)
            if validator:
                validator(data)
            receipt["status"] = "completed"
            return data
        except BaseException as exc:
            # Audit failures and interruptions, then preserve the exception for
            # its owning boundary. A cancelled request may already have been billed.
            if "usage" not in receipt:
                for current in meters:
                    current.usage_available = False
            receipt["error_type"] = type(exc).__name__
            receipt["status"] = self._failure_status(exc)
            receipt["error_message"] = "Model call failed: " + type(exc).__name__
            raise
        finally:
            receipt["wall_seconds"] = time.monotonic() - call_started
            self.store.put(f"model_calls/{call_id}.json", receipt)
            for mirror in self.audit_mirrors:
                mirror.put(f"model_calls/{call_id}.json", receipt)

    def _output_allowance(self, meters: list[CallBudget], payload: dict[str, Any]) -> int | None:
        """Preflight all scopes before reserving a possibly billed provider request."""
        input_bound = len(encode(payload))
        remaining = None
        for current in meters:
            current.check()
            if current.model_calls >= current.budget.max_model_calls:
                raise BudgetExceeded("model call budget exceeded")
            input_limit = current.budget.max_input_tokens
            output_limit = current.budget.max_output_tokens
            cost_limit = current.budget.max_cost
            if input_limit is not None and current.input_tokens + input_bound > input_limit:
                raise BudgetExceeded("conservative input token precheck exceeded")
            limits = []
            if output_limit is not None:
                limits.append(output_limit - current.output_tokens)
            pricing = current.budget.pricing
            if cost_limit is not None and pricing:
                funds = (
                    cost_limit
                    - current.cost
                    - (input_bound * pricing.input_per_million / 1_000_000)
                )
                if funds < 0:
                    raise BudgetExceeded("estimated input exceeds monetary budget")
                if pricing.output_per_million > 0:
                    limits.append(int(funds * 1_000_000 / pricing.output_per_million))
            if limits:
                cap = min(limits)
                remaining = cap if remaining is None else min(remaining, cap)
        if remaining is not None and remaining <= 0:
            raise BudgetExceeded("output/cost budget exhausted")
        return remaining

    def _account_response(
        self, result: dict[str, Any], receipt: dict[str, Any], meters: list[CallBudget]
    ) -> None:
        """Charge measured usage before validating content, including malformed JSON.

        Unknown usage prevents further calls with hard token or monetary limits.
        Content validation must follow accounting so a rejected response remains
        charged in every enclosing resource scope.
        """
        usage = result.get("usage")
        measured = (
            Usage.model_validate(usage) if usage is not None else Usage(token_usage_available=False)
        )
        receipt.update(
            {
                "usage": measured.model_dump(mode="json"),
                "response_sha256": digest(encode(result.get("data"))),
                "provider_request_id": result.get("provider_request_id"),
                "provider_model": result.get("provider_model"),
                "request_parameters": result.get("request_parameters", {}),
            }
        )
        for current in meters:
            current.input_tokens += measured.input_tokens
            current.output_tokens += measured.output_tokens
            current.usage_available &= measured.token_usage_available
            pricing = current.budget.pricing
            if pricing:
                current.cost += (
                    measured.input_tokens * pricing.input_per_million
                    + measured.output_tokens * pricing.output_per_million
                ) / 1_000_000
        pricing = next((m.budget.pricing for m in meters if m.budget.pricing), None)
        receipt["monetary_cost"] = (
            (
                measured.input_tokens * pricing.input_per_million
                + measured.output_tokens * pricing.output_per_million
            )
            / 1_000_000
            if pricing and measured.token_usage_available
            else None
        )
        receipt["currency"] = pricing.currency if pricing else None
        receipt["pricing"] = pricing.model_dump(mode="json") if pricing else None
        for current in meters:
            if not measured.token_usage_available and any(
                limit is not None
                for limit in (
                    current.budget.max_input_tokens,
                    current.budget.max_output_tokens,
                    current.budget.max_cost,
                )
            ):
                raise BudgetExceeded("cannot enforce token/cost limits without provider usage")
            current.check()

    @staticmethod
    def _failure_status(error: BaseException) -> str:
        """Classify a failure without exposing provider or private evaluation text."""
        if isinstance(error, ValidationError):
            return "validation_failed"
        if isinstance(error, ModelProtocolError):
            return "protocol_error"
        if isinstance(error, BudgetExceeded):
            return "budget_exceeded"
        category = type(error).__name__.lower()
        if isinstance(error, TimeoutError) or "timeout" in category:
            return "timeout"
        if "ratelimit" in category:
            return "rate_limited"
        if isinstance(error, ConnectionError) or "connection" in category:
            return "network_error"
        return "rejected" if isinstance(error, HEASError) else "failed"

    def structured(
        self,
        purpose: str,
        context: dict[str, Any],
        meter: CallBudget,
        cls: type[R],
        validate: Callable[[R], Any] | None = None,
    ) -> R:
        """Validate a concrete record with bounded, fully metered repair attempts.

        Args:
            purpose: Role-specific provider selection and audit label.
            context: Public evidence and instructions retained across repair attempts.
            meter: Shared resource scope charged for invalid outputs as well.
            cls: Record type whose schema governs every attempt.
            validate: Fixed Host check for evidence scope or other admission rules.

        Returns:
            An instance of the requested record type after all checks pass.

        Raises:
            ValidationError: Schema validation still fails after bounded repair.
            ModelProtocolError: Invalid JSON remains after bounded repair.
            HEASError: A fixed validator rejects the output; this is not repairable here.
        """
        request_id = uid("structured")
        original = {**context, "response_schema": cls.model_json_schema()}
        request = original
        for index in range(self.structured_repair_limit + 1):
            checked = _StructuredOutput[R]()

            def check(
                response_data: dict[str, Any], validation_state: _StructuredOutput[R] = checked
            ) -> None:
                """Validate the typed output and its evidence before accepting the receipt."""
                validation_state.data = response_data
                validated_record = cls.model_validate(response_data, strict=True)
                if validate:
                    validate(validated_record)
                validation_state.record = validated_record

            try:
                self.call(
                    purpose,
                    request,
                    meter,
                    validator=check,
                    request_group_id=request_id,
                    retry_index=index,
                )
                record = checked.record
                if record is None:
                    raise HEASError("model validator did not produce a typed record")
                if index:
                    self.store.put(
                        f"structured_calls/{request_id}/result.json",
                        {
                            "purpose": purpose,
                            "status": "repaired",
                            "repair_count": index,
                            "resource_usage": meter.usage().model_dump(mode="json"),
                        },
                    )
                return record
            except (ValidationError, ModelProtocolError) as exc:
                data: object = checked.data
                if data is None:
                    data = exc.data if isinstance(exc, ModelProtocolError) else {}
                errors = (
                    [
                        {"loc": list(e["loc"]), "type": e["type"]}
                        for e in exc.errors(include_input=False, include_url=False)
                    ]
                    if isinstance(exc, ValidationError)
                    else [{"loc": [], "type": "invalid_json"}]
                )
                self.store.put(
                    f"structured_calls/{request_id}/{index:06d}.json",
                    {
                        "purpose": purpose,
                        "status": "validation_failed",
                        "errors": errors,
                        "output": redact(data),
                        "repair_index": index,
                        "resource_usage": meter.usage().model_dump(mode="json"),
                    },
                )
                if index == self.structured_repair_limit:
                    raise
                # Archive full invalid output; bound the repair excerpt.
                request = {
                    **original,
                    "repair": {
                        "instruction": "Repair the previous output against the SAME schema. "
                        "All original safety and evidence constraints remain mandatory.",
                        "errors": errors[:12],
                        "previous_output_excerpt": encode(redact(data))[:1024].decode(
                            "utf-8", errors="replace"
                        ),
                    },
                }
            except HEASError as exc:
                self.store.put(
                    f"structured_calls/{request_id}/{index:06d}.json",
                    {
                        "purpose": purpose,
                        "status": "rejected_by_fixed_validator",
                        "error_type": type(exc).__name__,
                        "output": redact(checked.data),
                        "resource_usage": meter.usage().model_dump(mode="json"),
                    },
                )
                raise
        raise HEASError("unreachable structured recovery state")


class DeepSeekClient:
    """Configurable OpenAI-compatible JSON transport; no API keys in records."""

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0,
        tool_protocol: str = "json",
        supports_seed: bool = False,
        request_parameters: dict[str, Any] | None = None,
    ):
        from openai import OpenAI

        configured_model = model or os.getenv("HEAS_MODEL", "")
        key = api_key or os.getenv("HEAS_API_KEY", "")
        endpoint = base_url or os.getenv("HEAS_BASE_URL", "https://api.deepseek.com")
        if not configured_model or not key:
            raise HEASError("HEAS_MODEL and HEAS_API_KEY must be configured")
        self.model: str = configured_model
        self.identity = f"openai-compatible/{self.model}"
        if tool_protocol not in {"json", "native"}:
            raise HEASError("tool_protocol must be json or native")
        self.tool_protocol = tool_protocol
        self.supports_seed = supports_seed
        self.request_parameters = request_parameters or {}
        if not set(self.request_parameters).issubset(
            {"top_p", "frequency_penalty", "presence_penalty"}
        ):
            raise HEASError("unsupported model request parameter")
        self.settings: dict[str, Any] = {
            "model": self.model,
            "base_url": endpoint,
            "temperature": temperature,
            "max_retries": 0,
            "tool_protocol": tool_protocol,
            "supports_seed": supports_seed,
            "request_parameters": self.request_parameters,
            "determinism": "not_guaranteed",
        }
        self.client = OpenAI(api_key=key, base_url=endpoint, max_retries=0)
        self._key = key

    def fork(self) -> "DeepSeekClient":
        """Rebuild a provider client without sharing mutable session state across branches."""
        return DeepSeekClient(
            model=self.model,
            base_url=self.settings["base_url"],
            api_key=self._key,
            temperature=self.settings["temperature"],
            tool_protocol=self.tool_protocol,
            supports_seed=self.supports_seed,
            request_parameters=dict(self.request_parameters),
        )

    def complete(
        self,
        purpose: str,
        context: dict[str, Any],
        *,
        timeout: float,
        max_output_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Make one SDK request and retain usage even when response content is invalid.

        Transport exceptions propagate to ModelGateway for classification. Content
        failures return reserved protocol markers so billing evidence reaches the
        ledger before rejection. There are no hidden SDK retries.

        Args:
            purpose: Role label selecting JSON or native task protocol.
            context: Host-built public messages and approved tool schemas.
            timeout: Remaining wall-clock allowance for this request.
            max_output_tokens: Optional output cap computed by the gateway.

        Returns:
            A provider envelope containing decoded data, usage and request metadata.
        """
        kwargs: dict[str, Any] = {}
        if max_output_tokens is not None:
            kwargs["max_tokens"] = max_output_tokens
        kwargs.update(
            {
                "model": self.model,
                "timeout": max(0.001, timeout),
                "temperature": self.settings["temperature"],
                "response_format": {"type": "json_object"},
                "messages": ContextBuilder().messages(purpose, context),
            }
        )
        kwargs.update(self.request_parameters)
        if self.supports_seed and "random_seed" in context:
            kwargs["seed"] = context["random_seed"]
        if purpose == "task" and self.tool_protocol == "native":
            kwargs.pop("response_format", None)
            kwargs["messages"] = ContextBuilder().messages(purpose, context, native=True)
            schemas = context.get("tool_schemas", {})
            if schemas:
                kwargs["tools"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": context.get("tool_descriptions", {}).get(name, ""),
                            "parameters": schema,
                        },
                    }
                    for name, schema in schemas.items()
                ]
        response = self.client.chat.completions.create(**kwargs)
        usage = response.usage
        # A malformed completion may still be billed. Return a protocol marker
        # with its usage so the gateway can charge and archive it before rejection.
        message = response.choices[0].message if response.choices else None
        data = (
            self._response_data(
                message, native=purpose == "task" and self.tool_protocol == "native"
            )
            if message is not None
            else {"invalid_json_value": None}
        )
        return {
            "data": data,
            "usage": {
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
            }
            if usage is not None
            else None,
            "provider_request_id": response.id if isinstance(response.id, str) else None,
            "provider_model": response.model if isinstance(response.model, str) else self.model,
            "request_parameters": {
                k: kwargs[k]
                for k in ("model", "temperature", "seed", "max_tokens", "top_p")
                if k in kwargs
            },
        }

    @staticmethod
    def _response_data(message: ChatCompletionMessage, *, native: bool) -> dict[str, Any]:
        """Decode SDK content while preserving rejected data for the gateway ledger."""
        if not native:
            content = message.content
            try:
                value = json.loads(content) if content is not None else None
            except (ValueError, TypeError):
                return {"invalid_json_text": content}
            return value if isinstance(value, dict) else {"invalid_json_value": value}
        calls = []
        for call in message.tool_calls or []:
            function = getattr(call, "function", None)
            raw_arguments = getattr(function, "arguments", None)
            tool_id = getattr(function, "name", None)
            call_id = call.id
            if not isinstance(call_id, str) or not call_id:
                return {"invalid_json_value": None}
            if not isinstance(tool_id, str) or not tool_id:
                return {"invalid_json_value": None}
            if not isinstance(raw_arguments, str):
                return {"invalid_json_value": None}
            try:
                arguments = json.loads(raw_arguments)
                if not isinstance(arguments, dict):
                    arguments = None
            except (ValueError, TypeError):
                arguments = None
            try:
                calls.append(
                    NativeToolCall(
                        call_id=call_id,
                        tool_id=tool_id,
                        arguments=arguments,
                        raw_arguments=raw_arguments,
                    )
                )
            except ValidationError:
                return {"invalid_json_value": None}
        try:
            return NativeModelTurn(content=message.content, tool_calls=calls).model_dump(
                mode="json"
            )
        except ValidationError:
            return {"invalid_json_value": None}
