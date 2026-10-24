"""Bounded ModelClient gateway and configurable OpenAI-compatible transport."""

import json
import os
import time
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import ValidationError

from heas.core.domain import Budget, NativeModelTurn, NativeToolCall, Record, Usage, uid
from heas.core.errors import BudgetExceeded, HEASError, ModelProtocolError
from heas.core.ports import ModelClient
from heas.infrastructure.context import ContextBudgetManager, ContextBuilder
from heas.infrastructure.storage import FileStore, digest, encode, redact

R = TypeVar("R", bound=Record)


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
        if self.budget.max_input_tokens is not None:
            if self.input_tokens > self.budget.max_input_tokens:
                raise BudgetExceeded("input token budget exceeded")
        if self.budget.max_output_tokens is not None:
            if self.output_tokens > self.budget.max_output_tokens:
                raise BudgetExceeded("output token budget exceeded")
        if self.budget.max_cost is not None and self.cost > self.budget.max_cost:
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
        input_bound = len(encode(payload))
        remaining = None
        for current in meters:
            current.check()
            if current.model_calls >= current.budget.max_model_calls:
                raise BudgetExceeded("model call budget exceeded")
            if (
                current.budget.max_input_tokens is not None
                and current.input_tokens + input_bound > current.budget.max_input_tokens
            ):
                raise BudgetExceeded("conservative input token precheck exceeded")
            limits = []
            if current.budget.max_output_tokens is not None:
                limits.append(current.budget.max_output_tokens - current.output_tokens)
            pricing = current.budget.pricing
            if current.budget.max_cost is not None and pricing:
                funds = (
                    current.budget.max_cost
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
            usage = result.get("usage")
            measured = (
                Usage.model_validate(usage)
                if usage is not None
                else Usage(token_usage_available=False)
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
            data = result.get("data")
            if not isinstance(data, dict):
                raise ModelProtocolError("model response data must be a JSON object", data)
            if set(data).intersection({"invalid_json_text", "invalid_json_value"}):
                raise ModelProtocolError("model response contains invalid JSON", data)
            if validator:
                validator(data)
            receipt["status"] = "completed"
            return data
        except Exception as exc:
            if "usage" not in receipt:
                for current in meters:
                    current.usage_available = False
            receipt["error_type"] = type(exc).__name__
            category = type(exc).__name__.lower()
            receipt["status"] = (
                "validation_failed"
                if isinstance(exc, ValidationError)
                else "protocol_error"
                if isinstance(exc, ModelProtocolError)
                else "budget_exceeded"
                if isinstance(exc, BudgetExceeded)
                else "timeout"
                if isinstance(exc, TimeoutError) or "timeout" in category
                else "rate_limited"
                if "ratelimit" in category
                else "network_error"
                if isinstance(exc, ConnectionError) or "connection" in category
                else "rejected"
                if isinstance(exc, HEASError)
                else "failed"
            )
            receipt["error_message"] = "Model call failed: " + type(exc).__name__
            raise
        finally:
            receipt["wall_seconds"] = time.monotonic() - call_started
            self.store.put(f"model_calls/{call_id}.json", receipt)
            for mirror in self.audit_mirrors:
                mirror.put(f"model_calls/{call_id}.json", receipt)

    def structured(
        self,
        purpose: str,
        context: dict[str, Any],
        meter: CallBudget,
        cls: type[R],
        validate: Callable[[R], Any] | None = None,
    ) -> R:
        """Validate typed model output with bounded, fully metered repair attempts."""
        request_id = uid("structured")
        original = {**context, "response_schema": cls.model_json_schema()}
        request = original
        for index in range(self.structured_repair_limit + 1):
            checked: dict[str, Any] = {}

            def check(data: dict[str, Any], checked: dict[str, Any] = checked) -> None:
                """Validate the typed output and its evidence before accepting the receipt."""
                checked["data"] = data
                record = cls.model_validate(data, strict=True)
                if validate:
                    validate(record)
                checked["record"] = record

            try:
                self.call(
                    purpose,
                    request,
                    meter,
                    validator=check,
                    request_group_id=request_id,
                    retry_index=index,
                )
                record = checked["record"]
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
                data = checked.get("data", exc.data if isinstance(exc, ModelProtocolError) else {})
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
                        "output": redact(checked.get("data")),
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
        # One request only; controller records failures and enforces total call/time limits.
        """Perform one provider transport request and return its response with usage evidence."""
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
        message = response.choices[0].message
        if purpose == "task" and self.tool_protocol == "native":
            calls = []
            for call in message.tool_calls or []:
                try:
                    args = json.loads(call.function.arguments)
                    if not isinstance(args, dict):
                        args = None
                except (ValueError, TypeError):
                    args = None
                calls.append(
                    NativeToolCall(
                        call_id=call.id,
                        tool_id=call.function.name,
                        arguments=args,
                        raw_arguments=call.function.arguments,
                    )
                )
            data = NativeModelTurn(content=message.content, tool_calls=calls).model_dump(
                mode="json"
            )
        else:
            try:
                data = json.loads(message.content or "{}")
            except (ValueError, TypeError):
                data = {"invalid_json_text": message.content or ""}
            if not isinstance(data, dict):
                data = {"invalid_json_value": data}
        return {
            "data": data,
            "usage": {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }
            if response.usage
            else None,
            "provider_request_id": response.id if isinstance(response.id, str) else None,
            "provider_model": response.model if isinstance(response.model, str) else self.model,
            "request_parameters": {
                k: kwargs[k]
                for k in ("model", "temperature", "seed", "max_tokens", "top_p")
                if k in kwargs
            },
        }
