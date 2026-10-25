"""Bounded hierarchical semantic aggregation; complete raw evidence stays on disk."""

from collections.abc import Callable
from typing import Any

from pydantic import Field

from heas.core.domain import Budget, EvidenceRef, Record, uid
from heas.core.errors import BudgetExceeded
from heas.infrastructure.model import CallBudget, ModelGateway
from heas.infrastructure.storage import FileStore, encode


class AggregationConfig(Record, frozen=True):
    """Bound hierarchy size so experience reports retain evidence within context limits."""

    batch_size: int = Field(default=4, ge=2, le=32)
    max_context_bytes: int = Field(default=32000, ge=8192)
    max_excerpt_chars: int = Field(default=768, ge=64)
    max_evidence_refs: int = Field(default=16, ge=1)
    representative_events: int = Field(default=2, ge=1, le=16)


def bounded(value: Any, chars: int = 768, items: int = 8) -> Any:
    """Bound nested diagnostic content before constructing model-facing evidence packets."""
    if isinstance(value, str):
        return value[:chars]
    if isinstance(value, list):
        return [bounded(v, chars, items) for v in value[:items]]
    if isinstance(value, dict):
        return {k: bounded(v, chars, items) for k, v in list(value.items())[:24]}
    return value


class HierarchicalExperienceAggregator:
    """Compress large evidence windows through a traced hierarchy of bounded model calls."""

    def __init__(
        self, store: FileStore, gateway: ModelGateway, config: AggregationConfig | None = None
    ):
        self.store, self.gateway = store, gateway
        self.config = config or AggregationConfig()

    def aggregate(
        self,
        purpose: str,
        rows: list[dict[str, Any]],
        stats: dict[str, Any],
        cls: type[Record],
        budget: Budget,
        validate: Callable[[Record, set[tuple[str, str]]], None],
        allowed: set[tuple[str, str]],
        meter: CallBudget | None = None,
    ) -> tuple[Record, str]:
        """Aggregate bounded evidence while retaining traced source references and uncertainty."""
        config = self.config
        context_limit = min(config.max_context_bytes, self.gateway.context_budget.max_tokens - 2048)
        aggregation_id = uid("aggregation")
        meter = meter or CallBudget(budget)
        nodes: list[dict[str, Any]] = []
        level = 0
        current = rows
        while True:
            outputs = []
            for start in range(0, len(current), config.batch_size):
                group = current[start : start + config.batch_size]
                refs = [row["source_ref"] for row in group]
                context = {
                    "objective_stats": bounded(stats, config.max_excerpt_chars, 12),
                    "evidence_refs": refs[: config.max_evidence_refs],
                    "aggregation_level": level,
                    "analysis_rules": "Statistics are controller observations. Analyze recurring "
                    "patterns, capability gaps/interactions and testable evolution hypotheses. "
                    "Retain evidence and uncertainty. Injection is not verified contribution; "
                    "execution/materialization is not performance improvement. Input excerpts "
                    "are sampled, not exhaustive. Original evidence remains addressable.",
                    "records": bounded(group, config.max_excerpt_chars, 6),
                    "response_schema": cls.model_json_schema(),
                }
                if purpose == "task_reflection":
                    raise ValueError("single-run reflection uses its own evidence contract")
                for chars in (384, 192, 96):
                    if len(encode(context)) <= context_limit:
                        break
                    context["records"] = bounded(group, chars, 2)
                if len(encode(context)) > context_limit:
                    context["records"] = [{"source_ref": row["source_ref"]} for row in group]
                size = len(encode(context))
                if size > context_limit:
                    raise BudgetExceeded(
                        "aggregation schema/evidence exceeds configured context cap"
                    )
                analysis = self.gateway.structured(
                    purpose, context, meter, cls, lambda report: validate(report, allowed)
                )
                chunk_id = uid("summary_chunk")
                uri = f"analysis/{aggregation_id}/{chunk_id}.json"
                self.store.put(
                    uri,
                    {
                        "schema_version": "1.2",
                        "chunk_id": chunk_id,
                        "source_refs": refs,
                        "analysis": analysis.model_dump(mode="json"),
                        "level": level,
                        "input_bytes": size,
                    },
                )
                self.store.index("summary", chunk_id, uri)
                allowed.add(("summary", chunk_id))
                node = {
                    "source_ref": EvidenceRef(ref_type="summary", ref_id=chunk_id).model_dump(
                        mode="json"
                    ),
                    "analysis": analysis.model_dump(mode="json"),
                }
                outputs.append((node, analysis))
                nodes.append(
                    {
                        "chunk_id": chunk_id,
                        "uri": uri,
                        "level": level,
                        "input_bytes": size,
                        "source_refs": refs,
                    }
                )
            if len(outputs) == 1:
                manifest_uri = f"analysis/{aggregation_id}/manifest.json"
                self.store.put(
                    manifest_uri,
                    {
                        "schema_version": "1.2",
                        "nodes": nodes,
                        "config": config.model_dump(mode="json"),
                        "source_records": len(rows),
                        "resource_usage": meter.usage().model_dump(mode="json"),
                    },
                )
                return outputs[0][1], manifest_uri
            current = [summary_node for summary_node, _ in outputs]
            level += 1
