"""Evidence-first context assembly reusing the existing hierarchical aggregator."""

from typing import Any

from heas.core.domain import Claim, EvidenceRef, Record, uid
from heas.evidence.aggregation import AggregationConfig, HierarchicalExperienceAggregator, bounded
from heas.evidence.experience import validate_evidence
from heas.infrastructure.context import EvidenceRetriever
from heas.infrastructure.model import CallBudget, ModelGateway
from heas.infrastructure.storage import FileStore, encode


class HistoryAnalysis(Record, frozen=True):
    """Validate bounded historical evidence reports used by upper-level decisions."""

    highlights: list[Claim]
    failure_refs: list[EvidenceRef]
    capability_change_refs: list[EvidenceRef]
    limitations: list[str]


class HistoryContext:
    """Provide bounded historical evidence without dropping source references."""

    def __init__(self, store: FileStore, gateway: ModelGateway):
        self.store, self.gateway = store, gateway
        self.retriever = EvidenceRetriever(store)

    def prepare(
        self,
        purpose: str,
        context: dict[str, Any],
        rows: list[dict[str, Any]],
        meter: CallBudget,
    ) -> dict[str, Any]:
        """Prepare the bounded public context required by the next execution stage."""
        manifest_id = uid("context")
        uri = f"contexts/{manifest_id}.json"
        source_refs = [row["source_ref"] for row in rows]
        request = {**context, "supporting_evidence": rows, "context_manifest_ref": uri}
        limit = self.gateway.context_budget.max_tokens - 2048
        aggregation_ref = None
        if len(encode(request)) > limit and rows:
            scope = {(r["ref_type"], r["ref_id"]) for r in source_refs}
            analysis, aggregation_ref = HierarchicalExperienceAggregator(
                self.store, self.gateway, AggregationConfig(max_context_bytes=max(8192, limit))
            ).aggregate(
                "evidence_summary",
                rows,
                {
                    "source_records": len(rows),
                    "failure_records": sum(bool(r.get("failure")) for r in rows),
                },
                HistoryAnalysis,
                meter.budget,
                lambda record, allowed: validate_evidence(self.store, record, allowed),
                scope,
                meter=meter,
            )
            request["supporting_evidence"] = {
                "hierarchical_summary": bounded(analysis.model_dump(mode="json"), 384, 2),
                "aggregation_ref": aggregation_ref,
                "full_source_count": len(rows),
            }
        base_snapshot_ref = None
        if len(encode(request)) > limit:
            base_snapshot_ref = f"contexts/{manifest_id}_source.json"
            self.store.put(base_snapshot_ref, context)
            if "summaries" in request:
                request["summaries"] = [
                    {
                        k: bounded(s[k], 192, 1)
                        for k in (
                            "summary_id",
                            "genome_id",
                            "objective_stats_ref",
                            "feedback_coverage",
                            "weaknesses",
                            "evolution_hypotheses",
                        )
                        if k in s
                    }
                    for s in request["summaries"]
                ]
            if "parents" in request:
                request["parents"] = [
                    {
                        "genome_id": p["genome_id"],
                        "skills": [
                            {k: skill[k] for k in ("artifact_id", "requires_tools") if k in skill}
                            for skill in p["skills"]
                        ],
                        "memory": p["memory"],
                        "tools": [
                            {"tool_id": t["tool_id"], "version_id": t["version_id"]}
                            for t in p["tools"]
                        ],
                    }
                    for p in request["parents"]
                ]
            if "parent_capabilities" in request:
                request["parent_capabilities"] = bounded(request["parent_capabilities"], 192, 1)
            if "summary" in request:
                request["summary"] = bounded(request["summary"], 256, 1)
            request["full_context_ref"] = base_snapshot_ref
        if len(encode(request)) > limit and aggregation_ref:
            # The hierarchy retains every source; keep a failure/change example in the final view.
            critical = next((r for r in rows if r.get("failure")), rows[0])
            request["supporting_evidence"] = {
                "aggregation_ref": aggregation_ref,
                "full_source_count": len(rows),
                "critical_example": bounded(critical, 128, 1),
            }
        self.store.put(
            uri,
            {
                "schema_version": "1.2",
                "purpose": purpose,
                "source_refs": source_refs,
                "aggregation_ref": aggregation_ref,
                "base_snapshot_ref": base_snapshot_ref,
                "input_bytes": len(encode(request)),
                "source_records": len(rows),
                "sampling_limits": "Excerpts are not exhaustive; retrieve full records by ID.",
            },
        )
        self.gateway.context_budget.check(request)
        return request
