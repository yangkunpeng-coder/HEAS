"""Replaceable archive parent selection, independent of candidate storage or promotion."""

from typing import Any, Protocol

from heas.core.domain import (
    EvidenceRef as EvidenceRef,
)
from heas.core.domain import (
    EvolutionPolicy,
    ParentSelectionConfig,
    TaskExperienceSummary,
)
from heas.core.errors import SecurityError
from heas.infrastructure.storage import FileStore


class ParentSelectionPolicy(Protocol):
    """Define the replaceable host interface for evidence-covered parent selection."""

    version: str

    def select(
        self,
        store: FileStore,
        summaries: list[TaskExperienceSummary],
        policy: EvolutionPolicy,
    ) -> tuple[list[TaskExperienceSummary], dict[str, Any]]:
        """Select admissible capabilities or parents and retain the reasons for the choice."""
        ...


class ArchiveParentSelection:
    """Rank compatible evidence while retaining capability diversity and uncertainty."""

    version = "archive-evidence-diversity-v1"

    def select(
        self,
        store: FileStore,
        summaries: list[TaskExperienceSummary],
        policy: EvolutionPolicy,
    ) -> tuple[list[TaskExperienceSummary], dict[str, Any]]:
        """Select admissible capabilities or parents and retain the reasons for the choice."""
        from heas.agents.evolution.candidates import (
            CandidateRegistry,
            archived_runs,
            compare_windows,
            comparison_protocol,
            evaluated_metrics,
        )

        config = policy.parent_selection or ParentSelectionConfig()
        latest: dict[str, TaskExperienceSummary] = {}
        for summary in summaries:
            if (
                summary.genome_id not in latest
                or latest[summary.genome_id].created_at < summary.created_at
            ):
                latest[summary.genome_id] = summary
        candidates: list[dict[str, Any]] = []
        registry = CandidateRegistry(store)
        for genome_id, summary in latest.items():
            genome = store.genome(genome_id)
            features = {"skill:" + s.content_sha256 for s in genome.skills}
            features.add("memory:" + genome.memory.content_sha256)
            features.update("tool:" + t.tool_id + "@" + t.version_id for t in genome.tools)
            runs = archived_runs(store, summary.run_ids)
            if any(r.task_partition == "final_test" for r in runs):
                raise SecurityError("final_test evidence cannot influence parent selection")
            measurements = [evaluated_metrics(store, r) for r in runs]
            measurements = [
                None
                if m is not None
                and (
                    m[2] is False
                    or (
                        r.termination_reason in {"model_error", "environment_error", "cancelled"}
                        and m[2] is not True
                    )
                )
                else m
                for r, m in zip(runs, measurements, strict=True)
            ]
            primary = [r.evaluation.score for r in runs]
            current_protocol = comparison_protocol(store, runs)
            comparable_score: float | None = None
            contract: str | None = None
            if all(m is not None for m in measurements) and all(s is not None for s in primary):
                scores = [s for s in primary if s is not None]
                if len({(s.source, s.name, s.unit, s.direction) for s in scores}) == 1:
                    contract = current_protocol
                    comparable_score = sum(s.value for s in scores) / len(scores)
                    if scores[0].direction == "lower_is_better":
                        comparable_score = -comparable_score
            gain: float | None = None
            gain_status = "not_available"
            comparison_ref: str | None = None
            try:
                record = registry.get(genome_id)
                comparison_uri = record.comparison_ref
                comparison_ref = comparison_uri
                if comparison_uri is not None:
                    comparison = store.resolve(
                        store.json_record_ref(comparison_uri, "candidate_comparison")
                    )
                    previous_runs = archived_runs(store, comparison["new_run_ids"])
                    if comparison_protocol(store, previous_runs) == current_protocol:
                        checked = compare_windows(
                            store, comparison["old_run_ids"], comparison["new_run_ids"]
                        )
                        if checked["evaluation_status"] == "completed":
                            gain, gain_status = checked["improvement"], "compatible"
                    else:
                        gain_status = "incompatible_background_only"
            except ValueError:
                pass
            candidates.append(
                {
                    "summary": summary,
                    "features": features,
                    "score": comparable_score,
                    "contract": contract,
                    "gain": gain,
                    "gain_status": gain_status,
                    "comparison_ref": comparison_ref,
                }
            )
        selected: list[TaskExperienceSummary] = []
        ranking: list[dict[str, Any]] = []
        selected_features: set[str] = set()
        count = config.max_parents if policy.parent_selection else min(8, len(candidates))
        while candidates and len(selected) < count:

            def rank(parent_candidate: dict[str, Any]) -> float:
                peers = [
                    r
                    for r in candidates
                    if r["contract"] == parent_candidate["contract"] and r["score"] is not None
                ]
                # Rank only within compatible metric protocols; never mix raw score scales.
                percentile = (
                    sum(p["score"] < parent_candidate["score"] for p in peers)
                    / max(1, len(peers) - 1)
                    if parent_candidate["score"] is not None
                    else 0
                )
                novelty = len(parent_candidate["features"] - selected_features) / max(
                    1, len(parent_candidate["features"])
                )
                return (
                    config.performance_weight * percentile
                    + config.diversity_weight * novelty
                    + config.historical_gain_weight
                    * int(parent_candidate["gain"] is not None and parent_candidate["gain"] > 0)
                    + config.exploration_weight * int(parent_candidate["score"] is None)
                )

            selected_candidate = max(candidates, key=rank)
            ranking.append(
                {
                    "genome_id": selected_candidate["summary"].genome_id,
                    "selection_utility": rank(selected_candidate),
                    "objective_score_available": selected_candidate["score"] is not None,
                    "historical_gain": selected_candidate["gain"],
                    "historical_gain_status": selected_candidate["gain_status"],
                    "comparison_ref": selected_candidate["comparison_ref"],
                    "reason": "Compatible performance rank, capability novelty, historical "
                    "comparison evidence and exploration; utility is not objective fitness.",
                    "evidence_refs": [
                        {"ref_type": "summary", "ref_id": selected_candidate["summary"].summary_id}
                    ],
                }
            )
            selected.append(selected_candidate["summary"])
            selected_features.update(selected_candidate["features"])
            candidates.remove(selected_candidate)
        return selected, {
            "selector_version": self.version,
            "policy_id": policy.policy_id,
            "config": config.model_dump(mode="json"),
            "ranking": ranking,
            "source_summary_ids": [s.summary_id for s in summaries],
            "quality_status": "selection_is_not_promotion",
        }
