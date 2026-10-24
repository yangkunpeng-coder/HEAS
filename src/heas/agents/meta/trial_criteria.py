"""Host-owned conservative paired-trial rules; never editable by Meta-Evolution."""

import math
import random
import statistics
from typing import Any, Literal

from pydantic import Field

from heas.core.domain import Record


class PolicyTrialCriteria(Record):
    """Define fixed statistical and cost gates outside the meta-evolution search space."""

    version: str = "paired-conservative-v1"
    min_practical_improvement: float = Field(default=0.01, ge=0)
    min_independent_repeats: int = Field(default=2, ge=1)
    min_paired_samples: int = Field(default=6, ge=1)
    confidence_z: float = Field(default=1.96, ge=0)
    min_positive_repeat_fraction: float = Field(default=1, ge=0, le=1)
    max_relative_cost_increase: float = Field(default=0.25, ge=0)
    cost_metric: Literal["model_calls", "input_tokens", "monetary_cost", "wall_seconds"] = (
        "model_calls"
    )
    interval_method: Literal["normal", "paired_bootstrap"] = "normal"
    bootstrap_samples: int = Field(default=1000, ge=100, le=100000)
    bootstrap_seed: int = 23
    confidence_level: float = Field(default=0.95, gt=0, lt=1)


def assess_trial(
    comparisons: list[dict[str, Any]],
    sides: list[dict[str, Any]],
    criteria: PolicyTrialCriteria,
) -> dict[str, Any]:
    """Apply conservative paired-sample and cost gates fixed by the host."""
    result: dict[str, Any] = {
        "decision": "undecided",
        "criteria": criteria.model_dump(mode="json"),
        "mean_difference": None,
        "sample_count": 0,
        "independent_repeats": 0,
        "standard_error": None,
        "interval": None,
        "cost_change": None,
        "reasons": [],
        "uncertainty_method": "Normal approximation on independent repetition/seed means; "
        "small samples remain uncertain and this is not a research significance claim.",
    }
    if len(sides) != 2 or any(s.get("status") != "completed" for s in sides):
        result["reasons"].append("incomplete_trial")
        return result
    if any(s.get("resource_usage", {}).get("external_usage_unknown") for s in sides):
        result["reasons"].append("unmetered_external_evaluation_cost")
        return result
    if criteria.cost_metric == "wall_seconds" and any(
        s.get("resource_usage", {}).get("wall_seconds_overlap_possible") is not False for s in sides
    ):
        result["reasons"].append("uncertain_legacy_wall_accounting")
        return result
    if not comparisons or any(c["evaluation_status"] != "completed" for c in comparisons):
        result["reasons"].append("missing_independent_evaluation")
        return result
    grouped: dict[tuple[int, int], list[float]] = {}
    rounds = sides[0]["rounds"]
    if len(rounds) != len(comparisons):
        result["reasons"].append("incomplete_paired_windows")
        return result
    for comparison, row, paired_row in zip(comparisons, rounds, sides[1]["rounds"], strict=True):
        key = (row["repetition"], row["random_seed"])
        if key != (paired_row["repetition"], paired_row["random_seed"]):
            result["reasons"].append("unmatched_repetitions")
            return result
        values = comparison.get("paired_differences", [])
        if not values:
            result["reasons"].append("no_scalar_primary_metric")
            return result
        grouped.setdefault(key, []).extend(values)
    samples = [value for values in grouped.values() for value in values]
    repeat_means = [statistics.mean(values) for values in grouped.values()]
    mean = statistics.mean(repeat_means)
    se = (
        statistics.stdev(repeat_means) / math.sqrt(len(repeat_means))
        if len(repeat_means) > 1
        else None
    )
    interval = (
        [mean - criteria.confidence_z * se, mean + criteria.confidence_z * se]
        if se is not None
        else None
    )
    if criteria.interval_method == "paired_bootstrap" and len(repeat_means) > 1:
        rng = random.Random(criteria.bootstrap_seed)
        sampled = sorted(
            statistics.mean(rng.choices(repeat_means, k=len(repeat_means)))
            for _ in range(criteria.bootstrap_samples)
        )
        tail = (1 - criteria.confidence_level) / 2
        interval = [
            sampled[int(tail * (len(sampled) - 1))],
            sampled[int((1 - tail) * (len(sampled) - 1))],
        ]
        result["uncertainty_method"] = "Paired bootstrap of independent repetition/seed means."
    result.update(
        {
            "sample_count": len(samples),
            "independent_repeats": len(grouped),
            "mean_difference": mean,
            "repeat_means": repeat_means,
            "paired_differences": samples,
            "standard_error": se,
            "interval": interval,
        }
    )
    before = sides[0]["resource_usage"].get(criteria.cost_metric)
    after = sides[1]["resource_usage"].get(criteria.cost_metric)
    if before is None or after is None or (before == 0 and after > 0):
        result["reasons"].append("unknown_cost_change")
    else:
        cost = (after - before) / before if before else 0
        result["cost_change"] = {
            "metric": criteria.cost_metric,
            "old": before,
            "new": after,
            "relative_increase": cost,
        }
        if cost > criteria.max_relative_cost_increase:
            result["reasons"].append("cost_increase_exceeds_rule")
    if (
        len(grouped) < criteria.min_independent_repeats
        or len(samples) < criteria.min_paired_samples
    ):
        result["reasons"].append("insufficient_samples")
    threshold = criteria.min_practical_improvement
    fraction = sum(value > threshold for value in repeat_means) / len(repeat_means)
    if 0 < fraction < criteria.min_positive_repeat_fraction:
        result["reasons"].append("unstable_repetition_results")
    if result["reasons"]:
        return result
    if (
        interval is not None
        and interval[0] > threshold
        and fraction >= criteria.min_positive_repeat_fraction
    ):
        result["decision"] = "adopt"
    elif interval is not None and interval[1] <= threshold:
        result["decision"] = "reject"
    else:
        result["reasons"].append("uncertainty_overlaps_practical_threshold")
    return result
