"""Parse actual official trial JSON and normalized report.csv output schemas."""

import csv
import math
from pathlib import Path
from typing import Any

from benchmark_adapters.common.process import BenchmarkProtocolError

COUNT_COLUMNS = frozenset(
    {
        "num_steps",
        "num_tool_calls",
        "num_skill_calls",
        "input_tokens",
        "output_tokens",
        "num_skills_invoked",
        "num_skills_total",
    }
)


def trial_metrics(
    rows: list[dict[str, Any]], official_task_id: str
) -> tuple[dict[str, float], bool | None]:
    """Reject missing verifier receipts rather than accepting infrastructure zero rewards."""
    if not rows:
        return {}, None
    for row in rows:
        if row.get("task_id") != official_task_id:
            raise BenchmarkProtocolError("SkillLearnBench trial task identity mismatch")
        if row.get("error") or row.get("verifier_exit") is None:
            return {}, None
        if not isinstance(row.get("passed"), bool) or row.get("reward") not in {0, 1}:
            raise BenchmarkProtocolError("official trial is missing binary pass/reward fields")
        if row["passed"] != (row["reward"] == 1):
            raise BenchmarkProtocolError("official pass and reward fields disagree")
    rate = sum(trial_record["passed"] for trial_record in rows) / len(rows)
    return {"pass_rate": rate}, all(trial_record["passed"] for trial_record in rows)


def quality_metrics(path: Path) -> dict[str, float]:
    """Read the official normalized 0–100 report without rescaling or inventing columns."""
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise BenchmarkProtocolError("official report.csv must contain one aggregate row")
    result = {}
    for name, raw in rows[0].items():
        if raw in {"", None}:
            continue
        value = float(raw)
        if not math.isfinite(value):
            raise BenchmarkProtocolError("official quality metric is not finite")
        if name not in COUNT_COLUMNS and not 0 <= value <= 100:
            raise BenchmarkProtocolError("official normalized quality metric is outside 0–100")
        result[name] = value
    return result
