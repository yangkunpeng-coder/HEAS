"""Parse official verifier responses without substituting a HEAS scoring rule."""

import math
from typing import Any

from benchmark_adapters.common.process import BenchmarkProtocolError


def verifier_metrics(
    result: dict[str, Any], threshold: float
) -> tuple[dict[str, float], bool | None]:
    """Read flat or official nested rewards and preserve missing/failed evaluation."""
    if result.get("exception_info"):
        return {}, None
    raw = result.get("verifier_result", result)
    if not isinstance(raw, dict):
        raise BenchmarkProtocolError("official verifier result must be an object")
    error = str(raw.get("error", ""))
    if error and error != "no_code_extracted":
        return {}, None
    reward = raw.get("rewards", {}).get("reward") if isinstance(raw.get("rewards"), dict) else None
    if reward is None:
        reward = raw.get("reward")
    if reward is None:
        return {}, None
    if (
        isinstance(reward, bool)
        or not isinstance(reward, (int, float))
        or not math.isfinite(reward)
    ):
        raise BenchmarkProtocolError("official reward must be a finite number")
    if not math.isfinite(threshold):
        raise BenchmarkProtocolError("official pass threshold must be finite")
    return {"reward": float(reward)}, reward > threshold
