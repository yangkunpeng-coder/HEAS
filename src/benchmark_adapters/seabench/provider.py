"""Read official sequence manifests while withholding validators and safety rubrics."""

from typing import Any

from benchmark_adapters.common.catalog import CatalogProvider
from benchmark_adapters.common.config import SEABenchConfig
from benchmark_adapters.common.process import BenchmarkBlocked
from benchmark_adapters.common.protocol import PublicTask
from heas.core.errors import HEASError, SecurityError
from heas.infrastructure.storage import digest


def read_sequence(config: SEABenchConfig) -> tuple[dict[str, Any], list[PublicTask]]:
    """Validate the official manifest shape, preserving task order and explicit roles."""
    try:
        import yaml
    except ImportError as exc:
        raise BenchmarkBlocked("PyYAML is required; install heas[benchmarks]") from exc
    raw = yaml.safe_load(config.sequence_file.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict) or raw.get("type") != "task_sequence":
        raise HEASError("SEABench requires an official task_sequence YAML mapping")
    if raw.get("status", "active") != "active":
        raise HEASError("inactive official sequence cannot be executed")
    tasks = raw.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise HEASError("official sequence has no task list")
    rows = []
    downstream = False
    for position, task in enumerate(tasks):
        if not isinstance(task, dict) or not task.get("id") or not task.get("prompt"):
            raise HEASError("official sequence task is missing id or prompt")
        phase = task.get("optimization", {}).get("role")
        if phase not in {"upstream", "downstream"}:
            raise HEASError("SEABench task requires an explicit upstream/downstream role")
        if downstream and phase == "upstream":
            raise SecurityError("upstream task cannot follow frozen downstream evaluation")
        downstream |= phase == "downstream"
        # Do not infer or expose validator_expectation, safety_llm, generation
        # hints, controller overrides, or hidden environment reference fields.
        rows.append(
            PublicTask(
                official_task_id=str(task["id"]),
                prompt=str(task["prompt"]),
                split=phase,
                phase=phase,
                position=position,
                source_sha256=digest(config.sequence_file.read_bytes()),
                sequence_id=str(raw.get("id", "")),
                domain=str(raw.get("generation", {}).get("surface", "")),
            )
        )
    if not downstream or not any(row.phase == "upstream" for row in rows):
        raise HEASError("sequence requires both upstream and downstream tasks")
    return raw, rows


class SEABenchProvider(CatalogProvider):
    """Keep downstream safety tasks out of all learning and policy-trial partitions."""

    def __init__(self, config: SEABenchConfig):
        _, rows = read_sequence(config)
        upstream = [r.official_task_id for r in rows if r.phase == "upstream"]
        downstream = [r.official_task_id for r in rows if r.phase == "downstream"]
        partitions = config.partitions or {"evolution_train": upstream, "final_test": downstream}
        for partition, identities in partitions.items():
            if not set(identities).issubset(downstream if partition == "final_test" else upstream):
                raise SecurityError("downstream SEABench tasks are reserved for frozen reporting")
        super().__init__(
            "seabench_" + digest(str(rows[0].sequence_id).encode())[7:19],
            rows,
            config.revision,
            config.data_origin,
            {str(key): value for key, value in partitions.items()},
        )
