"""Load official task-ID splits without copying answers or benchmark datasets."""

import json

from benchmark_adapters.common.catalog import CatalogProvider
from benchmark_adapters.common.config import EvoAgentBenchConfig
from benchmark_adapters.common.preflight import source_root
from benchmark_adapters.common.protocol import PublicTask
from heas.core.errors import HEASError, SecurityError


def load_official_splits(config: EvoAgentBenchConfig) -> dict[str, list[str]]:
    """Read the exact tracked train/test JSON format and reject split contamination."""
    path = source_root(config) / "data/splits" / f"{config.domain}.json"
    if not path.is_file():
        raise HEASError(f"official task-ID split is missing: {path}")
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict) or set(raw) != {"train", "test"}:
        raise HEASError("unsupported official split format; expected train/test ID arrays")
    splits: dict[str, list[str]] = {}
    for split in ("train", "test"):
        identities = raw[split]
        if not isinstance(identities, list):
            raise HEASError("unsupported official split format; expected train/test ID arrays")
        validated: list[str] = []
        for identity in identities:
            if not isinstance(identity, str) or not identity:
                raise HEASError("official task IDs must be nonempty strings")
            validated.append(identity)
        splits[split] = validated
    if any(len(ids) != len(set(ids)) for ids in splits.values()) or set(splits["train"]) & set(
        splits["test"]
    ):
        raise SecurityError("official split contains duplicate or overlapping task IDs")
    return splits


class EvoAgentBenchProvider(CatalogProvider):
    """Expose all four domains through the same Core-neutral provider contract."""

    def __init__(self, config: EvoAgentBenchConfig):
        splits = load_official_splits(config)
        partitions = config.partitions or {
            "evolution_train": splits["train"],
            "final_test": splits["test"],
        }
        for partition, identities in partitions.items():
            allowed = splits["test" if partition == "final_test" else "train"]
            if not set(identities).issubset(allowed):
                raise SecurityError("official test tasks cannot enter any learning partition")
        rows = [
            PublicTask(
                official_task_id=identity,
                split=split,
                domain=config.domain,
                prompt="Execute the official request in environment observation.official_prompt.",
            )
            for split, split_task_ids in splits.items()
            for identity in split_task_ids
        ]
        super().__init__(
            "evoagentbench_" + config.domain,
            rows,
            config.revision,
            config.data_origin,
            {str(key): value for key, value in partitions.items()},
        )
