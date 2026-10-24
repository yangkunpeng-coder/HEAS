"""Read only official public instructions and task configuration, never tests/solution."""

import tomllib
from pathlib import Path

from benchmark_adapters.common.catalog import CatalogProvider
from benchmark_adapters.common.config import SkillLearnBenchConfig
from benchmark_adapters.common.preflight import source_root
from benchmark_adapters.common.protocol import PublicTask
from heas.core.errors import HEASError, SecurityError
from heas.infrastructure.storage import digest


def task_path(root: Path, native_id: str) -> Path:
    """Resolve the official parent/instance layout without allowing path traversal."""
    parts = native_id.split("/")
    if len(parts) != 2 or any(part in {"", ".", "..", "tests", "solution"} for part in parts):
        raise SecurityError("SkillLearnBench task IDs must use parent/instance form")
    path = (root / "tasks" / native_id).resolve()
    if not path.is_relative_to((root / "tasks").resolve()):
        raise SecurityError("official task path escapes the task root")
    return path


class SkillLearnBenchProvider(CatalogProvider):
    """Expose skill-evaluation tasks with host-defined, explicitly disjoint partitions."""

    def __init__(self, config: SkillLearnBenchConfig):
        if not config.partitions:
            raise HEASError(
                "SkillLearnBench has no universal official split; provide host partitions"
            )
        rows = []
        self.task_configs = {}
        root = source_root(config)
        for identity in config.task_ids:
            path = task_path(root, identity)
            instruction = path / "instruction.md"
            if not instruction.is_file() or not (path / "task.toml").is_file():
                raise HEASError("official instruction.md or task.toml is missing: " + identity)
            task_config = tomllib.loads((path / "task.toml").read_text(encoding="utf-8-sig"))
            self.task_configs[identity] = task_config
            rows.append(
                PublicTask(
                    official_task_id=identity,
                    split="host_defined",
                    source_sha256=digest(
                        instruction.read_bytes() + (path / "task.toml").read_bytes()
                    ),
                    prompt="Review the task request for the current genome's exported skills. "
                    "The official solver, not this planning run, executes the task:\n"
                    + instruction.read_text(encoding="utf-8-sig"),
                    domain=identity.split("/")[0],
                )
            )
        super().__init__(
            "skilllearnbench",
            rows,
            config.revision,
            config.data_origin,
            {str(key): value for key, value in config.partitions.items()},
        )
