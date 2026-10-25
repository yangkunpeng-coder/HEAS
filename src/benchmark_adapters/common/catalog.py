"""Public task catalogs with explicit host partition ownership."""

from collections.abc import Mapping

from benchmark_adapters.common.config import require_partition_map
from benchmark_adapters.common.protocol import PublicTask, required_text
from heas.core.domain import TaskSpec
from heas.core.errors import HEASError


class CatalogProvider:
    """Keep native identities and private official data out of the Core task schema."""

    def __init__(
        self,
        provider_id: str,
        rows: list[PublicTask],
        revision: str,
        origin: str,
        partitions: Mapping[str, list[str]],
    ):
        require_partition_map(partitions)
        self.provider_id, self.revision = provider_id, revision
        self.rows = rows
        self.tasks: dict[str, TaskSpec] = {}
        self.native: dict[str, str] = {}
        for row in rows:
            task = row.heas_task(provider_id, revision, origin)
            if row.official_task_id in self.native or task.task_id in self.tasks:
                raise HEASError("duplicate native benchmark task identity")
            self.native[row.official_task_id] = task.task_id
            self.tasks[task.task_id] = task
        self.partitions = {}
        for partition, ids in partitions.items():
            if not set(ids).issubset(self.native):
                raise HEASError("partition references an unknown official task")
            self.partitions[partition] = [self.native[identity] for identity in ids]

    def list_tasks(self, partition: str) -> list[TaskSpec]:
        """Return authorized tasks in catalog order, never alphabetical sequence order."""
        allowed = set(self.partitions.get(partition, []))
        return [task for task in self.tasks.values() if task.task_id in allowed]

    def get_task(self, task_id: str) -> TaskSpec:
        """Resolve a stable HEAS task identity without loading a private verifier row."""
        try:
            return self.tasks[task_id]
        except KeyError as exc:
            raise HEASError("task is absent from the benchmark catalog") from exc

    def native_id(self, task_id: str) -> str:
        """Return the original official identity for a Core execution receipt."""
        return required_text(self.get_task(task_id).metadata, "official_task_id")
