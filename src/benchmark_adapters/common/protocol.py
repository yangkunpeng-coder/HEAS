"""Public-only task and evaluation contracts at the benchmark/Core boundary."""

from typing import Any, Literal, Protocol

from pydantic import Field

from heas.core.domain import Id, Record, TaskSpec
from heas.core.errors import HEASError
from heas.infrastructure.storage import digest


def required_text(payload: dict[str, Any], field: str, *, allow_empty: bool = False) -> str:
    """Read required protocol text without coercing missing values into identities.

    Args:
        payload: Untrusted decoded worker data or task metadata.
        field: Host-defined field name, safe to include in diagnostics.
        allow_empty: Permit an empty answer while retaining string type validation.

    Raises:
        HEASError: The required field is missing, None, or incompatible text.
    """
    value = payload.get(field)
    if not isinstance(value, str) or (not allow_empty and not value):
        raise HEASError("required protocol text is missing or invalid: " + field)
    return value


class PublicTask(Record, frozen=True):
    """Expose task identity and instructions while retaining rubrics in the worker."""

    official_task_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    split: str
    domain: str = ""
    sequence_id: str | None = None
    position: int | None = None
    phase: Literal["upstream", "downstream"] | None = None
    source_sha256: str | None = None

    def heas_task(self, provider_id: str, revision: str, origin: str) -> TaskSpec:
        """Create a stable public specification without copying private official rows."""
        identity = (
            "task_"
            + digest(f"{provider_id}|{self.sequence_id}|{self.official_task_id}".encode())[7:39]
        )
        return TaskSpec(
            task_id=identity,
            provider_id=provider_id,
            instruction=self.prompt,
            environment_spec={"official_task_id": self.official_task_id},
            metadata={
                "benchmark_revision": revision,
                "data_origin": origin,
                "official_task_id": self.official_task_id,
                "official_split": self.split,
                "domain": self.domain,
                "sequence_id": self.sequence_id,
                "position": self.position,
                "phase": self.phase,
                "source_sha256": self.source_sha256,
            },
        )


class OfficialReceipt(Record, frozen=True):
    """Retain only finite official metrics, execution identity, and public provenance."""

    benchmark: str
    revision: str
    official_task_id: str
    run_id: Id
    genome_id: Id
    status: Literal["completed", "not_available", "failed"]
    metrics: dict[str, float] = Field(default_factory=dict)
    diagnostics: dict[str, float] = Field(default_factory=dict)
    success: bool | None = None
    measurement_target: Literal["task_agent", "exported_genome_skills"] = "task_agent"
    raw_sha256: str | None = None
    usage: dict[str, Any] | None = None
    private_feedback_withheld: bool = True


class RunnerBridge(Protocol):
    """Define the HEAS-owned RPC contract, not an invented official API."""

    def request(self, operation: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        """Send one bounded request to an independent official worker."""
        ...

    def close(self) -> None:
        """Release worker resources without modifying immutable source data."""
        ...
