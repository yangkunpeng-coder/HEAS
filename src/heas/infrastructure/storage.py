"""Serial filesystem repository: immutable records, hashes and resolvable evidence."""

import hashlib
import json
import os
import re
from pathlib import Path, PureWindowsPath
from typing import Any, TypeVar

from heas.core.domain import ArtifactRef, EvidenceRef, Record, SkillRef, TaskGenome, TraceEvent, uid
from heas.core.errors import IntegrityError, MissingEvidence, SecurityError

T = TypeVar("T", bound=Record)


def digest(data: bytes) -> str:
    """Compute a prefixed SHA-256 fingerprint of the exact stored bytes."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


def encode(data: Any) -> bytes:
    """Serialize records deterministically as sorted UTF-8 JSON for hashing and storage."""
    if isinstance(data, Record):
        data = data.model_dump(mode="json")
    return json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")


def redact(value: Any) -> Any:
    """Conservative credential scrubber; raw exception/request bodies are never logged."""
    # Custom provider credentials must receive the same protection as the default
    # HEAS key. Snapshot once per tree, replacing longer overlapping values first.
    labels = ("API_KEY", "TOKEN", "PASSWORD", "SECRET", "CREDENTIAL", "PRIVATE_KEY")
    secrets = sorted(
        {
            secret
            for name, secret in os.environ.items()
            if secret
            and any(name.upper() == label or name.upper().endswith("_" + label) for label in labels)
        },
        key=len,
        reverse=True,
    )

    def scrub(item: Any) -> Any:
        if isinstance(item, dict):
            return {
                key: "[REDACTED]"
                if any(
                    label in key.lower()
                    for label in (
                        "api_key",
                        "authorization",
                        "password",
                        "secret",
                        "access_token",
                        "refresh_token",
                        "private_key",
                    )
                )
                else scrub(content)
                for key, content in item.items()
            }
        if isinstance(item, list):
            return [scrub(content) for content in item]
        if isinstance(item, str):
            for secret in secrets:
                item = item.replace(secret, "[REDACTED]")
            item = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[REDACTED]", item)
            return re.sub(r"(?i)bearer\s+[A-Za-z0-9_.-]+", "Bearer [REDACTED]", item)
        return item

    return scrub(value)


class FileStore:
    """Persist immutable artifacts and resolve hash-checked evidence references."""

    def __init__(self, root: Path | str):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, uri: str) -> Path:
        """Resolve a workspace-relative path while rejecting traversal and symlink escapes."""
        win = PureWindowsPath(uri)
        if not uri or "\\" in uri or win.drive or win.is_absolute():
            raise SecurityError("workspace-relative forward-slash path required")
        candidate = Path(uri)
        if candidate.is_absolute() or any(p in {"..", "."} for p in uri.split("/")):
            raise SecurityError("path traversal rejected")
        resolved = (self.root / candidate).resolve()
        if not resolved.is_relative_to(self.root):
            raise SecurityError("symlink escapes workspace")
        return resolved

    def put_bytes(self, uri: str, data: bytes) -> None:
        """Commit immutable bytes atomically and reject conflicting existing content."""
        path = self.path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f".{path.name}.{uid('tmp')}")
        try:
            with temp.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            # Hard link publishes atomically without replacing an existing immutable object.
            os.link(temp, path)
        except FileExistsError as exc:
            raise IntegrityError(f"immutable object already exists: {uri}") from exc
        finally:
            temp.unlink(missing_ok=True)

    def put(self, uri: str, data: Any) -> None:
        """Persist a deterministically encoded value under the immutable write contract."""
        self.put_bytes(uri, encode(data))

    def read(self, uri: str) -> Any:
        """Read an existing persisted value without modifying its source artifact."""
        return json.loads(self.path(uri).read_text(encoding="utf-8"))

    def load(self, uri: str, cls: type[T]) -> T:
        """Load and validate persisted content without changing its inherited version."""
        return cls.model_validate(self.read(uri))

    def artifact(self, uri: str, artifact_id: str, version: str = "v1") -> ArtifactRef:
        """Create a versioned artifact and retain its content fingerprint."""
        return ArtifactRef(
            artifact_id=artifact_id,
            version_id=version,
            content_sha256=digest(self.path(uri).read_bytes()),
            uri=uri,
        )

    def verify(self, ref: ArtifactRef) -> bytes:
        """Load artifact bytes only after their declared content fingerprint is verified."""
        try:
            content = self.path(ref.uri).read_bytes()
        except OSError as exc:
            raise IntegrityError(f"missing artifact: {ref.artifact_id}") from exc
        if digest(content) != ref.content_sha256:
            raise IntegrityError(f"hash mismatch: {ref.artifact_id}")
        return content

    def index(
        self,
        ref_type: str,
        ref_id: str,
        uri: str,
        *,
        event: TraceEvent | None = None,
        offset: int | None = None,
        length: int | None = None,
    ) -> None:
        # EvidenceRef validates IDs and type before interpolating index paths.
        """Bind a typed evidence identity to its immutable relative storage path."""
        EvidenceRef(ref_type=ref_type, ref_id=ref_id)  # type: ignore[arg-type]
        content = encode(event) if event else self.path(uri).read_bytes()
        self.put(
            f"index/{ref_type}/{ref_id}.json",
            {
                "schema_version": "1.0",
                "uri": uri,
                "content_sha256": digest(content),
                "event_id": event.event_id if event else None,
                "offset": offset,
                "length": length,
            },
        )

    def resolve(self, ref: EvidenceRef) -> Any:
        """Resolve an evidence reference and verify the archived content hash."""
        try:
            item = self.read(f"index/{ref.ref_type}/{ref.ref_id}.json")
            if item["event_id"]:
                if item.get("offset") is not None:
                    try:
                        with self.path(item["uri"]).open("rb") as stream:
                            stream.seek(item["offset"])
                            event = TraceEvent.model_validate_json(stream.read(item["length"]))
                        matches = [event] if event.event_id == item["event_id"] else []
                    except ValueError as exc:
                        raise IntegrityError("trace event bytes are invalid or modified") from exc
                else:
                    matches = [e for e in self.trace(item["uri"]) if e.event_id == item["event_id"]]
                if len(matches) != 1 or digest(encode(matches[0])) != item["content_sha256"]:
                    raise IntegrityError("event missing, duplicated or modified")
                return matches[0].model_dump(mode="json")
            content = self.path(item["uri"]).read_bytes()
            if digest(content) != item["content_sha256"]:
                raise IntegrityError("evidence hash mismatch")
            if item["uri"].endswith(".md"):
                return content.decode("utf-8")
            return json.loads(content)
        except FileNotFoundError as exc:
            raise MissingEvidence(f"missing-evidence: {ref.ref_type}/{ref.ref_id}") from exc

    def save_record(self, uri: str, data: Record, ref_type: str, ref_id: str) -> None:
        """Persist an immutable record and publish its resolvable evidence index."""
        self.put(uri, data)
        self.index(ref_type, ref_id, uri)

    def append_event(self, event: TraceEvent) -> None:
        """Append one trace event and index its byte range for later verification."""
        uri = f"runs/{event.run_id}/trace.jsonl"
        path = self.path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.path(f"index/trace_event/{event.event_id}.json").exists():
            raise IntegrityError("duplicate event_id")
        line = (
            json.dumps(event.model_dump(mode="json"), ensure_ascii=False, sort_keys=True).encode(
                "utf-8"
            )
            + b"\n"
        )
        with path.open("ab") as stream:
            offset = stream.tell()
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        self.index("trace_event", event.event_id, uri, event=event, offset=offset, length=len(line))

    def trace(self, uri: str) -> list[TraceEvent]:
        """Load the run's archived event stream with per-event schema validation."""
        return [
            TraceEvent.model_validate_json(line)
            for line in self.path(uri).read_text(encoding="utf-8").splitlines()
            if line
        ]

    def recover_trace(self, run_id: str) -> dict[str, Any]:
        """Repair only an interrupted trace tail and rebuild its evidence offsets."""
        EvidenceRef(ref_type="task_run", ref_id=run_id)  # Validate ID before path use.
        uri = f"runs/{run_id}/trace.jsonl"
        path = self.path(uri)
        if not path.exists():
            return {"recovered_events": 0, "truncated_bytes": 0}
        original = path.read_bytes()
        offset, recovered = 0, 0
        seen: set[str] = set()
        for line in original.splitlines(keepends=True):
            try:
                event = TraceEvent.model_validate_json(line)
                if event.run_id != run_id or event.event_id in seen:
                    raise IntegrityError("invalid/duplicate event in interrupted trace")
            except ValueError:
                break
            if not line.endswith(b"\n"):
                break
            index_uri = f"index/trace_event/{event.event_id}.json"
            if not self.path(index_uri).exists():
                self.index(
                    "trace_event", event.event_id, uri, event=event, offset=offset, length=len(line)
                )
            seen.add(event.event_id)
            offset += len(line)
            recovered += 1
        if offset < len(original):
            self.put_bytes(f"quarantine/trace_recovery/{uid('recovery')}.bin", original)
            temp = path.with_name(uid("recovered"))
            temp.write_bytes(original[:offset])
            os.replace(temp, path)
        return {"recovered_events": recovered, "truncated_bytes": len(original) - offset}

    def genome(self, genome_id: str) -> TaskGenome:
        """Load an indexed immutable Genome snapshot with schema validation."""
        item = self.resolve(EvidenceRef(ref_type="artifact", ref_id=genome_id))
        genome = TaskGenome.model_validate(item)
        self.verify(genome.memory)
        for ref in genome.skills:
            self.verify(ref)
        return genome

    def commit_genome(self, genome: TaskGenome, memory: str) -> None:
        """Commit an independent genome and memory snapshot as immutable artifacts."""
        if self.path(f"index/artifact/{genome.genome_id}.json").exists():
            raise IntegrityError("genome identity already archived")
        for tool in genome.tools:
            if not self.path(tool.manifest_ref).is_file():
                raise SecurityError("tool manifest is missing")
            manifest_data = self.read(tool.manifest_ref)
            expected = tool.extensions.get("heas", {}).get("manifest_sha256")
            if (
                digest(self.path(tool.manifest_ref).read_bytes()) != expected
                or manifest_data.get("tool_id") != tool.tool_id
                or manifest_data.get("version_id") != tool.version_id
                or manifest_data.get("permissions") != tool.permissions
            ):
                raise SecurityError("tool manifest/binding integrity mismatch")
        for skill in genome.skills:
            manifest = self.load(skill.uri.replace("SKILL.md", "manifest.json"), SkillRef)
            if manifest != skill:
                raise IntegrityError("skill manifest mismatch")
            required = set(skill.requires_tools)
            if not required.issubset({t.tool_id for t in genome.tools}):
                raise IntegrityError("skill tool dependencies are missing")
        directory = f"task_agents/{genome.agent_id}/{genome.genome_id}"
        if genome.memory.uri != f"{directory}/MEMORY.md":
            raise IntegrityError("memory must be a private snapshot for this genome")
        if digest(memory.encode("utf-8")) != genome.memory.content_sha256:
            raise IntegrityError("memory hash mismatch")
        for ref in genome.skills:
            self.verify(ref)
        stage_uri = f".staging/{uid('genome')}"
        stage = self.path(stage_uri)
        self.put_bytes(f"{stage_uri}/MEMORY.md", memory.encode("utf-8"))
        self.put(f"{stage_uri}/genome.json", genome)
        destination = self.path(directory)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise IntegrityError("genome version already exists")
        stage.rename(destination)
        self.index("artifact", genome.genome_id, f"{directory}/genome.json")
        self.index("artifact", genome.memory.artifact_id, genome.memory.uri)

    def skill(
        self,
        content: str,
        requires_tools: list[str] | None = None,
        selection_tags: list[str] | None = None,
        *,
        applicability: str = "",
        conflicts_with: list[str] | None = None,
        constraints: dict[str, str] | None = None,
    ) -> SkillRef:
        """Persist a new Skill text version and return its content-addressed reference."""
        skill_id = uid("skill")
        uri = f"artifacts/skills/{skill_id}/v1/SKILL.md"
        self.put_bytes(uri, content.encode("utf-8"))
        ref = self.artifact(uri, skill_id)
        ref = SkillRef(
            **{
                **ref.model_dump(),
                "requires_tools": requires_tools or [],
                "selection_tags": selection_tags or [],
                "applicability": applicability,
                "conflicts_with": conflicts_with or [],
                "constraints": constraints or {},
            }
        )
        self.put(uri.replace("SKILL.md", "manifest.json"), ref)
        # Artifact evidence may contain Markdown; resolve artifact refs through verify().
        self.index("artifact", skill_id, uri.replace("SKILL.md", "manifest.json"))
        return ref
