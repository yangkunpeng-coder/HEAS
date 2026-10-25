"""Version-bound Markdown memory entries; read-only selection and controlled merge."""

import re
from typing import Any

from heas.core.domain import ArtifactRef, EvidenceRef, Record, TaskSpec
from heas.core.errors import HEASError
from heas.infrastructure.storage import FileStore, digest


class MemoryEntry(Record, frozen=True):
    """Represent one section of inherited textual memory with explicit provenance."""

    entry_id: str
    title: str
    text: str
    key: str | None = None
    source_ref: ArtifactRef


def sections(text: str) -> list[tuple[str, str, str | None]]:
    """Parse memory sections without interpreting observations as inherited truth."""
    entries: list[tuple[str, str, str | None]] = []
    title, key = "general", None
    lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if "\n".join(lines).strip():
                entries.append((title, "\n".join(lines).strip(), key))
            title, lines, key = line[3:].strip(), [], line[3:].strip().casefold()
        elif not line.startswith("# "):
            lines.append(line)
    if "\n".join(lines).strip():
        entries.append((title, "\n".join(lines).strip(), key))
    return entries


def explicit_conflicts(texts: list[str]) -> list[str]:
    """Identify explicitly represented memory conflicts for later host-guided handling."""
    directives: dict[str, str] = {}
    errors = []
    for text in texts:
        for line in text.splitlines():
            match = re.match(r"^\s*(always|never)\s+(.+?)\s*[.!]?\s*$", line, re.I)
            if match:
                polarity, action = match.group(1).lower(), match.group(2).rstrip(".!").casefold()
                if action in directives and directives[action] != polarity:
                    errors.append("contradictory directive: " + action)
                directives[action] = polarity
    return sorted(set(errors))


class MemoryManager:
    """Select inherited memory and merge explicit knowledge without silent conflicts."""

    def __init__(self, store: FileStore):
        self.store = store

    def read(self, ref: ArtifactRef) -> list[MemoryEntry]:
        """Read an existing persisted value without modifying its source artifact."""
        entries = []
        for title, text, key in sections(self.store.verify(ref).decode("utf-8")):
            entry_id = "memory_entry_" + digest((ref.artifact_id + title + text).encode())[7:31]
            entry = MemoryEntry(entry_id=entry_id, title=title, text=text, key=key, source_ref=ref)
            uri = f"analysis/memory_entries/{entry_id}.json"
            if not self.store.path(uri).exists():
                self.store.save_record(uri, entry, "memory_entry", entry_id)
            entries.append(entry)
        return entries

    def select(
        self,
        ref: ArtifactRef,
        task: TaskSpec,
        state: dict[str, Any],
        *,
        max_entries: int = 4,
        max_tokens: int = 2048,
    ) -> list[MemoryEntry]:
        """Select admissible capabilities or parents and retain the reasons for the choice."""
        tokens = set(re.findall(r"\w+", (task.instruction + " " + str(state)).lower()))
        ranked = sorted(
            enumerate(self.read(ref)),
            key=lambda pair: (
                -len(tokens.intersection(re.findall(r"\w+", pair[1].text.lower()))),
                pair[0],
            ),
        )
        selected: list[MemoryEntry] = []
        used = 0
        for _, entry in ranked:
            size = len(entry.text.encode("utf-8"))  # conservative token upper bound
            if len(selected) < max_entries and used + size <= max_tokens:
                selected.append(entry)
                used += size
        return selected

    @staticmethod
    def merge(texts: list[str]) -> str:
        """Construct inherited memory from explicit parents and preserve conflict provenance."""
        errors = explicit_conflicts(texts)
        seen: set[str] = set()
        keyed: dict[str, str] = {}
        merged: list[tuple[str, str]] = []
        for text in texts:
            for title, body, key in sections(text):
                normalized = " ".join(body.split()).casefold()
                if key and key in keyed and keyed[key] != normalized:
                    errors.append("conflicting memory entry key: " + key)
                if key:
                    keyed[key] = normalized
                if normalized not in seen:
                    merged.append((title, body))
                    seen.add(normalized)
        if errors:
            raise HEASError("; ".join(errors))
        return (
            "# Memory\n\n"
            + "\n\n".join(
                (
                    f"## {section_title}\n{section_body}"
                    if section_title != "general"
                    else section_body
                )
                for section_title, section_body in merged
            )
            + "\n"
        )

    def refs(self, entries: list[MemoryEntry]) -> list[EvidenceRef]:
        """Expose memory provenance references for capability and trace auditing."""
        return [EvidenceRef(ref_type="memory_entry", ref_id=e.entry_id) for e in entries]
