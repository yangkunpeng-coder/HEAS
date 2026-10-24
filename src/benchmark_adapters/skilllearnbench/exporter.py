"""Export immutable Genome Skill text into the official task/skill/SKILL.md layout."""

import json
from pathlib import Path
from typing import Any

from benchmark_adapters.skilllearnbench.provider import task_path
from heas.core.errors import SecurityError
from heas.infrastructure.storage import FileStore, digest


class SkillExporter:
    """Export skills only; never copy generated tool code into an executable sandbox."""

    def export(
        self, store: FileStore, genome_id: str, official_tasks: list[str], destination: Path
    ) -> dict[str, Any]:
        """Write a new versioned export and provenance manifest without overwriting assets."""
        if destination.exists():
            raise SecurityError("skill exports are immutable; choose a new destination")
        genome = store.genome(genome_id)
        capabilities = [
            {
                "artifact_id": ref.artifact_id,
                "content": store.verify(ref).decode("utf-8"),
                "content_sha256": ref.content_sha256,
            }
            for ref in genome.skills
        ]
        return export_capabilities(genome_id, capabilities, official_tasks, destination)


def export_capabilities(
    genome_id: str, capabilities: list[dict[str, str]], official_tasks: list[str], destination: Path
) -> dict[str, Any]:
    """Materialize the same format from a worker's already hash-checked public payload."""
    if destination.exists():
        raise SecurityError("skill export destination already exists")
    parents = set()
    for identity in official_tasks:
        task_path(destination, identity)  # Validate native identity before path interpolation.
        parents.add(identity.split("/")[0])
    files = []
    destination.mkdir(parents=True)
    for parent in sorted(parents):
        (destination / parent).mkdir()
        for capability in capabilities:
            body = capability["content"]
            if digest(body.encode()) != capability["content_sha256"]:
                raise SecurityError("export payload differs from its genome artifact")
            identity = "heas-" + capability["content_sha256"][7:31]
            if body.startswith("---\n") and "\n---\n" in body[4:]:
                body = body.split("\n---\n", 1)[1]
            content = (
                "---\nname: "
                + identity
                + "\ndescription: "
                + json.dumps("Procedural skill exported from HEAS " + genome_id)
                + "\n---\n\n"
                + body.rstrip()
                + "\n"
            )
            path = destination / parent / identity / "SKILL.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                continue  # Identical skill contents share one exported immutable file.
            path.write_text(content, encoding="utf-8", newline="\n")
            files.append(
                {
                    "path": path.relative_to(destination).as_posix(),
                    "source_artifact_id": capability["artifact_id"],
                    "source_sha256": capability["content_sha256"],
                    "export_sha256": digest(content.encode()),
                }
            )
    report = {
        "genome_id": genome_id,
        "mode": "exported_skills_official_solver",
        "files": files,
        "contains_executable_tool_code": False,
    }
    (destination / "heas_export.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
