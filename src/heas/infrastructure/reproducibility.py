"""Export immutable reproducibility metadata, without copying private task payloads."""

import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from heas.core.domain import RunPlan, now, uid
from heas.infrastructure.resources import ResourceLedger
from heas.infrastructure.storage import FileStore, digest, encode, redact


def export_manifest(store: FileStore, source_root: Path | None = None) -> str:
    """Export reproducibility metadata and hashes without copying private task data."""
    source_root = source_root or Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        revision = result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        revision = None
    versions: dict[str, str | None] = {}
    for name in ("heas", "pydantic", "openai"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    plans = []
    for path in store.path("plans").glob("*.json"):
        plan = store.load(path.relative_to(store.root).as_posix(), RunPlan)
        plans.append(
            {
                "run_plan_id": plan.run_plan_id,
                "genome_ids": plan.genome_ids,
                "partition": plan.task_partition,
                "task_suite_id": plan.task_suite_id,
                "task_suite_version": plan.task_suite_version,
                "evaluation_suite_id": plan.evaluation_suite_id,
                "task_evaluator_id": plan.task_evaluator_id,
                "evaluator_version": plan.evaluator_version,
                "random_seed": plan.random_seed,
                "task_set_sha256": digest(
                    encode({"provider": plan.task_provider_id, "tasks": plan.task_ids})
                ),
                "model_config_ref": plan.model_config_ref,
                "budget": plan.budget.model_dump(mode="json"),
                "determinism": "not_guaranteed",
            }
        )
    files: list[dict[str, Any]] = []
    for directory in (
        "task_agents",
        "policies",
        "plans",
        "runs",
        "evolution",
        "meta",
        "configs",
        "artifacts",
        "experiences",
        "analysis",
        "candidates",
        "model_calls",
        "suites",
        "contexts",
        "structured_calls",
        "model_requests",
        "tool_calls",
        "experiments",
        "benchmark_invocations",
        "benchmark_completions",
        "metrics",
    ):
        for path in store.path(directory).rglob("*"):
            if path.is_file() and path.suffix in {".json", ".jsonl", ".md"}:
                files.append(
                    {
                        "uri": path.relative_to(store.root).as_posix(),
                        "content_sha256": digest(path.read_bytes()),
                    }
                )
    source_hashes = {
        p.relative_to(source_root).as_posix(): digest(p.read_bytes())
        for p in sorted((source_root / "src").rglob("*.py"))
    }
    for name in ("pyproject.toml", "requirements-lock.txt"):
        if (source_root / name).is_file():
            source_hashes[name] = digest((source_root / name).read_bytes())
    run_snapshots = []
    for path in store.path("runs").glob("*/run.json"):
        run = store.read(path.relative_to(store.root).as_posix())
        feedback = (
            store.read(run["objective_feedback_ref"]) if run["objective_feedback_ref"] else {}
        )
        run_snapshots.append(
            {
                "run_id": run["run_id"],
                "genome_id": run["genome_id"],
                "task_instance_sha256": run.get("task_instance_sha256"),
                "evaluator_id": feedback.get("evaluator_id"),
                "evaluator_version": feedback.get("evaluator_version"),
                "evaluator_versions": feedback.get("evaluator_versions", {}),
            }
        )
    model_profiles = {}
    for path in store.path("model_calls").glob("*.json"):
        receipt = store.read(path.relative_to(store.root).as_posix())
        profile = {
            key: receipt.get(key)
            for key in (
                "purpose",
                "model_identity",
                "model_parameters",
                "provider_model",
                "request_parameters",
            )
        }
        model_profiles[digest(encode(profile))] = redact(profile)
    uri = f"exports/{uid('reproducibility')}.json"
    store.put(
        uri,
        {
            "schema_version": "1.2",
            "created_at": now().isoformat(),
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "package_versions": versions,
            "code_revision": revision,
            "code_revision_status": "available" if revision else "not_available",
            "source_hashes": source_hashes,
            "plans": plans,
            "run_snapshots": run_snapshots,
            "model_profiles": list(model_profiles.values()),
            "artifact_hashes": files,
            "resource_usage": ResourceLedger(store).report(),
            "determinism": "Check provider support and repeated outcomes; seed is not proof.",
            "private_payload_exported": False,
        },
    )
    return uri
