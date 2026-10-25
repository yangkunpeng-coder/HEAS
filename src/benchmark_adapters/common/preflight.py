"""Read-only environment checks; never start a container or contact a model."""

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from benchmark_adapters.common.config import IntegrationConfig


def source_root(config: IntegrationConfig) -> Path:
    """Resolve the benchmark subproject independently from its website checkout."""
    if config.benchmark == "evoagentbench" and (config.repository / "benchmark").is_dir():
        return config.repository / "benchmark"
    return config.repository


def preflight(config: IntegrationConfig) -> dict[str, Any]:
    """Report pinned-source, configuration, dependency, and execution readiness."""
    checks: list[dict[str, Any]] = []

    def check(check_name: str, ready: bool, detail: str) -> None:
        checks.append({"name": check_name, "ready": ready, "detail": detail})

    root = source_root(config)
    check("repository", root.is_dir(), str(root))
    actual = None
    if root.is_dir() and shutil.which("git"):
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        actual = result.stdout.strip() if result.returncode == 0 else None
    check(
        "revision",
        actual == config.revision,
        f"expected={config.revision}; actual={actual or 'unavailable'}",
    )
    clean = False
    if actual:
        result = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        clean = result.returncode == 0 and not result.stdout.strip()
    check("unmodified_source", clean, "tracked official source must match the pinned revision")
    requirements = {
        "evoagentbench": ["src/config.py", "src/domains/base.py", "src/runner.py"],
        "seabench": ["runtime/task_loader.py", "runtime/validation_dispatcher.py", "env_assets"],
        "skilllearnbench": ["evaluate_skills.py", "core/eval_runner.py", "tasks"],
    }[config.benchmark]
    for name in requirements:
        check("source:" + name, (root / name).exists(), "required official source/data path")
    check("yaml", importlib.util.find_spec("yaml") is not None, "install heas[benchmarks]")
    if config.benchmark == "evoagentbench":
        check("official_config", config.official_config.is_file(), str(config.official_config))
        if config.domain in {"software_engineering", "code_implementation"}:
            check("docker", shutil.which("docker") is not None, "no daemon/container is started")
        if config.domain == "code_implementation":
            check(
                "sandbox_prefix",
                bool(config.isolation_prefix),
                "official code verifier executes generated code; isolated worker required",
            )
        if config.domain == "information_retrieval":
            check(
                "java",
                shutil.which("java") is not None,
                "Java 21 and prepared search index required",
            )
    elif config.benchmark == "seabench":
        check("sequence_file", config.sequence_file.is_file(), str(config.sequence_file))
        check(
            "judge",
            bool(config.judge_model and os.getenv(config.judge_key_env)),
            "judge model/key required for official LLM evaluation; key values are never logged",
        )
    else:
        check(
            "docker", shutil.which("docker") is not None, "Docker and official task images required"
        )
        for name in ("anthropic", "rich", "dataclaw", "json_repair"):
            check(
                "dependency:" + name,
                importlib.util.find_spec(name) is not None,
                "official dependencies are optional and not installed by HEAS Core",
            )
        check(
            "solver_key",
            bool(os.getenv("ANTHROPIC_API_KEY")),
            "official evaluate_skills.py prechecks this key even for dry-run/help",
        )
    check("execution_enabled", config.allow_execution, "explicit host opt-in; default is false")
    if config.benchmark in {"seabench", "skilllearnbench"} or (
        config.benchmark == "evoagentbench"
        and config.domain in {"information_retrieval", "knowledge_work"}
    ):
        check(
            "paid_execution_enabled",
            config.allow_paid_judges,
            "official judge/solver requests require separate host opt-in",
        )
    return {
        "benchmark": config.benchmark,
        "revision": config.revision,
        "implementation_status": "implemented",
        "officially_verified": False,
        "ready": all(item["ready"] for item in checks),
        "checks": checks,
    }
