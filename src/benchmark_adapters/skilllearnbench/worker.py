"""Call evaluate_skills.py with HEAS exports; do not impersonate its solving agent."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from benchmark_adapters.common.config import SkillLearnBenchConfig
from benchmark_adapters.common.process import BenchmarkBlocked
from benchmark_adapters.common.protocol import OfficialReceipt, required_text
from benchmark_adapters.skilllearnbench.exporter import export_capabilities
from benchmark_adapters.skilllearnbench.results import COUNT_COLUMNS, quality_metrics, trial_metrics
from heas.core.errors import SecurityError
from heas.infrastructure.storage import digest, encode


def evaluation_command(
    config: SkillLearnBenchConfig, native_id: str, export: Path, trials: Path, *, has_skills: bool
) -> list[str]:
    """Build arguments verified against the pinned official argparse entry point."""
    command = [
        config.worker_python or sys.executable,
        str(config.repository / "evaluate_skills.py"),
        native_id,
        "--skill-path",
        str(export) if has_skills else "none",
        "--agent",
        config.solver_agent,
        "--model",
        config.solver_model,
        "--repeats",
        str(config.repeats),
        "--max-workers",
        "1",
        "--build-workers",
        "1",
        "--max-steps",
        str(config.max_steps),
        "--trials-dir",
        str(trials),
    ]
    command.extend(
        ["--judge-model", config.judge_model]
        if config.compute_quality_metrics
        else ["--skip-metrics"]
    )
    return command


class SkillLearnBenchWorker:
    """Evaluate a genome's skills with the official solver and retain separate provenance."""

    def __init__(self, config: SkillLearnBenchConfig, root: Path):
        self.config, self.root = config, root
        self.sessions: dict[str, str] = {}

    def handle(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Prepare evaluation requests or run the official bounded serial command."""
        session = required_text(payload, "session_id")
        if operation == "prepare":
            self.sessions[session] = required_text(payload, "official_task_id")
            return {
                "official_prompt": "Review the public task instruction. Independent evaluation "
                "uses the genome's exported skills with the official solving agent.",
                "allowed_actions": [],
            }
        if operation in {"cleanup", "abandon"}:
            self.sessions.pop(session, None)
            return {"cleaned": True}
        if operation != "evaluate":
            raise BenchmarkBlocked("HEAS is not registered as the official SkillLearnBench solver")
        if not self.config.allow_paid_judges:
            raise BenchmarkBlocked(
                "the official solving agent requires explicit paid execution opt-in"
            )
        native = self.sessions[session]
        if native != payload["official_task_id"]:
            raise SecurityError("official skill evaluation task identity mismatch")
        export = self.root / "exports" / (payload["genome_id"] + "_" + session)
        export_capabilities(payload["genome_id"], payload["capabilities"], [native], export)
        trials = self.root / "trials" / session
        command = evaluation_command(
            self.config, native, export, trials, has_skills=bool(payload["capabilities"])
        )
        completed = subprocess.run(
            command,
            cwd=self.config.repository,
            capture_output=True,
            text=True,
            timeout=self.config.timeout_seconds,
            check=False,
        )
        log = self.root / "logs" / (session + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise BenchmarkBlocked(
                "official evaluate_skills.py failed; inspect the private worker log"
            )
        rows = [json.loads(p.read_text(encoding="utf-8")) for p in trials.glob("**/result.json")]
        if len(rows) != self.config.repeats:
            raise BenchmarkBlocked("official trial outputs are missing or incomplete")
        metrics, success = trial_metrics(rows, native)
        diagnostics: dict[str, float] = {}
        if self.config.compute_quality_metrics:
            config_name = export.name if payload["capabilities"] else "no_skill"
            path = self.config.repository / "output/evaluation_reports" / config_name / "report.csv"
            for name, value in quality_metrics(path).items():
                (diagnostics if name in COUNT_COLUMNS else metrics)[name] = value
        raw = {"trials": rows, "metrics": metrics}
        return OfficialReceipt(
            benchmark="skilllearnbench",
            revision=self.config.revision,
            official_task_id=native,
            run_id=payload["run_id"],
            genome_id=payload["genome_id"],
            status="completed"
            if success is not None and "pass_rate" in metrics
            else "not_available",
            metrics=metrics,
            diagnostics=diagnostics,
            success=success,
            measurement_target="exported_genome_skills",
            raw_sha256=digest(encode(raw)),
            usage={
                "official_trial_usage": [r.get("token_usage") for r in rows],
                "judge_usage_available": False,
            },
        ).model_dump(mode="json")

    def close(self) -> None:
        """Release in-memory requests; official subprocesses own their container cleanup."""
        self.sessions.clear()
