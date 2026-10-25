"""Bridge official manifest loading and completion/safety judges, with sealed feedback."""

import importlib
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from benchmark_adapters.common.config import SEABenchConfig
from benchmark_adapters.common.process import BenchmarkBlocked
from benchmark_adapters.common.protocol import OfficialReceipt, required_text
from heas.core.errors import HEASError, SecurityError
from heas.infrastructure.storage import digest, encode


def scored_boolean(
    value: Any, detail: dict[str, Any], backend: str, feedback: str = ""
) -> bool | None:
    """Distinguish a scored failure from an official judge infrastructure failure.

    The pinned dispatcher returns False for missing credentials and malformed
    judge output as well as for real task failures. Empty judge scores are not
    objective zeros. This validates receipt completeness, not the judge threshold.
    """
    if not isinstance(value, bool):
        return None
    if backend == "hybrid" and feedback.startswith("[deterministic]"):
        return value
    if backend == "hybrid" and not detail.get("enabled") and feedback.startswith("deterministic:"):
        return value
    if backend.startswith("llm") or backend == "hybrid":
        scores = detail.get("scores")
        mean = detail.get("mean_score")
        if (
            not isinstance(scores, dict)
            or not scores
            or detail.get("result") not in {"PASS", "FAIL"}
            or not isinstance(mean, (int, float))
            or not math.isfinite(mean)
            or any(
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
                for score in scores.values()
            )
        ):
            return None
        return value
    if backend in {"deterministic", "python"}:
        if detail.get("passed") != value or feedback.startswith("python validator error:"):
            return None
        return value
    return None


class SEABenchWorker:
    """Keep sequence workspaces independent and safety judgments outside agent inputs."""

    def __init__(self, config: SEABenchConfig, root: Path):
        self.config, self.root = config, root
        sys.path.insert(0, str(config.repository))
        loader = importlib.import_module("runtime.task_loader")
        self.tasks = {
            task.task_id: task
            for task in loader.load_yaml_task_suite_from_files([config.sequence_file])
        }
        self.validator = importlib.import_module("runtime.validation_dispatcher")
        self.setup = importlib.import_module("runtime.task_setup")
        self.sessions: dict[str, tuple[Any, Path]] = {}
        self.sequences: dict[str, Path] = {}

    def handle(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Run public actions or private official evaluation for one correlated session."""
        session = required_text(payload, "session_id")
        if operation == "prepare":
            task = self.tasks[required_text(payload, "official_task_id")]
            sequence = payload.get("sequence")
            if sequence is None:
                sequence = {}
            if not isinstance(sequence, dict):
                raise HEASError("sequence context must be an object when supplied")
            sequence_id = (
                session
                if sequence.get("session_id") is None
                else required_text(sequence, "session_id")
            )
            identity = digest(sequence_id.encode())[7:39]
            if identity not in self.sequences:
                workspace = self.root / "public" / identity
                shutil.copytree(self.config.repository / "env_assets", workspace)
                if task.sequence_setup:
                    self.setup.apply_task_setup(
                        workspace_dir=workspace,
                        repo_root=self.config.repository,
                        setup=task.sequence_setup,
                    )
                self.sequences[identity] = workspace
            workspace = self.sequences[identity]
            self.sessions[session] = task, workspace
            return {
                "official_prompt": task.prompt,
                "task_input_sha256": digest(self.config.sequence_file.read_bytes()),
                "workspace": str(workspace),
                "phase": task.optimization_role,
                "allowed_actions": ["read_file", "write_file", "list_files"],
            }
        if operation in {"cleanup", "abandon"}:
            self.sessions.pop(session, None)
            return {"cleaned": True}
        task, workspace = self.sessions[session]
        if operation == "action":
            return self._action(workspace, payload)
        if operation != "evaluate" or task.task_id != payload["official_task_id"]:
            raise SecurityError("invalid SEABench evaluation request")
        llm = task.validation_backend in {"llm", "hybrid", "auto"} and (
            task.validator_expectation
            or task.validator_spec.get("llm")
            or task.validator_spec.get("safety_llm")
        )
        if llm and not self.config.allow_paid_judges:
            raise BenchmarkBlocked("official SEABench judge requires explicit paid-judge opt-in")
        self.validator.initialize_validation_dispatcher(
            list(self.tasks.values()),
            [self.config.sequence_file],
            workspace,
            {
                "api_key": os.getenv(self.config.judge_key_env, ""),
                "judge_model": self.config.judge_model,
                "base_url": self.config.judge_base_url,
            },
        )
        passed, feedback, backend, observer, detail = self.validator.evaluate_task_answer(
            task, required_text(payload, "answer", allow_empty=True), workspace
        )
        raw = {
            "passed": passed,
            "feedback": feedback,
            "backend": backend,
            "observer": observer,
            "validator": detail,
        }
        private = self.root / "verifier" / session
        private.mkdir(parents=True, exist_ok=False)
        (private / "official_result.json").write_bytes(encode(raw))
        completion = scored_boolean(passed, detail, backend, feedback)
        metrics = {"completion_pass": float(completion)} if completion is not None else {}
        # Hidden safety feedback is never exposed upstream, including in scores.
        safety = scored_boolean(observer.get("passed"), observer, observer.get("backend", "none"))
        if payload["task_partition"] == "final_test" and safety is not None:
            metrics["safety_pass"] = float(safety)
        return OfficialReceipt(
            benchmark="seabench",
            revision=self.config.revision,
            official_task_id=task.task_id,
            run_id=payload["run_id"],
            genome_id=payload["genome_id"],
            status="completed" if completion is not None else "not_available",
            metrics=metrics,
            success=completion,
            raw_sha256=digest(encode(raw)),
        ).model_dump(mode="json")

    @staticmethod
    def _action(workspace: Path, payload: dict[str, Any]) -> dict[str, Any]:
        path = (workspace / payload["path"]).resolve()
        if not path.is_relative_to(workspace.resolve()):
            raise SecurityError("SEABench action escapes the synthetic workspace")
        action = payload["action"]
        if action == "read_file":
            return {"content": path.read_text(encoding="utf-8")[:100000]}
        if action == "list_files":
            return {"files": [p.relative_to(workspace).as_posix() for p in path.iterdir()][:1000]}
        if action == "write_file":
            # Capability changes must use L2 and create a new genome. Files in the
            # official sandbox are task artifacts, never executable HEAS extensions.
            if path.suffix in {".py", ".sh", ".exe", ".dll", ".bat", ".cmd"}:
                raise SecurityError("generated executable proposals require trusted review")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload["content"], encoding="utf-8", newline="\n")
            return {"written": path.relative_to(workspace).as_posix()}
        raise SecurityError("only approved sandbox file actions are enabled")

    def close(self) -> None:
        """Drop live sessions while retaining task artifacts for private verification."""
        self.sessions.clear()
