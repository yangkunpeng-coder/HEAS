"""Invoke pinned official DomainAdapter lifecycle methods in a separate process."""

import argparse
import asyncio
import importlib
import subprocess
import sys
from pathlib import Path
from typing import Any

from benchmark_adapters.common.config import EvoAgentBenchConfig
from benchmark_adapters.common.preflight import source_root
from benchmark_adapters.common.process import BenchmarkBlocked
from benchmark_adapters.common.protocol import OfficialReceipt, required_text
from benchmark_adapters.evoagentbench.results import verifier_metrics
from heas.core.errors import SecurityError
from heas.infrastructure.storage import digest, encode


class EvoAgentBenchWorker:
    """Reuse official setup/build_prompt/verify/cleanup; never replace its verifier."""

    def __init__(self, config: EvoAgentBenchConfig, root: Path):
        self.config, self.root = config, root
        official = source_root(config)
        sys.path.insert(0, str(official / "src"))
        module = importlib.import_module("config")
        module.load_config(str(config.official_config))
        self.domain = module.get_domain(config.domain)
        if config.domain == "knowledge_work":
            # GDPVal otherwise uses a shared configured workspace_root across arms.
            module.get_domain_config(config.domain)["workspace_root"] = str(root / "public")
        self.args = argparse.Namespace(
            split="train",
            task=None,
            trials=1,
            parallel=1,
            live=False,
            disk_budget=None,
            max_retries=0,
            parallel_trials=False,
            submit_interval=0,
        )
        self.domain.initialize(self.args)
        self.tasks: dict[str, dict[str, Any]] = {}
        self.sessions: dict[str, tuple[dict[str, Any], dict[str, Any], Path]] = {}

    def _task(self, native_id: str) -> dict[str, Any]:
        if native_id not in self.tasks:
            from benchmark_adapters.evoagentbench.provider import load_official_splits

            splits = load_official_splits(self.config)
            partition = next((name for name, ids in splits.items() if native_id in ids), None)
            if partition is None:
                raise SecurityError("unknown official task identity")
            args = argparse.Namespace(**{**vars(self.args), "task": native_id, "split": partition})
            rows = self.domain.load_tasks(args)
            if len(rows) != 1 or required_text(rows[0], "name") != native_id:
                raise BenchmarkBlocked("official data are absent or task selection is ambiguous")
            self.tasks[native_id] = rows[0]
        return dict(self.tasks[native_id])

    def handle(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Dispatch the HEAS-owned bridge operations to verified official methods."""
        session = required_text(payload, "session_id")
        if operation == "prepare":
            task = self._task(required_text(payload, "official_task_id"))
            job = self.root / "trials" / session
            trial_dir = job / f"{task['name']}__trial_1"
            trial_dir.mkdir(parents=True, exist_ok=False)
            task["_job_dir"] = job
            self.domain.pre_task_trials(task)
            env = self.domain.setup(task, "heas_" + session[-12:], 1)
            self.sessions[session] = task, env, trial_dir
            return {
                "official_prompt": self.domain.build_prompt(task, env),
                "task_input_sha256": digest(
                    encode(
                        {
                            key: task[key]
                            for key in (
                                "name",
                                "question",
                                "question_content",
                                "starter_code",
                                "problem_statement",
                                "repo",
                                "hints_text",
                                "prompt",
                                "reference_files",
                                "reference_file_urls",
                            )
                            if key in task
                        }
                    )
                ),
                "workspace": env.get("workspace_dir"),
                "allowed_actions": [
                    "read_file",
                    "write_file",
                    "list_files",
                    "mcp_call",
                    "container_exec",
                ],
            }
        if operation in {"cleanup", "abandon"}:
            self._cleanup(session)
            return {"cleaned": True}
        task, env, trial_dir = self.sessions[session]
        if operation == "action":
            return self._action(env, payload)
        if operation != "evaluate":
            raise SecurityError("unsupported benchmark worker operation")
        if required_text(task, "name") != required_text(payload, "official_task_id"):
            raise SecurityError("verifier task differs from the prepared task")
        if self.config.domain == "code_implementation" and not self.config.isolation_prefix:
            raise BenchmarkBlocked("generated solutions may only be verified in an isolated worker")
        if (
            self.config.domain in {"knowledge_work", "information_retrieval"}
            and not self.config.allow_paid_judges
        ):
            raise BenchmarkBlocked("official judge calls require an explicit paid-judge opt-in")
        raw = self.domain.verify(
            task,
            env,
            trial_dir,
            agent_result={
                "response": payload["answer"],
                "completion_status": payload["termination_reason"],
                "returncode": 0 if payload["termination_reason"] == "final_answer" else 1,
            },
        )
        metrics, success = verifier_metrics(raw, self.domain.pass_threshold())
        (trial_dir / "official_verifier.json").write_bytes(encode(raw))
        return OfficialReceipt(
            benchmark="evoagentbench",
            revision=self.config.revision,
            official_task_id=required_text(task, "name"),
            run_id=payload["run_id"],
            genome_id=payload["genome_id"],
            status="completed" if metrics else "not_available",
            metrics=metrics,
            success=success,
            raw_sha256=digest(encode(raw)),
        ).model_dump(mode="json")

    def _action(self, env: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        action = payload["action"]
        if action == "container_exec":
            container = env.get("container_name")
            if not container:
                raise SecurityError("this official environment has no approved execution container")
            result = subprocess.run(
                ["docker", "exec", container, "bash", "-lc", payload["command"]],
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                check=False,
            )
            return {
                "stdout": result.stdout[-100000:],
                "stderr": result.stderr[-20000:],
                "returncode": result.returncode,
            }
        if action == "mcp_call":
            name = payload["tool_name"]
            if name not in self.config.mcp_tool_allowlist:
                raise SecurityError("MCP tool was not approved by the host")
            servers = env.get("mcp_servers", {})
            if not servers:
                raise BenchmarkBlocked("official search server is not configured")
            return asyncio.run(
                self._mcp(next(iter(servers.values()))["url"], name, payload["arguments"])
            )
        workspace = env.get("workspace_dir")
        if not workspace:
            raise BenchmarkBlocked("this official domain has no public file workspace")
        root = Path(workspace).resolve()
        path = (root / payload["path"]).resolve()
        if not path.is_relative_to(root):
            raise SecurityError("benchmark action escapes its public workspace")
        if action == "read_file":
            return {"content": path.read_text(encoding="utf-8")[:100000]}
        if action == "list_files":
            return {"files": [p.relative_to(root).as_posix() for p in path.iterdir()][:1000]}
        if action == "write_file":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload["content"], encoding="utf-8", newline="\n")
            return {"written": path.relative_to(root).as_posix()}
        raise SecurityError("unsupported workspace action")

    @staticmethod
    async def _mcp(url: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            from mcp import ClientSession
            from mcp.client.sse import sse_client
        except ImportError as exc:
            raise BenchmarkBlocked(
                "MCP SDK is unavailable; install heas[mcp] in the configured worker interpreter"
            ) from exc

        async with sse_client(url) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments=arguments)
                return {
                    "content": [item.model_dump(mode="json") for item in result.content],
                    "is_error": result.isError,
                }

    def _cleanup(self, session: str) -> None:
        state = self.sessions.pop(session, None)
        if state:
            task, env, _ = state
            try:
                self.domain.cleanup(task, env)
            finally:
                self.domain.post_task_trials(task)

    def close(self) -> None:
        """Pair official lifecycle cleanup/finalize even after an interrupted request."""
        for session in list(self.sessions):
            self._cleanup(session)
        self.domain.finalize()
