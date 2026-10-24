"""Bounded JSON-line IPC with an explicitly enabled, independently owned worker."""

import json
import os
import queue
import subprocess
import sys
import threading
from contextlib import suppress
from pathlib import Path
from typing import IO, Any

from benchmark_adapters.common.config import IntegrationConfig
from benchmark_adapters.common.preflight import preflight
from heas.core.domain import uid
from heas.core.errors import HEASError, SecurityError
from heas.infrastructure.storage import encode


class BenchmarkBlocked(HEASError):
    """Indicate missing official dependencies or an unapproved execution boundary."""


class BenchmarkProtocolError(HEASError):
    """Indicate an incomplete, mismatched, or invalid official bridge response."""


class ProcessBridge:
    """Own one serial worker connection for an application or isolated trial branch.

    Any transport or protocol failure permanently closes this connection. The
    Host must inspect partial effects before explicitly constructing another
    bridge; requests are never replayed here.
    """

    max_response_chars = 2_000_000

    def __init__(self, config: IntegrationConfig, worker_root: Path):
        self.config, self.worker_root = config, worker_root
        self.process: subprocess.Popen[str] | None = None
        self.responses: queue.Queue[str | None] = queue.Queue(maxsize=2)
        self.log: IO[str] | None = None
        self.closed = False

    def _start(self) -> None:
        report = preflight(self.config)
        if not report["ready"]:
            missing = [c["name"] for c in report["checks"] if not c["ready"]]
            raise BenchmarkBlocked("official worker unavailable: " + ", ".join(missing))
        self.worker_root.mkdir(parents=True, exist_ok=True)
        config_path = self.worker_root / "worker_config.json"
        config_path.write_bytes(encode(self.config))
        interpreter = self.config.worker_python or sys.executable
        command = [
            *self.config.isolation_prefix,
            interpreter,
            "-m",
            "benchmark_adapters.worker",
            "--config",
            str(config_path),
            "--root",
            str(self.worker_root),
        ]
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        if self.config.offline_data:
            env.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
        self.log = (self.worker_root / "worker.log").open("a", encoding="utf-8")
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

        process = self.process

        def read() -> None:
            assert process and process.stdout
            while not self.closed:
                # Bound allocation even when a worker omits its line terminator.
                line = process.stdout.readline(self.max_response_chars + 1)
                while not self.closed:
                    try:
                        self.responses.put(line or None, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                if not line or len(line) > self.max_response_chars:
                    return

        threading.Thread(target=read, daemon=True).start()

    def request(self, operation: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        """Exchange one correlated JSON request and reject uncertain worker state.

        Args:
            operation: Host-approved worker operation, potentially with side effects.
            payload: JSON data passed to that operation.
            timeout: Maximum seconds to await a response after writing the request.

        Returns:
            The result object from a completed, matching worker receipt.

        Raises:
            BenchmarkBlocked: If the bridge is closed or the worker reports failure.
            BenchmarkProtocolError: If the response violates the wire contract.
            SecurityError: If the response belongs to a different request.
            TimeoutError: If no receipt arrives; partial effects are not replayed.
        """
        if self.closed:
            raise BenchmarkBlocked("worker is closed; inspect effects before creating a new bridge")
        try:
            if self.process is None:
                self._start()
            assert self.process and self.process.stdin
            identity = uid("rpc")
            self.process.stdin.write(
                json.dumps({"id": identity, "operation": operation, "payload": payload}) + "\n"
            )
            self.process.stdin.flush()
            try:
                line = self.responses.get(timeout=timeout)
            except queue.Empty as exc:
                raise TimeoutError(
                    "official worker request timed out; effects were not replayed"
                ) from exc
            return self._decode_response(line, identity)
        except BaseException:
            # Even cancellation may follow a completed side effect with a lost receipt.
            self.close()
            raise

    def _decode_response(self, line: str | None, identity: str) -> dict[str, Any]:
        if line is None or len(line) > self.max_response_chars:
            raise BenchmarkProtocolError("official worker exited or exceeded response bounds")

        def reject_constant(value: str) -> None:
            raise ValueError("non-finite JSON constant")

        try:
            response = json.loads(line, parse_constant=reject_constant)
        except (ValueError, RecursionError) as exc:
            raise BenchmarkProtocolError("official worker response is not valid JSON") from exc
        if not isinstance(response, dict):
            raise BenchmarkProtocolError("official worker response must be an object")
        if response.get("id") != identity:
            raise SecurityError("official worker response identity mismatch")
        if response.get("status") == "failed":
            error_type = response.get("error_type")
            if (
                not isinstance(error_type, str)
                or not error_type.isidentifier()
                or len(error_type) > 100
            ):
                raise BenchmarkProtocolError("official worker failure requires an error type")
            raise BenchmarkBlocked("official worker failed: " + error_type)
        if response.get("status") != "completed":
            raise BenchmarkProtocolError("official worker response has an unknown status")
        if not isinstance(response.get("result"), dict):
            raise BenchmarkProtocolError("official worker result must be an object")
        return response["result"]

    def close(self) -> None:
        """Stop the process; retain logs and all potentially partial official artifacts."""
        self.closed = True
        if self.process:
            process = self.process
            if process.poll() is None:
                if process.stdin:
                    with suppress(OSError, ValueError):
                        process.stdin.close()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    with suppress(OSError):
                        process.terminate()
                        process.kill()
                    with suppress(OSError, subprocess.TimeoutExpired):
                        process.wait(timeout=5)
            stdout = getattr(process, "stdout", None)
            if stdout:
                with suppress(OSError, ValueError):
                    stdout.close()
            self.process = None
        if self.log:
            self.log.close()
            self.log = None
