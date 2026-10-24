"""Serve the HEAS-owned JSON-line protocol around lazily imported official projects."""

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

from benchmark_adapters.common.config import load_config


def build_worker(config_path: Path, root: Path) -> Any:
    """Select a concrete bridge without importing any official project into HEAS Core."""
    config = load_config(config_path)
    root.mkdir(parents=True, exist_ok=True)
    if config.benchmark == "evoagentbench":
        from benchmark_adapters.evoagentbench.worker import EvoAgentBenchWorker

        return EvoAgentBenchWorker(config, root)
    if config.benchmark == "seabench":
        from benchmark_adapters.seabench.worker import SEABenchWorker

        return SEABenchWorker(config, root)
    from benchmark_adapters.skilllearnbench.worker import SkillLearnBenchWorker

    return SkillLearnBenchWorker(config, root)


def main() -> None:
    """Redirect third-party output to a private log, preserving a strict public IPC channel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    channel = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        worker = build_worker(args.config, args.root)
    try:
        for line in sys.stdin:
            request = json.loads(line)
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    result = worker.handle(request["operation"], request["payload"])
                response = {"id": request["id"], "status": "completed", "result": result}
            except Exception as exc:
                # Third-party exception text can contain credentials or answer keys.
                response = {
                    "id": request["id"],
                    "status": "failed",
                    "error_type": type(exc).__name__,
                }
            channel.write(json.dumps(response, ensure_ascii=False, allow_nan=False) + "\n")
            channel.flush()
    finally:
        with contextlib.redirect_stdout(sys.stderr):
            worker.close()


if __name__ == "__main__":
    main()
