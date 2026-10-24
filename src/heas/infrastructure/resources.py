"""Whole-workspace resource accounting, including task, L2/L3 and analysis calls."""

from collections import Counter
from typing import Any

from heas.core.domain import Usage
from heas.infrastructure.storage import FileStore


class ResourceCheckpoint(set[str]):
    """Identify records already accounted for at a resource window boundary."""

    def __init__(self, calls: set[str], runs: set[str], tools: set[str] | None = None):
        super().__init__(calls)
        self.runs = runs
        self.tools = tools or set()


class ResourceLedger:
    """Reconstruct known and unknown resource use from durable execution receipts."""

    def __init__(self, store: FileStore):
        self.store = store

    def checkpoint(self) -> ResourceCheckpoint:
        """Capture durable record identities at the start of a resource window."""
        return ResourceCheckpoint(
            {
                p.name
                for directory in ("model_calls", "model_requests")
                for p in self.store.path(directory).glob("*.json")
            },
            {p.parent.name for p in self.store.path("runs").glob("*/run.json")},
            {p.name for p in self.store.path("tool_calls").glob("*.json")},
        )

    def report(
        self, after: ResourceCheckpoint | None = None, *, scope_id: str | None = None
    ) -> dict[str, Any]:
        """Read evidence and resource accounting without making model calls."""
        calls = [
            self.store.read(p.relative_to(self.store.root).as_posix())
            for p in self.store.path("model_calls").glob("*.json")
            if after is None or p.name not in after
        ]
        calls.extend(
            {
                **self.store.read(p.relative_to(self.store.root).as_posix()),
                "status": "interrupted_usage_unknown",
            }
            for p in self.store.path("model_requests").glob("*.json")
            if not self.store.path(f"model_calls/{p.name}").exists()
            and (after is None or p.name not in after)
        )
        runs = [
            self.store.read(p.relative_to(self.store.root).as_posix())
            for p in self.store.path("runs").glob("*/run.json")
            if after is None or p.parent.name not in after.runs
        ]
        tools = [
            self.store.read(p.relative_to(self.store.root).as_posix())
            for p in self.store.path("tool_calls").glob("*.json")
            if after is None or p.name not in after.tools
        ]
        if scope_id:
            calls = [c for c in calls if scope_id in c.get("budget_scope_ids", [])]
            runs = [
                r
                for r in runs
                if scope_id in r.get("extensions", {}).get("heas", {}).get("budget_scope_ids", [])
            ]
            tools = [t for t in tools if scope_id in t.get("budget_scope_ids", [])]
        external = [
            self.store.read(p.relative_to(self.store.root).as_posix())
            for p in self.store.path("benchmark_invocations").glob("*.json")
        ]
        if scope_id:
            external = [r for r in external if scope_id in r.get("budget_scope_ids", [])]
        elif after is not None:
            external = [r for r in external if r["run_id"] not in after.runs]
        unknown = sum(
            not c.get("usage", {}).get("token_usage_available", True) or "usage" not in c
            for c in calls
        )
        known_input = sum(c.get("usage", {}).get("input_tokens", 0) for c in calls)
        known_output = sum(c.get("usage", {}).get("output_tokens", 0) for c in calls)
        currencies = {c.get("currency") for c in calls if c.get("currency")}
        priced = bool(calls) and all(c.get("monetary_cost") is not None for c in calls)
        run_ids = {run["run_id"] for run in runs}
        external_times = {
            row["run_id"]: self.store.read(f"benchmark_completions/{row['run_id']}.json")[
                "wall_seconds"
            ]
            for row in external
            if self.store.path(f"benchmark_completions/{row['run_id']}.json").exists()
        }
        external_wall = sum(external_times.values())

        def covered_by_run(call: dict[str, Any]) -> bool:
            return any(
                scope == f"run:{identity}"
                for identity in run_ids
                for scope in call.get("budget_scope_ids", [])
            )

        # Run wall time already covers nested selector calls and evaluation.
        # Keep their diagnostic totals, adding only otherwise unaccounted spans.
        remaining_external_wall = sum(
            value for identity, value in external_times.items() if identity not in run_ids
        )
        legacy_overlap = bool(runs) and any(
            call["purpose"] == "skill_selection"
            and not any(scope.startswith("run:") for scope in call.get("budget_scope_ids", []))
            for call in calls
        )
        return {
            "schema_version": "1.1",
            "model_calls": len(calls),
            "input_tokens": None if unknown or external else known_input,
            "output_tokens": None if unknown or external else known_output,
            "known_input_tokens": known_input,
            "known_output_tokens": known_output,
            "token_accounting_methods": sorted(
                {
                    c.get("usage", {})
                    .get("extensions", {})
                    .get("heas", {})
                    .get("token_accounting", "provider_reported")
                    for c in calls
                }
            ),
            "failed_calls": sum(c["status"] != "completed" for c in calls),
            "calls_by_purpose": dict(Counter(c["purpose"] for c in calls)),
            "monetary_cost": sum(c["monetary_cost"] for c in calls)
            if priced and len(currencies) == 1 and not external
            else None,
            "currency": next(iter(currencies)) if len(currencies) == 1 else None,
            "unknown_usage_calls": unknown,
            "external_invocations": len(external),
            "external_usage_unknown": bool(external),
            "known_external_wall_seconds": external_wall,
            "wall_seconds_overlap_possible": legacy_overlap,
            "tool_calls": max(
                sum(r["usage"]["tool_calls"] for r in runs),
                len(tools),
            ),
            "task_wall_seconds": sum(r["usage"]["wall_seconds"] for r in runs),
            "model_wall_seconds": sum(c.get("wall_seconds", 0) for c in calls),
            "wall_seconds": sum(r["usage"]["wall_seconds"] for r in runs)
            + sum(
                c.get("wall_seconds", 0)
                for c in calls
                if c["purpose"] != "task" and not covered_by_run(c)
            )
            + remaining_external_wall,
        }

    def usage(self, after: ResourceCheckpoint) -> Usage:
        """Return measured usage while preserving unknown token or monetary accounting."""
        report = self.report(after)
        return Usage(
            model_calls=report["model_calls"],
            input_tokens=report["known_input_tokens"],
            output_tokens=report["known_output_tokens"],
            tool_calls=report["tool_calls"],
            wall_seconds=report["wall_seconds"],
            token_usage_available=report["unknown_usage_calls"] == 0
            and not report["external_usage_unknown"],
            monetary_cost=report["monetary_cost"],
        )
