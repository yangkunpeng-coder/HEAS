"""Serial batch journal; completed units resume without replaying tool side effects."""

from collections.abc import Callable

from heas.core.domain import RunPlan, TaskRunRecord, TaskSpec, now, uid
from heas.core.errors import HEASError, RecoveryRequired
from heas.infrastructure.storage import FileStore, digest


class SerialBatchScheduler:
    """Journal serial execution units so completed side effects are not replayed."""

    def __init__(self, store: FileStore):
        self.store = store

    def execute(
        self,
        plan: RunPlan,
        tasks: list[TaskSpec],
        run: Callable[..., TaskRunRecord],
        *,
        resume: bool = False,
    ) -> list[TaskRunRecord]:
        """Journal serial task units while preserving one frozen genome per batch.

        Completed units can be reused during explicit recovery. Retries follow the
        Host plan and retain their own run identities; uncertain effects are never
        silently treated as completed work.
        """
        directory = f"batches/{plan.run_plan_id}"
        results: list[TaskRunRecord] = []
        all_run_ids: list[str] = []
        reused = 0

        def report(status: str, error_type: str | None = None) -> None:
            """Read evidence and resource accounting without making model calls."""
            index = len(list(self.store.path(f"{directory}/reports").glob("*.json"))) + 1
            self.store.put(
                f"{directory}/reports/{index:06d}.json",
                {
                    "schema_version": "1.2",
                    "run_plan_id": plan.run_plan_id,
                    "status": status,
                    "error_type": error_type,
                    "recorded_at": now().isoformat(),
                    "terminal_run_ids": [r.run_id for r in results],
                    "attempt_run_ids": all_run_ids,
                    "reused_units": reused,
                    "task_count": len(tasks),
                },
            )

        try:
            for genome_id in plan.genome_ids:
                frozen = self.store.genome(genome_id)
                for task in tasks:
                    for attempt in range(plan.attempts_per_task):
                        unit_id = digest(f"{genome_id}|{task.task_id}|{attempt}".encode())[7:]
                        unit_uri = f"{directory}/units/{unit_id}"
                        completed = f"{unit_uri}/completed.json"
                        if self.store.path(completed).exists():
                            if not resume:
                                raise HEASError("batch already started; use resume=True")
                            item = self.store.read(completed)
                            results.append(
                                self.store.load(f"runs/{item['run_id']}/run.json", TaskRunRecord)
                            )
                            all_run_ids.extend(item["attempt_run_ids"])
                            reused += 1
                            continue
                        starts = [
                            self.store.read(p.relative_to(self.store.root).as_posix())
                            for p in self.store.path(f"{unit_uri}/attempts").glob("*.json")
                        ]
                        starts.sort(key=lambda attempt_entry: attempt_entry["retry_index"])
                        unit_runs = [attempt_entry["run_id"] for attempt_entry in starts]
                        retry = 0
                        terminal = None
                        if starts:
                            last = starts[-1]
                            run_uri = f"runs/{last['run_id']}/run.json"
                            if self.store.path(run_uri).exists():
                                previous = self.store.load(run_uri, TaskRunRecord)
                                if (
                                    previous.termination_reason
                                    not in plan.retry_termination_reasons
                                    or last["retry_index"] >= plan.max_retries
                                ):
                                    terminal = previous
                                retry = last["retry_index"] + 1
                            else:
                                if not resume or not plan.retry_interrupted:
                                    raise RecoveryRequired(
                                        "interrupted unit may have tool side effects; "
                                        "explicit retry_interrupted authorization required"
                                    )
                                self.store.recover_trace(last["run_id"])
                                retry = last["retry_index"] + 1
                        while terminal is None:
                            if retry > plan.max_retries:
                                raise HEASError("interrupted unit exhausted its retry budget")
                            if self.store.genome(genome_id) != frozen:
                                raise HEASError("frozen genome changed during resumed batch")
                            run_id = uid("run")
                            self.store.put(
                                f"{unit_uri}/attempts/{retry:04d}.json",
                                {"schema_version": "1.2", "run_id": run_id, "retry_index": retry},
                            )
                            unit_runs.append(run_id)
                            current = run(
                                frozen, task, plan, attempt, run_id=run_id, retry_index=retry
                            )
                            if (
                                current.termination_reason in plan.retry_termination_reasons
                                and retry < plan.max_retries
                            ):
                                retry += 1
                            else:
                                terminal = current
                        self.store.put(
                            completed,
                            {
                                "schema_version": "1.2",
                                "run_id": terminal.run_id,
                                "attempt_run_ids": unit_runs,
                            },
                        )
                        results.append(terminal)
                        all_run_ids.extend(unit_runs)
                        report("running")
                self.store.genome(genome_id)
            report("completed")
            return results
        except BaseException as exc:
            report(
                "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
                type(exc).__name__,
            )
            raise
