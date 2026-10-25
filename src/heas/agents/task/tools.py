"""Host-approved tool metadata and executors, independent of task semantics."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from heas.core.domain import Record, ToolBinding, heas_ext
from heas.core.errors import SecurityError
from heas.core.ports import ToolExecutor
from heas.infrastructure.storage import FileStore, digest


class ToolManifest(Record, frozen=True):
    """Record host approval, implementation identity, and allowed execution permissions."""

    tool_id: str
    version_id: str
    description: str
    parameters: dict[str, Any]
    permissions: list[str]
    implementation_sha256: str
    selection_tags: list[str] = []
    requires_tools: list[str] = []


class ToolRegistry:
    """Expose only host-approved, versioned tool implementations and permissions."""

    def __init__(self, store: FileStore, allowed_permissions: frozenset[str] = frozenset()):
        self.store = store
        self.allowed_permissions = allowed_permissions
        self.entries: dict[tuple[str, str], tuple[ToolManifest, type[BaseModel], ToolExecutor]] = {}

    def register(
        self,
        tool_id: str,
        version_id: str,
        description: str,
        arguments: type[BaseModel],
        executor: ToolExecutor,
        *,
        approval_id: str,
        permissions: tuple[str, ...] = (),
        implementation_sha256: str | None = None,
        selection_tags: tuple[str, ...] = (),
        requires_tools: tuple[str, ...] = (),
    ) -> ToolBinding:
        """Register a new host-owned artifact or approved binding, preserving prior identities."""
        if not approval_id or not set(permissions).issubset(self.allowed_permissions):
            raise SecurityError("trusted approval and permitted capabilities required")
        if not isinstance(arguments, type) or not issubclass(arguments, BaseModel):
            raise SecurityError("approved tool arguments require a Pydantic model class")
        if not isinstance(executor, ToolExecutor) or not callable(executor.execute):
            raise SecurityError("approved tool executor requires a callable execute method")
        implementation = implementation_sha256 or getattr(executor, "expected_sha256", None)
        if not implementation:
            raise SecurityError("approved executors require a versioned implementation digest")
        manifest = ToolManifest(
            tool_id=tool_id,
            version_id=version_id,
            description=description,
            parameters=arguments.model_json_schema(),
            permissions=list(permissions),
            implementation_sha256=implementation,
            selection_tags=list(selection_tags),
            requires_tools=list(requires_tools),
        )
        uri = f"artifacts/tools/{tool_id}/{version_id}/manifest.json"
        binding = ToolBinding(
            tool_id=tool_id, version_id=version_id, manifest_ref=uri, permissions=list(permissions)
        )
        if self.store.path(uri).exists():
            if self.store.load(uri, ToolManifest) != manifest:
                raise SecurityError("registered manifest conflicts with archived version")
        else:
            self.store.put(uri, manifest)
        self.entries[(tool_id, version_id)] = (manifest, arguments, executor)
        return ToolBinding(
            **{
                **binding.model_dump(),
                "extensions": {
                    "heas": {
                        "manifest_sha256": digest(self.store.path(uri).read_bytes()),
                        "approval_id": approval_id,
                    }
                },
            }
        )

    def validate(self, binding: ToolBinding) -> None:
        """Reject a capability configuration that violates its fixed host contract."""
        item = self.entries.get((binding.tool_id, binding.version_id))
        if item is None:
            raise SecurityError("tool/version is not approved in this host registry")
        if digest(self.store.path(binding.manifest_ref).read_bytes()) != heas_ext(binding).get(
            "manifest_sha256"
        ):
            raise SecurityError("tool manifest hash mismatch")
        manifest = self.store.load(binding.manifest_ref, ToolManifest)
        if manifest != item[0] or binding.permissions != manifest.permissions:
            raise SecurityError("binding does not match host-approved manifest")
        if digest(self.store.path(binding.manifest_ref).read_bytes()) != heas_ext(binding).get(
            "manifest_sha256"
        ):
            raise SecurityError("tool manifest hash mismatch")
        if not set(binding.permissions).issubset(self.allowed_permissions):
            raise SecurityError("tool exceeds host permissions")

    def schemas_for(self, bindings: list[ToolBinding]) -> dict[str, Any]:
        """Return schemas only after validating every requested tool binding."""
        for binding in bindings:
            self.validate(binding)
        return {b.tool_id: self.entries[(b.tool_id, b.version_id)][0].parameters for b in bindings}

    def validate_dependencies(self, bindings: list[ToolBinding]) -> None:
        """Reject missing or cyclic dependencies among the selected tools."""
        available = {b.tool_id: b for b in bindings}

        def visit(current_tool_id: str, chain: set[str]) -> None:
            if current_tool_id not in available or current_tool_id in chain:
                raise SecurityError("missing or cyclic tool dependency")
            binding = available[current_tool_id]
            self.validate(binding)
            for dep in self.entries[(current_tool_id, binding.version_id)][0].requires_tools:
                visit(dep, chain | {current_tool_id})

        for tool_id in available:
            visit(tool_id, set())

    def execute_binding(self, binding: ToolBinding, args: dict[str, Any], *, timeout: float) -> Any:
        """Invoke the implementation matching a validated host approval manifest."""
        self.validate(binding)
        _, arguments, executor = self.entries[(binding.tool_id, binding.version_id)]
        checked = arguments.model_validate(args)
        return executor.execute(binding.tool_id, checked.model_dump(), timeout=timeout)


class PythonScriptExecutor:
    """Child process for a host-reviewed script, never an Agent-supplied path.

    Not an OS sandbox: host integration must approve code and permissions.
    """

    def __init__(self, script: Path, expected_sha256: str):
        self.script = script.resolve()
        self.expected_sha256 = expected_sha256

    def execute(self, tool_id: str, args: dict[str, Any], *, timeout: float) -> Any:
        """Run the reviewed script only while its bytes match the Host-approved hash.

        The registry supplies validated arguments before this subprocess boundary.
        A timeout ends the call; process separation alone is not an OS sandbox.
        """
        if digest(self.script.read_bytes()) != self.expected_sha256:
            raise SecurityError("approved executor code changed")
        result = subprocess.run(
            [sys.executable, "-I", str(self.script)],
            input=json.dumps({"tool_id": tool_id, "args": args}),
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=True,
            timeout=max(0.001, timeout),
            env={"PYTHONIOENCODING": "utf-8"},
        )
        return json.loads(result.stdout)
