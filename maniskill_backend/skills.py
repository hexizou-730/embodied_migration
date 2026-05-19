"""High-level skill API surface exposed to LMP programs.

The concrete ManiSkill implementation will come later. For now these interfaces
let us build prompts, static checks, and oracle/source-copy program shapes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class SkillResult:
    ok: bool
    message: str = ""
    info: dict[str, Any] | None = None

    def __bool__(self) -> bool:
        return self.ok


class SceneLike(Protocol):
    def get_object(self, name: str) -> Any:
        ...

    def get_region(self, name: str) -> Any:
        ...


class RobotSkillAPI(Protocol):
    def grasp(self, obj: Any) -> bool:
        ...

    def align_to_target(self, obj: Any, target: Any, tolerance: float) -> bool:
        ...

    def insert(self, obj: Any, target: Any, speed: float) -> bool:
        ...

    def place(self, obj: Any, target: Any) -> bool:
        ...

    def hook_object(self, tool: Any, obj: Any) -> bool:
        ...

    def pull_with_tool(self, tool: Any, obj: Any, target: Any) -> bool:
        ...


class UnimplementedManiSkillRobot:
    """Placeholder that makes missing runtime skills explicit during execution."""

    def __getattr__(self, name: str) -> Any:
        raise NotImplementedError(
            f"Skill {name!r} is not implemented yet. Use this API surface for "
            "prompt/static development, then implement the concrete ManiSkill wrapper."
        )
