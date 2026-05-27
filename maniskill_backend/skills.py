"""High-level PullCube skill API surface exposed to LMP programs."""

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
    def pull(self, obj: Any, target: Any) -> bool:
        ...


class UnimplementedManiSkillRobot:
    """Placeholder that makes missing runtime skills explicit during execution."""

    def __getattr__(self, name: str) -> Any:
        raise NotImplementedError(
            f"Skill {name!r} is not implemented yet. Use this API surface for "
            "prompt/static development, then implement the concrete ManiSkill wrapper."
        )
