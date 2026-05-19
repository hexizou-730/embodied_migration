"""Lazy ManiSkill environment adapter.

This file intentionally avoids importing ManiSkill at module import time. That
keeps WSL2/static-development workflows usable even when Vulkan or GPU rendering
is not ready.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class StepResult:
    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


def can_import_maniskill() -> Tuple[bool, str]:
    try:
        import gymnasium  # noqa: F401
        import mani_skill  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on local install
        return False, repr(exc)
    return True, "ok"


class ManiSkillEnvAdapter:
    def __init__(
        self,
        env_id: str,
        robot_uid: Optional[str] = None,
        render_mode: Optional[str] = None,
        **make_kwargs: Any,
    ) -> None:
        self.env_id = env_id
        self.robot_uid = robot_uid
        self.render_mode = render_mode
        self.make_kwargs = dict(make_kwargs)
        self.env = None

    def make(self) -> Any:
        if self.env is not None:
            return self.env

        import gymnasium as gym
        import mani_skill.envs  # noqa: F401  # registers ManiSkill envs

        kwargs = dict(self.make_kwargs)
        if self.robot_uid is not None:
            kwargs["robot_uids"] = self.robot_uid
        if self.render_mode is not None:
            kwargs["render_mode"] = self.render_mode

        self.env = gym.make(self.env_id, **kwargs)
        return self.env

    def reset(self, seed: Optional[int] = None) -> Tuple[Any, Dict[str, Any]]:
        env = self.make()
        return env.reset(seed=seed)

    def step(self, action: Any) -> StepResult:
        env = self.make()
        obs, reward, terminated, truncated, info = env.step(action)
        return StepResult(obs, float(reward), bool(terminated), bool(truncated), info)

    def render(self) -> Any:
        env = self.make()
        return env.render()

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None

