"""Task-neutral ManiSkill runtime exposed to generated adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SkillTarget:
    name: str
    kind: str


class ManiSkillSceneAdapter:
    """Minimal scene vocabulary used by generated high-level programs."""

    def get_object(self, name: str) -> SkillTarget:
        return SkillTarget(name=name, kind="object")

    def get_region(self, name: str) -> SkillTarget:
        return SkillTarget(name=name, kind="region")


class ManiSkillDeltaEERobot:
    """Generic delta end-effector execution helpers with no task policy."""

    def __init__(
        self,
        env: Any,
        *,
        move_steps: int = 14,
        settle_steps: int = 10,
        max_delta_m: float = 0.07,
        gripper_open: float = 1.0,
        gripper_close: float = -1.0,
        control_mode: Optional[str] = None,
    ) -> None:
        self.env = env
        self.move_steps = move_steps
        self.settle_steps = settle_steps
        self.max_delta_m = max_delta_m
        self.gripper_open = gripper_open
        self.gripper_close = gripper_close
        self.control_mode = control_mode
        self.last_info: Dict[str, Any] = {}
        self.terminated = False
        self.truncated = False
        self.events: List[Dict[str, Any]] = []
        self._validate_action_space()

    def _validate_action_space(self) -> None:
        space = getattr(self.env, "action_space", None)
        shape = getattr(space, "shape", None)
        if not shape or len(shape) != 1 or int(shape[0]) < 3:
            raise RuntimeError(
                "Adapter requires a one-dimensional Box-like action space "
                f"with at least three entries, got {shape!r}."
            )

    def execution_log(self) -> List[Dict[str, Any]]:
        return list(self.events)

    def _move_towards(self, target_pos: np.ndarray, *, gripper: float, steps: int) -> None:
        for _ in range(max(1, steps)):
            if self._early_stop():
                return
            delta = np.asarray(target_pos, dtype=np.float32) - self._tcp_pos()
            if np.linalg.norm(delta) < 0.01:
                return
            command = np.clip(delta / self.max_delta_m, -1.0, 1.0)
            self._step(self._make_action(command, gripper=gripper))

    def _repeat_action(self, delta_xyz: np.ndarray, *, gripper: float, steps: int) -> None:
        action = self._make_action(delta_xyz, gripper=gripper)
        for _ in range(max(1, steps)):
            if self._early_stop():
                return
            self._step(action)

    def _early_stop(self) -> bool:
        return bool(self.terminated or self.truncated)

    def _make_action(self, delta_xyz: np.ndarray, *, gripper: float = 0.0) -> Any:
        space = self.env.action_space
        action = np.zeros(space.shape, dtype=getattr(space, "dtype", np.float32))
        flat = action.reshape(-1)
        flat[:3] = self._world_delta_to_action_delta(delta_xyz)
        if flat.size >= 4:
            flat[-1] = float(gripper)
        low = getattr(space, "low", None)
        high = getattr(space, "high", None)
        if low is not None and high is not None:
            action = np.clip(action, low, high)
        return action

    def _robot_root_quat(self) -> Optional[np.ndarray]:
        """Return the root-to-world rotation used by root-frame EE controllers."""

        agent = getattr(self._base_env(), "agent", None)
        robot = getattr(agent, "robot", None)
        pose = getattr(robot, "pose", None)
        if pose is None:
            get_pose = getattr(robot, "get_pose", None)
            pose = get_pose() if callable(get_pose) else None
        quat = getattr(pose, "q", None)
        if quat is None:
            return None
        return _first_vector(quat, 4)

    def _world_delta_to_action_delta(self, delta_xyz: np.ndarray) -> np.ndarray:
        """Express a normalized world-frame translation in the controller root frame."""

        delta = _first_vector(delta_xyz, 3)
        if not str(self.control_mode or "").startswith("pd_ee_delta_"):
            return delta
        quat = self._robot_root_quat()
        if quat is None:
            return delta
        return _quat_rotate_wxyz(_quat_conjugate_wxyz(quat), delta)

    def _step(self, action: Any) -> None:
        _, _, terminated, truncated, info = self.env.step(action)
        self.last_info = dict(info or {})
        self.terminated = self.terminated or _scalar_bool(terminated)
        self.truncated = self.truncated or _scalar_bool(truncated)

    def _base_env(self) -> Any:
        return getattr(self.env, "unwrapped", self.env)

    def _tcp_pos(self) -> np.ndarray:
        agent = self._base_env().agent
        tcp_pose = getattr(agent, "tcp_pose", None)
        if tcp_pose is not None:
            return _first_vector(tcp_pose.p, 3)
        tcp = getattr(agent, "tcp", None)
        if tcp is not None and getattr(tcp, "pose", None) is not None:
            return _first_vector(tcp.pose.p, 3)
        raise RuntimeError("Could not read the ManiSkill agent TCP pose.")

    def _log(self, api: str, args: Dict[str, Any], result: Any, ok: bool, message: str = "") -> bool:
        self.events.append(
            {
                "step": len(self.events) + 1,
                "api": api,
                "args": dict(args),
                "result": bool(result),
                "ok": bool(ok),
                "message": message,
                "failure_type": "" if ok else "execution failure",
            }
        )
        return bool(ok)

    def _fail(self, api: str, args: Dict[str, Any], message: str) -> bool:
        return self._log(api, args, False, False, message)


def _to_numpy(value: Any) -> np.ndarray:
    try:
        import torch

        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(value)


def _scalar_bool(value: Any) -> bool:
    try:
        return bool(_to_numpy(value).reshape(-1)[0])
    except Exception:
        return bool(value)


def _first_vector(value: Any, width: int) -> np.ndarray:
    """Normalize a single-environment batched vector to a flat float32 value."""

    array = np.asarray(_to_numpy(value), dtype=np.float32)
    if array.ndim == 0 or array.size < width:
        raise ValueError(f"Expected a vector with at least {width} values, got {array.shape!r}.")
    if array.ndim > 1 and array.shape[-1] >= width:
        return array.reshape(-1, array.shape[-1])[0, :width].copy()
    return array.reshape(-1)[:width].copy()


def _quat_conjugate_wxyz(quat: np.ndarray) -> np.ndarray:
    value = _first_vector(quat, 4)
    norm = float(np.linalg.norm(value))
    if norm < 1e-8:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    value /= norm
    value[1:] *= -1.0
    return value


def _quat_rotate_wxyz(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q = _first_vector(quat, 4)
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        return _first_vector(vector, 3)
    q /= norm
    v = _first_vector(vector, 3)
    q_xyz = q[1:]
    twice_cross = 2.0 * np.cross(q_xyz, v)
    return (v + q[0] * twice_cross + np.cross(q_xyz, twice_cross)).astype(np.float32)


_ENTITY_ALIAS_GROUPS = (
    ("cube", "object", "obj"),
    ("goal", "target", "goal_region", "target_region"),
)


class ManiSkillDynamicRobot(ManiSkillDeltaEERobot):
    """Task-neutral helpers available to dynamically generated adapters.

    The class exposes observations and action execution, but it never implements
    a task policy or overrides ManiSkill's official success criterion.
    """

    def __init__(self, env: Any, *, control_mode: str, robot_uid: str, **kwargs: Any) -> None:
        self.robot_uid = robot_uid
        self.action_steps = 0
        super().__init__(env, control_mode=control_mode, **kwargs)

    def _step(self, action: Any) -> None:
        super()._step(action)
        self.action_steps += 1

    def _validate_action_space(self) -> None:
        space = getattr(self.env, "action_space", None)
        shape = getattr(space, "shape", None)
        if not shape or len(shape) != 1 or int(shape[0]) < 3:
            raise RuntimeError(
                "Dynamic adapter requires a one-dimensional Box-like action space "
                f"with at least three entries, got {shape!r}."
            )

    def _make_action(self, delta_xyz: np.ndarray, *, gripper: float = 0.0) -> Any:
        """Map normalized world xyz plus gripper into the current action space."""

        return super()._make_action(delta_xyz, gripper=gripper)

    def _entity_catalog(self) -> Dict[str, Any]:
        """Return actor-like public task fields without traversing object graphs."""

        base = self._base_env()
        catalog: Dict[str, Any] = {}
        for name, value in vars(base).items():
            if name.startswith("_") or name in {"agent", "scene"}:
                continue
            self._add_entity(catalog, name, value)
        return catalog

    @staticmethod
    def _add_entity(catalog: Dict[str, Any], path: str, value: Any) -> None:
        pose = getattr(value, "pose", None)
        if pose is not None and getattr(pose, "p", None) is not None:
            catalog[path] = value
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value[:32]):
                ManiSkillDynamicRobot._add_entity(catalog, f"{path}[{index}]", item)
        elif isinstance(value, dict):
            for key, item in list(value.items())[:32]:
                ManiSkillDynamicRobot._add_entity(catalog, f"{path}[{key!r}]", item)

    def _entity(self, name: str) -> Any:
        catalog = self._entity_catalog()
        if name in catalog:
            return catalog[name]
        matches = [value for path, value in catalog.items() if path.rsplit(".", 1)[-1] == name]
        if len(matches) == 1:
            return matches[0]
        alias_matches = self._entity_alias_matches(name, catalog)
        if len(alias_matches) == 1:
            return alias_matches[0]
        direct = getattr(self._base_env(), name, None)
        if direct is not None:
            return direct
        available = ", ".join(sorted(catalog)[:40])
        aliases = self._entity_aliases(catalog)
        raise AttributeError(
            f"Unknown or ambiguous task entity {name!r}. Available pose entities: {available}. "
            f"Unambiguous semantic aliases: {aliases}"
        )

    @staticmethod
    def _entity_alias_matches(name: str, catalog: Dict[str, Any]) -> list[Any]:
        """Resolve only common, unambiguous task names such as cube -> obj."""

        group = next((items for items in _ENTITY_ALIAS_GROUPS if name in items), ())
        if not group:
            return []
        matching_keys = [key for key in group if key in catalog]
        return [catalog[key] for key in matching_keys]

    @classmethod
    def _entity_aliases(cls, catalog: Dict[str, Any]) -> Dict[str, str]:
        aliases: Dict[str, str] = {}
        for group in _ENTITY_ALIAS_GROUPS:
            matching_keys = [key for key in group if key in catalog]
            if len(matching_keys) != 1:
                continue
            resolved = matching_keys[0]
            for alias in group:
                if alias not in catalog:
                    aliases[alias] = resolved
        return aliases

    def _actor(self, name: str) -> Any:
        return self._entity(name)

    def _actor_pos(self, name: str) -> np.ndarray:
        return _first_vector(self._entity(name).pose.p, 3)

    def _entity_pos(self, name: str) -> np.ndarray:
        return self._actor_pos(name)

    def _entity_quat(self, name: str) -> np.ndarray:
        return _first_vector(self._entity(name).pose.q, 4)

    def _region_pos(self, name: str) -> np.ndarray:
        return self._actor_pos(name)

    def _is_grasping_entity(self, name: str) -> bool:
        try:
            return _scalar_bool(self._base_env().agent.is_grasping(self._entity(name)))
        except Exception:
            return False

    def _official_evaluation(self) -> Dict[str, Any]:
        result = dict(self._base_env().evaluate() or {})
        self.last_info = {**self.last_info, **result}
        return result

    def _official_success(self) -> bool:
        result = self._official_evaluation()
        return _scalar_bool(result.get("success", False))

    def _snapshot(self) -> Dict[str, Any]:
        entities = {}
        for name, actor in self._entity_catalog().items():
            try:
                entities[name] = {
                    "position": np.round(_to_numpy(actor.pose.p), 6).tolist(),
                    "quaternion": np.round(_to_numpy(actor.pose.q), 6).tolist(),
                }
            except Exception:
                continue
        try:
            tcp = np.round(self._tcp_pos(), 6).tolist()
        except Exception:
            tcp = None
        return {
            "tcp": tcp,
            "entities": entities,
            "entity_aliases": self._entity_aliases(self._entity_catalog()),
            "official_evaluation": _jsonable(self._official_evaluation()),
            "terminated": bool(self.terminated),
            "truncated": bool(self.truncated),
        }

    def _record_state(self, phase: str, **values: Any) -> None:
        self.events.append(
            {
                "step": len(self.events) + 1,
                "api": "state",
                "phase": str(phase),
                "values": _jsonable(values),
                "result": True,
                "ok": True,
                "message": "",
                "failure_type": "",
            }
        )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        array = _to_numpy(value)
        if array.size == 1:
            return array.reshape(-1)[0].item()
        return array.tolist()
    except Exception:
        return repr(value)


def iter_pose_entities(base_env: Any) -> Iterable[Tuple[str, Any]]:
    """Public helper used by environment discovery without constructing a robot."""

    catalog: Dict[str, Any] = {}
    for name, value in vars(base_env).items():
        if name.startswith("_") or name in {"agent", "scene"}:
            continue
        ManiSkillDynamicRobot._add_entity(catalog, name, value)
    return tuple(sorted(catalog.items()))
