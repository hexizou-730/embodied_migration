"""Generic runtime surface for adapters synthesized for unregistered tasks."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import numpy as np

from .skill_adapter import ManiSkillDeltaEERobot, _scalar_bool, _to_numpy


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
        """Conservative fallback mapping; generated adapters may override it."""

        space = self.env.action_space
        action = np.zeros(space.shape, dtype=getattr(space, "dtype", np.float32))
        flat = action.reshape(-1)
        flat[:3] = np.asarray(delta_xyz, dtype=np.float32).reshape(-1)[:3]
        if flat.size >= 4:
            flat[-1] = float(gripper)
        low = getattr(space, "low", None)
        high = getattr(space, "high", None)
        if low is not None and high is not None:
            action = np.clip(action, low, high)
        return action

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
        direct = getattr(self._base_env(), name, None)
        if direct is not None:
            return direct
        available = ", ".join(sorted(catalog)[:40])
        raise AttributeError(f"Unknown task entity {name!r}. Available pose entities: {available}")

    def _actor(self, name: str) -> Any:
        return self._entity(name)

    def _actor_pos(self, name: str) -> np.ndarray:
        return _to_numpy(self._entity(name).pose.p)

    def _entity_pos(self, name: str) -> np.ndarray:
        return self._actor_pos(name)

    def _entity_quat(self, name: str) -> np.ndarray:
        return _to_numpy(self._entity(name).pose.q)

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
