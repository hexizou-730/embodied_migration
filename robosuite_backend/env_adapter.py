"""Optional utilities for launching real robosuite environments.

The migration experiment can run symbolically without robosuite installed. When
robosuite is available, these helpers open the corresponding MuJoCo scene so a
demo can show the richer task environment alongside the LMP execution logs.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from robosuite_backend.profiles import RobosuiteProfile
from robosuite_backend.tasks import RobosuiteTask


def availability_message() -> str:
    try:
        import mujoco  # noqa: F401
        import robosuite  # noqa: F401
    except Exception as exc:
        return (
            "robosuite backend is not installed. Install with:\n"
            "  conda activate em\n"
            "  python -m pip install -r requirements-robosuite.txt\n"
            f"Import error: {type(exc).__name__}: {exc}"
        )
    return "robosuite backend is available"


def is_available() -> bool:
    return availability_message() == "robosuite backend is available"


def make_env(
    task: RobosuiteTask,
    profile: RobosuiteProfile,
    has_renderer: bool = False,
):
    """Create a robosuite env for visual inspection.

    This does not solve the low-level MuJoCo task; LMP execution happens through
    the high-level skill wrapper in `symbolic.py`.
    """
    try:
        import robosuite as suite
    except Exception as exc:
        raise RuntimeError(availability_message()) from exc

    controller_configs = None
    try:
        from robosuite.controllers import load_composite_controller_config

        controller_configs = load_composite_controller_config(controller="BASIC")
    except Exception:
        # robosuite versions differ. Let suite.make fall back to defaults.
        controller_configs = None

    kwargs = {
        "robots": list(profile.robosuite_robots) if len(profile.robosuite_robots) > 1 else profile.robosuite_robots[0],
        "has_renderer": has_renderer,
        "has_offscreen_renderer": False,
        "use_object_obs": True,
        "use_camera_obs": False,
        "control_freq": 20,
        "horizon": 1000,
        "ignore_done": True,
        "hard_reset": True,
    }
    if profile.env_configuration not in {"single-robot", "mobile"}:
        kwargs["env_configuration"] = profile.env_configuration
    if controller_configs is not None:
        kwargs["controller_configs"] = controller_configs
    return suite.make(task.robosuite_env, **kwargs)


def preview_env(env, steps: int = 120, realtime: bool = False) -> None:
    """Render a passive preview with zero actions."""
    low, _high = env.action_spec
    action = np.zeros_like(low)
    for _ in range(steps):
        env.step(action)
        if getattr(env, "has_renderer", False):
            env.render()
        if realtime:
            time.sleep(0.05)


def animate_env(env, task_name: str = "two_arm_lift", seconds: float = 12.0,
                realtime: bool = False) -> None:
    """Run a simple scripted low-level motion so the GUI is visibly active.

    This animation is for demonstration only. The paper-side success checker is
    still driven by the high-level migration skills.
    """
    low, high = env.action_spec
    action_dim = int(low.shape[0])
    steps = max(1, int(seconds * 20))
    print(f"\n[GUI] Animating robosuite robot for {seconds:.0f}s ({task_name})...")
    for i in range(steps):
        phase = i / max(1, steps - 1)
        if task_name == "two_arm_lift":
            action = _two_arm_lift_action(action_dim, phase)
        elif task_name == "two_arm_handover":
            action = _two_arm_handover_action(action_dim, phase)
        elif task_name == "two_arm_peg_in_hole":
            action = _two_arm_peg_action(action_dim, phase)
        else:
            action = _generic_motion_action(action_dim, phase)
        action = np.clip(action, low, high)
        env.step(action)
        if getattr(env, "has_renderer", False):
            env.render()
        if realtime:
            time.sleep(0.05)


def _split_arm_action(action_dim: int) -> tuple[np.ndarray, slice, slice]:
    action = np.zeros(action_dim)
    if action_dim >= 14:
        return action, slice(0, 7), slice(7, 14)
    half = max(1, action_dim // 2)
    return action, slice(0, half), slice(half, action_dim)


def _set_arm_delta(action: np.ndarray, arm_slice: slice, xyz=(0.0, 0.0, 0.0),
                   grip: float = 0.0) -> None:
    arm = action[arm_slice]
    if len(arm) >= 3:
        arm[0:3] = xyz
    if len(arm) >= 7:
        arm[6] = grip
    elif len(arm) >= 1:
        arm[-1] = grip


def _two_arm_lift_action(action_dim: int, phase: float) -> np.ndarray:
    action, left, right = _split_arm_action(action_dim)
    if phase < 0.35:
        # Approach the two pot handles from opposite sides.
        _set_arm_delta(action, left, xyz=(0.10, -0.14, -0.06), grip=1.0)
        _set_arm_delta(action, right, xyz=(0.10, 0.14, -0.06), grip=1.0)
    elif phase < 0.55:
        # Close grippers and settle.
        _set_arm_delta(action, left, xyz=(0.02, -0.04, -0.02), grip=-1.0)
        _set_arm_delta(action, right, xyz=(0.02, 0.04, -0.02), grip=-1.0)
    elif phase < 0.85:
        # Coordinated upward lift.
        _set_arm_delta(action, left, xyz=(0.00, 0.00, 0.18), grip=-1.0)
        _set_arm_delta(action, right, xyz=(0.00, 0.00, 0.18), grip=-1.0)
    else:
        _set_arm_delta(action, left, xyz=(0.00, 0.00, 0.02), grip=-1.0)
        _set_arm_delta(action, right, xyz=(0.00, 0.00, 0.02), grip=-1.0)
    return action


def _two_arm_handover_action(action_dim: int, phase: float) -> np.ndarray:
    action, left, right = _split_arm_action(action_dim)
    if phase < 0.35:
        _set_arm_delta(action, right, xyz=(0.12, -0.10, -0.05), grip=1.0)
        _set_arm_delta(action, left, xyz=(0.02, 0.08, 0.02), grip=1.0)
    elif phase < 0.65:
        _set_arm_delta(action, right, xyz=(-0.10, 0.12, 0.04), grip=-1.0)
        _set_arm_delta(action, left, xyz=(0.08, -0.08, 0.02), grip=1.0)
    else:
        _set_arm_delta(action, right, xyz=(0.00, 0.00, 0.00), grip=1.0)
        _set_arm_delta(action, left, xyz=(0.12, 0.05, 0.02), grip=-1.0)
    return action


def _two_arm_peg_action(action_dim: int, phase: float) -> np.ndarray:
    action, left, right = _split_arm_action(action_dim)
    if phase < 0.35:
        _set_arm_delta(action, left, xyz=(0.06, 0.10, -0.04), grip=-1.0)
        _set_arm_delta(action, right, xyz=(0.12, -0.10, -0.04), grip=-1.0)
    elif phase < 0.70:
        _set_arm_delta(action, left, xyz=(0.00, 0.02, 0.04), grip=-1.0)
        _set_arm_delta(action, right, xyz=(-0.08, 0.10, 0.02), grip=-1.0)
    else:
        _set_arm_delta(action, left, xyz=(0.00, 0.00, 0.00), grip=-1.0)
        _set_arm_delta(action, right, xyz=(-0.04, 0.02, -0.08), grip=-1.0)
    return action


def _generic_motion_action(action_dim: int, phase: float) -> np.ndarray:
    action = np.zeros(action_dim)
    t = 2 * np.pi * phase
    n = min(6, action_dim)
    action[:n] = 0.12 * np.sin(t + np.arange(n) * 0.7)
    return action


def hold_env(env, seconds: float = 30.0) -> None:
    """Keep a rendered robosuite window alive for demonstrations."""
    if seconds <= 0:
        return
    steps = max(1, int(seconds * 20))
    print(f"\n[GUI] Keeping robosuite viewer open for {seconds:.0f}s...")
    preview_env(env, steps=steps, realtime=True)


def close_env(env: Optional[object]) -> None:
    if env is None:
        return
    close = getattr(env, "close", None)
    if callable(close):
        close()
