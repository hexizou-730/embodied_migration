"""Small ManiSkill smoke check.

This module is intentionally conservative: it checks whether ManiSkill can be
imported, an environment can be created, reset works, and one zero-action step
can run. Rendering is optional and reported separately.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from typing import Any, Dict, Iterable, Optional

from .env_adapter import ManiSkillEnvAdapter, can_import_maniskill


def summarize_value(value: Any) -> Dict[str, Any]:
    """Return a compact, JSON-friendly summary of an observation-like value."""

    try:
        import numpy as np
        import torch
    except Exception:  # pragma: no cover - optional dependencies
        np = None
        torch = None

    if isinstance(value, dict):
        return {
            "type": "dict",
            "keys": sorted(str(key) for key in value.keys())[:20],
        }
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "len": len(value)}
    if np is not None and isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if torch is not None and torch.is_tensor(value):
        return {
            "type": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    return {"type": type(value).__name__, "repr": repr(value)[:200]}


def make_zero_action(env: Any) -> Any:
    action_space = getattr(env, "action_space", None)
    if action_space is None:
        raise RuntimeError("env has no action_space")
    try:
        import numpy as np

        return np.zeros(action_space.shape, dtype=action_space.dtype)
    except Exception:
        sample = action_space.sample()
        try:
            return sample * 0
        except Exception:
            return sample


def run_check(
    *,
    env_id: str = "PullCube-v1",
    robot_uid: Optional[str] = None,
    obs_mode: str = "state",
    control_mode: Optional[str] = None,
    render: bool = False,
    seed: int = 0,
) -> Dict[str, Any]:
    graphics_preflight = diagnose_graphics_stack()
    ok, message = can_import_maniskill()
    result: Dict[str, Any] = {
        "import_ok": ok,
        "import_message": message,
        "env_id": env_id,
        "robot_uid": robot_uid,
        "obs_mode": obs_mode,
        "control_mode": control_mode,
    }
    result["graphics_preflight"] = graphics_preflight
    if not ok:
        return result

    make_kwargs: Dict[str, Any] = {"obs_mode": obs_mode}
    if control_mode:
        make_kwargs["control_mode"] = control_mode
    adapter = ManiSkillEnvAdapter(
        env_id=env_id,
        robot_uid=robot_uid,
        render_mode="human" if render else None,
        **make_kwargs,
    )
    try:
        env = adapter.make()
        result["make_ok"] = True
        result["action_space"] = repr(getattr(env, "action_space", None))

        obs, info = adapter.reset(seed=seed)
        result["reset_ok"] = True
        result["obs"] = summarize_value(obs)
        result["reset_info"] = summarize_value(info)
        result["reset_info_keys"] = sorted(str(key) for key in info.keys())[:20]

        step = adapter.step(make_zero_action(env))
        result["step_ok"] = True
        result["reward"] = step.reward
        result["terminated"] = step.terminated
        result["truncated"] = step.truncated
        result["step_info"] = summarize_value(step.info)
        result["step_info_keys"] = sorted(str(key) for key in step.info.keys())[:20]
        result["success_value"] = _first_existing(step.info, ("success", "is_success"))

        if render:
            try:
                frame = adapter.render()
                result["render_ok"] = True
                result["render"] = summarize_value(frame)
            except Exception as exc:  # pragma: no cover - depends on local GPU
                result["render_ok"] = False
                result["render_error"] = repr(exc)
    except Exception as exc:  # pragma: no cover - depends on local install/GPU
        result["error"] = repr(exc)
        if _looks_like_vulkan_error(exc):
            result["diagnosis"] = result["graphics_preflight"]
    finally:
        adapter.close()
    return result


def _looks_like_vulkan_error(exc: Exception) -> bool:
    message = repr(exc).lower()
    return "vulkan" in message or "vk::" in message or "rendering device" in message


def diagnose_graphics_stack() -> Dict[str, Any]:
    nvidia = _run_command(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    vulkan = _run_command(["vulkaninfo", "--summary"])
    vulkan_text = (vulkan.get("stdout", "") + "\n" + vulkan.get("stderr", "")).lower()
    has_llvmpipe = "llvmpipe" in vulkan_text
    has_nvidia_vulkan = "nvidia" in vulkan_text and "llvmpipe" not in vulkan_text

    notes = []
    if nvidia.get("ok"):
        notes.append("CUDA/GPU is visible to WSL through nvidia-smi.")
    else:
        notes.append("nvidia-smi is not working in this shell, so CUDA visibility is not confirmed.")
    if has_llvmpipe and not has_nvidia_vulkan:
        notes.append("Vulkan only exposes llvmpipe CPU rendering, not the NVIDIA GPU.")
        notes.append("SAPIEN/ManiSkill simulation cannot run here until Vulkan GPU rendering works.")
    if "wsl" in _run_command(["uname", "-r"]).get("stdout", "").lower():
        notes.append("Do not install a Linux NVIDIA display driver inside WSL; use the Windows driver/WSLg path or native Ubuntu.")

    return {
        "nvidia_smi": _compact_command_result(nvidia),
        "vulkan_devices": _extract_vulkan_devices(vulkan.get("stdout", "")),
        "notes": notes,
    }


def _run_command(command: list[str]) -> Dict[str, Any]:
    try:
        completed = subprocess.run(command, check=False, text=True, capture_output=True, timeout=20)
    except FileNotFoundError:
        return {"ok": False, "error": f"command not found: {command[0]}"}
    except Exception as exc:  # pragma: no cover - host dependent
        return {"ok": False, "error": repr(exc)}
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _compact_command_result(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key in {"ok", "returncode", "error", "stdout"} and value not in {"", None}
    }


def _extract_vulkan_devices(output: str) -> list[str]:
    devices = []
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("deviceName") or stripped.startswith("deviceType") or stripped.startswith("driverName"):
            devices.append(stripped)
    return devices[:30]


def _first_existing(info: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in info:
            value = info[key]
            try:
                if hasattr(value, "item"):
                    return value.item()
            except Exception:
                pass
            return value
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether ManiSkill can run locally.")
    parser.add_argument("--env", default="PullCube-v1")
    parser.add_argument("--robot", default="", help="Optional robot uid; empty uses task default.")
    parser.add_argument("--obs-mode", default="state")
    parser.add_argument("--control-mode", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_check(
        env_id=args.env,
        robot_uid=args.robot or None,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode or None,
        render=args.render,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=repr))


if __name__ == "__main__":
    main()
