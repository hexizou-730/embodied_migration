"""Preflight the frozen paper cases before spending simulation or LLM budget."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from maniskill_backend.cases import FullMigrationCase, get_full_migration_case
from maniskill_backend.env_adapter import ManiSkillEnvAdapter, can_import_maniskill
from maniskill_backend.embodiment_contracts import get_embodiment_contract, observe_runtime_contract
from maniskill_backend.sim_check import make_zero_action
from maniskill_backend.skill_adapter import (
    ManiSkillFetchPickCubeRobot,
    ManiSkillPickCubeRobot,
    ManiSkillPullCubeRobot,
    ManiSkillPushCubeRobot,
)
from maniskill_backend.tasks import get_task_spec
from maniskill_backend.stack_pyramid import ManiSkillStackPyramidRobot


REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _static_case_check(case: FullMigrationCase) -> Dict[str, Any]:
    paths = {
        "program": REPO_ROOT / case.target_program_path,
        "seed_adapter": REPO_ROOT / case.seed_adapter_path,
        "target_adapter": REPO_ROOT / case.target_adapter_path,
    }
    checks: Dict[str, Any] = {
        name: {
            "path": str(path.relative_to(REPO_ROOT)),
            "exists": path.is_file(),
            "sha256": _sha256(path),
        }
        for name, path in paths.items()
    }
    module_error = ""
    build_robot_ok = False
    try:
        module = importlib.import_module(case.target_adapter_module)
        build_robot_ok = callable(getattr(module, "build_robot", None))
    except Exception as exc:
        module_error = repr(exc)
    checks["target_module"] = {
        "module": case.target_adapter_module,
        "import_ok": not module_error,
        "build_robot_ok": build_robot_ok,
        "error": module_error,
    }
    ok = all(item.get("exists") for name, item in checks.items() if name != "target_module")
    ok = ok and not module_error and build_robot_ok
    return {"ok": bool(ok), "checks": checks}


def _source_adapter(case: FullMigrationCase, env: Any) -> Any:
    if case.task_id == "stack_pyramid":
        cls = ManiSkillStackPyramidRobot
    elif case.task_id == "pull_cube":
        cls = ManiSkillPullCubeRobot
    elif case.task_id == "push_cube":
        cls = ManiSkillPushCubeRobot
    elif case.source_robot == "fetch":
        cls = ManiSkillFetchPickCubeRobot
    else:
        cls = ManiSkillPickCubeRobot
    return cls(env, robot_uid=case.source_robot, control_mode=case.source_control_mode)


def _runtime_role_check(
    case: FullMigrationCase,
    *,
    role: str,
    seed: int,
    sim_backend: str,
    render_backend: str,
) -> Dict[str, Any]:
    robot_uid = case.source_robot if role == "source" else case.target_robot
    control_mode = case.source_control_mode if role == "source" else case.target_control_mode
    task = get_task_spec(case.task_id)
    adapter = ManiSkillEnvAdapter(
        task.maniskill_env_id,
        robot_uid=robot_uid,
        obs_mode="state",
        control_mode=control_mode,
        sim_backend=sim_backend,
        render_backend=render_backend,
        max_episode_steps=case.max_episode_steps,
        **({"reward_mode": "sparse"} if case.task_id == "stack_pyramid" else {}),
    )
    result: Dict[str, Any] = {
        "role": role,
        "robot_uid": robot_uid,
        "control_mode": control_mode,
        "env_id": task.maniskill_env_id,
    }
    try:
        env = adapter.make()
        result["make_ok"] = True
        result["action_space"] = repr(getattr(env, "action_space", None))
        shape = getattr(getattr(env, "action_space", None), "shape", None)
        result["action_shape"] = list(shape) if shape is not None else None
        contract = get_embodiment_contract(robot_uid, control_mode)
        result["runtime_contract_observation"] = observe_runtime_contract(env, contract)
        result["runtime_contract_ok"] = result["runtime_contract_observation"]["valid"]
        _, reset_info = adapter.reset(seed=seed)
        result["reset_ok"] = True

        if role == "target":
            module = importlib.import_module(case.target_adapter_module)
            robot = module.build_robot(env, control_mode=control_mode, robot_uid=robot_uid)
        else:
            robot = _source_adapter(case, env)
        result["adapter_build_ok"] = robot is not None

        step = adapter.step(make_zero_action(env))
        result["step_ok"] = True
        result["success_signal_present"] = any(
            key in step.info or key in reset_info for key in ("success", "is_success")
        )
        result["ok"] = all(
            result.get(key) is True
            for key in (
                "make_ok",
                "reset_ok",
                "adapter_build_ok",
                "step_ok",
                "success_signal_present",
                "runtime_contract_ok",
            )
        )
    except Exception as exc:
        result["ok"] = False
        result["error"] = repr(exc)
    finally:
        adapter.close()
    return result


def run_paper_preflight(
    case_ids: Iterable[str],
    *,
    seed: int = 0,
    sim_backend: str = "auto",
    render_backend: str = "gpu",
    static_only: bool = False,
) -> Dict[str, Any]:
    import_ok, import_message = can_import_maniskill()
    rows = []
    for case_id in case_ids:
        case = get_full_migration_case(case_id)
        static = _static_case_check(case)
        runtime: list[Dict[str, Any]] = []
        if not static_only and import_ok:
            for role in ("source", "target"):
                runtime.append(
                    _runtime_role_check(
                        case,
                        role=role,
                        seed=seed,
                        sim_backend=sim_backend,
                        render_backend=render_backend,
                    )
                )
        runtime_ok: bool | None
        if static_only:
            runtime_ok = None
        elif not import_ok:
            runtime_ok = False
        else:
            runtime_ok = all(item.get("ok") is True for item in runtime)
        rows.append(
            {
                "case_id": case.case_id,
                "task_id": case.task_id,
                "source_robot": case.source_robot,
                "target_robot": case.target_robot,
                "support_status": case.support_status,
                "support_source_url": case.support_source_url,
                "static_ok": static["ok"],
                "runtime_ok": runtime_ok,
                "static": static,
                "runtime": runtime,
            }
        )
    ready = all(row["static_ok"] and (static_only or row["runtime_ok"] is True) for row in rows)
    return {
        "schema": "embodied_migration_paper_preflight.v1",
        "ready": ready,
        "static_only": static_only,
        "maniskill_import_ok": import_ok,
        "maniskill_import_message": import_message,
        "seed": seed,
        "sim_backend": sim_backend,
        "render_backend": render_backend,
        "rows": rows,
    }


def preflight_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Paper Benchmark Preflight",
        "",
        f"- ready: `{payload.get('ready')}`",
        f"- static only: `{payload.get('static_only')}`",
        f"- ManiSkill import: `{payload.get('maniskill_import_ok')}`",
        "",
        "| Case | Transfer | Support | Static | Runtime | Error |",
        "|---|---|---|---|---|---|",
    ]
    for row in payload.get("rows") or []:
        errors = [str(item.get("error")) for item in row.get("runtime") or [] if item.get("error")]
        module_error = ((row.get("static") or {}).get("checks") or {}).get("target_module", {}).get("error")
        if module_error:
            errors.append(str(module_error))
        lines.append(
            f"| `{row.get('case_id')}` | {row.get('source_robot')} -> {row.get('target_robot')} | "
            f"`{row.get('support_status')}` | {row.get('static_ok')} | {row.get('runtime_ok')} | "
            f"{'<br>'.join(errors)[:300]} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_paper_preflight(payload: Mapping[str, Any], output_dir: Path) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "preflight.json"
    md_path = output_dir / "preflight.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=repr), encoding="utf-8")
    md_path.write_text(preflight_markdown(payload), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}
