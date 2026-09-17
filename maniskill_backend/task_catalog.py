"""Load and validate the target-independent 20-task benchmark catalog."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_ROOT = REPO_ROOT / "benchmark_tasks"
CATALOG_PATH = CATALOG_ROOT / "catalog.json"
TASK_ID = re.compile(r"t\d{2}_[a-z0-9_]+")
ENV_ID = re.compile(r"[A-Za-z0-9_-]+-v\d+")
FORBIDDEN_TASK_KEYS = {"target_robot", "target_adapter", "migration_case", "case_id"}


def load_task_catalog(root: Path = CATALOG_ROOT) -> dict[str, Any]:
    index = _read_json(root / "catalog.json")
    profiles = _read_json(root / "difficulty_profiles.json")
    tasks = []
    for entry in index.get("tasks", []):
        task = _read_json(root / entry["path"])
        if task.get("task_id") != entry.get("task_id"):
            raise ValueError(f"Catalog task_id mismatch for {entry['path']}")
        tasks.append(task)
    payload = {**index, "tasks": tasks, "difficulty_profiles": profiles["profiles"]}
    validate_task_catalog(payload)
    return payload


def validate_task_catalog(catalog: dict[str, Any]) -> None:
    if catalog.get("schema") != "task_catalog.v1":
        raise ValueError("Task catalog must use schema task_catalog.v1")
    tasks = catalog.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 20:
        raise ValueError("Task catalog must contain exactly 20 tasks")
    profiles = catalog.get("difficulty_profiles")
    if not isinstance(profiles, list) or len(profiles) != 6:
        raise ValueError("Difficulty profile catalog must contain exactly six profiles")
    profile_ids = [profile.get("profile_id") for profile in profiles]
    if len(set(profile_ids)) != 6:
        raise ValueError("Difficulty profile IDs must be unique")

    task_ids: set[str] = set()
    env_ids: set[str] = set()
    for task in tasks:
        _validate_task(task, profile_ids)
        task_id = task["task_id"]
        env_id = task["env_id"]
        if task_id in task_ids:
            raise ValueError(f"Duplicate task_id: {task_id}")
        if env_id in env_ids:
            raise ValueError(f"Duplicate env_id: {env_id}")
        task_ids.add(task_id)
        env_ids.add(env_id)


def _validate_task(task: dict[str, Any], profile_ids: list[str]) -> None:
    required = {
        "schema",
        "task_id",
        "env_id",
        "source_robot",
        "required_capabilities",
        "success_signal",
        "episode_limit",
        "difficulty_variants",
    }
    missing = required - set(task)
    if missing:
        raise ValueError(f"{task.get('task_id', '<unknown>')} missing fields: {sorted(missing)}")
    forbidden = FORBIDDEN_TASK_KEYS & set(task)
    if forbidden:
        raise ValueError(f"{task['task_id']} contains target-specific keys: {sorted(forbidden)}")
    if task["schema"] != "benchmark_task.v1" or not TASK_ID.fullmatch(task["task_id"]):
        raise ValueError(f"Invalid task schema or ID: {task.get('task_id')}")
    if not ENV_ID.fullmatch(task["env_id"]):
        raise ValueError(f"Invalid ManiSkill env_id: {task['env_id']}")
    if not isinstance(task["source_robot"], str) or not task["source_robot"]:
        raise ValueError(f"{task['task_id']} needs one source robot UID")
    capabilities = task["required_capabilities"]
    if not isinstance(capabilities, list) or not capabilities or len(capabilities) != len(set(capabilities)):
        raise ValueError(f"{task['task_id']} capabilities must be a non-empty unique list")
    success = task["success_signal"]
    if success.get("authority") != "env.unwrapped.evaluate()" or success.get("key") != "success":
        raise ValueError(f"{task['task_id']} must use ManiSkill's official success signal")
    if not isinstance(task["episode_limit"], int) or task["episode_limit"] <= 0:
        raise ValueError(f"{task['task_id']} has invalid episode_limit")

    variants = task["difficulty_variants"]
    if not isinstance(variants, list) or len(variants) != 6:
        raise ValueError(f"{task['task_id']} must define exactly six variants")
    if [variant.get("profile") for variant in variants] != profile_ids:
        raise ValueError(f"{task['task_id']} variants must follow the shared difficulty order")
    if [variant.get("difficulty") for variant in variants] != list(range(1, 7)):
        raise ValueError(f"{task['task_id']} difficulty levels must be 1 through 6")
    if any(not isinstance(variant.get("episode_limit"), int) or variant["episode_limit"] <= 0 for variant in variants):
        raise ValueError(f"{task['task_id']} variant episode limits must be positive integers")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    catalog = load_task_catalog()
    print(f"valid task catalog: {len(catalog['tasks'])} tasks x 6 variants")
    for task in catalog["tasks"]:
        print(
            f"{task['task_id']}: {task['env_id']} | source={task['source_robot']} | "
            f"limit={task['episode_limit']}"
        )


if __name__ == "__main__":
    main()
