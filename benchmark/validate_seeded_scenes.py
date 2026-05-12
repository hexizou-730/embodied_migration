"""
Validate seeded Mobile + Dual-arm migration scenes with scripted oracle policies.

This module does not call an LLM. It checks that randomized Stage-5 layouts are
physically meaningful before a costly benchmark run starts.
"""
import argparse
from typing import Callable, Dict, Iterable, List, Tuple

import pybullet as p

from robots import make_robot
from benchmark.run_benchmark import (
    TASKS_BIMANUAL,
    TASKS_MOBILITY,
    setup_scene,
)


TASK_GROUPS = {
    "mobility": TASKS_MOBILITY,
    "bimanual": TASKS_BIMANUAL,
    "migration": TASKS_MOBILITY + TASKS_BIMANUAL,
    "all": TASKS_MOBILITY + TASKS_BIMANUAL,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robots", nargs="+", default=["dual_arm", "mobile_dual_arm", "dual_franka"],
                    choices=["mobile", "dual_arm", "mobile_dual_arm", "dual_franka", "kuka", "franka"])
    ap.add_argument("--tasks", default="migration",
                    choices=["mobility", "bimanual", "migration", "all"])
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--scene-variant", default="seeded", choices=["fixed", "seeded"])
    ap.add_argument("--allow-unexpected", action="store_true",
                    help="Print mismatches but exit 0.")
    args = ap.parse_args()

    tasks = TASK_GROUPS[args.tasks]
    mismatches = validate(
        robots=args.robots,
        tasks=tasks,
        trials=args.trials,
        seed_base=args.seed_base,
        scene_variant=args.scene_variant,
    )
    if mismatches and not args.allow_unexpected:
        raise SystemExit(1)


def validate(
    robots: Iterable[str],
    tasks,
    trials: int,
    seed_base: int,
    scene_variant: str,
) -> List[Dict[str, object]]:
    mismatches: List[Dict[str, object]] = []
    total = 0
    print("Stage-5 seeded-scene validation (no LLM)")
    print(f"robots={list(robots)} tasks={[task[0] for task in tasks]} "
          f"trials={trials} scene_variant={scene_variant} seed_base={seed_base}")
    for robot_name in robots:
        for task in tasks:
            task_name = task[0]
            for trial in range(trials):
                scene_seed = seed_base + trial if scene_variant == "seeded" else None
                total += 1
                result = run_oracle_trial(robot_name, task, scene_variant, scene_seed)
                status = "OK" if result["matches_expectation"] else "MISMATCH"
                icon = "✓" if result["matches_expectation"] else "✗"
                print(
                    f"  {icon} {status:<8} robot={robot_name:<8} "
                    f"task={task_name:<30} seed={scene_seed!s:<5} "
                    f"expected={result['expected_success']} actual={result['checker_success']}"
                )
                if not result["matches_expectation"]:
                    mismatches.append(result)

    print(f"validated={total} mismatches={len(mismatches)}")
    return mismatches


def run_oracle_trial(
    robot_name: str,
    task,
    scene_variant: str,
    scene_seed,
) -> Dict[str, object]:
    task_name, _, checker = task
    p.connect(p.DIRECT)
    try:
        scene = setup_scene(
            robot_name,
            scene_variant=scene_variant,
            scene_seed=scene_seed,
            task_name=task_name,
        )
        robot = make_robot(robot_name)
        robot.scene = scene
        for _ in range(240):
            p.stepSimulation()

        policy_ok = oracle_policy(robot_name, task_name, robot, scene)
        for _ in range(120):
            p.stepSimulation()
        checker_success, (_, actual) = checker(scene)
        expected = expected_success(robot_name, task_name)
        return {
            "robot": robot_name,
            "task": task_name,
            "scene_variant": scene_variant,
            "scene_seed": scene_seed,
            "expected_success": expected,
            "policy_ok": policy_ok,
            "checker_success": checker_success,
            "matches_expectation": bool(checker_success) == bool(expected),
            "actual": actual,
            "action_failures": (
                robot.get_action_failures()
                if hasattr(robot, "get_action_failures") else []
            ),
        }
    finally:
        p.disconnect()


def expected_success(robot_name: str, task_name: str) -> bool:
    if robot_name == "mobile" and task_name in {task[0] for task in TASKS_BIMANUAL}:
        return False
    return robot_name in {"mobile", "dual_arm", "mobile_dual_arm", "dual_franka"}


def oracle_policy(robot_name: str, task_name: str, robot, scene) -> bool:
    if robot_name == "mobile":
        return mobile_oracle(task_name, robot, scene)
    if robot_name == "mobile_dual_arm":
        navigate_mobile_to_table(robot, scene)
        return dual_arm_oracle(task_name, robot, scene)
    if robot_name in {"dual_arm", "dual_franka"}:
        return dual_arm_oracle(task_name, robot, scene)
    return False


def mobile_oracle(task_name: str, robot, scene) -> bool:
    navigate_mobile_to_table(robot, scene)
    if task_name == "wide_blue_to_tray":
        return mobile_move_to_tray(robot, scene, "blue block")
    if task_name == "collect_red_and_blue_to_tray":
        ok = True
        offsets = {
            "red block": (-0.035, -0.025, 0.0),
            "blue block": (0.035, 0.025, 0.0),
        }
        for name in ("red block", "blue block"):
            ok = mobile_move_to_tray(robot, scene, name, offsets[name]) and ok
        return ok
    if task_name in {
        "hold_red_while_place_green",
        "lift_red_and_green_together",
        "lift_red_green_together_to_tray",
    }:
        # This deliberately attempts the impossible single-arm version. The
        # checker should remain false, confirming the intended capability gap.
        red = scene.get_object_position("red block")
        green = scene.get_object_position("green block")
        return bool(robot.pick(red) and robot.pick(green))
    return False


def mobile_move_to_tray(robot, scene, object_name: str, tray_offset=(0.0, 0.0, 0.0)) -> bool:
    pos = scene.get_object_position(object_name)
    tray = scene.get_object_position("yellow tray")
    return robot.pick_and_place(
        pos,
        tray + [tray_offset[0], tray_offset[1], 0.05 + tray_offset[2]],
        place_release_height=0.005,
    )


def navigate_mobile_to_table(robot, scene) -> bool:
    table_x, table_y = float(scene.table_position[0]), float(scene.table_position[1])
    clearance = float(robot.capability_card.nav_min_clearance_m)
    if robot.capability_card.has_dual_arms:
        return robot.navigate_to(table_x, table_y + clearance)
    return robot.navigate_to(table_x - 0.05, table_y + clearance)


def dual_arm_oracle(task_name: str, robot, scene) -> bool:
    if task_name == "wide_blue_to_tray":
        return dual_move_to_tray(robot, scene, "blue block")
    if task_name == "collect_red_and_blue_to_tray":
        ok = True
        for name in ("red block", "blue block"):
            ok = dual_move_to_tray(robot, scene, name) and ok
        return ok
    if task_name == "hold_red_while_place_green":
        red = scene.get_object_position("red block")
        green = scene.get_object_position("green block")
        tray = scene.get_object_position("yellow tray")
        red_arm, green_arm = assign_pair_for(robot, scene)
        ok = robot.pick_with_arm(red_arm, red)
        ok = robot.pick_with_arm(green_arm, green) and ok
        ok = robot.place_with_arm(
            green_arm,
            tray + [0, 0, 0.05],
            pre_release_height=robot.capability_card.recommended_release_height_m,
        ) and ok
        return ok
    if task_name == "lift_red_and_green_together":
        red = scene.get_object_position("red block")
        green = scene.get_object_position("green block")
        return bool(robot.lift_two_objects(red, green))
    if task_name == "lift_red_green_together_to_tray":
        red = scene.get_object_position("red block")
        green = scene.get_object_position("green block")
        tray = scene.get_object_position("yellow tray")
        red_target = tray + [-0.03, -0.03, 0.05]
        green_target = tray + [0.03, 0.03, 0.05]
        return bool(
            robot.lift_two_objects(red, green)
            and robot.place_two_objects(
                red_target,
                green_target,
                pre_release_height=robot.capability_card.recommended_release_height_m,
            )
        )
    return False


def dual_move_to_tray(robot, scene, object_name: str) -> bool:
    pos = scene.get_object_position(object_name)
    tray = scene.get_object_position("yellow tray")
    arm = robot.choose_arm_for(pos)
    return robot.pick_and_place_with_arm(
        arm,
        pos,
        tray + [0, 0, 0.05],
        place_release_height=robot.capability_card.recommended_release_height_m,
    )


def arm_pair_for(scene) -> Tuple[str, str]:
    red = scene.get_object_position("red block")
    green = scene.get_object_position("green block")
    if red[1] <= green[1]:
        return "left", "right"
    return "right", "left"


def assign_pair_for(robot, scene) -> Tuple[str, str]:
    red = scene.get_object_position("red block")
    green = scene.get_object_position("green block")
    if hasattr(robot, "_assign_two_arms"):
        return robot._assign_two_arms(red, green)
    return arm_pair_for(scene)


if __name__ == "__main__":
    main()
