"""Runner for complex robosuite source-to-target program migration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from lmp import execute_lmp, extract_code_or_text
from lmp.failure_report import build_failure_report
from llm_client import chat
from robosuite_backend.profiles import RobosuiteProfile, get_profile
from robosuite_backend.prompting import SYSTEM_PROMPT, build_migration_prompt
from robosuite_backend.symbolic import RobosuiteSkillRobot, RobosuiteSymbolicScene
from robosuite_backend.tasks import RobosuiteTask, get_task
from robosuite_backend.trajectory_robot import RobosuiteTrajectoryRobot


ORACLE_TARGET_PATCHES: Dict[Tuple[str, str], str] = {
    ("two_arm_lift", "rs_dual_iiwa"): """robot.set_grip_force(0.85)
left_ok = robot.grasp_pot_handle('left', 'left_handle')
right_ok = robot.grasp_pot_handle('right', 'right_handle')
if left_ok and right_ok:
    ok = robot.lift_pot(lift_height=0.16, keep_level=True)
    ret_val = 'success' if ok else 'failure'
else:
    ret_val = 'failure'""",
    ("two_arm_lift", "rs_baxter"): """left_ok = robot.grasp_pot_handle('left', 'left_handle')
right_ok = robot.grasp_pot_handle('right', 'right_handle')
if left_ok and right_ok:
    ok = robot.lift_pot(lift_height=0.16, keep_level=True)
    ret_val = 'success' if ok else 'failure'
else:
    ret_val = 'failure'""",
    ("two_arm_lift", "rs_mobile_tiago"): """ret_val = 'refuse_requires_dual_arm'""",
    ("two_arm_handover", "rs_mobile_tiago"): """ret_val = 'refuse_requires_dual_arm'""",
    ("two_arm_peg_in_hole", "rs_mobile_tiago"): """ret_val = 'refuse_requires_dual_arm'""",
    ("two_arm_handover", "rs_baxter"): """pick_arm = robot.choose_arm_for('hammer')
other_arm = robot.other_arm(pick_arm)
picked = robot.pick_hammer(pick_arm)
if picked:
    ready = robot.move_to_handover_pose(clearance=0.10)
    handed = ready and robot.handover_object(pick_arm, other_arm, object_name='hammer')
    if handed:
        ok = robot.place_hammer_on_target(other_arm)
        ret_val = 'success' if ok else 'failure'
    else:
        ret_val = 'failure'
else:
    ret_val = 'failure'""",
    ("two_arm_handover", "rs_dual_iiwa"): """pick_arm = robot.choose_arm_for('hammer')
other_arm = robot.other_arm(pick_arm)
picked = robot.pick_hammer(pick_arm)
if picked:
    handed = robot.handover_object(pick_arm, other_arm, object_name='hammer')
    if handed:
        ok = robot.place_hammer_on_target(other_arm)
        ret_val = 'success' if ok else 'failure'
    else:
        ret_val = 'failure'
else:
    ret_val = 'failure'""",
    ("two_arm_peg_in_hole", "rs_dual_iiwa"): """board_arm = 'left'
peg_arm = robot.other_arm(board_arm)
board_ok = robot.hold_board(board_arm)
peg_ok = robot.grasp_peg(peg_arm)
if board_ok and peg_ok:
    aligned = robot.align_peg_to_hole(tolerance=0.015)
    if aligned:
        ok = robot.insert_peg(speed=0.02)
        ret_val = 'success' if ok else 'failure'
    else:
        ret_val = 'failure'
else:
    ret_val = 'failure'""",
    ("two_arm_peg_in_hole", "rs_baxter"): """board_arm = 'left'
peg_arm = robot.other_arm(board_arm)
board_ok = robot.hold_board(board_arm)
peg_ok = robot.grasp_peg(peg_arm)
if board_ok and peg_ok:
    aligned = robot.align_peg_to_hole(tolerance=0.02)
    if aligned:
        ok = robot.insert_peg(speed=0.02)
        ret_val = 'success' if ok else 'failure'
    else:
        ret_val = 'failure'
else:
    ret_val = 'failure'""",
}


@dataclass
class MigrationAttempt:
    attempt: int
    code: str
    exec_ok: bool
    exec_message: str
    action_failures: List[str]
    ret_val: object
    checker_success: bool
    expected: Dict[str, object]
    actual: Dict[str, object]


@dataclass
class MigrationResult:
    task: str
    source: str
    target: str
    planner: str
    success: bool
    final_reason: str
    attempts: List[MigrationAttempt] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"task={self.task} source={self.source} target={self.target} "
            f"planner={self.planner} success={self.success} reason={self.final_reason} "
            f"attempts={len(self.attempts)}"
        )


def make_symbolic_pair(
    task_name: str,
    target_name: str,
    real_env=None,
    render: bool = False,
    realtime: bool = False,
    assist_grasp: bool = True,
) -> Tuple[RobosuiteTask, RobosuiteProfile, RobosuiteProfile, RobosuiteSymbolicScene, RobosuiteSkillRobot]:
    task = get_task(task_name)
    source_profile = get_profile(task.source_robot)
    target_profile = get_profile(target_name)
    scene = RobosuiteSymbolicScene(task)
    if real_env is None:
        robot = RobosuiteSkillRobot(target_profile, scene)
    else:
        robot = RobosuiteTrajectoryRobot(
            target_profile,
            scene,
            env=real_env,
            render=render,
            realtime=realtime,
            assist_grasp=assist_grasp,
        )
    return task, source_profile, target_profile, scene, robot


def run_migration_trial(
    task_name: str,
    target_name: str,
    client=None,
    planner: str = "oracle",
    use_card: bool = True,
    use_failure_report: bool = True,
    max_attempts: int = 3,
    verbose: bool = True,
    real_env=None,
    render: bool = False,
    realtime: bool = False,
    assist_grasp: bool = True,
) -> MigrationResult:
    task, source_profile, target_profile, scene, robot = make_symbolic_pair(
        task_name,
        target_name,
        real_env=real_env,
        render=render,
        realtime=realtime,
        assist_grasp=assist_grasp,
    )
    source_code = task.source_program
    planner = planner.lower()
    last_report = None
    attempts: List[MigrationAttempt] = []

    if planner == "source-copy":
        max_attempts = 1
    elif planner == "oracle":
        max_attempts = 1
    elif planner != "llm":
        raise ValueError("planner must be one of: llm, oracle, source-copy")

    for attempt_idx in range(1, max_attempts + 1):
        scene.reset()
        robot.reset_action_log()
        if hasattr(robot, "reset_real_env"):
            robot.reset_real_env()

        if planner == "source-copy":
            code = source_code
        elif planner == "oracle":
            code = ORACLE_TARGET_PATCHES.get((task.name, target_profile.name), source_code)
        else:
            if client is None:
                raise RuntimeError("planner='llm' requires a live LLM client")
            prompt = build_migration_prompt(
                task=task,
                source_profile=source_profile,
                target_profile=target_profile,
                target_robot=robot,
                target_scene=scene,
                source_code=source_code,
                use_capability_card=use_card,
                failure_report=last_report if use_failure_report else None,
            )
            raw = chat(client, system=SYSTEM_PROMPT, user=prompt)
            code = extract_code_or_text(raw)

        exec_ctx = {"robot": robot, "scene": scene}
        if target_profile.has_mobile_base:
            exec_ctx["mobile"] = robot
        if len(target_profile.arm_names) >= 2:
            exec_ctx["left"] = robot
            exec_ctx["right"] = robot

        ok, msg, locals_dict = execute_lmp(code, exec_ctx, verbose=verbose)
        action_failures = robot.get_action_failures()
        ret_val = locals_dict.get("ret_val")
        checker_success, expected, actual = scene.check_success()
        ret_val_failed = ret_val is False or ret_val == "failure"
        action_ok = not action_failures and not ret_val_failed
        impossible_refusal_ok = len(target_profile.arm_names) < 2 and ret_val == "refuse_requires_dual_arm"
        success = bool((ok and checker_success and action_ok) or (ok and impossible_refusal_ok))
        attempts.append(
            MigrationAttempt(
                attempt=attempt_idx,
                code=code,
                exec_ok=ok,
                exec_message=msg,
                action_failures=action_failures,
                ret_val=ret_val,
                checker_success=checker_success,
                expected=expected,
                actual=actual,
            )
        )

        if success:
            return MigrationResult(
                task=task.name,
                source=source_profile.name,
                target=target_profile.name,
                planner=planner,
                success=True,
                final_reason=f"success_on_attempt_{attempt_idx}",
                attempts=attempts,
            )

        if not use_failure_report or attempt_idx == max_attempts:
            if not ok:
                reason = "exec-fail"
            elif action_failures:
                reason = "action-fail"
            elif ret_val_failed:
                reason = "ret-val-fail"
            else:
                reason = "check-fail"
            return MigrationResult(
                task=task.name,
                source=source_profile.name,
                target=target_profile.name,
                planner=planner,
                success=False,
                final_reason=reason,
                attempts=attempts,
            )

        actual_with_failures = dict(actual)
        actual_with_failures["action_failures"] = action_failures
        actual_with_failures["ret_val"] = ret_val
        expected_with_action = dict(expected)
        expected_with_action["robot_action_success"] = True
        last_report = build_failure_report(
            task_name=task.name,
            instruction=task.instruction,
            robot_name=target_profile.name,
            expected=expected_with_action,
            actual=actual_with_failures,
            code_raised=not ok,
            traceback_str=msg if not ok else None,
        )
        if verbose:
            print(f"\n⚠️  Attempt {attempt_idx} failed; retrying with Failure Report.")

    return MigrationResult(
        task=task.name,
        source=source_profile.name,
        target=target_profile.name,
        planner=planner,
        success=False,
        final_reason="max_attempts_exceeded",
        attempts=attempts,
    )
