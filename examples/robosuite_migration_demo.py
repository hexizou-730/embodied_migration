"""Interactive-ish demo for complex robosuite source-program migration.

Examples:
  python -m examples.robosuite_migration_demo --task two_arm_lift --target rs_dual_iiwa --planner oracle
  python -m examples.robosuite_migration_demo --task two_arm_lift --target rs_dual_iiwa --planner llm
  python -m examples.robosuite_migration_demo --task two_arm_handover --target rs_baxter --planner source-copy
  python -m examples.robosuite_migration_demo --show-env --gui
"""
from __future__ import annotations

import argparse
from dataclasses import asdict

from dotenv import load_dotenv

from llm_client import make_client
from robosuite_backend.env_adapter import (
    animate_env,
    availability_message,
    close_env,
    hold_env,
    make_env,
    preview_env,
)
from robosuite_backend.migration import run_migration_trial
from robosuite_backend.profiles import profile_names
from robosuite_backend.tasks import get_task, task_names


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=task_names(), default="two_arm_lift")
    ap.add_argument("--target", choices=profile_names(), default="rs_dual_iiwa")
    ap.add_argument("--planner", choices=["llm", "oracle", "source-copy"], default="oracle",
                    help="llm calls OpenRouter; oracle uses the built-in target patch; "
                         "source-copy executes the source program unchanged on the target.")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--no-card", action="store_true",
                    help="For planner=llm, omit source/target capability cards from the prompt.")
    ap.add_argument("--no-retry", action="store_true")
    ap.add_argument("--show-env", action="store_true",
                    help="Also launch the real robosuite/MuJoCo task scene if installed.")
    ap.add_argument("--gui", action="store_true",
                    help="Render robosuite viewer during --show-env preview.")
    ap.add_argument("--real-control", action="store_true",
                    help="Execute supported high-level skills through real robosuite "
                         "controller trajectories instead of symbolic-only state updates. "
                         "Implemented for TwoArmLift, TwoArmHandover, and TwoArmPegInHole.")
    ap.add_argument("--no-assist-grasp", action="store_true",
                    help="Disable the abstract grasp constraint used after the real "
                         "controller reaches both pot handles.")
    ap.add_argument("--animate", choices=["none", "task", "wiggle"], default="none",
                    help="When --show-env --gui is used, send low-level actions so "
                         "the robosuite robot visibly moves. 'task' uses a simple "
                         "task-shaped motion; 'wiggle' uses a generic joint motion.")
    ap.add_argument("--animate-seconds", type=float, default=12.0,
                    help="Duration of the GUI animation when --animate is not none.")
    ap.add_argument("--hold-seconds", type=float, default=None,
                    help="Seconds to keep the robosuite GUI open after printing results. "
                         "Defaults to 30 when --gui is used, otherwise 0.")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    task = get_task(args.task)
    print("\n" + "=" * 72)
    print("Complex Robosuite Program Migration Demo")
    print("=" * 72)
    print(task.describe())
    print(f"Source robot: {task.source_robot}")
    print(f"Target robot: {args.target}")
    print(f"Planner: {args.planner}")

    env = None
    if args.show_env or args.real_control:
        print("\n[robosuite availability]")
        print(availability_message())
        try:
            from robosuite_backend.profiles import get_profile

            env = make_env(task, get_profile(args.target), has_renderer=args.gui)
            print(f"✅ robosuite env launched: {task.robosuite_env}")
            if args.show_env:
                preview_env(env, steps=40 if args.gui else 1, realtime=args.gui)
            if args.gui and args.animate != "none" and not args.real_control:
                animation_task = args.task if args.animate == "task" else "generic"
                animate_env(env, task_name=animation_task, seconds=args.animate_seconds, realtime=True)
        except Exception as exc:
            print(f"⚠️  Could not launch robosuite env: {exc}")
            env = None

    client = make_client() if args.planner == "llm" else None
    result = run_migration_trial(
        task_name=args.task,
        target_name=args.target,
        client=client,
        planner=args.planner,
        use_card=not args.no_card,
        use_failure_report=not args.no_retry,
        max_attempts=args.attempts,
        verbose=not args.quiet,
        real_env=env if args.real_control else None,
        render=args.gui,
        realtime=args.gui,
        assist_grasp=not args.no_assist_grasp,
    )
    print("\n" + "-" * 72)
    print(result.summary())
    for attempt in result.attempts:
        print(f"\nAttempt {attempt.attempt}:")
        print(f"  exec_ok={attempt.exec_ok} ret_val={attempt.ret_val!r}")
        print(f"  checker_success={attempt.checker_success}")
        print(f"  action_failures={attempt.action_failures}")
        print(f"  actual={attempt.actual}")
        if args.quiet:
            continue
        print("  code:")
        for line in attempt.code.splitlines():
            print(f"    {line}")

    hold_seconds = args.hold_seconds
    if hold_seconds is None:
        hold_seconds = 30.0 if (args.show_env and args.gui and env is not None) else 0.0
    if env is not None and args.gui:
        hold_env(env, seconds=hold_seconds)
    close_env(env)
    print("\nResult dict:")
    print(asdict(result))


if __name__ == "__main__":
    main()
