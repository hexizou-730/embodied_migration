"""
主入口: 跨具身代码迁移 demo (B+A 支持)。

用法:
    python main.py --robot kuka                 # 默认启用 B+A
    python main.py --robot franka --no-card     # 关闭方法 A (只用 B)
    python main.py --robot kuka --mode baseline # 完全朴素 baseline

交互模式下输入自然语言指令, LLM 生成代码, 自动执行, 失败时用结构化反馈重试。
"""
import argparse
import sys
import pybullet as p
from dotenv import load_dotenv

from robots import make_robot
from perception import TabletopScene
from prompts import SYSTEM_PROMPT, build_user_prompt
from lmp import execute_lmp, extract_code_or_text
from lmp.failure_report import FailureReport
from llm_client import make_client, chat

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()


def build_scene_and_robot(robot_name: str):
    print(f"\n🌍 Launching PyBullet (robot={robot_name})...")
    p.connect(p.GUI)

    if robot_name == "mobile":
        # 远距离场景: Husky 必须先 navigate 才能够到桌子
        p.resetDebugVisualizerCamera(
            cameraDistance=3.5, cameraYaw=45, cameraPitch=-25,
            cameraTargetPosition=[0.7, 0.1, 0.4],
        )
        scene = TabletopScene(table_position=(1.5, 0.0, 0.0))
        scene.add_cube("red block", [1.5, -0.10, 0.65], color=(1, 0, 0, 1))
        scene.add_cube("green block", [1.60, 0.24, 0.65], color=(0, 1, 0, 1))
        scene.add_cube("blue block", [1.45, 0.10, 0.65], color=(0, 0, 1, 1))
        scene.add_tray("yellow tray", [1.3, 0.3, 0.63], color=(1, 1, 0, 1))
    elif robot_name == "mobile_dual_arm":
        # 远距离双臂场景: mobile dual-arm 必须先 navigate, 再双臂协作抓放
        p.resetDebugVisualizerCamera(
            cameraDistance=3.8, cameraYaw=45, cameraPitch=-25,
            cameraTargetPosition=[1.25, 0.1, 0.55],
        )
        scene = TabletopScene(table_position=(1.5, 0.0, 0.0))
        scene.add_cube("red block", [1.45, -0.10, 0.65], color=(1, 0, 0, 1))
        scene.add_cube("green block", [1.55, 0.15, 0.65], color=(0, 1, 0, 1))
        scene.add_cube("blue block", [1.50, -0.25, 0.65], color=(0, 0, 1, 1))
        scene.add_tray("yellow tray", [1.30, 0.12, 0.63], color=(1, 1, 0, 1))
    elif robot_name in {"dual_arm", "dual_franka"}:
        # 近距离双臂场景: 物体覆盖左右臂共享工作区
        p.resetDebugVisualizerCamera(
            cameraDistance=3.5, cameraYaw=45, cameraPitch=-25,
            cameraTargetPosition=[0.35, 0.0, 0.55],
        )
        scene = TabletopScene(table_position=(0.0, 0.0, 0.0))
        scene.add_cube("red block", [0.5, -0.10, 0.65], color=(1, 0, 0, 1))
        scene.add_cube("green block", [0.55, 0.15, 0.65], color=(0, 1, 0, 1))
        scene.add_cube("blue block", [0.45, -0.25, 0.65], color=(0, 0, 1, 1))
        scene.add_tray("yellow tray", [0.35, 0.12, 0.63], color=(1, 1, 0, 1))
    else:
        # 近距离场景 (KUKA / Franka)
        scene = TabletopScene(table_position=(0.0, 0.0, 0.0))
        scene.add_cube("red block", [0.5, 0.0, 0.65], color=(1, 0, 0, 1))
        scene.add_cube("green block", [0.55, 0.15, 0.65], color=(0, 1, 0, 1))
        scene.add_cube("blue block", [0.45, -0.15, 0.65], color=(0, 0, 1, 1))
        scene.add_tray("yellow tray", [0.3, 0.3, 0.63], color=(1, 1, 0, 1))

    robot = make_robot(robot_name)
    robot.scene = scene

    for _ in range(240):
        p.stepSimulation()

    print(f"✅ Scene ready. {robot.describe()}")
    print(scene.describe())
    return scene, robot


def run_instruction(scene, robot, instruction: str, client,
                    use_card: bool = True, use_retry: bool = True):
    """LLM 生成代码 → 执行。失败时(如启用 retry)把 traceback 作为 FailureReport 反馈。"""
    max_attempts = 3 if use_retry else 1
    last_report = None

    for attempt in range(1, max_attempts + 1):
        if last_report is None:
            print(f"\n🧠 [Attempt {attempt}/{max_attempts}] Asking LLM to generate code"
                  f" (capability_card={'ON' if use_card else 'OFF'})...")
        else:
            print(f"\n🧠 [Attempt {attempt}/{max_attempts}] Retrying with failure report...")

        user_prompt = build_user_prompt(
            robot=robot,
            scene_description=scene.describe(),
            instruction=instruction,
            use_capability_card=use_card,
            failure_report=last_report,
        )

        raw = chat(client, system=SYSTEM_PROMPT, user=user_prompt)
        code = extract_code_or_text(raw)

        # mobile 机器人需要把 'mobile' 名字也注入到 exec 上下文 (与 robot 同对象)
        exec_ctx = {"robot": robot, "scene": scene}
        if getattr(robot, "capability_card", None) and robot.capability_card.has_mobile_base:
            exec_ctx["mobile"] = robot
        if getattr(robot, "capability_card", None) and robot.capability_card.has_dual_arms:
            exec_ctx["left"] = robot.left
            exec_ctx["right"] = robot.right
        if hasattr(robot, "reset_action_log"):
            robot.reset_action_log()
        ok, msg, exec_locals = execute_lmp(code, exec_ctx, verbose=True)
        action_failures = (
            robot.get_action_failures()
            if hasattr(robot, "get_action_failures") else []
        )
        ret_val = exec_locals.get("ret_val")
        ret_val_failed = ret_val is False or ret_val == "failure"

        if ok and not action_failures and not ret_val_failed:
            print(f"\n✅ Instruction code executed on attempt {attempt}.")
            return True

        if ok:
            # 代码没抛异常, 但机器人 API 返回过 False 或 ret_val 明确失败。
            failure_text = "; ".join(action_failures) or f"ret_val={ret_val!r}"
            last_report = FailureReport(
                task_name="user_instruction",
                instruction=instruction,
                robot_name=robot.embodiment_name,
                code_raised=False,
                expected={"robot_action_success": True},
                actual={
                    "robot_action_success": False,
                    "action_failures": action_failures,
                    "ret_val": ret_val,
                },
                diagnosis=[
                    "The Python snippet ran without an exception, but a robot API call returned False.",
                    f"Failure detail: {failure_text}",
                ],
                suggestions=[
                    "Check the boolean return values from robot.pick(), robot.place(), and robot.pick_and_place().",
                    "For a mobile robot, navigate to a reachable table-side standoff pose before picking or placing; do not park directly on top of the target or under the table. A safe default is table_x - 0.05, table_y + 0.70.",
                    "Use mobile.is_reachable(target) after navigation and adjust the base pose if it is False.",
                    "For a dual-arm robot, choose an arm with robot.choose_arm_for(target) or robot.is_reachable_by(arm, target). For simultaneous two-object transport, use robot.lift_two_objects(pos_a, pos_b) followed by robot.place_two_objects(target_a, target_b).",
                ],
            )
        else:
            # 执行报错 -> 构造简化版失败报告
            last_report = FailureReport(
                task_name="user_instruction",
                instruction=instruction,
                robot_name=robot.embodiment_name,
                code_raised=True,
                traceback=msg,
                diagnosis=["Code raised an exception during execution."],
                suggestions=["Fix the syntax/API usage indicated by the traceback."],
            )
        print(f"\n⚠️  Attempt {attempt} failed, will retry with structured feedback.")

    print(f"\n❌ Giving up after {max_attempts} attempts.")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--robot",
        choices=["kuka", "franka", "mobile", "dual_arm", "mobile_dual_arm", "dual_franka"],
        default="kuka",
    )
    ap.add_argument("--instruction", default=None,
                    help="One-shot instruction; otherwise enters REPL.")
    ap.add_argument("--mode", choices=["baseline", "b", "ba"], default="ba",
                    help="baseline: 1 attempt, no card. "
                         "b: retry+feedback, no card. "
                         "ba: retry+feedback+capability card (default).")
    ap.add_argument("--no-card", action="store_true",
                    help="Disable capability card (equivalent to mode=b).")
    ap.add_argument("--no-retry", action="store_true",
                    help="Disable retry (equivalent to 1 attempt).")
    args = ap.parse_args()

    # 从 mode 推导出开关, --no-card / --no-retry 可覆盖
    use_card = args.mode == "ba"
    use_retry = args.mode != "baseline"
    if args.no_card:
        use_card = False
    if args.no_retry:
        use_retry = False

    scene, robot = build_scene_and_robot(args.robot)
    client = make_client()

    print(f"\n⚙️  Config: capability_card={use_card}, retry={use_retry}")

    if args.instruction:
        run_instruction(scene, robot, args.instruction, client,
                        use_card=use_card, use_retry=use_retry)
        input("\n[Press Enter to close]")
        return

    print("\n💬 Enter a natural-language instruction. Type 'exit' to quit.")
    print("   Example: put the red block on the yellow tray")
    while True:
        try:
            line = input("\n👉 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye.")
            break
        if not line:
            continue
        if line.lower() in {"exit", "quit"}:
            break
        run_instruction(scene, robot, line, client,
                        use_card=use_card, use_retry=use_retry)


if __name__ == "__main__":
    main()
