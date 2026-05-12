"""
Smoke test: 不调用 LLM, 直接手写 LMP 代码验证物理层。

v5 改动:
- 加入 'mobile' 选项
- 加入 'dual_arm' 选项
- 加入 'mobile_dual_arm' 选项
- 加入 'dual_franka' 选项
- 桌子放远 (x=1.5), 这样 mobile 必须先 navigate 才能 pick
- mobile 用专门的 LMP (含 navigate_to + pick_and_place)
- dual_arm / mobile_dual_arm / dual_franka 跑同一个双块入盘任务
- KUKA / Franka 用近距离场景 (x=0), 维持原行为

用法:
    python -m examples.smoke_test --robot kuka
    python -m examples.smoke_test --robot franka
    python -m examples.smoke_test --robot mobile
    python -m examples.smoke_test --robot dual_arm --gui
    python -m examples.smoke_test --robot mobile_dual_arm --gui
    python -m examples.smoke_test --robot dual_franka --gui
"""
import argparse
import time
import pybullet as p

from robots import make_robot
from perception import TabletopScene
from lmp import execute_lmp


# ============================================================
# 手写 LMP: 静态机器人 vs mobile 机器人 vs dual-arm 机器人
# ============================================================

# 静态机器人 (KUKA / Franka): 桌子在原点, 近距离, 不需要 navigate
LMP_FIXED_BASE = """
red_pos = scene.get_object_position('red block')
tray_pos = scene.get_object_position('yellow tray')
target = tray_pos + np.array([0, 0, 0.05])
success = robot.pick_and_place(red_pos, target, place_release_height=0.005)
ret_val = 'success' if success else 'failure'
"""

# 移动机器人: 桌子在 (1.5, 0), 默认 husky 在原点, 必须先 navigate 才能够到
LMP_MOBILE = """
# 取桌面位置 (用于决定停车点)
table_x, table_y = float(scene.table_position[0]), float(scene.table_position[1])

red_pos = scene.get_object_position('red block')
tray_pos = scene.get_object_position('yellow tray')

# Step 1: 开到桌子侧边, 保持约 0.70m standoff, 避免视觉上钻进桌子
# Husky 朝向桌面 (navigate_to 默认行为)
mobile.navigate_to(table_x - 0.05, table_y + 0.70)

# Step 2: 抓
pick_ok = robot.pick(red_pos)

# Step 3: 放 (导航后的位置应该够得到 tray, 因为 tray 在桌面同一区域)
target = tray_pos + np.array([0, 0, 0.05])
success = False
if pick_ok:
    success = robot.place(target, pre_release_height=0.005)
ret_val = 'success' if success else 'failure'
"""

# 双臂机器人: 同时抓起两个方块, 同时放到托盘
LMP_DUAL_ARM = """
red_pos = scene.get_object_position('red block')
green_pos = scene.get_object_position('green block')
tray_pos = scene.get_object_position('yellow tray')

red_target = tray_pos + np.array([-0.03, -0.03, 0.05])
green_target = tray_pos + np.array([0.03, 0.03, 0.05])

lift_ok = robot.lift_two_objects(red_pos, green_pos)
place_ok = False
if lift_ok:
    place_ok = robot.place_two_objects(
        red_target,
        green_target,
        pre_release_height=0.005,
    )
ret_val = 'success' if (lift_ok and place_ok) else 'failure'
"""

# 双 Franka: 同一个双臂任务, 但释放高度从 parallel-jaw 能力卡读取
LMP_DUAL_FRANKA = """
red_pos = scene.get_object_position('red block')
green_pos = scene.get_object_position('green block')
tray_pos = scene.get_object_position('yellow tray')

red_target = tray_pos + np.array([-0.03, -0.03, 0.05])
green_target = tray_pos + np.array([0.03, 0.03, 0.05])
release_h = robot.capability_card.recommended_release_height_m

lift_ok = robot.lift_two_objects(red_pos, green_pos)
place_ok = False
if lift_ok:
    place_ok = robot.place_two_objects(
        red_target,
        green_target,
        pre_release_height=release_h,
    )
ret_val = 'success' if (lift_ok and place_ok) else 'failure'
"""

# 移动双臂机器人: 同一个双臂任务, 但必须先导航到桌边
LMP_MOBILE_DUAL_ARM = """
table_x, table_y = float(scene.table_position[0]), float(scene.table_position[1])
mobile.navigate_to(table_x, table_y + robot.capability_card.nav_min_clearance_m)

red_pos = scene.get_object_position('red block')
green_pos = scene.get_object_position('green block')
tray_pos = scene.get_object_position('yellow tray')

red_target = tray_pos + np.array([-0.03, -0.03, 0.05])
green_target = tray_pos + np.array([0.03, 0.03, 0.05])

lift_ok = robot.lift_two_objects(red_pos, green_pos)
place_ok = False
if lift_ok:
    place_ok = robot.place_two_objects(
        red_target,
        green_target,
        pre_release_height=0.005,
    )
ret_val = 'success' if (lift_ok and place_ok) else 'failure'
"""


def setup_scene_for(robot_name: str) -> TabletopScene:
    """根据机器人类型选择不同的场景布局。"""
    if robot_name == "mobile":
        # 远距离场景: 桌子在 (1.5, 0), Husky 默认从原点出发
        scene = TabletopScene(table_position=(1.5, 0.0, 0.0))
        scene.add_cube("red block", [1.5, 0.0, 0.65], color=(1, 0, 0, 1))
        scene.add_cube("green block", [1.55, 0.15, 0.65], color=(0, 1, 0, 1))
        scene.add_tray("yellow tray", [1.3, 0.3, 0.63], color=(1, 1, 0, 1))
    elif robot_name in {"dual_arm", "dual_franka"}:
        # 双臂近距离场景: 物体分布在左右臂共享工作区内
        scene = TabletopScene(table_position=(0.0, 0.0, 0.0))
        scene.add_cube("red block", [0.5, -0.10, 0.65], color=(1, 0, 0, 1))
        scene.add_cube("green block", [0.55, 0.15, 0.65], color=(0, 1, 0, 1))
        scene.add_tray("yellow tray", [0.35, 0.12, 0.63], color=(1, 1, 0, 1))
    elif robot_name == "mobile_dual_arm":
        # 远距离双臂场景: 任务与 dual_arm 相同, 但需要先导航
        scene = TabletopScene(table_position=(1.5, 0.0, 0.0))
        scene.add_cube("red block", [1.45, -0.10, 0.65], color=(1, 0, 0, 1))
        scene.add_cube("green block", [1.55, 0.15, 0.65], color=(0, 1, 0, 1))
        scene.add_tray("yellow tray", [1.30, 0.12, 0.63], color=(1, 1, 0, 1))
    else:
        # 近距离场景 (KUKA / Franka): 桌子在原点, 维持 v3 行为
        scene = TabletopScene(table_position=(0.0, 0.0, 0.0))
        scene.add_cube("red block", [0.5, 0.0, 0.65], color=(1, 0, 0, 1))
        scene.add_cube("green block", [0.55, 0.15, 0.65], color=(0, 1, 0, 1))
        scene.add_tray("yellow tray", [0.3, 0.3, 0.63], color=(1, 1, 0, 1))
    return scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--robot",
        choices=["kuka", "franka", "mobile", "dual_arm", "mobile_dual_arm", "dual_franka"],
        default="kuka",
    )
    ap.add_argument("--gui", action="store_true",
                    help="Show the PyBullet GUI. Defaults to headless DIRECT mode.")
    args = ap.parse_args()

    p.connect(p.GUI if args.gui else p.DIRECT)

    # 给 mobile / dual_arm 设置一个更好的相机视角
    if args.robot in {"mobile", "dual_arm", "mobile_dual_arm", "dual_franka"} and args.gui:
        target = {
            "mobile": [0.7, 0.1, 0.4],
            "dual_arm": [0.35, 0.0, 0.55],
            "mobile_dual_arm": [1.25, 0.1, 0.55],
            "dual_franka": [0.35, 0.0, 0.55],
        }[args.robot]
        p.resetDebugVisualizerCamera(
            cameraDistance=3.5,
            cameraYaw=45,
            cameraPitch=-25,
            cameraTargetPosition=target,
        )

    scene = setup_scene_for(args.robot)
    robot = make_robot(args.robot)
    robot.scene = scene

    for _ in range(240):
        p.stepSimulation()

    print(f"\n✅ Scene ready. {robot.describe()}")
    print(scene.describe())

    # 选对应的 LMP
    if args.robot == "mobile":
        code = LMP_MOBILE
    elif args.robot == "mobile_dual_arm":
        code = LMP_MOBILE_DUAL_ARM
    elif args.robot == "dual_franka":
        code = LMP_DUAL_FRANKA
    elif args.robot == "dual_arm":
        code = LMP_DUAL_ARM
    else:
        code = LMP_FIXED_BASE

    # 给 embodiment-specific 代码注入别名
    exec_ctx = {"robot": robot, "scene": scene}
    if args.robot in {"mobile", "mobile_dual_arm"}:
        exec_ctx["mobile"] = robot
    if args.robot in {"dual_arm", "mobile_dual_arm", "dual_franka"}:
        exec_ctx["left"] = robot.left
        exec_ctx["right"] = robot.right

    ok, msg, locals_dict = execute_lmp(code, exec_ctx, verbose=True)
    print(f"\nResult: ok={ok}, {msg}")
    print(f"ret_val = {locals_dict.get('ret_val')!r}")

    tray = scene.get_object_position("yellow tray")
    red = scene.get_object_position("red block")
    red_dist = float(((red[:2] - tray[:2]) ** 2).sum() ** 0.5)
    if args.robot in {"dual_arm", "mobile_dual_arm", "dual_franka"}:
        green = scene.get_object_position("green block")
        green_dist = float(((green[:2] - tray[:2]) ** 2).sum() ** 0.5)
        physical_success = red_dist < 0.20 and green_dist < 0.20
        print(f"physical_success = {physical_success} "
              f"(red/tray = {red_dist:.3f}m, green/tray = {green_dist:.3f}m)")
    else:
        physical_success = red_dist < 0.15
        print(f"physical_success = {physical_success} "
              f"(red/tray horizontal distance = {red_dist:.3f}m)")
    if not ok or locals_dict.get("ret_val") != "success" or not physical_success:
        p.disconnect()
        raise SystemExit(1)

    if args.gui:
        print("\n[Press Ctrl+C to close the PyBullet window]")
        try:
            while True:
                p.stepSimulation()
                time.sleep(1.0 / 240)
        except KeyboardInterrupt:
            pass
    p.disconnect()


if __name__ == "__main__":
    main()
