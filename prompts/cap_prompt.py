"""
CaP-style few-shot prompt + B+A 扩展。

v3 改动:
- 支持注入 Capability Card (方法 A, 静态先验)
- 支持注入 Failure Report (方法 B, 运行时修正反馈)
- API hint 更新: place 新增 pre_release_height 参数, 让 LLM 能显式控制
"""
from typing import Optional
from robots.base_robot import BaseRobot
from lmp.failure_report import FailureReport


# ============================================================
# 系统提示
# ============================================================
SYSTEM_PROMPT = """You are a robot control code generator. You will be given:
1. A description of the current robot embodiment (type, DoF, gripper).
2. (Optionally) A Capability Card describing embodiment-specific priors.
3. A description of the scene (object names and positions).
4. A natural-language instruction from the user.
5. (Optionally) A Failure Report from a previous failed attempt.

Your job: output a short Python snippet that uses the provided APIs to
accomplish the instruction. Rules:
  - Output ONLY code inside ```python ... ``` fences. No prose.
  - Do NOT use 'import'. numpy is pre-imported as 'np'.
  - Call the provided APIs (robot.*, scene.*). Do not invent new functions.
  - Robot APIs return bool. Store and check those return values; do not ignore failures.
  - Keep the code short and readable. Use loops/conditionals when appropriate.
  - If a Capability Card is provided, respect its implications (e.g. release height).
  - If a Failure Report is provided, analyze the diagnosis and fix the identified issue.
  - If the instruction requires two objects to be lifted at the same time and
    the dual-arm coordinated API is available, use robot.lift_two_objects(...).
  - Coordinates are world-frame meters from PyBullet. Table surface is around z=0.63.
"""


# ============================================================
# Few-shot 示例
# ============================================================
FEW_SHOT_EXAMPLES = """# Example 1: Put the red apple into the blue tray.
apple_pos = scene.get_object_position('red apple')
tray_pos = scene.get_object_position('blue tray')
robot.pick_and_place(apple_pos, tray_pos + np.array([0, 0, 0.05]))

# Example 2: Move the green block 10cm to the right (precise placement).
# Use a low release height to minimize bounce.
block_pos = scene.get_object_position('green block')
target = block_pos + np.array([0.1, 0, 0])
robot.pick_and_place(block_pos, target, place_release_height=0.005)

# Example 3: Stack all blocks on top of the first one.
# Stacking requires a low release height for stability.
names = [n for n in scene.get_object_names() if 'block' in n]
base_pos = scene.get_object_position(names[0])
for i, name in enumerate(names[1:]):
    src = scene.get_object_position(name)
    dst = base_pos + np.array([0, 0, 0.05 * (i + 1)])
    robot.pick_and_place(src, dst, place_release_height=0.005)

# Example 4: Find and report the position of the leftmost object.
names = scene.get_object_names()
positions = np.array([scene.get_object_position(n) for n in names])
leftmost_idx = int(np.argmin(positions[:, 0]))
ret_val = names[leftmost_idx]
"""


# ============================================================
# Prompt 构造 (支持 B+A)
# ============================================================
def build_user_prompt(
    robot: BaseRobot,
    scene_description: str,
    instruction: str,
    use_capability_card: bool = True,
    include_few_shot: bool = True,
    failure_report: Optional[FailureReport] = None,
    last_error: str = "",
) -> str:
    """组装 prompt。

    Args:
        use_capability_card: True 则注入 CapabilityCard (方法 A)
        include_few_shot: True 则注入 CaP few-shot 示例; False 则仅给 API + scene + instruction
        failure_report: 若提供, 注入结构化失败报告 (方法 B)
        last_error: 兼容旧调用: 原始 traceback 字符串
    """
    api_hint = _api_hint_for(robot)

    parts = [
        f"## Current Robot\n{robot.describe()}",
    ]

    # --- 方法 A: 注入 Capability Card ---
    if use_capability_card:
        parts.append(
            f"\n## {robot.capability_card.to_prompt_section()}"
        )

    parts.extend([
        f"\n## Available APIs\n{api_hint}",
        f"\n## Scene\n{scene_description}",
    ])

    if include_few_shot:
        parts.append(f"\n## Few-shot examples\n```python\n{FEW_SHOT_EXAMPLES}```")

    # --- 方法 B: 注入结构化 Failure Report ---
    if failure_report is not None:
        parts.append(f"\n## {failure_report.to_prompt_section()}")
    elif last_error:
        parts.append(
            f"\n## Previous attempt failed with:\n```\n{last_error}\n```\n"
            "Please analyze the error and output a CORRECTED code snippet."
        )

    parts.append(f"\n## Instruction\n{instruction}\n\nNow output the code:")
    return "\n".join(parts)


def _api_hint_for(robot: BaseRobot) -> str:
    common = """# High-level (embodiment-agnostic):
robot.pick(position_3d) -> bool
robot.place(target_position_3d, pre_release_height=None) -> bool
    # pre_release_height defaults to capability_card.recommended_release_height_m.
    # Set a smaller value (e.g. 0.005) for precise placement or stacking.
robot.pick_and_place(src_pos_3d, target_pos_3d, place_release_height=None) -> bool
    # place_release_height is forwarded to place().
robot.move_ee_to(position_3d, orientation=None) -> bool
robot.get_ee_pose() -> (position_3d, quaternion_4d)

# Gripper (semantics depend on embodiment):
robot.activate_gripper()   # """ + _gripper_semantics(robot, "activate") + """
robot.release_gripper()    # """ + _gripper_semantics(robot, "release") + """

# Scene (perception):
scene.get_object_names() -> list[str]
scene.get_object_position(name: str) -> np.ndarray of shape (3,)
"""

    extras = []

    # ---- v4: mobile 机器人的额外 API ----
    if getattr(robot, "capability_card", None) and robot.capability_card.has_mobile_base:
        mobile_extra = """
# === Mobile-base API (only available on this robot) ===
# `mobile` is the SAME object as `robot` but exposes navigation methods.
mobile.get_base_position() -> np.ndarray of shape (3,)   # (x, y, theta)
mobile.navigate_to(x: float, y: float, theta: float = None) -> bool
    # Move the base to (x, y). Wait for completion before returning.
    # Navigate to a table-side standoff pose, e.g. 0.70-0.90m from the table center.
    # Do NOT navigate to the target object's exact (x, y).
    # Do NOT park under the table or at a table corner.
    # For this tabletop scene, a good default is:
    #   table_x, table_y = float(scene.table_position[0]), float(scene.table_position[1])
    #   mobile.navigate_to(table_x, table_y + robot.capability_card.nav_min_clearance_m)
mobile.is_reachable(target_position_3d) -> bool
    # Check if `target` is within the arm's workspace at the CURRENT base pose.
    # If False, you MUST navigate_to() first before pick/place.
"""
        extras.append(mobile_extra)

    # ---- v5: dual-arm 机器人的额外 API ----
    if getattr(robot, "capability_card", None) and robot.capability_card.has_dual_arms:
        dual_arm_extra = """
# === Dual-arm API (only available on this robot) ===
robot.left.pick(position_3d) -> bool
robot.left.place(target_position_3d, pre_release_height=None) -> bool
robot.right.pick(position_3d) -> bool
robot.right.place(target_position_3d, pre_release_height=None) -> bool
robot.choose_arm_for(target_position_3d) -> 'left' or 'right'
    # Convenience helper. Returns a good arm name based on workspace and object side.
robot.is_reachable_by(arm_name: str, target_position_3d) -> bool
    # Check whether 'left' or 'right' can reach the target without moving the base.
robot.pick_with_arm(arm_name: str, src_pos_3d) -> bool
robot.place_with_arm(arm_name: str, target_pos_3d, pre_release_height=None) -> bool
robot.pick_and_place_with_arm(arm_name: str, src_pos_3d, target_pos_3d, place_release_height=None) -> bool
robot.hold_with_arm(arm_name: str, object_name_or_position) -> bool
robot.lift_two_objects(first_pos_3d, second_pos_3d, lift_height=0.18) -> bool
    # Coordinated bimanual lift. Use this for instructions like "lift both
    # blocks at the same time"; separate pick_with_arm calls are sequential.
robot.pick_two_objects(first_pos_3d, second_pos_3d, lift_height=0.18) -> bool
    # Alias of lift_two_objects.
robot.place_two_objects(first_target_pos_3d, second_target_pos_3d, pre_release_height=0.005) -> bool
    # Coordinated bimanual place. If called after lift_two_objects(a, b),
    # first_target is assigned to the arm holding object a and second_target
    # to the arm holding object b.
robot.pick_and_place_two_objects(first_pos_3d, second_pos_3d, first_target_pos_3d, second_target_pos_3d, place_release_height=0.005) -> bool
    # Coordinated two-object transport: lift both, then place both.
robot.release_arm(arm_name: str)
    # One arm can hold an object while the other arm manipulates another object.
"""
        extras.append(dual_arm_extra)
    return common + "".join(extras)


def _gripper_semantics(robot: BaseRobot, action: str) -> str:
    g = robot.gripper_type
    if "suction" in g:
        g = "suction"
    elif "parallel_jaw" in g:
        g = "parallel_jaw"
    if action == "activate":
        return {
            "suction": "turn suction ON (attaches nearest object within 15cm)",
            "parallel_jaw": "close fingers (grips nearest object within 15cm)",
        }.get(g, "engage gripper")
    else:
        return {
            "suction": "turn suction OFF (releases the attached object)",
            "parallel_jaw": "open fingers (releases any held object)",
        }.get(g, "release gripper")
