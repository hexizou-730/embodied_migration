"""
Cross-Embodiment Migration Benchmark.

支持 strict ablation 模式:
  - 'api':          API-only prompt, 1 次尝试
  - 'fewshot':      API + few-shot, 1 次尝试
  - 'card':         Capability Card, 1 次尝试
  - 'failure':      Failure Report retry, 最多 3 次尝试
  - 'card_failure': Capability Card + Failure Report retry, 最多 3 次尝试

旧别名仍可用:
  - 'baseline' -> 'fewshot'
  - 'b'        -> 'failure'
  - 'ba'       -> 'card_failure'

用法:
    python -m benchmark.run_benchmark                    # dual_arm × mobile_dual_arm × dual_franka
    python -m benchmark.run_benchmark --modes card_failure
    python -m benchmark.run_benchmark --tasks bimanual --robots dual_arm dual_franka
    python -m benchmark.run_benchmark --modes api fewshot card failure card_failure --trials 3
    python -m benchmark.run_benchmark --gui              # 带 GUI, 慢
"""
import argparse
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pybullet as p
from dotenv import load_dotenv

from robots import make_robot
from perception import TabletopScene
from prompts import SYSTEM_PROMPT, build_user_prompt
from lmp import execute_lmp, extract_code_or_text
from lmp.failure_report import build_failure_report
from llm_client import DEFAULT_MODEL, make_client, chat
from benchmark.experiment_logging import (
    BenchmarkLogger,
    analyze_code,
    choose_failure_type,
    choose_failure_subtype,
    extract_error_excerpt,
    jsonable,
    summary_row_from_record,
    snapshot_scene,
)
from benchmark.llm_cache import LLMResponseCache, cached_chat


load_dotenv()


MODE_CONFIGS = {
    # Strict ablation modes for paper experiments.
    "api": {
        "canonical": "api",
        "use_capability_card": False,
        "include_few_shot": False,
        "use_failure_report": False,
        "max_attempts": 1,
        "description": "API-only prompt, 1 attempt, no examples/card/feedback",
    },
    "fewshot": {
        "canonical": "fewshot",
        "use_capability_card": False,
        "include_few_shot": True,
        "use_failure_report": False,
        "max_attempts": 1,
        "description": "API + few-shot prompt, 1 attempt",
    },
    "card": {
        "canonical": "card",
        "use_capability_card": True,
        "include_few_shot": True,
        "use_failure_report": False,
        "max_attempts": 1,
        "description": "Capability Card only, 1 attempt",
    },
    "failure": {
        "canonical": "failure",
        "use_capability_card": False,
        "include_few_shot": True,
        "use_failure_report": True,
        "max_attempts": 3,
        "description": "Failure Report retry only, no Capability Card",
    },
    "card_failure": {
        "canonical": "card_failure",
        "use_capability_card": True,
        "include_few_shot": True,
        "use_failure_report": True,
        "max_attempts": 3,
        "description": "Capability Card + Failure Report retry",
    },
    # Legacy aliases kept for older commands/results.
    "baseline": {
        "canonical": "fewshot",
        "use_capability_card": False,
        "include_few_shot": True,
        "use_failure_report": False,
        "max_attempts": 1,
        "description": "Legacy alias for fewshot",
    },
    "b": {
        "canonical": "failure",
        "use_capability_card": False,
        "include_few_shot": True,
        "use_failure_report": True,
        "max_attempts": 3,
        "description": "Legacy alias for failure",
    },
    "ba": {
        "canonical": "card_failure",
        "use_capability_card": True,
        "include_few_shot": True,
        "use_failure_report": True,
        "max_attempts": 3,
        "description": "Legacy alias for card_failure",
    },
}

STRICT_MODES = ["api", "fewshot", "card", "failure", "card_failure"]


def _mode_config(mode: str) -> Dict[str, object]:
    if mode not in MODE_CONFIGS:
        raise ValueError(f"Unknown mode '{mode}'. Available: {sorted(MODE_CONFIGS)}")
    return MODE_CONFIGS[mode]


# ============================================================
# Task 定义: (name, instruction, checker)
#
# checker(scene) -> (success: bool, expected: dict, actual: dict)
#   expected/actual 字典用于构建 FailureReport (方法 B 的输入)
# ============================================================

def task_pick_red_to_tray():
    def check(scene):
        red = scene.get_object_position("red block")
        tray = scene.get_object_position("yellow tray")
        horiz_dist = float(np.linalg.norm(red[:2] - tray[:2]))
        success = horiz_dist < 0.15
        return success, (
            {"red_block_near_tray": True,
             "red_block_position": tuple(round(float(v), 3) for v in red)},
            {"red_block_near_tray": horiz_dist < 0.15,
             "red_block_position": tuple(round(float(v), 3) for v in red),
             "horizontal_distance_to_tray": round(horiz_dist, 3)},
        )
    return ("pick_red_to_tray", "Put the red block into the yellow tray.", check)


def task_move_green_right():
    """要求绿方块向右移动至少 5cm (相对初始位置)。
    用 scene.object_initial_positions 获取初始位置。
    """
    def check(scene):
        green = scene.get_object_position("green block")
        initial = scene.object_initial_positions.get("green block")
        if initial is None:
            # 兜底: 假设初始 x 是 0.55 或 1.55
            initial_x = 0.55 if green[0] < 1.0 else 1.55
        else:
            initial_x = initial[0]
        # 移动至少 5cm
        success = green[0] > initial_x + 0.05
        return success, (
            {"green_block_x": f">{initial_x + 0.05:.2f}"},
            {"green_block_x": round(float(green[0]), 3),
             "initial_x": round(initial_x, 3)},
        )
    return ("move_green_right", "Move the green block 10 cm to the right.", check)


def task_stack_two():
    def check(scene):
        red = scene.get_object_position("red block")
        blue = scene.get_object_position("blue block")
        horiz = float(np.linalg.norm(red[:2] - blue[:2]))
        stacked = blue[2] > red[2] + 0.02
        success = horiz < 0.08 and stacked
        return success, (
            {"blue_on_red": True,
             "blue_block_position": tuple(round(float(v), 3) for v in red + np.array([0, 0, 0.05]))},
            {"blue_on_red": success,
             "blue_block_position": tuple(round(float(v), 3) for v in blue),
             "horizontal_offset": round(horiz, 3)},
        )
    return ("stack_two", "Stack the blue block on top of the red block.", check)


def task_report_leftmost():
    """纯推理任务, 不涉及物理执行。"""
    def check(scene):
        # 这个任务在 run_trial 里特殊处理 (看 ret_val)
        return True, ({}, {})
    return (
        "report_leftmost",
        "What is the name of the leftmost block? Set ret_val to its name.",
        check,
    )


def task_refuse_rotate_object():
    """Capability-conditioned refusal: all current robots cannot rotate in hand."""
    def check(scene):
        # 这个任务在 run_trial 里特殊处理 (看 ret_val)
        return True, ({}, {})

    return (
        "refuse_rotate_object",
        "Rotate the red block by 90 degrees in hand. "
        "If the current robot cannot rotate objects in hand, do not attempt motion; "
        "set ret_val = 'refuse_cannot_rotate'.",
        check,
    )


def task_refuse_missing_object():
    """Scene-conditioned refusal: requested object is absent."""
    def check(scene):
        # 这个任务在 run_trial 里特殊处理 (看 ret_val)
        return True, ({}, {})

    return (
        "refuse_missing_object",
        "Pick up the purple block. If the purple block is not present in the scene, "
        "do not call pick/place; set ret_val = 'refuse_missing_object'.",
        check,
    )


# ============================================================
# v6 Mobile + Dual-arm 任务: 让 embodiment 差异进入任务本身
# ============================================================

def task_wide_blue_to_tray():
    """宽桌面取放: mobile 需要选择能同时覆盖 blue 和 tray 的停车点。"""
    def check(scene):
        blue = scene.get_object_position("blue block")
        tray = scene.get_object_position("yellow tray")
        horiz_dist = float(np.linalg.norm(blue[:2] - tray[:2]))
        success = horiz_dist < 0.15
        return success, (
            {"blue_block_near_tray": True},
            {"blue_block_near_tray": success,
             "horizontal_distance_to_tray": round(horiz_dist, 3),
             "blue_block_position": tuple(round(float(v), 3) for v in blue)},
        )

    return (
        "wide_blue_to_tray",
        "Move the blue block into the yellow tray. The block and tray are far apart "
        "on the tabletop, so choose a base pose or arm that can reach both before "
        "starting the pick-and-place.",
        check,
    )


def task_collect_red_and_blue_to_tray():
    """顺序多物体任务: 需要循环/返回值检查/稳定低释放。"""
    def check(scene):
        red = scene.get_object_position("red block")
        blue = scene.get_object_position("blue block")
        tray = scene.get_object_position("yellow tray")
        red_dist = float(np.linalg.norm(red[:2] - tray[:2]))
        blue_dist = float(np.linalg.norm(blue[:2] - tray[:2]))
        success = red_dist < 0.15 and blue_dist < 0.15
        return success, (
            {"red_and_blue_near_tray": True},
            {"red_distance_to_tray": round(red_dist, 3),
             "blue_distance_to_tray": round(blue_dist, 3),
             "red_and_blue_near_tray": success},
        )

    return (
        "collect_red_and_blue_to_tray",
        "Put both the red block and the blue block into the yellow tray. "
        "Check each robot action result and use a low release height.",
        check,
    )


def task_hold_red_while_place_green():
    """双臂协作任务: 一只手保持红块悬空, 另一只手放绿块。"""
    def check(scene):
        red = scene.get_object_position("red block")
        green = scene.get_object_position("green block")
        tray = scene.get_object_position("yellow tray")
        green_dist = float(np.linalg.norm(green[:2] - tray[:2]))
        red_lifted = float(red[2]) > float(scene.table_top_z) + 0.08
        green_in_tray = green_dist < 0.15
        success = red_lifted and green_in_tray
        return success, (
            {"red_block_lifted_while_green_in_tray": True},
            {"red_z": round(float(red[2]), 3),
             "red_lifted": red_lifted,
             "green_distance_to_tray": round(green_dist, 3),
             "green_in_tray": green_in_tray},
        )

    return (
        "hold_red_while_place_green",
        "Use one arm to hold the red block above the table while the other arm "
        "places the green block into the yellow tray. Keep the red block held "
        "in the air at the end.",
        check,
    )


def task_lift_red_and_green_together():
    """双臂同时持物任务: 单臂机器人无法在最终状态同时保持两个物体悬空。"""
    def check(scene):
        red = scene.get_object_position("red block")
        green = scene.get_object_position("green block")
        red_lifted = float(red[2]) > float(scene.table_top_z) + 0.08
        green_lifted = float(green[2]) > float(scene.table_top_z) + 0.08
        success = red_lifted and green_lifted
        return success, (
            {"red_and_green_lifted_together": True},
            {"red_z": round(float(red[2]), 3),
             "green_z": round(float(green[2]), 3),
             "red_lifted": red_lifted,
             "green_lifted": green_lifted},
        )

    return (
        "lift_red_and_green_together",
        "Lift the red block and the green block at the same time, using one arm "
        "for each block. Keep both blocks held above the table at the end.",
        check,
    )


def task_lift_red_green_together_to_tray():
    """共同双臂迁移任务: 同时拿起红/绿方块, 再同时放入托盘。"""
    def check(scene):
        red = scene.get_object_position("red block")
        green = scene.get_object_position("green block")
        tray = scene.get_object_position("yellow tray")
        red_dist = float(np.linalg.norm(red[:2] - tray[:2]))
        green_dist = float(np.linalg.norm(green[:2] - tray[:2]))
        red_in_tray = red_dist < 0.20
        green_in_tray = green_dist < 0.20
        red_released_low = float(red[2]) < float(scene.table_top_z) + 0.10
        green_released_low = float(green[2]) < float(scene.table_top_z) + 0.10
        success = red_in_tray and green_in_tray and red_released_low and green_released_low
        return success, (
            {"red_and_green_in_yellow_tray": True},
            {"red_distance_to_tray": round(red_dist, 3),
             "green_distance_to_tray": round(green_dist, 3),
             "red_z": round(float(red[2]), 3),
             "green_z": round(float(green[2]), 3),
             "red_in_tray": red_in_tray,
             "green_in_tray": green_in_tray,
             "red_released_low": red_released_low,
             "green_released_low": green_released_low},
        )

    return (
        "lift_red_green_together_to_tray",
        "Pick up the red block and the green block at the same time, using one "
        "arm for each block, then place both blocks into the yellow tray at the "
        "same time. Use coordinated two-arm APIs; if the robot has a mobile "
        "base, navigate to a safe table-side standoff before lifting.",
        check,
    )


# ============================================================
# v4 空间几何任务: 需要 LLM 现场写 numpy 几何计算
# (Tool Calling 范式做不到, 必须 Code Generation)
# ============================================================

def task_arrange_line():
    """3 个方块排成水平直线, 间距 8cm, 中心在桌面中心。"""
    def check(scene):
        names = ["red block", "green block", "blue block"]
        positions = [scene.get_object_position(n) for n in names]
        positions_sorted = sorted(positions, key=lambda p: p[0])
        center_x = float(np.mean([p[0] for p in positions_sorted]))
        center_y = float(np.mean([p[1] for p in positions_sorted]))
        # 期望中心: 桌面中心 (table_position 的 x, y)
        target_cx = float(scene.table_position[0])
        if target_cx == 0.0:
            target_cx = 0.5  # 静态机器人场景下桌子在原点, 但物体集中在 x=0.5
        target_cy = 0.0
        ys = np.array([p[1] for p in positions_sorted])
        gaps = np.diff([p[0] for p in positions_sorted])
        center_ok = abs(center_x - target_cx) < 0.05 and abs(center_y - target_cy) < 0.05
        line_ok = float(np.std(ys)) < 0.03
        spacing_ok = all(abs(g - 0.08) < 0.025 for g in gaps)
        success = center_ok and line_ok and spacing_ok
        return success, (
            {"layout": "horizontal_line", "spacing_m": 0.08,
             "center": (round(target_cx, 2), round(target_cy, 2))},
            {"layout_actual": "line" if line_ok else "scattered",
             "center_actual": (round(center_x, 3), round(center_y, 3)),
             "spacings_actual": [round(float(g), 3) for g in gaps],
             "y_std": round(float(np.std(ys)), 3)},
        )

    # 对应的指令也要因机器人不同而不同 (mobile 场景下中心是 1.5)
    return (
        "arrange_line",
        "Arrange the three blocks into a horizontal line, 8cm apart, "
        "centered at the center of the table. All blocks should have the same y coordinate.",
        check,
    )


def task_arrange_triangle():
    """3 个方块放到正三角形顶点, 边长 12cm, 中心在 tray 位置。"""
    def check(scene):
        names = ["red block", "green block", "blue block"]
        positions = np.array([scene.get_object_position(n)[:2] for n in names])
        center = np.mean(positions, axis=0)
        tray = scene.get_object_position("yellow tray")[:2]
        # 各点到中心的距离应相近, 且约等于 0.12 / sqrt(3)
        expected_radius = 0.12 / (3 ** 0.5)
        radii = np.linalg.norm(positions - center, axis=1)
        radius_ok = all(abs(r - expected_radius) < 0.03 for r in radii)
        # 中心接近 tray
        center_ok = float(np.linalg.norm(center - tray)) < 0.06
        # 三角形规整 (相邻角都 ~ 120°)
        success = radius_ok and center_ok
        return success, (
            {"layout": "equilateral_triangle", "side_m": 0.12, "center_at": "tray"},
            {"radii_actual": [round(float(r), 3) for r in radii],
             "expected_radius": round(expected_radius, 3),
             "center_offset_from_tray": round(float(np.linalg.norm(center - tray)), 3)},
        )
    return (
        "arrange_triangle",
        "Place the three blocks at the vertices of an equilateral triangle "
        "with side 12cm, centered at the yellow tray position.",
        check,
    )


def task_arrange_circle():
    """3 个方块均匀分布在以 tray 为中心、半径 10cm 的圆上。"""
    def check(scene):
        names = ["red block", "green block", "blue block"]
        positions = np.array([scene.get_object_position(n)[:2] for n in names])
        tray = scene.get_object_position("yellow tray")[:2]
        radii = np.linalg.norm(positions - tray, axis=1)
        radius_ok = all(abs(r - 0.10) < 0.025 for r in radii)
        # 三个角度均匀分布 (120° 间距)
        angles = np.array([np.arctan2(p[1] - tray[1], p[0] - tray[0]) for p in positions])
        angles_sorted = sorted(angles.tolist())
        gaps = [angles_sorted[(i+1) % 3] - angles_sorted[i] for i in range(3)]
        gaps[-1] += 2 * np.pi  # 处理环绕
        gaps_uniform = all(abs(g - 2*np.pi/3) < 0.4 for g in gaps)
        success = radius_ok and gaps_uniform
        return success, (
            {"layout": "circle", "radius_m": 0.10, "n_points": 3, "center_at": "tray"},
            {"radii_actual": [round(float(r), 3) for r in radii],
             "angle_gaps_rad": [round(float(g), 3) for g in gaps]},
        )
    return (
        "arrange_circle",
        "Place the three blocks evenly on a circle of radius 10cm centered at the yellow tray. "
        "The blocks should be 120 degrees apart.",
        check,
    )


def task_mirror_layout():
    """把当前布局沿桌面中心的垂直线镜像。
    红方块 (在中心) 不动, 绿方块和蓝方块的 x 互换。
    """
    def check(scene):
        red = scene.get_object_position("red block")
        green = scene.get_object_position("green block")
        blue = scene.get_object_position("blue block")
        # 用初始位置作为镜像参考
        green_init = scene.object_initial_positions.get("green block", green)
        blue_init = scene.object_initial_positions.get("blue block", blue)
        cx = float(scene.table_position[0])
        if cx == 0.0:
            cx = 0.5

        # 期望: green x = 2*cx - green_init_x, y 不变
        # 期望: blue  x = 2*cx - blue_init_x,  y 不变
        green_expected_x = 2 * cx - green_init[0]
        blue_expected_x = 2 * cx - blue_init[0]
        green_ok = abs(green[0] - green_expected_x) < 0.04 and abs(green[1] - green_init[1]) < 0.04
        blue_ok = abs(blue[0] - blue_expected_x) < 0.04 and abs(blue[1] - blue_init[1]) < 0.04
        red_ok = abs(red[0] - cx) < 0.06
        success = green_ok and blue_ok and red_ok
        return success, (
            {"green_after_mirror_x": round(green_expected_x, 3),
             "blue_after_mirror_x": round(blue_expected_x, 3),
             "mirror_axis_x": round(cx, 3)},
            {"green_actual": (round(float(green[0]), 3), round(float(green[1]), 3)),
             "blue_actual": (round(float(blue[0]), 3), round(float(blue[1]), 3))},
        )
    return (
        "mirror_layout",
        "Mirror the current block layout across the line passing through the table center "
        "(perpendicular to the x-axis). Each block should end up at the position obtained "
        "by reflecting its current x coordinate around the table center (y stays the same).",
        check,
    )


def task_sort_left_to_right():
    """把方块按颜色顺序 (red, green, blue) 排在 y=0 上, 间距 10cm 围绕桌面中心。"""
    def check(scene):
        red = scene.get_object_position("red block")
        green = scene.get_object_position("green block")
        blue = scene.get_object_position("blue block")
        cx = float(scene.table_position[0])
        if cx == 0.0:
            cx = 0.5
        targets_x = [cx - 0.1, cx, cx + 0.1]
        positions = [red, green, blue]
        all_ok = True
        for pos, tx in zip(positions, targets_x):
            if abs(pos[0] - tx) > 0.04 or abs(pos[1]) > 0.04:
                all_ok = False
                break
        return all_ok, (
            {"order": "red, green, blue (left to right)",
             "target_x": [round(t, 2) for t in targets_x], "target_y": 0.0},
            {"red": (round(float(red[0]), 3), round(float(red[1]), 3)),
             "green": (round(float(green[0]), 3), round(float(green[1]), 3)),
             "blue": (round(float(blue[0]), 3), round(float(blue[1]), 3))},
        )
    return (
        "sort_left_to_right",
        "Place the blocks in a row along y=0, sorted by color order: "
        "red on the left, green in the middle, blue on the right, 10cm apart, "
        "centered around the table center.",
        check,
    )


# ============================================================
# Task families (v4)
# ============================================================
TASKS_BASIC = [
    task_pick_red_to_tray(),
    task_move_green_right(),
    task_stack_two(),
    task_report_leftmost(),
]

TASKS_SPATIAL_GEOMETRIC = [
    task_arrange_line(),
    task_arrange_triangle(),
    task_arrange_circle(),
    task_mirror_layout(),
    task_sort_left_to_right(),
]

TASKS_REFUSAL = [
    task_refuse_rotate_object(),
    task_refuse_missing_object(),
]

TASKS_MOBILITY = [
    task_wide_blue_to_tray(),
    task_collect_red_and_blue_to_tray(),
]

TASKS_BIMANUAL = [
    task_hold_red_while_place_green(),
    task_lift_red_and_green_together(),
    task_lift_red_green_together_to_tray(),
]

TASK_FAMILY_BY_NAME = {
    **{task[0]: "basic" for task in TASKS_BASIC},
    **{task[0]: "geometric" for task in TASKS_SPATIAL_GEOMETRIC},
    **{task[0]: "refusal" for task in TASKS_REFUSAL},
    **{task[0]: "mobility" for task in TASKS_MOBILITY},
    **{task[0]: "bimanual" for task in TASKS_BIMANUAL},
}

RETVAL_TASK_EXPECTATIONS = {
    "refuse_rotate_object": "refuse_cannot_rotate",
    "refuse_missing_object": "refuse_missing_object",
}

# 默认全部任务
TASKS_MIGRATION = TASKS_MOBILITY + TASKS_BIMANUAL
BENCHMARK_TASKS = (
    TASKS_BASIC
    + TASKS_SPATIAL_GEOMETRIC
    + TASKS_REFUSAL
    + TASKS_MOBILITY
    + TASKS_BIMANUAL
)


def task_family(task_name: str) -> str:
    return TASK_FAMILY_BY_NAME.get(task_name, "unknown")


# ============================================================
# 场景构造 (v5: 支持固定布局和 seeded randomized layouts)
# ============================================================
def setup_scene(
    robot_name: str = "kuka",
    scene_variant: str = "fixed",
    scene_seed: Optional[int] = None,
    task_name: Optional[str] = None,
):
    """构造场景。

    - mobile / mobile_dual_arm 机器人: 桌子放远 (x=1.5), 必须 navigate 才能到达
    - dual_arm / dual_franka / 其他固定机器人: 桌子在原点附近
    - scene_variant='fixed': 使用旧的固定布局, 兼容前几阶段实验
    - scene_variant='seeded': 根据 scene_seed 生成可复现的随机初始布局
    """
    if robot_name in {"mobile", "mobile_dual_arm"}:
        table_position = (1.5, 0.0, 0.0)
    else:
        table_position = (0.0, 0.0, 0.0)

    scene = TabletopScene(table_position=table_position)
    positions = _layout_positions(robot_name, scene_variant, scene_seed, task_name=task_name)
    scene.scene_variant = scene_variant
    scene.scene_seed = scene_seed
    scene.add_cube("red block", positions["red block"], color=(1, 0, 0, 1))
    scene.add_cube("green block", positions["green block"], color=(0, 1, 0, 1))
    scene.add_cube("blue block", positions["blue block"], color=(0, 0, 1, 1))
    scene.add_tray("yellow tray", positions["yellow tray"], color=(1, 1, 0, 1))
    return scene


def _layout_positions(
    robot_name: str,
    scene_variant: str = "fixed",
    scene_seed: Optional[int] = None,
    task_name: Optional[str] = None,
) -> Dict[str, List[float]]:
    """Return object positions for a deterministic scene variant."""
    if robot_name in {"mobile", "mobile_dual_arm"}:
        cx = 1.5
        fixed = {
            "red block": [1.5, 0.0, 0.65],
            "green block": [1.55, 0.15, 0.65],
            "blue block": [1.45, -0.15, 0.65],
            "yellow tray": [1.3, 0.3, 0.63],
        }
        if task_name in {"wide_blue_to_tray", "collect_red_and_blue_to_tray"}:
            fixed = {
                "red block": [1.45, -0.05, 0.65],
                "green block": [1.55, 0.18, 0.65],
                "blue block": [1.50, 0.05, 0.65],
                "yellow tray": [1.30, 0.30, 0.63],
            }
        elif task_name in {
            "hold_red_while_place_green",
            "lift_red_and_green_together",
            "lift_red_green_together_to_tray",
        }:
            fixed = {
                "red block": [1.45, -0.10, 0.65],
                "green block": [1.55, 0.15, 0.65],
                "blue block": [1.50, -0.25, 0.65],
                "yellow tray": [1.30, 0.12, 0.63],
            }
    elif robot_name in {"dual_arm", "dual_franka"}:
        cx = 0.5
        fixed = {
            "red block": [0.5, -0.10, 0.65],
            "green block": [0.55, 0.15, 0.65],
            "blue block": [0.45, -0.25, 0.65],
            "yellow tray": [0.35, 0.12, 0.63],
        }
        if task_name in {"wide_blue_to_tray", "collect_red_and_blue_to_tray"}:
            fixed = {
                "red block": [0.45, -0.05, 0.65],
                "green block": [0.55, 0.18, 0.65],
                "blue block": [0.55, 0.22, 0.65],
                "yellow tray": [0.35, 0.12, 0.63],
            }
        elif task_name in {
            "hold_red_while_place_green",
            "lift_red_and_green_together",
            "lift_red_green_together_to_tray",
        }:
            fixed = {
                "red block": [0.5, -0.10, 0.65],
                "green block": [0.55, 0.15, 0.65],
                "blue block": [0.45, -0.25, 0.65],
                "yellow tray": [0.35, 0.12, 0.63],
            }
    else:
        cx = 0.5
        fixed = {
            "red block": [0.5, 0.0, 0.65],
            "green block": [0.55, 0.15, 0.65],
            "blue block": [0.45, -0.15, 0.65],
            "yellow tray": [0.3, 0.3, 0.63],
        }

    if scene_variant == "fixed":
        return fixed
    if scene_variant != "seeded":
        raise ValueError("scene_variant must be 'fixed' or 'seeded'")

    rng = np.random.default_rng(0 if scene_seed is None else scene_seed)
    block_x_range = (cx - 0.14, cx + 0.14)
    block_y_range = (-0.20, 0.20)
    tray_x_range = (cx - 0.22, cx + 0.22)
    tray_y_range = (-0.25, 0.25)
    min_block_dist = 0.11
    min_tray_dist = 0.14

    if task_name in {
        "wide_blue_to_tray",
        "collect_red_and_blue_to_tray",
        "hold_red_while_place_green",
        "lift_red_and_green_together",
        "lift_red_green_together_to_tray",
    }:
        # Stage-5 seeded migration layouts should vary object positions while
        # staying inside the empirically reliable manipulation workspace. This
        # prevents random seeds from turning a capability-comparison task into
        # a trivial IK-failure task.
        if robot_name in {"mobile", "mobile_dual_arm"}:
            block_x_range = (cx - 0.08, cx + 0.08)
            block_y_range = (-0.08, 0.22)
            tray_x_range = (cx - 0.22, cx - 0.06)
            tray_y_range = (0.08, 0.30)
            min_block_dist = 0.10
            min_tray_dist = 0.12
        elif robot_name in {"dual_arm", "dual_franka"}:
            block_x_range = (cx - 0.08, cx + 0.08)
            block_y_range = (-0.16, 0.22)
            tray_x_range = (cx - 0.20, cx - 0.08)
            tray_y_range = (0.04, 0.18)
            min_block_dist = 0.10
            min_tray_dist = 0.12

    if robot_name in {"mobile", "mobile_dual_arm", "dual_arm", "dual_franka"} and task_name in {
        "wide_blue_to_tray",
        "collect_red_and_blue_to_tray",
        "hold_red_while_place_green",
        "lift_red_and_green_together",
        "lift_red_green_together_to_tray",
    }:
        return _seeded_migration_layout(robot_name, task_name, cx, rng)

    block_names = ["red block", "green block", "blue block"]
    block_positions = _sample_nonoverlapping_xy(
        rng,
        count=len(block_names),
        x_range=block_x_range,
        y_range=block_y_range,
        min_dist=min_block_dist,
    )
    tray_xy = _sample_tray_xy(
        rng,
        block_positions,
        x_range=tray_x_range,
        y_range=tray_y_range,
        min_dist=min_tray_dist,
    )
    out = {
        name: [float(x), float(y), 0.65]
        for name, (x, y) in zip(block_names, block_positions)
    }
    out["yellow tray"] = [float(tray_xy[0]), float(tray_xy[1]), 0.63]
    return out


def _seeded_migration_layout(robot_name: str, task_name: str, cx: float, rng) -> Dict[str, List[float]]:
    """Task-constrained seeded layouts for the Mobile/Dual-arm main experiment.

    These layouts still vary by seed, but keep objects in reliable manipulation
    regions so Stage-5 measures capability adaptation rather than random IK
    edge cases.
    """
    if task_name in {"wide_blue_to_tray", "collect_red_and_blue_to_tray"}:
        if robot_name == "mobile":
            anchors = {
                "red block": (cx - 0.02, -0.03),
                "green block": (cx + 0.05, 0.18),
                "blue block": (cx + 0.02, 0.06),
                "yellow tray": (cx - 0.17, 0.26),
            }
        else:
            anchors = {
                "red block": (cx - 0.05, -0.05),
                "green block": (cx + 0.06, 0.04),
                "blue block": (cx + 0.05, 0.20),
                "yellow tray": (cx - 0.15, 0.12),
            }
    elif task_name in {
        "hold_red_while_place_green",
        "lift_red_and_green_together",
        "lift_red_green_together_to_tray",
    }:
        if robot_name in {"mobile", "mobile_dual_arm"}:
            anchors = {
                "red block": (cx - 0.03, -0.08),
                "green block": (cx + 0.04, 0.14),
                "blue block": (cx + 0.02, -0.22),
                "yellow tray": (cx - 0.17, 0.18),
            }
        else:
            anchors = {
                "red block": (cx - 0.03, -0.10),
                "green block": (cx + 0.04, 0.14),
                "blue block": (cx - 0.06, -0.23),
                "yellow tray": (cx - 0.15, 0.12),
            }
    else:
        raise ValueError(f"Unsupported migration task: {task_name}")

    noise = 0.025
    out: Dict[str, List[float]] = {}
    for name, (x, y) in anchors.items():
        jx = float(rng.uniform(-noise, noise))
        jy = float(rng.uniform(-noise, noise))
        z = 0.63 if "tray" in name else 0.65
        out[name] = [float(x + jx), float(y + jy), z]
    return out


def _sample_nonoverlapping_xy(
    rng,
    count: int,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    min_dist: float,
) -> List[np.ndarray]:
    points: List[np.ndarray] = []
    for _ in range(count):
        for _attempt in range(200):
            candidate = np.array([
                rng.uniform(*x_range),
                rng.uniform(*y_range),
            ])
            if all(np.linalg.norm(candidate - p) >= min_dist for p in points):
                points.append(candidate)
                break
        else:
            raise RuntimeError("Failed to sample non-overlapping scene layout")
    return points


def _sample_tray_xy(
    rng,
    block_positions: List[np.ndarray],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    min_dist: float,
) -> np.ndarray:
    for _attempt in range(200):
        candidate = np.array([
            rng.uniform(*x_range),
            rng.uniform(*y_range),
        ])
        if all(np.linalg.norm(candidate - p) >= min_dist for p in block_positions):
            return candidate
    # Deterministic fallback in the upper-left part of the table.
    return np.array([x_range[0], y_range[1]])


def expected_ret_val_for_task(task_name: str, scene) -> Optional[str]:
    if task_name == "report_leftmost":
        block_names = [n for n in scene.get_object_names() if "block" in n]
        return min(block_names, key=lambda n: float(scene.get_object_position(n)[0]))
    return RETVAL_TASK_EXPECTATIONS.get(task_name)


# ============================================================
# Trial runner: 支持严格 ablation 模式
# ============================================================
def run_trial(
    robot_name: str,
    task: Tuple[str, str, Callable],
    client,
    mode: str = "ba",
    headless: bool = True,
    verbose: bool = False,
    logger: Optional[BenchmarkLogger] = None,
    trial_id: Optional[str] = None,
    trial_index: int = 1,
    scene_variant: str = "fixed",
    scene_seed: Optional[int] = None,
    llm_model: str = DEFAULT_MODEL,
    llm_temperature: float = 0.0,
    llm_cache: Optional[LLMResponseCache] = None,
) -> Tuple[bool, str]:
    """在指定机器人上跑一个任务一次, 使用指定 ablation 模式。"""
    task_name, instruction, checker = task
    family = task_family(task_name)
    mode_config = _mode_config(mode)
    canonical_mode = str(mode_config["canonical"])
    use_card = bool(mode_config["use_capability_card"])
    include_few_shot = bool(mode_config["include_few_shot"])
    use_failure_report = bool(mode_config["use_failure_report"])
    max_attempts = int(mode_config["max_attempts"])
    trial_id = trial_id or f"{mode}_{robot_name}_{task_name}_{trial_index}"

    record = {
        "run_id": logger.run_id if logger else None,
        "trial_id": trial_id,
        "trial_index": trial_index,
        "mode": mode,
        "canonical_mode": canonical_mode,
        "mode_config": mode_config,
        "robot": robot_name,
        "task": task_name,
        "task_family": family,
        "scene_variant": scene_variant,
        "scene_seed": scene_seed,
        "instruction": instruction,
        "headless": headless,
        "use_capability_card": use_card,
        "include_few_shot": include_few_shot,
        "use_failure_report": use_failure_report,
        "max_attempts": max_attempts,
        "llm_model": llm_model,
        "llm_temperature": llm_temperature,
        "llm_cache_enabled": bool(llm_cache and llm_cache.enabled),
        "attempts": [],
        "success": False,
        "info": "",
        "final_reason": "",
        "failure_type": "",
        "failure_subtype": "",
        "error_excerpt": "",
        "llm_error": "",
        "exec_error": False,
        "check_failure": False,
        "action_failure": False,
        "ret_val_failed": False,
        "expected": {},
        "actual": {},
    }
    scene = None

    def finish(success: bool, info: str, final_reason: str,
               expected=None, actual=None) -> Tuple[bool, str]:
        record["success"] = bool(success)
        record["info"] = info
        record["final_reason"] = final_reason
        if expected is not None:
            record["expected"] = expected
        if actual is not None:
            record["actual"] = actual
        if scene is not None:
            record["final_scene"] = snapshot_scene(scene)
        record["attempt_count"] = len(record["attempts"])
        if success:
            record["failure_type"] = ""
            record["failure_subtype"] = ""
            record["error_excerpt"] = ""
        else:
            record["failure_type"] = choose_failure_type(record)
            record["failure_subtype"] = choose_failure_subtype(record)
            record["error_excerpt"] = extract_error_excerpt(record)

        if logger is not None:
            logger.write_trial(record)
            logger.add_summary_row(summary_row_from_record(record))
        return success, info

    p.connect(p.DIRECT if headless else p.GUI)
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

        record["robot_description"] = robot.describe()
        record["initial_scene"] = snapshot_scene(scene)
        ret_val_expectation = expected_ret_val_for_task(task_name, scene)
        if ret_val_expectation is not None:
            record["ret_val_expected"] = ret_val_expectation
        if getattr(robot, "capability_card", None) is not None:
            record["capability_card_prompt"] = robot.capability_card.to_prompt_section()

        last_report = None  # 方法 B 的失败报告

        for attempt in range(1, max_attempts + 1):
            # --- 构造 prompt ---
            user_prompt = build_user_prompt(
                robot=robot,
                scene_description=scene.describe(),
                instruction=instruction,
                use_capability_card=use_card,
                include_few_shot=include_few_shot,
                failure_report=last_report if use_failure_report else None,
            )

            # --- 调 LLM ---
            try:
                raw, llm_info = cached_chat(
                    client=client,
                    system=SYSTEM_PROMPT,
                    user=user_prompt,
                    model=llm_model,
                    temperature=llm_temperature,
                    cache=llm_cache,
                    chat_fn=chat,
                )
            except Exception as e:
                if verbose:
                    print(f"    [trial] LLM call failed: {e}")
                record["llm_error"] = str(e)
                return finish(False, f"llm-error: {e}", "llm_error")

            code = extract_code_or_text(raw)
            artifact_paths = (
                logger.save_attempt_artifacts(trial_id, attempt, user_prompt, raw, code)
                if logger is not None else {}
            )
            if verbose:
                print(f"    [trial attempt {attempt}] generated code:")
                for line in code.splitlines():
                    print(f"      {line}")

            # 给 mobile 机器人额外注入 mobile 别名
            exec_ctx = {"robot": robot, "scene": scene}
            if getattr(robot, "capability_card", None) and robot.capability_card.has_mobile_base:
                exec_ctx["mobile"] = robot
            if getattr(robot, "capability_card", None) and robot.capability_card.has_dual_arms:
                exec_ctx["left"] = robot.left
                exec_ctx["right"] = robot.right

            # --- 执行代码 ---
            if hasattr(robot, "reset_action_log"):
                robot.reset_action_log()
            ok, msg, locals_dict = execute_lmp(
                code, exec_ctx, verbose=False,
            )
            action_failures = (
                robot.get_action_failures()
                if hasattr(robot, "get_action_failures") else []
            )
            ret_val = locals_dict.get("ret_val")
            ret_val_failed = ret_val is False or ret_val == "failure"

            # --- 等物理稳定 ---
            for _ in range(120):
                p.stepSimulation()

            attempt_record = {
                "attempt": attempt,
                "prompt": user_prompt,
                "raw_response": raw,
                "code": code,
                "artifact_paths": artifact_paths,
                "llm_model": llm_model,
                "llm_temperature": llm_temperature,
                "llm_cache_enabled": llm_info.get("cache_enabled", False),
                "llm_cache_hit": llm_info.get("cache_hit", False),
                "llm_cache_key": llm_info.get("cache_key", ""),
                "llm_cache_path": llm_info.get("cache_path", ""),
                "code_features": analyze_code(code),
                "exec_ok": ok,
                "exec_message": msg,
                "ret_val": jsonable(ret_val),
                "locals": jsonable(locals_dict),
                "action_failures": action_failures,
                "scene_after_execution": snapshot_scene(scene),
            }
            record["attempts"].append(attempt_record)
            if not ok:
                record["exec_error"] = True
            if action_failures:
                record["action_failure"] = True
            if ret_val_failed:
                record["ret_val_failed"] = True

            # --- ret_val 任务: 纯推理 / refusal accuracy ---
            expected_ret_val = ret_val_expectation
            if expected_ret_val is not None:
                actual_ret_val = locals_dict.get("ret_val")
                attempt_record["ret_val_expected"] = expected_ret_val
                attempt_record["ret_val_success"] = bool(ok and actual_ret_val == expected_ret_val)
                if ok and actual_ret_val == expected_ret_val:
                    return finish(
                        True,
                        f"ret_val={actual_ret_val}",
                        f"success_on_attempt_{attempt}",
                        expected={"ret_val": expected_ret_val},
                        actual={"ret_val": actual_ret_val},
                    )
                record["ret_val_failed"] = True
                if not use_failure_report or attempt == max_attempts:
                    reason = "exec-fail" if not ok else "ret-val-fail"
                    return finish(
                        False,
                        f"ret_val={actual_ret_val}",
                        reason,
                        expected={"ret_val": expected_ret_val},
                        actual={"ret_val": actual_ret_val},
                    )
                # 重试: 构造简单报告
                last_report = build_failure_report(
                    task_name=task_name, instruction=instruction,
                    robot_name=robot_name,
                    expected={"ret_val": expected_ret_val},
                    actual={"ret_val": actual_ret_val},
                    code_raised=not ok, traceback_str=msg if not ok else None,
                )
                continue

            # --- 常规任务: 跑 checker ---
            if ok:
                success, (expected, actual) = checker(scene)
                attempt_record["checker_success"] = success
                attempt_record["expected"] = expected
                attempt_record["actual"] = actual
                action_ok = not action_failures and not ret_val_failed
                if success and action_ok:
                    return finish(
                        True,
                        f"pass on attempt {attempt}",
                        f"success_on_attempt_{attempt}",
                        expected=expected,
                        actual=actual,
                    )
            else:
                success = False
                expected, actual = {}, {}
                attempt_record["checker_success"] = False
                attempt_record["expected"] = expected
                attempt_record["actual"] = actual

            if not success or action_failures or ret_val_failed:
                record["check_failure"] = record["check_failure"] or bool(ok and not success)

            # 失败分支
            if not use_failure_report or attempt == max_attempts:
                if not ok:
                    reason = "exec-fail"
                elif action_failures:
                    reason = "action-fail"
                elif ret_val_failed:
                    reason = "ret-val-fail"
                else:
                    reason = "check-fail"
                return finish(
                    False,
                    f"{reason} after {attempt} attempt(s)",
                    reason,
                    expected=expected,
                    actual=actual,
                )

            # 构造 FailureReport 供下一轮 LLM 使用
            if ok and (action_failures or ret_val_failed):
                actual = dict(actual)
                actual["action_failures"] = action_failures
                actual["ret_val"] = ret_val
                expected = dict(expected)
                expected["robot_action_success"] = True
            last_report = build_failure_report(
                task_name=task_name,
                instruction=instruction,
                robot_name=robot_name,
                expected=expected,
                actual=actual,
                code_raised=not ok,
                traceback_str=msg if not ok else None,
            )
            if verbose:
                print(f"    [trial] attempt {attempt} failed, will retry with report")

        return finish(False, "max-attempts-exceeded", "max_attempts_exceeded")
    finally:
        p.disconnect()


# ============================================================
# Benchmark 主流程
# ============================================================
def run_benchmark(
    robots=("dual_arm", "mobile_dual_arm", "dual_franka"),
    modes=tuple(STRICT_MODES),
    n_trials=1,
    headless=True,
    verbose=False,
    log_dir="results/runs",
    run_id=None,
    enable_logging=True,
    scene_variant="fixed",
    seed_base=0,
    llm_model=DEFAULT_MODEL,
    llm_temperature=0.0,
    use_cache=True,
    cache_dir="results/llm_cache",
    resume=False,
    offline_cache_only=False,
):
    client = None
    llm_cache = LLMResponseCache(root=cache_dir, enabled=use_cache)
    logger = BenchmarkLogger(root=log_dir, run_id=run_id, enabled=enable_logging)

    # results[mode][robot][task] = success_rate (0-1)
    results: Dict[str, Dict[str, Dict[str, float]]] = {
        m: {r: {} for r in robots} for m in modes
    }

    task_names = [t[0] for t in BENCHMARK_TASKS]
    logger.write_metadata({
        "robots": list(robots),
        "modes": list(modes),
        "mode_configs": {m: _mode_config(m) for m in modes},
        "n_trials": n_trials,
        "headless": headless,
        "scene_variant": scene_variant,
        "seed_base": seed_base,
        "scene_seeds": [
            seed_base + i if scene_variant == "seeded" else None
            for i in range(n_trials)
        ],
        "llm_model": llm_model,
        "llm_temperature": llm_temperature,
        "llm_cache_enabled": use_cache,
        "llm_cache_dir": cache_dir,
        "resume": resume,
        "offline_cache_only": offline_cache_only,
        "task_names": task_names,
        "task_families": {name: task_family(name) for name in task_names},
    })
    if enable_logging:
        print(f"\n📝 Logging run artifacts to: {logger.path_for_display()}")

    trial_seq = 0
    for mode in modes:
        print(f"\n{'═' * 70}")
        print(f"  MODE: {mode.upper()}  "
              f"({_mode_description(mode)})")
        print(f"{'═' * 70}")

        for robot_name in robots:
            print(f"\n🤖 Robot: {robot_name}")
            for task in BENCHMARK_TASKS:
                name = task[0]
                successes = 0
                for trial in range(n_trials):
                    scene_seed = seed_base + trial if scene_variant == "seeded" else None
                    trial_seq += 1
                    trial_id = (
                        f"{trial_seq:05d}_{mode}_{robot_name}_{name}_"
                        f"trial{trial + 1:03d}"
                    )
                    if enable_logging and resume:
                        existing = logger.read_trial(trial_id)
                        if _is_complete_trial_record(existing):
                            logger.add_summary_row(summary_row_from_record(existing))
                            if existing.get("success"):
                                successes += 1
                            print(f"   [{name:<20}] trial {trial+1}: ⏭️  resumed")
                            continue

                    if client is None and not offline_cache_only:
                        client = make_client()

                    ok, info = run_trial(
                        robot_name, task, client,
                        mode=mode, headless=headless, verbose=verbose,
                        logger=logger if enable_logging else None,
                        trial_id=trial_id,
                        trial_index=trial + 1,
                        scene_variant=scene_variant,
                        scene_seed=scene_seed,
                        llm_model=llm_model,
                        llm_temperature=llm_temperature,
                        llm_cache=llm_cache,
                    )
                    icon = "✅" if ok else "❌"
                    print(f"   [{name:<20}] trial {trial+1}: {icon}  {info}")
                    if ok:
                        successes += 1
                results[mode][robot_name][name] = successes / n_trials

    _print_summary(results, modes, robots, task_names)
    logger.write_summary()
    if enable_logging:
        print(f"📝 Wrote summary.csv and trial artifacts to: {logger.path_for_display()}")
        print(f"📈 Analyze tables with: python -m benchmark.analyze_results {logger.path_for_display()}")
    return results


def _is_complete_trial_record(record) -> bool:
    return bool(
        record
        and record.get("trial_id")
        and record.get("final_reason")
        and isinstance(record.get("attempts"), list)
    )


def _mode_description(mode):
    return str(_mode_config(mode)["description"])


def _print_summary(results, modes, robots, task_names):
    """打印三档对比表 + Migration Score 随模式的演变。"""
    print(f"\n\n{'═' * 70}")
    print("  📊 Cross-Embodiment Migration Ablation Results")
    print(f"{'═' * 70}\n")

    # 表头
    header = f"{'Task':<22}"
    for mode in modes:
        for r in robots:
            header += f"{mode}/{r[:3]:<7}"
    print(header)
    print("─" * len(header))

    # 每个任务一行, 每格一个成功率
    for t in task_names:
        row = f"{t:<22}"
        for mode in modes:
            for r in robots:
                rate = results[mode][r][t]
                row += f"{rate:<10.1%}"
        print(row)

    # Migration Score: 两个机器人都通过的任务数
    print("\n" + "─" * len(header))
    print(f"{'Migration Score':<22}", end="")
    for mode in modes:
        for r in robots:
            pass  # 不放在这
        shared = sum(
            1 for t in task_names
            if all(results[mode][r][t] > 0.5 for r in robots)
        )
        # 左对齐占一个 mode 的宽度 (len(robots) 个单元格)
        cell = f"{shared}/{len(task_names)}"
        width = 10 * len(robots)
        print(f"{cell:<{width}}", end="")
    print()

    # 总体成功率 (所有 robot × task 的平均)
    print(f"{'Overall success':<22}", end="")
    for mode in modes:
        for r in robots:
            pass
        total = sum(results[mode][r][t] for r in robots for t in task_names)
        avg = total / (len(robots) * len(task_names))
        cell = f"{avg:.1%}"
        width = 10 * len(robots)
        print(f"{cell:<{width}}", end="")
    print("\n")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--robots", nargs="+", default=["dual_arm", "mobile_dual_arm", "dual_franka"],
                    choices=["kuka", "franka", "mobile", "dual_arm", "mobile_dual_arm", "dual_franka"],
                    help="Robots to evaluate. Default is the current paper direction: "
                         "dual_arm vs mobile_dual_arm vs dual_franka.")
    ap.add_argument("--modes", nargs="+", default=STRICT_MODES,
                    choices=sorted(MODE_CONFIGS),
                    help="Ablation modes. Strict modes: api fewshot card failure card_failure. "
                         "Legacy aliases: baseline b ba.")
    ap.add_argument("--tasks", default="migration",
                    choices=["all", "basic", "geometric", "refusal", "mobility", "bimanual", "migration"],
                    help="'basic' = 4 simple tasks (v3 set). "
                         "'geometric' = 5 spatial-geometric tasks (v4 new). "
                         "'refusal' = 2 capability/scene refusal tasks. "
                         "'mobility' = mobile-base navigation/workspace tasks. "
                         "'bimanual' = dual-arm simultaneous/hold-while-place tasks. "
                         "'migration' = mobility + bimanual (default). "
                         "'all' = all task families.")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--scene-variant", default="fixed",
                    choices=["fixed", "seeded"],
                    help="Scene layout policy. 'fixed' reproduces the original layout; "
                         "'seeded' samples deterministic randomized layouts.")
    ap.add_argument("--seed-base", type=int, default=0,
                    help="Base seed for --scene-variant seeded. Trial i uses seed_base + i.")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="LLM model id. Defaults to EM_MODEL or llm_client.DEFAULT_MODEL.")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="LLM sampling temperature.")
    ap.add_argument("--cache-dir", default="results/llm_cache",
                    help="Directory for prompt-response cache files.")
    ap.add_argument("--no-cache", action="store_true",
                    help="Disable LLM response cache.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip completed trial JSON files when reusing a run id.")
    ap.add_argument("--offline-cache-only", action="store_true",
                    help="Do not create a live LLM client; fail on cache miss.")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--log-dir", default="results/runs",
                    help="Directory where timestamped benchmark run artifacts are written.")
    ap.add_argument("--run-id", default=None,
                    help="Optional run id. Defaults to a timestamp.")
    ap.add_argument("--no-log", action="store_true",
                    help="Disable JSON/CSV/artifact logging.")
    args = ap.parse_args()

    # task family 过滤
    if args.tasks == "basic":
        BENCHMARK_TASKS = TASKS_BASIC
    elif args.tasks == "geometric":
        BENCHMARK_TASKS = TASKS_SPATIAL_GEOMETRIC
    elif args.tasks == "refusal":
        BENCHMARK_TASKS = TASKS_REFUSAL
    elif args.tasks == "mobility":
        BENCHMARK_TASKS = TASKS_MOBILITY
    elif args.tasks == "bimanual":
        BENCHMARK_TASKS = TASKS_BIMANUAL
    elif args.tasks == "migration":
        BENCHMARK_TASKS = TASKS_MIGRATION
    # else: 用模块级默认 (all)

    run_benchmark(
        robots=tuple(args.robots),
        modes=tuple(args.modes),
        n_trials=args.trials,
        headless=not args.gui,
        verbose=args.verbose,
        log_dir=args.log_dir,
        run_id=args.run_id,
        enable_logging=not args.no_log,
        scene_variant=args.scene_variant,
        seed_base=args.seed_base,
        llm_model=args.model,
        llm_temperature=args.temperature,
        use_cache=not args.no_cache,
        cache_dir=args.cache_dir,
        resume=args.resume,
        offline_cache_only=args.offline_cache_only,
    )
