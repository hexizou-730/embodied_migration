"""
Smoke test for B+A infrastructure (no LLM calls).

验证:
1. 每个机器人有正确的 capability_card
2. CapabilityCard.to_prompt_section() 能生成可读文本
3. FailureReport 的自动诊断能工作
4. build_user_prompt 能同时注入 card 和 report
5. Mobile / Dual-arm prompt hints 与 strict ablation mode 配置正确

用法:
    python -m examples.test_capability_card
"""
from robots import make_robot, ROBOT_REGISTRY
from capabilities import CapabilityCard
from lmp.failure_report import build_failure_report
from prompts import build_user_prompt

import pybullet as p


def test_all_robots_have_cards():
    print("─" * 70)
    print("TEST 1: Every robot class declares a capability_card")
    print("─" * 70)
    for name, cls in ROBOT_REGISTRY.items():
        card = cls.capability_card
        assert isinstance(card, CapabilityCard), f"{name}: wrong type"
        assert card.grasp_mechanism != "unknown", f"{name}: grasp_mechanism not set"
        print(f"  ✅ {name:<10} grasp={card.grasp_mechanism:<14} "
              f"stable_stacked={card.stable_when_stacked} "
              f"rec_release={card.recommended_release_height_m}m")


def test_card_to_prompt():
    print("\n" + "─" * 70)
    print("TEST 2: CapabilityCard.to_prompt_section() produces readable text")
    print("─" * 70)
    # 不用实例化机器人 (不用启动 pybullet), 直接从类上取 card
    from robots.kuka_robot import KukaRobot
    section = KukaRobot.capability_card.to_prompt_section()
    print(section)
    assert "suction" in section
    assert "Implications" in section
    print("\n  ✅ Card section rendered OK")


def test_failure_report():
    print("\n" + "─" * 70)
    print("TEST 3: FailureReport auto-diagnoses position deviation")
    print("─" * 70)
    report = build_failure_report(
        task_name="move_green_right",
        instruction="Move the green block 10 cm to the right.",
        robot_name="KUKA iiwa",
        expected={"green_block_position": (0.65, 0.15, 0.65)},
        actual={"green_block_position": (0.58, 0.13, 0.63)},
    )
    section = report.to_prompt_section()
    print(section)
    assert len(report.diagnosis) > 0, "should have auto-diagnosed deviation"
    assert len(report.suggestions) > 0, "should have auto-generated suggestions"
    print("\n  ✅ Report rendered OK, diagnosis count:", len(report.diagnosis))


def test_prompt_integration():
    print("\n" + "─" * 70)
    print("TEST 4: build_user_prompt wires card + report together")
    print("─" * 70)

    # 需要真实的机器人对象 (有 scene), 起一个最小的 pybullet DIRECT
    p.connect(p.DIRECT)
    try:
        from perception import TabletopScene
        scene = TabletopScene()
        scene.add_cube("red block", [0.5, 0.0, 0.65], color=(1, 0, 0, 1))

        robot = make_robot("kuka")
        robot.scene = scene
        for _ in range(60):
            p.stepSimulation()

        report = build_failure_report(
            task_name="test",
            instruction="demo",
            robot_name="KUKA iiwa",
            expected={"x": 1.0}, actual={"x": 0.5},
        )

        prompt_ba = build_user_prompt(
            robot=robot, scene_description=scene.describe(),
            instruction="Put the red block in the tray",
            use_capability_card=True, failure_report=report,
        )
        assert "Capability Card" in prompt_ba
        assert "Failure Report" in prompt_ba
        print(f"  ✅ +B+A prompt length: {len(prompt_ba)} chars "
              f"(contains both Card and Report)")

        prompt_baseline = build_user_prompt(
            robot=robot, scene_description=scene.describe(),
            instruction="Put the red block in the tray",
            use_capability_card=False, failure_report=None,
        )
        assert "Capability Card" not in prompt_baseline
        assert "Failure Report" not in prompt_baseline
        print(f"  ✅ baseline prompt length: {len(prompt_baseline)} chars "
              f"(neither Card nor Report)")

        delta = len(prompt_ba) - len(prompt_baseline)
        print(f"  📏 Extra context added by B+A: +{delta} chars")
    finally:
        p.disconnect()


def test_mobile_dual_prompt_hints_and_modes():
    print("\n" + "─" * 70)
    print("TEST 5: Mobile/Dual-arm prompt hints + strict ablation modes")
    print("─" * 70)

    from perception import TabletopScene
    from benchmark.run_benchmark import STRICT_MODES, _mode_config

    expected_modes = ["api", "fewshot", "card", "failure", "card_failure"]
    assert STRICT_MODES == expected_modes
    assert _mode_config("api")["include_few_shot"] is False
    assert _mode_config("api")["use_capability_card"] is False
    assert _mode_config("card")["use_capability_card"] is True
    assert _mode_config("failure")["use_failure_report"] is True
    assert _mode_config("failure")["use_capability_card"] is False
    assert _mode_config("card_failure")["use_failure_report"] is True
    assert _mode_config("card_failure")["use_capability_card"] is True

    p.connect(p.DIRECT)
    try:
        scene = TabletopScene()
        scene.add_cube("red block", [0.5, -0.1, 0.65], color=(1, 0, 0, 1))
        scene.add_cube("green block", [0.55, 0.15, 0.65], color=(0, 1, 0, 1))
        scene.add_tray("yellow tray", [0.35, 0.12, 0.63], color=(1, 1, 0, 1))

        mobile = make_robot("mobile")
        mobile.scene = scene
        mobile_prompt = build_user_prompt(
            robot=mobile,
            scene_description=scene.describe(),
            instruction="Put the red block in the yellow tray.",
            use_capability_card=True,
        )
        assert "Mobile-base API" in mobile_prompt
        assert "mobile.navigate_to" in mobile_prompt

        dual = make_robot("dual_arm")
        dual.scene = scene
        dual_prompt = build_user_prompt(
            robot=dual,
            scene_description=scene.describe(),
            instruction="Lift the red and green blocks at the same time.",
            use_capability_card=True,
        )
        assert "Dual-arm API" in dual_prompt
        assert "pick_with_arm" in dual_prompt
        assert "lift_two_objects" in dual_prompt
        assert "has_dual_arms: True" in dual_prompt
        assert "can_coordinate_arms: True" in dual_prompt

        mobile_dual = make_robot("mobile_dual_arm")
        mobile_dual.scene = scene
        mobile_dual_prompt = build_user_prompt(
            robot=mobile_dual,
            scene_description=scene.describe(),
            instruction=(
                "Pick up the red and green blocks at the same time, "
                "then place both into the tray at the same time."
            ),
            use_capability_card=True,
        )
        assert "Mobile-base API" in mobile_dual_prompt
        assert "mobile.navigate_to" in mobile_dual_prompt
        assert "Dual-arm API" in mobile_dual_prompt
        assert "lift_two_objects" in mobile_dual_prompt
        assert "place_two_objects" in mobile_dual_prompt

        dual_franka = make_robot("dual_franka")
        dual_franka.scene = scene
        dual_franka_prompt = build_user_prompt(
            robot=dual_franka,
            scene_description=scene.describe(),
            instruction=(
                "Pick up the red and green blocks at the same time, "
                "then place both into the tray at the same time."
            ),
            use_capability_card=True,
        )
        assert "Dual-arm API" in dual_franka_prompt
        assert "parallel_jaw" in dual_franka_prompt
        assert "lift_two_objects" in dual_franka_prompt
        assert "place_two_objects" in dual_franka_prompt

        print("  ✅ strict modes and Mobile/Dual-arm prompt hints are wired")
    finally:
        p.disconnect()


if __name__ == "__main__":
    test_all_robots_have_cards()
    test_card_to_prompt()
    test_failure_report()
    test_prompt_integration()
    test_mobile_dual_prompt_hints_and_modes()
    print("\n" + "═" * 70)
    print("  ✅ All smoke tests passed. B+A infrastructure is wired correctly.")
    print("═" * 70)
