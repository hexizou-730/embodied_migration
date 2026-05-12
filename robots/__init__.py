from robots.base_robot import BaseRobot
from robots.kuka_robot import KukaRobot
from robots.franka_robot import FrankaRobot
from robots.mobile_robot import MobileManipulatorRobot
from robots.dual_arm_robot import DualArmRobot
from robots.mobile_dual_arm_robot import MobileDualArmRobot
from robots.dual_franka_robot import DualFrankaRobot

ROBOT_REGISTRY = {
    "kuka": KukaRobot,
    "franka": FrankaRobot,
    "mobile": MobileManipulatorRobot,
    "dual_arm": DualArmRobot,
    "mobile_dual_arm": MobileDualArmRobot,
    "dual_franka": DualFrankaRobot,
}


def make_robot(name: str, **kwargs) -> BaseRobot:
    name = name.lower()
    if name not in ROBOT_REGISTRY:
        raise ValueError(f"Unknown robot '{name}'. Available: {list(ROBOT_REGISTRY)}")
    return ROBOT_REGISTRY[name](**kwargs)
