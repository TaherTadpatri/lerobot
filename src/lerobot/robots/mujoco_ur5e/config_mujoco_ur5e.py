"""Configuration class for MuJoCo UR5e Simulation Robot."""

from dataclasses import dataclass

from ..config import RobotConfig


@RobotConfig.register_subclass("mujoco_ur5e")
@dataclass
class MuJoCoUR5eRobotConfig(RobotConfig):
    """Configuration for MuJoCo UR5e Pick & Place environment robot wrapper."""

    # Hugging Face Hub dataset / environment path
    hub_path: str = "castanetnicolas/UR5e_robosuite_pick_place"

    # Environment task name
    task: str = "pick_place_can"

    # Control frequency (FPS)
    fps: int = 20

    # Whether to launch native interactive MuJoCo passive 3D GUI window
    render_gui: bool = True
