"""Script to test and verify UR5e robot arm joints and gripper in MuJoCo simulation."""

import math
import time

import mujoco
import mujoco.viewer
import numpy as np

from lerobot.envs import make_env
from lerobot.envs.configs import HubEnvConfig

# 1. Configure Hugging Face Hub environment with Pick & Place task
cfg = HubEnvConfig(
    hub_path="castanetnicolas/UR5e_robosuite_pick_place",
    task="pick_place_can",
)

print("=" * 65)
print("     UR5e Robot Joint Diagnostic & Functional Test")
print("=" * 65)
print("Loading UR5e simulation environment...")

envs = make_env(cfg, trust_remote_code=True)
vec_env = envs[next(iter(envs))][0]
obs, info = vec_env.reset()

env_wrapper = vec_env.envs[0]
robosuite_env = env_wrapper._env
sim = robosuite_env.sim
m = sim.model._model
d = sim.data._data

# UR5e 6-DOF joint labels
JOINT_NAMES = [
    "Shoulder Pan (Joint 1)",
    "Shoulder Lift (Joint 2)",
    "Elbow (Joint 3)",
    "Wrist 1 (Joint 4)",
    "Wrist 2 (Joint 5)",
    "Wrist 3 (Joint 6)",
]

initial_qpos = np.copy(sim.data.qpos[:6])

print("\nUR5e Arm Joints Detected:")
for idx, name in enumerate(JOINT_NAMES):
    init_deg = math.degrees(initial_qpos[idx])
    print(f"  [{idx}] {name:25s} | Init Angle: {initial_qpos[idx]:.4f} rad ({init_deg:.1f}°)")

print("\nStarting MuJoCo Passive Viewer...")
print("Diagnostic sequence:")
print("  - Sequentially oscillates each joint (Joint 0 -> Joint 5) with a smooth sine wave.")
print("  - Cycles the gripper open/close at the end of each pass.")
print("  - Close the interactive viewer GUI window to finish the test.\n")

steps_per_joint = 120  # ~1.2 seconds per joint test phase
total_joints = 6


def run_joint_diagnostic():
    """Run interactive visual joint diagnostic loop in MuJoCo viewer."""
    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Disable Group 0 (collision geometry) & Enable Group 1 (visual CAD meshes)
        viewer._opt.geomgroup[0] = 0
        viewer._opt.geomgroup[1] = 1

        current_phase_idx = -1
        step_count = 0

        while viewer.is_running():
            joint_test_step = step_count % (steps_per_joint * (total_joints + 1))
            joint_idx = joint_test_step // steps_per_joint

            if joint_idx != current_phase_idx:
                current_phase_idx = joint_idx
                if current_phase_idx < total_joints:
                    j_name = JOINT_NAMES[current_phase_idx]
                    print(f"--> [Testing Joint {current_phase_idx}]: {j_name}")
                else:
                    print("--> [Testing Gripper]: Cycling open and close...")

            # Reset robot joints to baseline pose before applying current joint offset
            for j in range(6):
                sim.data.qpos[j] = initial_qpos[j]

            phase = (joint_test_step % steps_per_joint) / steps_per_joint * 2.0 * math.pi
            amplitude = 0.35  # ±0.35 radians deflection (~20 degrees)

            gripper_val = 0.0
            if current_phase_idx < total_joints:
                # Apply sinusoidal joint displacement to active joint
                sim.data.qpos[current_phase_idx] = initial_qpos[current_phase_idx] + amplitude * math.sin(
                    phase
                )
            else:
                # Cycle gripper value between -1.0 (close) and 1.0 (open)
                gripper_val = math.sin(phase)

            # Update kinematics computations
            sim.forward()

            # Step physics simulation step
            env_action = np.zeros((1, 7), dtype=np.float32)
            env_action[0, -1] = gripper_val
            obs, reward, terminated, truncated, info = vec_env.step(env_action)

            viewer.sync()
            time.sleep(0.01)
            step_count += 1

    vec_env.close()
    print("\nJoint diagnostic test complete. Environment closed.")


if __name__ == "__main__":
    run_joint_diagnostic()
