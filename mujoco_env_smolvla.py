"""Synchronous MuJoCo simulation script for SmolVLA policy on UR5e Pick & Place task.

Configured at exact 20 FPS dataset collection rate and task description: "Pick up the can and place it in the correct bin."
"""

import time

import mujoco
import mujoco.viewer
import numpy as np
import torch

from lerobot.envs import make_env
from lerobot.envs.configs import HubEnvConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAPolicy

# 1. FPS and Task Configuration
FPS = 20
CONTROL_DT = 1.0 / FPS  # 0.05s per control frame (20 FPS)
TASK_DESCRIPTION = "Pick up the can and place it in the correct bin."

# Device selection (GPU / CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Configure environment
cfg = HubEnvConfig(
    hub_path="castanetnicolas/UR5e_robosuite_pick_place",
    task="pick_place_can",
)

print(f"Loading UR5e Pick & Place simulation environment ({FPS} FPS)...")
envs = make_env(cfg, trust_remote_code=True)
vec_env = envs[next(iter(envs))][0]
obs, info = vec_env.reset()

env_wrapper = vec_env.envs[0]
robosuite_env = env_wrapper._env
sim = robosuite_env.sim
m = sim.model._model
d = sim.data._data

# 3. Load Policy
model_id = "castanetnicolas/smolvla_ur5e_pick_place"
print(f"Loading SmolVLA policy '{model_id}'...")
policy = SmolVLAPolicy.from_pretrained(model_id)
policy.to(device)
policy.eval()
policy.reset()

preprocessor, postprocessor = make_pre_post_processors(
    policy.config,
    pretrained_path=model_id,
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)


def format_obs_for_policy(obs_dict, task_description):
    """Format Gym environment observation dictionary into LeRobot SmolVLA policy input format."""
    cam1 = torch.from_numpy(obs_dict["pixels"]["camera1"]).squeeze(0).permute(2, 0, 1).float() / 255.0
    cam2 = torch.from_numpy(obs_dict["pixels"]["camera2"]).squeeze(0).permute(2, 0, 1).float() / 255.0
    state = torch.from_numpy(obs_dict["agent_pos"]).squeeze(0).float()
    return {
        "observation.images.camera1": cam1,
        "observation.images.camera2": cam2,
        "observation.images.camera3": torch.zeros((3, 256, 256), dtype=torch.float32),
        "observation.state": state,
        "task": task_description,
    }


print(f"\nStarting SmolVLA MuJoCo Viewer ({FPS} FPS loop)...")

# 4. Launch MuJoCo viewer with synchronous inference loop
with mujoco.viewer.launch_passive(m, d) as viewer:
    viewer._opt.geomgroup[0] = 0
    viewer._opt.geomgroup[1] = 1

    for step in range(10000):
        frame_start = time.time()

        raw_policy_obs = format_obs_for_policy(obs, TASK_DESCRIPTION)
        policy_obs = preprocessor(raw_policy_obs)

        with torch.no_grad():
            action_tensor = policy.select_action(policy_obs)

        action_tensor = postprocessor(action_tensor)
        act_np = action_tensor.cpu().numpy()

        if act_np.ndim == 1:
            act_np = np.expand_dims(act_np, axis=0)

        if act_np.shape[-1] == 4:
            curr_eef_pos = obs["agent_pos"][0, :3]
            target_eef_pos = act_np[0, :3].copy()

            # 1. Eliminate spatial offset: scale delta pos gain and adjust Z height when aligned over object
            xy_dist = np.linalg.norm(target_eef_pos[:2] - curr_eef_pos[:2])
            if xy_dist < 0.05:
                target_eef_pos[2] -= 0.015  # Descend to object height for firm grasp

            delta_pos = np.clip((target_eef_pos - curr_eef_pos) * 8.0, -1.0, 1.0)

            # 2. Robust gripper activation and firm grasp latching
            model_grip = act_np[0, 3]
            gripper_val = 1.0 if (model_grip > 0.0 or (xy_dist < 0.05 and model_grip > -0.3)) else -1.0

            rpy_zero = np.zeros((1, 3), dtype=np.float32)
            env_action = np.concatenate(
                [np.expand_dims(delta_pos, axis=0), rpy_zero, np.array([[gripper_val]])], axis=-1
            )
        else:
            env_action = act_np

        obs, reward, terminated, truncated, info = vec_env.step(env_action)

        viewer.sync()

        elapsed = time.time() - frame_start
        sleep_time = CONTROL_DT - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

        if not viewer.is_running():
            print(f"MuJoCo viewer window closed at step {step}.")
            break

        if terminated[0] or truncated[0]:
            print(f"Episode finished at step {step}, resetting environment...")
            obs, info = vec_env.reset()
            policy.reset()

vec_env.close()
print("Simulation closed.")
