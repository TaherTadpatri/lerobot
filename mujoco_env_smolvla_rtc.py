"""MuJoCo simulation script executing SmolVLA policy with Real-Time Control (RTC) & Temporal Action Chunking.

Instead of querying the VLM single-frame-by-single-frame (which causes open/close flickering and drift),
this script predicts 50-step action chunks and executes smooth multi-step action queues.
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

# 1. Device selection (GPU / CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Configure Hugging Face Hub environment with Pick & Place Can task
cfg = HubEnvConfig(
    hub_path="castanetnicolas/UR5e_robosuite_pick_place",
    task="pick_place_can",
)

print("Loading UR5e Pick & Place simulation environment...")
envs = make_env(cfg, trust_remote_code=True)
vec_env = envs[next(iter(envs))][0]
obs, info = vec_env.reset()

# Access environment wrapper and raw robosuite simulation instance
env_wrapper = vec_env.envs[0]
robosuite_env = env_wrapper._env
sim = robosuite_env.sim
m = sim.model._model
d = sim.data._data

# 3. Load pretrained SmolVLA Policy from Hugging Face Hub
model_id = "castanetnicolas/smolvla_ur5e_pick_place"
print(f"Loading pretrained SmolVLA policy '{model_id}'...")
policy = SmolVLAPolicy.from_pretrained(model_id)
policy.to(device)
policy.eval()
policy.reset()

# Create policy pre- and post-processors
preprocessor, postprocessor = make_pre_post_processors(
    policy.config,
    pretrained_path=model_id,
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)

print(f"\nLoaded Task: '{env_wrapper.task}' ({env_wrapper.task_description})")
print(f"Policy Device: {device}")
print("Native MuJoCo GUI viewer started. Running SmolVLA Temporal Action Chunking (RTC) rollout...\n")


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


# Temporal Action Chunking settings
execution_horizon = 8  # Execute 8 steps per VLM chunk prediction

# 4. Launch native MuJoCo passive viewer GUI window and run RTC chunked policy loop
with mujoco.viewer.launch_passive(m, d) as viewer:
    viewer._opt.geomgroup[0] = 0
    viewer._opt.geomgroup[1] = 1

    step = 0
    while viewer.is_running() and step < 10000:
        # Step A: Query VLM for a 50-step action chunk
        raw_policy_obs = format_obs_for_policy(obs, env_wrapper.task_description)
        policy_obs = preprocessor(raw_policy_obs)

        with torch.no_grad():
            action_chunk = policy.predict_action_chunk(policy_obs)

        chunk_tensor = postprocessor(action_chunk)
        chunk_np = chunk_tensor.cpu().numpy().squeeze(0)  # Shape: (50, 4)

        # Step B: Execute execution_horizon steps open-loop from predicted chunk
        steps_to_exec = min(execution_horizon, len(chunk_np))
        for i in range(steps_to_exec):
            pred_act = chunk_np[i]
            curr_eef_pos = obs["agent_pos"][0, :3]
            target_eef_pos = pred_act[:3]

            # Delta target position command
            delta_pos = (target_eef_pos - curr_eef_pos) * 3.5

            # Gripper mapping (+1.0 close, -1.0 open in Robosuite)
            model_grip = pred_act[3]
            gripper_val = 1.0 if model_grip > 0.0 else -1.0

            rpy_zero = np.zeros((1, 3), dtype=np.float32)
            env_action = np.concatenate(
                [np.expand_dims(delta_pos, axis=0), rpy_zero, np.array([[gripper_val]])], axis=-1
            )

            obs, reward, terminated, truncated, info = vec_env.step(env_action)
            step += 1

            viewer.sync()
            time.sleep(0.01)

            if not viewer.is_running():
                print(f"MuJoCo viewer window closed at step {step}.")
                break

            if terminated[0] or truncated[0]:
                print(f"Episode finished at step {step}, resetting environment...")
                obs, info = vec_env.reset()
                policy.reset()
                break

vec_env.close()
print("Simulation finished.")
