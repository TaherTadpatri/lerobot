"""MuJoCo simulation script running SmolVLA policy on UR5e pick & place environment."""

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

# Create policy pre- and post-processors (tokenizing text, normalizing inputs, device placement, action unnormalization)
preprocessor, postprocessor = make_pre_post_processors(
    policy.config,
    pretrained_path=model_id,
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)

print(f"\nLoaded Task: '{env_wrapper.task}' ({env_wrapper.task_description})")
print(f"Policy Device: {device}")
print("Native MuJoCo GUI viewer started. Running SmolVLA policy inference...\n")


def format_obs_for_policy(obs_dict, task_description):
    """Format Gym environment observation dictionary into LeRobot SmolVLA policy input format."""
    # Convert uint8 image pixels to float32 tensors scaled to [0, 1]
    cam1 = torch.from_numpy(obs_dict["pixels"]["camera1"]).squeeze(0).permute(2, 0, 1).float() / 255.0
    cam2 = torch.from_numpy(obs_dict["pixels"]["camera2"]).squeeze(0).permute(2, 0, 1).float() / 255.0
    # State observation vector (13-D agent_pos)
    state = torch.from_numpy(obs_dict["agent_pos"]).squeeze(0).float()
    return {
        "observation.images.camera1": cam1,
        "observation.images.camera2": cam2,
        "observation.images.camera3": torch.zeros((3, 256, 256), dtype=torch.float32),
        "observation.state": state,
        "task": task_description,
    }


# 4. Launch native MuJoCo passive viewer GUI window and run policy loop
with mujoco.viewer.launch_passive(m, d) as viewer:
    # Disable Group 0 (collision primitives) and enable Group 1 (detailed visual CAD meshes)
    viewer._opt.geomgroup[0] = 0
    viewer._opt.geomgroup[1] = 1

    for step in range(10000):
        # Format current environment observation for SmolVLA policy
        raw_policy_obs = format_obs_for_policy(obs, env_wrapper.task_description)

        # Preprocess observations (apply image transforms, language tokenization, normalization, device placement)
        policy_obs = preprocessor(raw_policy_obs)

        # Infer action chunk from SmolVLA policy
        with torch.no_grad():
            action_tensor = policy.select_action(policy_obs)

        # Postprocess action (unnormalization to physical units)
        action_tensor = postprocessor(action_tensor)
        act_np = action_tensor.cpu().numpy()

        if act_np.ndim == 1:
            act_np = np.expand_dims(act_np, axis=0)

        # Format 4-D absolute policy action [x, y, z, gripper] into environment 7-D action [x, y, z, r, p, y, gripper]
        if act_np.shape[-1] == 4:
            xyz = act_np[:, :3]
            gripper = act_np[:, 3:]
            rpy_zero = np.zeros((act_np.shape[0], 3), dtype=np.float32)
            env_action = np.concatenate([xyz, rpy_zero, gripper], axis=-1)
        else:
            env_action = act_np

        # Step simulation with predicted policy action
        obs, reward, terminated, truncated, info = vec_env.step(env_action)

        viewer.sync()
        time.sleep(0.01)

        if not viewer.is_running():
            print(f"MuJoCo viewer window closed at step {step}.")
            break

        if terminated[0] or truncated[0]:
            print(f"Episode finished at step {step}, resetting environment...")
            obs, info = vec_env.reset()
            policy.reset()

vec_env.close()
print("Environment closed successfully.")
