"""MuJoCo simulation script running ACT policy on UR5e pick & place environment."""

import time

import mujoco
import mujoco.viewer
import numpy as np
import torch

from lerobot.envs import make_env
from lerobot.envs.configs import HubEnvConfig
from lerobot.policies.act.modeling_act import ACTPolicy

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

# 3. Load pretrained ACT Policy from Hugging Face Hub
print("Loading pretrained ACT policy 'castanetnicolas/act_ur5e_pick_place'...")
policy = ACTPolicy.from_pretrained("castanetnicolas/act_ur5e_pick_place")
policy.to(device)
policy.eval()
policy.reset()

print(f"\nLoaded Task: '{env_wrapper.task}' ({env_wrapper.task_description})")
print(f"Policy Device: {device}")
print("Native MuJoCo GUI viewer started. Running ACT policy inference...\n")


def format_obs_for_policy(obs_dict, device):
    """Format Gym environment observation dictionary into LeRobot ACT policy input format."""
    cam1 = torch.from_numpy(obs_dict["pixels"]["camera1"]).permute(0, 3, 1, 2).float().to(device) / 255.0
    cam2 = torch.from_numpy(obs_dict["pixels"]["camera2"]).permute(0, 3, 1, 2).float().to(device) / 255.0
    state = torch.from_numpy(obs_dict["agent_pos"]).float().to(device)
    return {
        "observation.images.camera1": cam1,
        "observation.images.camera2": cam2,
        "observation.state": state,
    }


# 4. Launch native MuJoCo passive viewer GUI window and run policy loop
with mujoco.viewer.launch_passive(m, d) as viewer:
    # Disable Group 0 (collision primitives) and enable Group 1 (detailed visual CAD meshes)
    viewer._opt.geomgroup[0] = 0
    viewer._opt.geomgroup[1] = 1

    for step in range(10000):
        # Format current environment observation for ACT policy
        policy_obs = format_obs_for_policy(obs, device)

        # Infer action chunk from ACT policy
        with torch.no_grad():
            action_tensor = policy.select_action(policy_obs)

        act_np = action_tensor.cpu().numpy()

        # Format 4-D absolute policy action [x, y, z, gripper] into environment action
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
