"""Asynchronous MuJoCo simulation script for SmolVLA policy to eliminate inference latency damping."""

import queue
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch

from lerobot.envs import make_env
from lerobot.envs.configs import HubEnvConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAPolicy

# 1. Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Configure Hugging Face Hub environment
cfg = HubEnvConfig(
    hub_path="castanetnicolas/UR5e_robosuite_pick_place",
    task="pick_place_can",
)

print("Loading UR5e Pick & Place simulation environment...")
envs = make_env(cfg, trust_remote_code=True)
vec_env = envs[next(iter(envs))][0]
obs, info = vec_env.reset()

env_wrapper = vec_env.envs[0]
robosuite_env = env_wrapper._env
sim = robosuite_env.sim
m = sim.model._model
d = sim.data._data

# 3. Load SmolVLA Policy
model_id = "castanetnicolas/smolvla_ur5e_pick_place"
print(f"Loading pretrained SmolVLA policy '{model_id}'...")
policy = SmolVLAPolicy.from_pretrained(model_id).to(device).eval()

preprocessor, postprocessor = make_pre_post_processors(
    policy.config,
    pretrained_path=model_id,
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)

# Shared thread-safe action queue & observation state for async inference
action_queue: queue.Queue = queue.Queue(maxsize=100)
latest_obs_lock = threading.Lock()
latest_raw_obs = None
running = True


def format_obs_for_policy(obs_dict, task_description):
    """Format Gym observation dictionary into LeRobot SmolVLA policy input format."""
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


def policy_async_worker():
    """Background thread running SmolVLA neural network inference without blocking MuJoCo physics."""
    global latest_raw_obs, running

    while running:
        if latest_raw_obs is None:
            time.sleep(0.005)
            continue

        with latest_obs_lock:
            current_obs = latest_raw_obs

        # Preprocess observations
        policy_obs = preprocessor(current_obs)

        # Infer action chunk asynchronously
        with torch.no_grad():
            actions_chunk = policy.predict_action_chunk(policy_obs)

        actions_chunk = postprocessor(actions_chunk).cpu().numpy()

        # Push action chunk into queue
        for act in actions_chunk[0]:
            if not action_queue.full():
                action_queue.put(act)

        time.sleep(0.01)


# Start background inference thread
latest_raw_obs = format_obs_for_policy(obs, env_wrapper.task_description)
worker_thread = threading.Thread(target=policy_async_worker, daemon=True)
worker_thread.start()

print("\nStarting Async SmolVLA MuJoCo Viewer (Physics running smoothly at full FPS)...")

# 4. Launch MuJoCo viewer with non-blocking async action consumer loop
with mujoco.viewer.launch_passive(m, d) as viewer:
    viewer._opt.geomgroup[0] = 0
    viewer._opt.geomgroup[1] = 1

    for step in range(10000):
        # Fetch next action from queue if available; fallback to zero action if queue is filling up
        if not action_queue.empty():
            act_np = action_queue.get()
            if act_np.ndim == 1:
                act_np = np.expand_dims(act_np, axis=0)

            if act_np.shape[-1] == 4:
                curr_eef_pos = obs["agent_pos"][0, :3]
                target_eef_pos = act_np[0, :3]
                delta_pos = (target_eef_pos - curr_eef_pos) * 5.0
                gripper = act_np[:, 3:]
                rpy_zero = np.zeros((act_np.shape[0], 3), dtype=np.float32)
                env_action = np.concatenate([np.expand_dims(delta_pos, axis=0), rpy_zero, gripper], axis=-1)
            else:
                env_action = act_np
        else:
            env_action = np.zeros((1, 7), dtype=np.float32)

        # Step simulation physics
        obs, reward, terminated, truncated, info = vec_env.step(env_action)

        # Update latest observation for background inference thread
        with latest_obs_lock:
            latest_raw_obs = format_obs_for_policy(obs, env_wrapper.task_description)

        viewer.sync()
        time.sleep(0.01)

        if not viewer.is_running():
            print(f"MuJoCo viewer window closed at step {step}.")
            break

        if terminated[0] or truncated[0]:
            print(f"Episode finished at step {step}, resetting environment...")
            obs, info = vec_env.reset()
            policy.reset()
            action_queue.queue.clear()

running = False
vec_env.close()
print("Async simulation environment closed.")
