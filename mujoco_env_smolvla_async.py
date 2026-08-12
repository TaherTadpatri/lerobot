"""Asynchronous non-blocking MuJoCo simulation script for SmolVLA policy on UR5e robot.

Offloads SmolVLA neural network inference to a background worker thread to eliminate FPS drop/damping,
while the main thread steps MuJoCo physics smoothly at 30+ FPS.
"""

import contextlib
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

# 1. Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Configure environment
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

# Global thread-safe queues and locks
obs_lock = threading.Lock()
latest_obs = obs
action_queue: queue.Queue = queue.Queue(maxsize=10)
stop_event = threading.Event()


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


def policy_async_worker():
    """Background worker thread running SmolVLA neural network inference in parallel."""
    global latest_obs
    print("Background SmolVLA inference thread started.")
    while not stop_event.is_set():
        with obs_lock:
            current_obs = latest_obs

        raw_policy_obs = format_obs_for_policy(current_obs, env_wrapper.task_description)
        policy_obs = preprocessor(raw_policy_obs)

        with torch.no_grad():
            action_tensor = policy.select_action(policy_obs)

        action_tensor = postprocessor(action_tensor)
        act_np = action_tensor.cpu().numpy()

        if action_queue.full():
            with contextlib.suppress(queue.Empty):
                action_queue.get_nowait()
        action_queue.put(act_np)
        time.sleep(0.005)


worker_thread = threading.Thread(target=policy_async_worker, daemon=True)
worker_thread.start()

print("\nStarting Async SmolVLA MuJoCo Viewer (Physics running smoothly at full FPS)...")

last_gripper_state = 1.0  # Start open (+1.0 in Robosuite)

# 4. Launch MuJoCo viewer with non-blocking async action consumer loop
with mujoco.viewer.launch_passive(m, d) as viewer:
    viewer._opt.geomgroup[0] = 0
    viewer._opt.geomgroup[1] = 1

    for step in range(10000):
        if not action_queue.empty():
            act_np = action_queue.get()
            if act_np.ndim == 1:
                act_np = np.expand_dims(act_np, axis=0)

            if act_np.shape[-1] == 4:
                curr_eef_pos = obs["agent_pos"][0, :3]
                target_eef_pos = act_np[0, :3]
                delta_pos = np.clip((target_eef_pos - curr_eef_pos) * 10.0, -1.0, 1.0)

                # Binarize and latch gripper action to prevent random open/close flickering
                model_grip = act_np[0, 3]
                if model_grip > 0.1:
                    env_grip = -1.0  # Close gripper in Robosuite
                elif model_grip < -0.1:
                    env_grip = 1.0  # Open gripper in Robosuite
                else:
                    env_grip = last_gripper_state
                last_gripper_state = env_grip

                rpy_zero = np.zeros((1, 3), dtype=np.float32)
                env_action = np.concatenate(
                    [np.expand_dims(delta_pos, axis=0), rpy_zero, np.array([[env_grip]])], axis=-1
                )
            else:
                env_action = act_np
        else:
            env_action = np.zeros((1, 7), dtype=np.float32)

        # Step simulation physics
        obs, reward, terminated, truncated, info = vec_env.step(env_action)

        with obs_lock:
            latest_obs = obs

        viewer.sync()
        time.sleep(0.01)

        if not viewer.is_running():
            print(f"MuJoCo viewer window closed at step {step}.")
            stop_event.set()
            break

        if terminated[0] or truncated[0]:
            print(f"Episode finished at step {step}, resetting environment...")
            obs, info = vec_env.reset()
            policy.reset()
            last_gripper_state = 1.0
            with obs_lock:
                latest_obs = obs

vec_env.close()
print("Simulation closed.")
