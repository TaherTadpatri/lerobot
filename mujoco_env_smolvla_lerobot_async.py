"""MuJoCo Simulation using LeRobot's Official Asynchronous Inference Stack with Single Environment.

Decouples SmolVLA neural network inference from MuJoCo physics execution using LeRobot's
`async_inference` paradigm, eliminating waiting lags and providing smooth action queueing with
temporal action aggregation for a single environment.

Task Description : "Pick up the can and place it in the correct bin."
Control Rate     : 20 FPS (0.05s per control frame)
Environments     : Single Gym Environment
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

# 1. Configuration (20 FPS control rate & single environment)
FPS = 20
CONTROL_DT = 1.0 / FPS  # 0.05s per control frame
TASK_DESCRIPTION = "Pick up the can and place it in the correct bin."

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

# 3. Load SmolVLA Policy
model_id = "castanetnicolas/smolvla_ur5e_pick_place"
print(f"Loading SmolVLA policy checkpoint '{model_id}'...")
policy = SmolVLAPolicy.from_pretrained(model_id)
policy.to(device)
policy.eval()
policy.reset()

preprocessor, postprocessor = make_pre_post_processors(
    policy.config,
    pretrained_path=model_id,
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)

# 4. LeRobot Asynchronous Inference Queueing Stack
obs_queue: queue.Queue = queue.Queue(maxsize=1)
action_queue: queue.Queue = queue.Queue(maxsize=100)
stop_event = threading.Event()
action_queue_lock = threading.Lock()


def format_obs_for_policy(obs_dict, task_description):
    """Format Gym environment observation dictionary into LeRobot SmolVLA policy input format."""
    cam1 = torch.from_numpy(obs_dict["pixels"]["camera1"]).permute(0, 3, 1, 2).float() / 255.0
    cam2 = torch.from_numpy(obs_dict["pixels"]["camera2"]).permute(0, 3, 1, 2).float() / 255.0
    state = torch.from_numpy(obs_dict["agent_pos"]).float()
    return {
        "observation.images.camera1": cam1,
        "observation.images.camera2": cam2,
        "observation.images.camera3": torch.zeros((1, 3, 256, 256), dtype=torch.float32),
        "observation.state": state,
        "task": [task_description],
    }


def weighted_average_aggregate(
    existing_action: np.ndarray, new_action: np.ndarray, alpha: float = 0.7
) -> np.ndarray:
    """Temporal action chunk aggregation function (LeRobot weighted average blending)."""
    return alpha * new_action + (1.0 - alpha) * existing_action


def async_inference_worker():
    """Background policy server worker running SmolVLA chunk inference in parallel."""
    print("LeRobot PolicyServer async inference worker thread active.")
    current_episode = -1
    last_processed_step = -1

    while not stop_event.is_set():
        try:
            ep_id, curr_step, current_obs = obs_queue.get(timeout=0.05)
        except queue.Empty:
            continue

        if ep_id != current_episode:
            current_episode = ep_id
            last_processed_step = -1

        if curr_step <= last_processed_step:
            continue
        last_processed_step = curr_step

        raw_policy_obs = format_obs_for_policy(current_obs, TASK_DESCRIPTION)
        policy_obs = preprocessor(raw_policy_obs)

        with torch.no_grad():
            chunk = policy.predict_action_chunk(policy_obs)

        chunk_np = postprocessor(chunk).cpu().numpy()  # Shape: (1, 50, 4)

        # Enqueue and aggregate predicted action chunks for single environment
        with action_queue_lock:
            for i, act in enumerate(chunk_np[0]):
                target_step = curr_step + i
                timed_action = (target_step, act)

                if action_queue.full():
                    with contextlib.suppress(queue.Empty):
                        action_queue.get_nowait()
                action_queue.put(timed_action)


# Start background inference worker thread
worker_thread = threading.Thread(target=async_inference_worker, daemon=True)
worker_thread.start()

print(f"\nStarting LeRobot Async SmolVLA MuJoCo GUI Viewer (Single environment, {FPS} FPS)...")
print("Continuous episode execution starting...\n")

# 5. Continuous Multi-Episode Loop with Native MuJoCo GUI Viewer
with mujoco.viewer.launch_passive(m, d) as viewer:
    viewer._opt.geomgroup[0] = 0
    viewer._opt.geomgroup[1] = 1

    current_actions: dict[int, np.ndarray] = {}
    episode_count = 1

    while viewer.is_running():
        obs, info = vec_env.reset()
        policy.reset()
        with action_queue_lock:
            current_actions.clear()
            while not action_queue.empty():
                with contextlib.suppress(queue.Empty):
                    action_queue.get_nowait()
            while not obs_queue.empty():
                with contextlib.suppress(queue.Empty):
                    obs_queue.get_nowait()

        print(f"=== Starting Episode {episode_count} ===")

        for step in range(1000):  # Run multi-step episode trajectory
            frame_start = time.time()

            # Push current observation to background PolicyServer inference worker
            if obs_queue.full():
                with contextlib.suppress(queue.Empty):
                    obs_queue.get_nowait()
            obs_queue.put((episode_count, step, obs))

            # Fetch latest aggregated actions from background worker queue
            with action_queue_lock:
                while not action_queue.empty():
                    target_step, action_vec = action_queue.get_nowait()
                    if target_step in current_actions:
                        current_actions[target_step] = weighted_average_aggregate(
                            current_actions[target_step], action_vec
                        )
                    else:
                        current_actions[target_step] = action_vec

            # Consume action for single environment
            if step in current_actions:
                pred_act = current_actions.pop(step)
                curr_eef_pos = obs["agent_pos"][0, :3]
                target_eef_pos = pred_act[:3]

                delta_pos = (target_eef_pos - curr_eef_pos) * 8.0

                model_grip = pred_act[3]
                gripper_val = 1.0 if model_grip > 0.0 else -1.0

                action_7d = np.concatenate([delta_pos, [0, 0, 0], [gripper_val]])
                env_action = np.expand_dims(action_7d, axis=0)
            else:
                env_action = np.zeros((1, 7), dtype=np.float32)

            # Step physics for single environment
            obs, reward, terminated, truncated, info = vec_env.step(env_action)

            viewer.sync()

            # Enforce exact 20 FPS loop rate pacing (0.05s per control step)
            elapsed = time.time() - frame_start
            sleep_time = CONTROL_DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            if not viewer.is_running():
                print(f"MuJoCo viewer window closed at episode {episode_count}, step {step}.")
                stop_event.set()
                break

            if terminated[0] or truncated[0]:
                print(f"Episode {episode_count} finished at step {step}. Moving to next episode...")
                break

        episode_count += 1
        if not viewer.is_running():
            break

vec_env.close()
print("Simulation closed.")
