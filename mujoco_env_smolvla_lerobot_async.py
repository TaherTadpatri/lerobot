"""MuJoCo Simulation using LeRobot's Official Asynchronous Inference Stack with 2 Parallel Environments.

Decouples SmolVLA neural network inference from MuJoCo physics execution using LeRobot's
`async_inference` paradigm, eliminating waiting lags and providing smooth action queueing with
temporal action aggregation across 2 parallel vectorized environments (`n_envs=2`).

Task Description : "Pick up the can and place it in the correct bin."
Control Rate     : 20 FPS (0.05s per control frame)
Environments     : 2 Parallel Gym Vectorized Environments
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

# 1. Configuration (20 FPS control rate & 2 vectorized environments)
NUM_ENVS = 1
FPS = 20
CONTROL_DT = 1.0 / FPS  # 0.05s per control frame
TASK_DESCRIPTION = "Pick up the can and place it in the correct bin."

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Configure environment with n_envs = 2
cfg = HubEnvConfig(
    hub_path="castanetnicolas/UR5e_robosuite_pick_place",
    task="pick_place_can",
)

print(f"Loading UR5e Pick & Place simulation environment ({NUM_ENVS} parallel envs, {FPS} FPS)...")
envs = make_env(cfg, n_envs=NUM_ENVS, trust_remote_code=True)
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


def format_obs_batch_for_policy(obs_dict, task_description, b_size):
    """Format Gym environment batch observation dictionary into LeRobot SmolVLA policy input format."""
    cam1 = torch.from_numpy(obs_dict["pixels"]["camera1"]).permute(0, 3, 1, 2).float() / 255.0
    cam2 = torch.from_numpy(obs_dict["pixels"]["camera2"]).permute(0, 3, 1, 2).float() / 255.0
    state = torch.from_numpy(obs_dict["agent_pos"]).float()
    return {
        "observation.images.camera1": cam1,
        "observation.images.camera2": cam2,
        "observation.images.camera3": torch.zeros((b_size, 3, 256, 256), dtype=torch.float32),
        "observation.state": state,
        "task": [task_description] * b_size,
    }


def weighted_average_aggregate(
    existing_action: np.ndarray, new_action: np.ndarray, alpha: float = 0.7
) -> np.ndarray:
    """Temporal action chunk aggregation function (LeRobot weighted average blending)."""
    return alpha * new_action + (1.0 - alpha) * existing_action


def async_inference_worker():
    """Background policy server worker running batched SmolVLA chunk inference in parallel."""
    print("LeRobot PolicyServer async inference worker thread active.")
    last_processed_step = -1

    while not stop_event.is_set():
        try:
            curr_step, current_obs = obs_queue.get(timeout=0.05)
        except queue.Empty:
            continue

        if curr_step <= last_processed_step:
            continue
        last_processed_step = curr_step

        raw_policy_obs = format_obs_batch_for_policy(current_obs, TASK_DESCRIPTION, NUM_ENVS)
        policy_obs = preprocessor(raw_policy_obs)

        with torch.no_grad():
            chunk = policy.predict_action_chunk(policy_obs)

        chunk_np = postprocessor(chunk).cpu().numpy()  # Shape: (NUM_ENVS, 50, 4)

        # Enqueue and aggregate predicted action chunks for all environments
        with action_queue_lock:
            for e in range(NUM_ENVS):
                for i, act in enumerate(chunk_np[e]):
                    target_step = curr_step + i
                    timed_action = (e, target_step, act)

                    if action_queue.full():
                        with contextlib.suppress(queue.Empty):
                            action_queue.get_nowait()
                    action_queue.put(timed_action)


# Start background inference worker thread
worker_thread = threading.Thread(target=async_inference_worker, daemon=True)
worker_thread.start()

print(f"\nStarting LeRobot Async SmolVLA MuJoCo GUI Viewer ({NUM_ENVS} parallel environments, {FPS} FPS)...")
print("Multiple continuous episodes running in parallel...\n")

# 5. Continuous Multi-Episode Loop with Native MuJoCo GUI Viewer
with mujoco.viewer.launch_passive(m, d) as viewer:
    viewer._opt.geomgroup[0] = 0
    viewer._opt.geomgroup[1] = 1

    current_actions: dict[tuple[int, int], np.ndarray] = {}
    episode_count = 1

    while viewer.is_running():
        obs, info = vec_env.reset()
        policy.reset()
        with action_queue_lock:
            current_actions.clear()

        print(f"=== Starting Episode {episode_count} across {NUM_ENVS} environments ===")

        for step in range(1000):  # Run multi-step episode trajectory
            frame_start = time.time()

            # Push current batch observation to background PolicyServer inference worker
            if obs_queue.full():
                with contextlib.suppress(queue.Empty):
                    obs_queue.get_nowait()
            obs_queue.put((step, obs))

            # Fetch latest aggregated actions from background worker queue
            with action_queue_lock:
                while not action_queue.empty():
                    env_idx, target_step, action_vec = action_queue.get_nowait()
                    key = (env_idx, target_step)
                    if key in current_actions:
                        current_actions[key] = weighted_average_aggregate(current_actions[key], action_vec)
                    else:
                        current_actions[key] = action_vec

            # Consume actions for all 2 parallel environments
            env_actions = []
            for e in range(NUM_ENVS):
                key = (e, step)
                if key in current_actions:
                    pred_act = current_actions.pop(key)
                    curr_eef_pos = obs["agent_pos"][e, :3]
                    target_eef_pos = pred_act[:3].copy()

                    # Apply downward Z offset boost when centered over object to ensure firm grasp
                    xy_dist = np.linalg.norm(target_eef_pos[:2] - curr_eef_pos[:2])
                    if xy_dist < 0.05:
                        target_eef_pos[2] -= 0.015

                    delta_pos = (target_eef_pos - curr_eef_pos) * 8.0

                    model_grip = pred_act[3]
                    gripper_val = 1.0 if model_grip > 0.0 else -1.0

                    action_7d = np.concatenate([delta_pos, [0, 0, 0], [gripper_val]])
                    env_actions.append(action_7d)
                else:
                    env_actions.append(np.zeros(7, dtype=np.float32))

            # Step physics for both environments
            obs, reward, terminated, truncated, info = vec_env.step(np.array(env_actions))

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

            if np.any(terminated) or np.any(truncated):
                print(f"Episode {episode_count} finished at step {step}. Moving to next episode...")
                break

        episode_count += 1
        if not viewer.is_running():
            break

vec_env.close()
print("Simulation closed.")
