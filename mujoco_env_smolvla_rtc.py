"""MuJoCo simulation script executing SmolVLA policy with Real-Time Control (RTC).

Multithreading & Temporal Action Chunking (Pure End-to-End VLA Evaluation).

Features:
1. Pure End-to-End VLA Evaluation: No privileged simulation ground-truth position reads
   or hardcoded gripper rules are used during control. The SmolVLA neural network
   operates strictly from camera observations (camera1, camera2) and robot proprioception (agent_pos).
2. Multithreaded RTC Architecture: Asynchronous VLA neural network chunk prediction
   runs in a background worker thread (vla_inference_worker), while the main thread
   steps MuJoCo physics and renders the GUI viewer smoothly at 20 FPS (CONTROL_DT = 0.05s).
3. Calibrated Proportional OSC Control: Converts SmolVLA predicted absolute EEF position
   targets into normalized [-1.0, 1.0] delta actions scaled against Robosuite's OSC_POSE
   controller limit (POS_OUTPUT_MAX = 0.05m/step).
"""

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

# 1. FPS and Task Configuration
FPS = 20
CONTROL_DT = 1.0 / FPS  # 0.05 seconds per step (20 FPS)
TASK_DESCRIPTION = "Pick up the can and place it in the correct bin."

# Device selection (GPU / CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Configure Hugging Face Hub environment with Pick & Place Can task
cfg = HubEnvConfig(
    hub_path="castanetnicolas/UR5e_robosuite_pick_place",
    task="pick_place_can",
)

print(f"Loading UR5e Pick & Place simulation environment (Control Rate: {FPS} FPS)...")
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

print(f"\nLoaded Task Prompt: '{TASK_DESCRIPTION}'")
print(f"Control Frequency : {FPS} FPS ({CONTROL_DT:.3f}s per step)")
print(f"Policy Device     : {device}\n")


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


# Global thread synchronization variables
obs_lock = threading.Lock()
chunk_lock = threading.Lock()

latest_obs_raw = obs
latest_action_chunk = None
stop_event = threading.Event()


def vla_inference_worker():
    """Background thread running SmolVLA VLA neural network action chunk inference asynchronously."""
    global latest_obs_raw, latest_action_chunk

    print("Background VLA inference worker thread started.")
    while not stop_event.is_set():
        with obs_lock:
            current_obs = latest_obs_raw

        raw_policy_obs = format_obs_for_policy(current_obs, TASK_DESCRIPTION)
        policy_obs = preprocessor(raw_policy_obs)

        with torch.no_grad():
            action_chunk = policy.predict_action_chunk(policy_obs)

        chunk_tensor = postprocessor(action_chunk)
        chunk_np = chunk_tensor.cpu().numpy().squeeze(0)  # Shape: (50, 4)

        with chunk_lock:
            latest_action_chunk = chunk_np

        time.sleep(0.01)  # Yield CPU to main thread physics loop


# Start background VLA inference thread
vla_thread = threading.Thread(target=vla_inference_worker, daemon=True)
vla_thread.start()

# Wait for first VLA prediction to be ready
print("Waiting for initial VLA action chunk prediction...")
while latest_action_chunk is None:
    time.sleep(0.05)
print("Initial action chunk received. Starting simulation & rendering GUI viewer...\n")

# Controller proportional gain: maps physical meters/step to normalized [-1.0, 1.0] OSC_POSE action
POS_SCALE = 20.0  # 1.0 / 0.05m max OSC step size
# 4. Launch native MuJoCo passive viewer GUI window and run main physics loop at 20 FPS
with mujoco.viewer.launch_passive(m, d) as viewer:
    viewer._opt.geomgroup[0] = 0
    viewer._opt.geomgroup[1] = 1

    step = 0
    chunk_step_idx = 0

    while viewer.is_running() and step < 2000:
        frame_start_time = time.time()

        # Update background worker observation
        with obs_lock:
            latest_obs_raw = obs

        # Get latest predicted action chunk from background worker
        with chunk_lock:
            current_chunk = latest_action_chunk

        if current_chunk is not None:
            pred_act = current_chunk[min(chunk_step_idx, len(current_chunk) - 1)]
            chunk_step_idx += 1
            if chunk_step_idx >= len(current_chunk):
                chunk_step_idx = 0  # Cycle chunk steps if chunk hasn't updated yet
        else:
            pred_act = np.zeros(4, dtype=np.float32)

        # Pure End-to-End VLA Policy Control (No privileged simulation reads)
        curr_eef = obs["agent_pos"][0, :3]
        target_eef = pred_act[:3]
        model_grip = pred_act[3]

        # Gripper action directly from VLA policy prediction
        gripper_val = 1.0 if model_grip > 0.0 else -1.0
        # print(f"model : gripper : {model_grip} , robot: gripper , {gripper_val}" )
        # End-effector delta position mapped into normalized [-1.0, 1.0] action space
        delta_pos = np.clip((target_eef - curr_eef) * POS_SCALE, -1.0, 1.0)
        env_action = np.concatenate(
            [np.expand_dims(delta_pos, axis=0), np.zeros((1, 3)), np.array([[gripper_val]])], axis=-1
        )
        print(
            f"Step {step:04d} | EEF: {curr_eef.round(3)} | VLA Target EEF: {target_eef.round(3)} | VLA Grip: {model_grip:.3f} -> Env Grip: {gripper_val:.1f}"
        )

        # if step % 20 == 0:
        #     print(
        #         f"Step {step:04d} | EEF: {curr_eef.round(3)} | VLA Target EEF: {target_eef.round(3)} | VLA Grip: {model_grip:.3f} -> Env Grip: {gripper_val:.1f}"
        #     )

        obs, reward, terminated, truncated, info = vec_env.step(env_action)
        step += 1

        viewer.sync()

        # Maintain exact 20 FPS loop pacing (0.05s per control frame)
        elapsed = time.time() - frame_start_time
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

stop_event.set()
vec_env.close()
print("Simulation finished.")
