"""MuJoCo UR5e Robot Implementation with Threaded Physics & GUI Viewer Rendering."""

import contextlib
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np

from lerobot.envs import make_env
from lerobot.envs.configs import HubEnvConfig

from ..robot import Robot
from .config_mujoco_ur5e import MuJoCoUR5eRobotConfig


class MuJoCoUR5eRobot(Robot):
    """LeRobot Robot interface wrapper around MuJoCo Robosuite UR5e Pick & Place simulation.

    Features threaded physics execution and GUI viewer rendering running independently at
    20 FPS in a background daemon thread, eliminating stop-and-start stuttering during gRPC
    client-server async inference.
    """

    config_class = MuJoCoUR5eRobotConfig
    name = "mujoco_ur5e"

    def __init__(self, config: MuJoCoUR5eRobotConfig):
        super().__init__(config)
        self.config: MuJoCoUR5eRobotConfig = config
        self._is_connected = False
        self.vec_env = None
        self.sim = None
        self.model = None
        self.data = None
        self.viewer = None
        self.last_obs = None

        # Threading & locks for asynchronous physics execution
        self.obs_lock = threading.Lock()
        self.act_lock = threading.Lock()
        self.physics_thread = None
        self.stop_event = threading.Event()
        self.current_target_action = np.zeros((1, 7), dtype=np.float32)

        # Proportional position gain scale mapping spatial error into normalized [-1.0, 1.0] OSC space
        self.POS_SCALE = getattr(self.config, "pos_scale", 8.0)

    @property
    def observation_features(self) -> dict:
        state_fts = {f"state_{i}": float for i in range(13)}
        cam_fts = {
            "camera1": (256, 256, 3),
            "camera2": (256, 256, 3),
        }
        return {**state_fts, **cam_fts}

    @property
    def action_features(self) -> dict:
        return {f"action_{i}": float for i in range(4)}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        """Connect to MuJoCo UR5e simulation environment and start background physics thread."""
        if self._is_connected:
            return

        print(f"Connecting to MuJoCo UR5e simulation environment (Task: '{self.config.task}')...")
        hub_cfg = HubEnvConfig(
            hub_path=self.config.hub_path,
            task=self.config.task,
        )
        envs = make_env(hub_cfg, trust_remote_code=True)
        self.vec_env = envs[next(iter(envs))][0]
        initial_obs, _ = self.vec_env.reset()

        with self.obs_lock:
            self.last_obs = initial_obs

        # Access Robosuite simulation internal pointers
        env_wrapper = self.vec_env.envs[0]
        robosuite_env = env_wrapper._env
        self.sim = robosuite_env.sim
        self.model = self.sim.model._model
        self.data = self.sim.data._data

        # Launch native MuJoCo passive 3D GUI viewer window if enabled
        if self.config.render_gui:
            print("Launching interactive native MuJoCo passive 3D GUI window...")
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer._opt.geomgroup[0] = 0  # Hide collision geometry primitives
            self.viewer._opt.geomgroup[1] = 1  # Show detailed CAD visual meshes
            self.viewer.sync()

        # Start dedicated background physics loop running independently at 20 FPS
        self.stop_event.clear()
        self.physics_thread = threading.Thread(target=self._physics_loop, daemon=True)
        self.physics_thread.start()

        self._is_connected = True
        print("MuJoCo UR5e Robot connected successfully with active background 20 FPS physics loop.\n")

    def _physics_loop(self) -> None:
        """Dedicated background thread stepping MuJoCo physics & updating 3D GUI at 20 FPS."""
        control_dt = 1.0 / self.config.fps

        while not self.stop_event.is_set():
            start_time = time.time()

            with self.act_lock:
                env_act = self.current_target_action.copy()

            # Step simulation physics
            obs, reward, terminated, truncated, info = self.vec_env.step(env_act)

            with self.obs_lock:
                self.last_obs = obs

            # Sync interactive 3D GUI viewer smoothly
            if self.viewer is not None and self.viewer.is_running():
                self.viewer.sync()

            # Reset environment on episode completion
            if terminated[0] or truncated[0]:
                print("MuJoCo simulation episode completed, resetting environment...")
                reset_obs, _ = self.vec_env.reset()
                with self.obs_lock:
                    self.last_obs = reset_obs

            # Maintain exact 20 FPS loop pacing (0.05s per control frame)
            elapsed = time.time() - start_time
            sleep_time = control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_observation(self) -> dict:
        """Retrieve current observation dictionary matching observation_features (non-blocking)."""
        if not self._is_connected:
            raise RuntimeError("Robot is not connected. Call connect() first.")

        with self.obs_lock:
            current_obs = self.last_obs

        cam1 = current_obs["pixels"]["camera1"].squeeze(0)
        cam2 = current_obs["pixels"]["camera2"].squeeze(0)
        state = current_obs["agent_pos"].squeeze(0)

        obs_dict = {
            "camera1": cam1,
            "camera2": cam2,
        }
        for i in range(13):
            obs_dict[f"state_{i}"] = float(state[i])

        return obs_dict

    def send_action(self, action: dict | np.ndarray) -> dict:
        """Convert policy actions into Operational Space Control (OSC) delta commands for MuJoCo (non-blocking)."""
        if not self._is_connected:
            raise RuntimeError("Robot is not connected. Call connect() first.")

        # Extract raw action array
        if isinstance(action, dict):
            act_keys = sorted(action.keys())
            act_arr = np.array([float(action[k]) for k in act_keys], dtype=np.float32)
        else:
            act_arr = np.asarray(action, dtype=np.float32)

        if act_arr.ndim == 2:
            act_arr = act_arr.squeeze(0)

        if len(act_arr) == 4:
            # Absolute target position [x, y, z] + gripper
            target_eef = act_arr[:3]
            model_grip = act_arr[3]

            with self.obs_lock:
                curr_eef = self.last_obs["agent_pos"][0, :3]

            # Compute relative delta position mapped into normalized [-1.0, 1.0] action space
            delta_pos = np.clip((target_eef - curr_eef) * self.POS_SCALE, -1.0, 1.0)
            gripper_val = 1.0 if model_grip > 0.0 else -1.0

            env_act = np.concatenate(
                [np.expand_dims(delta_pos, 0), np.zeros((1, 3)), np.array([[gripper_val]])], axis=-1
            )
        elif len(act_arr) == 7:
            # Direct 7D OSC pose delta action
            env_act = np.expand_dims(act_arr, axis=0)
        else:
            env_act = np.expand_dims(act_arr, axis=0)

        # Thread-safely set target action for the background physics thread
        with self.act_lock:
            self.current_target_action = env_act

        return {"action": env_act.squeeze(0)}

    def disconnect(self) -> None:
        """Stop background physics thread and close viewer window."""
        self.stop_event.set()
        if self.physics_thread is not None:
            self.physics_thread.join(timeout=1.0)
            self.physics_thread = None

        if self.viewer is not None:
            with contextlib.suppress(Exception):
                self.viewer.close()
            self.viewer = None

        if self.vec_env is not None:
            with contextlib.suppress(Exception):
                self.vec_env.close()
            self.vec_env = None

        self._is_connected = False
        print("MuJoCo UR5e Robot disconnected.")
