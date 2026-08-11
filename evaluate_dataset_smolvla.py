"""Script to evaluate SmolVLA policy predictions against ground-truth dataset actions."""

import argparse
import time

import numpy as np
import torch
from tqdm import trange

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAPolicy


def evaluate_policy_on_dataset(
    dataset_id: str = "castanetnicolas/UR5e_sim_pick_place_can_100_absolute_EEF_actions",
    model_id: str = "castanetnicolas/smolvla_ur5e_pick_place",
    num_episodes: int = 10,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Evaluate SmolVLA policy on dataset episodes and compute action error metrics."""
    device = torch.device(device_str)

    print("=" * 70)
    print("         SmolVLA Policy Offline Dataset Evaluation")
    print("=" * 70)
    print(f"Dataset ID : {dataset_id}")
    print(f"Model ID   : {model_id}")
    print(f"Device     : {device}")
    print(f"Episodes   : {num_episodes}\n")

    print("Loading LeRobot Dataset...")
    dataset = LeRobotDataset(dataset_id)
    print(f"Dataset loaded: {dataset.num_episodes} total episodes, {dataset.num_frames} total frames.\n")

    print("Loading SmolVLA Policy...")
    policy = SmolVLAPolicy.from_pretrained(model_id).to(device).eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=model_id,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    total_frames = 0
    all_abs_errors = []
    all_sq_errors = []

    num_eval_episodes = min(num_episodes, dataset.num_episodes)

    start_time = time.time()

    for ep_idx in trange(num_eval_episodes, desc="Evaluating Episodes"):
        ep = dataset.meta.episodes[ep_idx]
        from_idx = ep["dataset_from_index"]
        to_idx = ep["dataset_to_index"]

        policy.reset()

        for frame_idx in range(from_idx, to_idx):
            sample = dataset[frame_idx]

            raw_obs = {
                "observation.images.camera1": sample["observation.images.camera1"],
                "observation.images.camera2": sample["observation.images.camera2"],
                "observation.images.camera3": torch.zeros((3, 256, 256), dtype=torch.float32),
                "observation.state": sample["observation.state"],
                "task": sample["task"],
            }
            gt_action = sample["action"].numpy()

            # Preprocess observation & infer action prediction
            proc_obs = preprocessor(raw_obs)
            with torch.no_grad():
                pred_action_tensor = policy.select_action(proc_obs)

            pred_action = postprocessor(pred_action_tensor).cpu().numpy().squeeze(0)

            # Record errors
            abs_err = np.abs(pred_action - gt_action)
            sq_err = (pred_action - gt_action) ** 2

            all_abs_errors.append(abs_err)
            all_sq_errors.append(sq_err)
            total_frames += 1

    elapsed = time.time() - start_time
    all_abs_errors_np = np.array(all_abs_errors)
    all_sq_errors_np = np.array(all_sq_errors)

    mae_total = np.mean(all_abs_errors_np)
    rmse_total = np.sqrt(np.mean(all_sq_errors_np))

    mae_per_dim = np.mean(all_abs_errors_np, axis=0)

    print("\n" + "=" * 70)
    print("                     EVALUATION RESULTS")
    print("=" * 70)
    print(f"Evaluated Episodes : {num_eval_episodes} / {dataset.num_episodes}")
    print(f"Evaluated Frames   : {total_frames}")
    print(f"Elapsed Time       : {elapsed:.2f}s ({total_frames / elapsed:.1f} frames/sec)")
    print("-" * 70)
    print(f"Overall Action MAE : {mae_total:.5f}")
    print(f"Overall Action RMSE: {rmse_total:.5f}")
    print("-" * 70)
    print("Per-Dimension MAE:")
    dim_labels = ["Position X", "Position Y", "Position Z", "Gripper"]
    for i, label in enumerate(dim_labels):
        if i < len(mae_per_dim):
            print(f"  - {label:12s}: {mae_per_dim[i]:.5f}")
    print("=" * 70)


def main():
    """CLI entry point for evaluate_dataset_smolvla."""
    parser = argparse.ArgumentParser(description="Evaluate SmolVLA policy on dataset")
    parser.add_argument(
        "--dataset_id",
        type=str,
        default="castanetnicolas/UR5e_sim_pick_place_can_100_absolute_EEF_actions",
        help="HuggingFace dataset repo ID",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="castanetnicolas/smolvla_ur5e_pick_place",
        help="HuggingFace policy model ID",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu/mps)",
    )
    args = parser.parse_args()

    evaluate_policy_on_dataset(
        dataset_id=args.dataset_id,
        model_id=args.model_id,
        num_episodes=args.num_episodes,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
