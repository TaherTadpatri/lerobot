# Why Direct Inference with `pi0_base` on Isaac Sim UR5e Fails Out-of-the-Box

This document provides a detailed breakdown of why performing direct zero-shot inference using the pre-trained `lerobot/pi0_base` checkpoint on a custom **UR5e** robot inside **Isaac Sim** does not work as expected, and outlines the key technical mismatches you must resolve.

---

## 1. Dataset Normalization Issue ($\mu$ and $\sigma$)

### Physical Limits vs. Empirical Dataset Statistics

A common misconception in robotics ML is that because a physical UR5e and an Isaac Sim UR5e share identical kinematic joint limits (e.g., $[-2\pi, +2\pi]$ radians for arm joints), their state spaces are automatically compatible.

However, machine learning policies like $\mathbf{\pi_0}$ (Pi0) **do not normalize input states using URDF joint limits**. Instead, they normalize inputs using the **empirical mean ($\mu$) and standard deviation ($\sigma$)** computed across all frames of the dataset used during training:

$$\text{state}_{\text{normalized}} = \frac{\text{state}_{\text{raw}} - \mu_{\text{dataset}}}{\sigma_{\text{dataset}}}$$

### Why This Breaks in Direct Inference

1. **Restricted Task Workspaces**: In a training dataset, the robot operates within a localized region of its workspace (e.g., table surface directly in front). The joint angles observed during training might only span a small fraction of the total range (e.g., shoulder pan between $-30^\circ$ and $+30^\circ$, where $\mu = 0^\circ, \sigma = 15^\circ$).
2. **Out-of-Distribution Inputs**: If your Isaac Sim UR5e starts at a pose where the shoulder pan is rotated to $+90^\circ$ ($1.57 \text{ rad}$):
   $$\text{state}_{\text{normalized}} = \frac{90^\circ - 0^\circ}{15^\circ} = +6.0 \text{ std. deviations}$$
   An input tensor with values $6.0$ standard deviations away from the mean represents a severe out-of-distribution state for the VLM/action expert. As a result, the model outputs saturated, erratic, or near-zero action predictions.

---

## 2. Reset / Home Pose Angle Calibration

### Frame Calibration and Zero Offsets

Even when two robots are kinematically identical (6-DOF UR5e arms), their numerical joint angle representations ($q_0, q_1, \dots, q_5$) depend heavily on how the URDF / Isaac Sim stage defines the **Zero Pose ($0^\circ$)** and **Reset/Home Pose**.

### Specific Discrepancies

- **Zero Position Definition**:
  - In URDF $A$, `shoulder_pan_joint = 0.0` points directly forward along the positive X-axis.
  - In URDF $B$ or an Isaac Sim asset imported from USD, `shoulder_pan_joint = 0.0` might point to the side along the Y-axis.
  - A $90^\circ$ ($\frac{\pi}{2} \text{ rad}$) offset shifts every raw joint value fed into $\pi_0$.
- **Default Reset Poses**:
  - If the model was trained on trajectories starting from a specific "ready pose" (e.g., arm elbow bent at $90^\circ$, hovering over a table), but your Isaac Sim scene resets the arm to a straight upright position ($[0, -90^\circ, 0, -90^\circ, 0, 0]$), the initial input state will be invalid.

---

## 3. Gripper State Encoding Mismatch (The 7th Dimension)

The 7th dimension of the `observation.state` vector for a single-arm setup represents the gripper. While arm joints are typically angles in radians, gripper representations vary drastically across physical datasets and simulation environments:

| Source Environment             | Gripper State Representation            | Range                              |
| :----------------------------- | :-------------------------------------- | :--------------------------------- |
| **Real Robot (Robotiq 2F-85)** | Motor tick position or normalized width | $[0, 255]$ or $[0.0, 1.0]$         |
| **Metric Teleop Dataset**      | Linear gap distance between fingers     | $[0.0\text{ m}, 0.085\text{ m}]$   |
| **Isaac Sim Articulation**     | Angular joint position of knuckle link  | $[0.0\text{ rad}, 0.8\text{ rad}]$ |

### Impact on Inference

If Isaac Sim passes a raw knuckle angle of `0.7` rad, but `pi0_base` expects a normalized scalar where `1.0` is open and `0.0` is closed (or distance in meters `0.085`), the policy receives a completely wrong gripper dimension. This distorts both the gripper action predictions and the cross-attention layers in $\pi_0$.

---

## 4. Other Contributing Factors

1. **Camera View & Visual Domain Gap**: $\pi_0$ is a Vision-Language-Action foundation model. Synthetic images rendered in Isaac Sim have different lighting, background textures, and camera intrinsics/extrinsics compared to real-world pre-training data.
2. **Missing Language Instruction**: $\pi_0$ conditions actions on a text prompt (e.g., `"pick up the object"`). Passing an empty text string causes uninitialized text embeddings.
3. **Action Horizon & Execution Frequency**: $\pi_0$ predicts action chunks (e.g., 50 steps at 50 Hz). Stepping Isaac Sim without executing the action horizon properly causes stuttering or freezes.

---

## 5. Recommended Action Plan

### Option A: Inspecting & Aligning Policy Normalization Stats (Quick Inspection)

You can inspect the exact mean and std expected by your loaded policy in Python:

```python
from lerobot.policies.pi0 import PI0Policy

policy = PI0Policy.from_pretrained("lerobot/pi0_base")

# Inspect expected normalization stats
if hasattr(policy, "dataset_stats") and policy.dataset_stats:
    print("State Mean:", policy.dataset_stats["observation.state"]["mean"])
    print("State Std:",  policy.dataset_stats["observation.state"]["std"])
```

### Option B: Fine-Tuning on Isaac Sim Data (Recommended Solution)

To achieve reliable zero-shot control in your specific Isaac Sim environment:

1. Record **20–50 demonstration trajectories** of your UR5e in Isaac Sim using script or teleop.
2. Save the demonstrations as a LeRobot dataset (which automatically computes the exact `dataset_stats` for your scene).
3. Fine-tune `lerobot/pi0_base`:
   ```bash
   lerobot-train \
       --dataset.repo_id=your_username/ur5e_isaac_sim \
       --policy.type=pi0 \
       --policy.pretrained_path=lerobot/pi0_base \
       --output_dir=./outputs/pi0_ur5e_isaac \
       --steps=3000 \
       --policy.device=cuda
   ```
