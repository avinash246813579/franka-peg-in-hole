"""
Record demonstrations for the peg-in-hole perception net.

Drives the privileged-state ScriptedExpertPegV1 across N episodes with
randomized hole positions, and captures per-step observations during the
SCAN and APPROACH phases (when the hole is visible to the wrist camera —
before the peg starts occluding it during DESCEND/INSERT).

Output: one .npz file per (episode, env) trajectory in --output_dir.
Each file contains:
    rgb        : (T, 224, 224, 3) uint8  — wrist camera RGB
    depth      : (T, 224, 224)    float32 — distance to image plane (m)
    ee_pos_b   : (T, 3)           float32 — panda_hand position in robot frame
    ee_quat_b  : (T, 4)           float32 — panda_hand orientation (w, x, y, z)
    hole_xy_b  : (T, 2)           float32 — true hole xy in robot frame (the label)
    phase      : (T,)             uint8   — 0 = SCAN, 1 = APPROACH

Usage:
    /isaac-sim/python.sh record_demos_peg.py \
        --num_envs 4 --num_episodes 20 \
        --output_dir /workspace/imitation/demos_peg --seed 12345 --enable_cameras
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_episodes", type=int, default=20,
                    help="Number of episode resets to run. Total trajectories = num_envs * num_episodes.")
parser.add_argument("--max_steps_per_episode", type=int, default=220,
                    help="Stop each episode at this step. Default 220 = SCAN(100) + APPROACH(80) + buffer.")
parser.add_argument("--record_until_phase", type=int, default=1,
                    help="Record frames while expert phase <= this value. Default 1 = SCAN+APPROACH only.")
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--output_dir", type=str, default="/workspace/imitation/demos_peg")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import warp as wp  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402

from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_camera_env_cfg_peg_v1 import (  # noqa: E402
    FrankaPegInsertV1EnvCfg,
    HOLE_DEPTH,
    HOLE_NOMINAL_POS,
)
from scripted_expert_peg_v1 import ScriptedExpertPegV1  # noqa: E402
from peg_fixed_joint_helper import PegFixedJointAttacher  # noqa: E402


def _get_camera_rgb_depth(env):
    """Read wrist camera RGB (uint8, drops alpha) and depth from env scene."""
    cam = env.scene["wrist_camera"]
    rgb = cam.data.output["rgb"].cpu().numpy()  # (N, H, W, 4) uint8
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]  # drop alpha
    depth = cam.data.output["distance_to_image_plane"].cpu().numpy()  # (N, H, W) float32
    return rgb, depth


def _get_ee_pose_b(env):
    """Read panda_hand pose in robot frame."""
    body_state_w = wp.to_torch(env.scene["robot"].data.body_state_w)
    hand_idx = env.scene["robot"].body_names.index("panda_hand")
    ee_pos_w = body_state_w[:, hand_idx, :3]
    ee_quat_w = body_state_w[:, hand_idx, 3:7]
    root_pos_w = wp.to_torch(env.scene["robot"].data.root_pos_w)
    root_quat_w = wp.to_torch(env.scene["robot"].data.root_quat_w)
    ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
    return ee_pos_b, ee_quat_b


def _get_hole_xy_b(env, hole_z_top):
    """Read true hole xy in robot frame (per env). Returns (N, 3) with z replaced by hole_z_top."""
    hole_base_w = wp.to_torch(env.scene["hole_base"].data.root_pos_w)
    root_pos_w = wp.to_torch(env.scene["robot"].data.root_pos_w)
    root_quat_w = wp.to_torch(env.scene["robot"].data.root_quat_w)
    hole_pos_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, hole_base_w)
    hole_pos_b = hole_pos_b.clone()
    hole_pos_b[:, 2] = hole_z_top
    return hole_pos_b


def main():
    cfg = FrankaPegInsertV1EnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = "cuda:0"
    cfg.seed = args.seed
    cfg.episode_length_s = (args.max_steps_per_episode + 30) * 0.02

    env = ManagerBasedRLEnv(cfg=cfg)
    obs, _ = env.reset()

    N = args.num_envs
    dev = env.device

    hole_z_top = HOLE_NOMINAL_POS[2] + HOLE_DEPTH / 2

    expert = ScriptedExpertPegV1(
        env,
        hole_pos_b=(HOLE_NOMINAL_POS[0], HOLE_NOMINAL_POS[1], hole_z_top),
        seed=args.seed,
    )
    attach = PegFixedJointAttacher(env, peg_name="object", offset_z=-0.125)
    attach.attach()

    os.makedirs(args.output_dir, exist_ok=True)
    print()
    print("=" * 70)
    print(f"Recording demos: {args.num_episodes} episodes x {N} envs = {args.num_episodes * N} trajectories")
    print(f"Recording until expert.phase <= {args.record_until_phase}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 70, flush=True)

    total_traj = 0
    total_frames = 0

    for ep_idx in range(args.num_episodes):
        # Reset env (re-randomizes hole xy via reset event) and expert state
        obs, _ = env.reset()
        expert.reset()

        # Per-env buffers (lists; converted to arrays at end)
        rgb_buf = [[] for _ in range(N)]
        depth_buf = [[] for _ in range(N)]
        ee_pos_buf = [[] for _ in range(N)]
        ee_quat_buf = [[] for _ in range(N)]
        hole_xy_buf = [[] for _ in range(N)]
        phase_buf = [[] for _ in range(N)]

        for step in range(args.max_steps_per_episode):
            # Per-env hole xy in robot frame (the LABEL — true hole position)
            hole_pos_b = _get_hole_xy_b(env, hole_z_top)

            # Record observations BEFORE step (matches the action that will be taken)
            phase_now = expert.phase.cpu().numpy()  # (N,) int
            recordable = phase_now <= args.record_until_phase  # mask per env
            if recordable.any():
                rgb, depth = _get_camera_rgb_depth(env)
                ee_pos_b, ee_quat_b = _get_ee_pose_b(env)
                hole_xy = hole_pos_b[:, :2].cpu().numpy()  # (N, 2)

                for i in range(N):
                    if recordable[i]:
                        rgb_buf[i].append(rgb[i])  # (H, W, 3) uint8
                        depth_buf[i].append(depth[i])  # (H, W) float32
                        ee_pos_buf[i].append(ee_pos_b[i].cpu().numpy())  # (3,)
                        ee_quat_buf[i].append(ee_quat_b[i].cpu().numpy())  # (4,)
                        hole_xy_buf[i].append(hole_xy[i])  # (2,)
                        phase_buf[i].append(int(phase_now[i]))

            # Step with privileged hole pose (no noise — clean labels)
            action = expert.get_action(obs, hole_pos_b_override=hole_pos_b)
            obs, _, _, _, _ = env.step(action)

        # Save one NPZ per (episode, env)
        for i in range(N):
            T = len(rgb_buf[i])
            if T == 0:
                print(f"  ep{ep_idx:03d}_env{i}: skipped (0 frames)")
                continue
            out_path = os.path.join(args.output_dir, f"ep{ep_idx:03d}_env{i}.npz")
            np.savez_compressed(
                out_path,
                rgb=np.stack(rgb_buf[i], axis=0).astype(np.uint8),         # (T,H,W,3)
                depth=np.stack(depth_buf[i], axis=0).astype(np.float32),    # (T,H,W)
                ee_pos_b=np.stack(ee_pos_buf[i], axis=0).astype(np.float32),  # (T,3)
                ee_quat_b=np.stack(ee_quat_buf[i], axis=0).astype(np.float32),  # (T,4)
                hole_xy_b=np.stack(hole_xy_buf[i], axis=0).astype(np.float32),  # (T,2)
                phase=np.array(phase_buf[i], dtype=np.uint8),              # (T,)
            )
            total_traj += 1
            total_frames += T

        print(f"  ep{ep_idx:03d}: saved {N} trajectories, "
              f"frames per env = {[len(r) for r in rgb_buf]}", flush=True)

    print()
    print("=" * 70)
    print(f"DONE. Saved {total_traj} trajectories, {total_frames} total frames.")
    print(f"Output: {args.output_dir}")
    print("=" * 70, flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
