"""
End-to-end validation: scripted controller driven by perception-net-predicted
hole xy (instead of the privileged sim-truth hole pose).

Architecture during a single episode:
    SCAN     -> at each step, run the perception net on (rgb, ee_pose)
                and update a running average of predicted hole_xy.
    APPROACH -> continue updating the running average. At the END of APPROACH,
                LATCH the running average as the final hole-xy estimate.
    DESCEND  -> peg starts occluding the hole; perception net predictions are
    INSERT      no longer reliable. Use the LATCHED estimate for the rest
                of the episode. (The latched estimate is fed to the expert
                via `hole_pos_b_override` like in the noise-injection sweeps.)

Reports per-env final success and the perception MAE relative to the true hole
position. This is the apples-to-apples comparison to the noise-injection sweep:
  - Noise sweep used SYNTHETIC Gaussian noise on the true hole pose.
  - This script uses REAL noise from a trained perception net.

Usage:
    /isaac-sim/python.sh validate_perception_peg.py \
        --ckpt /workspace/imitation/perception_net_peg.pt \
        --num_envs 4 --num_episodes 5 --seed 12345 --enable_cameras
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default="/workspace/imitation/perception_net_peg.pt")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_episodes", type=int, default=5)
parser.add_argument("--max_steps_per_episode", type=int, default=500)
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--latch_at_phase", type=int, default=2,
                    help="Latch the running-average prediction when expert phase reaches this value "
                         "(default 2 = DESCEND_TO_CONTACT). Predictions stop updating after this.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import warp as wp  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402

from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_camera_env_cfg_peg_v1 import (  # noqa: E402
    FrankaPegInsertV1EnvCfg,
    HOLE_DEPTH,
    HOLE_INNER_HALF_X,
    HOLE_INNER_HALF_Y,
    HOLE_NOMINAL_POS,
)
from scripted_expert_peg_v1 import ScriptedExpertPegV1  # noqa: E402
from peg_fixed_joint_helper import PegFixedJointAttacher  # noqa: E402
from train_perception_peg import PegPerceptionNet  # noqa: E402


def _get_camera_image_torch(env, device, use_depth=False):
    """Read wrist camera RGB (and optionally depth) as one tensor on `device`.
    Returns shape (N, 3, H, W) RGB-only or (N, 4, H, W) RGB+depth, float in [0,1].
    """
    cam = env.scene["wrist_camera"]
    rgb = cam.data.output["rgb"]
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    rgb = rgb.permute(0, 3, 1, 2).float() / 255.0  # (N, 3, H, W)
    if not use_depth:
        return rgb.to(device)
    depth = cam.data.output["distance_to_image_plane"]  # (N, H, W) or (N, H, W, 1)
    if depth.dim() == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    depth = torch.where(torch.isfinite(depth), depth, torch.tensor(2.0, device=depth.device))
    depth = torch.clamp(depth, 0.05, 2.0)
    depth = (depth - 0.05) / (2.0 - 0.05)  # [0,1]
    depth = depth.unsqueeze(1)  # (N, 1, H, W)
    image = torch.cat([rgb, depth], dim=1)  # (N, 4, H, W)
    return image.to(device)


def _get_ee_pose_b_torch(env, device):
    body_state_w = wp.to_torch(env.scene["robot"].data.body_state_w)
    hand_idx = env.scene["robot"].body_names.index("panda_hand")
    ee_pos_w = body_state_w[:, hand_idx, :3]
    ee_quat_w = body_state_w[:, hand_idx, 3:7]
    root_pos_w = wp.to_torch(env.scene["robot"].data.root_pos_w)
    root_quat_w = wp.to_torch(env.scene["robot"].data.root_quat_w)
    ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
    return torch.cat([ee_pos_b, ee_quat_b], dim=-1).to(device)  # (N, 7)


def _get_hole_xy_b_true(env, hole_z_top):
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
    device = torch.device("cuda:0")

    hole_z_top = HOLE_NOMINAL_POS[2] + HOLE_DEPTH / 2

    expert = ScriptedExpertPegV1(
        env,
        hole_pos_b=(HOLE_NOMINAL_POS[0], HOLE_NOMINAL_POS[1], hole_z_top),
        seed=args.seed,
    )
    attach = PegFixedJointAttacher(env, peg_name="object", offset_z=-0.125)
    attach.attach()

    # Load perception net
    print(f"Loading perception net from {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg_ckpt = ckpt.get("config", {})
    in_ch = cfg_ckpt.get("in_ch", 3)
    use_depth = cfg_ckpt.get("use_depth", in_ch == 4)
    model = PegPerceptionNet(in_ch=in_ch).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  ckpt config: {cfg_ckpt}  (use_depth={use_depth})")
    print()

    peg = env.scene["object"]

    total_success = 0
    total_envs = 0
    all_pred_errors_mm = []  # MAE in mm per (episode, env, frame)
    all_latched_errors_mm = []  # final latched error per (episode, env)

    for ep_idx in range(args.num_episodes):
        obs, _ = env.reset()
        expert.reset()

        running_sum = torch.zeros((N, 2), device=device)
        running_count = torch.zeros((N,), device=device)
        latched = torch.zeros((N, 2), device=device)
        latched_done = torch.zeros((N,), dtype=torch.bool, device=device)
        ep_pred_errors = []  # list of (N,) per-step

        # True hole xy at episode start (doesn't change during episode)
        true_hole = _get_hole_xy_b_true(env, hole_z_top)
        true_xy = true_hole[:, :2]

        for step in range(args.max_steps_per_episode):
            # Run perception only during SCAN+APPROACH; otherwise use latched value
            phase_now = expert.phase  # (N,) long, on device
            in_perception_phase = phase_now < args.latch_at_phase  # (N,) bool

            if in_perception_phase.any():
                with torch.no_grad():
                    image = _get_camera_image_torch(env, device, use_depth=use_depth)
                    ee_pose = _get_ee_pose_b_torch(env, device)
                    pred = model(image, ee_pose)  # (N, 2)

                # Update running average only for envs still in perception phase
                m = in_perception_phase.float().unsqueeze(-1)  # (N, 1)
                running_sum = running_sum + m * pred
                running_count = running_count + in_perception_phase.float()

                # Record per-step prediction error in mm
                step_err_mm = (pred - true_xy).abs().mean(dim=-1) * 1000.0  # (N,)
                ep_pred_errors.append(step_err_mm.cpu().numpy())

            # When an env first leaves the perception phase, latch its running average
            newly_latched = (~in_perception_phase) & (~latched_done)
            if newly_latched.any():
                avg = running_sum / running_count.clamp(min=1).unsqueeze(-1)
                # Where newly latched, set latched value
                m = newly_latched.float().unsqueeze(-1)
                latched = latched * (1 - m) + avg * m
                latched_done = latched_done | newly_latched

            # Pick the hole xy to feed the expert this step
            #   - in perception phase: use running average so far (gives some smoothness)
            #   - after latch: use latched value
            avg_so_far = running_sum / running_count.clamp(min=1).unsqueeze(-1)
            m = latched_done.float().unsqueeze(-1)
            hole_xy_for_controller = latched * m + avg_so_far * (1 - m)

            # Build full hole_pos_b override: (xy from perception, z from constant)
            hole_pos_b_override = torch.zeros((N, 3), device=device)
            hole_pos_b_override[:, :2] = hole_xy_for_controller
            hole_pos_b_override[:, 2] = hole_z_top

            action = expert.get_action(obs, hole_pos_b_override=hole_pos_b_override.to(dev))
            obs, _, _, _, _ = env.step(action)

        # Episode-end success check
        peg_final_w = wp.to_torch(peg.data.root_pos_w)
        root_pos_w = wp.to_torch(env.scene["robot"].data.root_pos_w)
        root_quat_w = wp.to_torch(env.scene["robot"].data.root_quat_w)
        peg_final_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, peg_final_w)
        true_final = _get_hole_xy_b_true(env, hole_z_top)
        inside_xy = (
            (peg_final_b[:, 0] - true_final[:, 0]).abs() < HOLE_INNER_HALF_X
        ) & (
            (peg_final_b[:, 1] - true_final[:, 1]).abs() < HOLE_INNER_HALF_Y
        )
        below_top = peg_final_b[:, 2] < hole_z_top
        success = (inside_xy & below_top).cpu().numpy()

        # Final latched perception error per env
        latched_err_mm = (latched - true_xy).abs().mean(dim=-1).cpu().numpy() * 1000.0

        for i in range(N):
            all_latched_errors_mm.append(float(latched_err_mm[i]))
        all_pred_errors_mm.extend([err for arr in ep_pred_errors for err in arr.tolist()])

        ep_success = int(success.sum())
        total_success += ep_success
        total_envs += N

        print(f"  ep{ep_idx:03d}: success {ep_success}/{N}  "
              f"latched_err_mm {[f'{e:.2f}' for e in latched_err_mm]}",
              flush=True)

    print()
    print("=" * 70)
    print(f"FINAL: {total_success}/{total_envs} ({100*total_success/total_envs:.1f}%) "
          f"across {args.num_episodes} episodes x {N} envs at 1mm/side clearance")
    print(f"Perception MAE (per-frame, mm):  mean={np.mean(all_pred_errors_mm):.2f}  "
          f"median={np.median(all_pred_errors_mm):.2f}  "
          f"p90={np.percentile(all_pred_errors_mm, 90):.2f}")
    print(f"Perception MAE (final latched, mm):  mean={np.mean(all_latched_errors_mm):.2f}  "
          f"median={np.median(all_latched_errors_mm):.2f}  "
          f"max={np.max(all_latched_errors_mm):.2f}")
    print("=" * 70, flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
