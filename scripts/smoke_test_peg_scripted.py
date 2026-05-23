"""
Full scripted-controller smoke test with privileged hole pose.

Runs the ScriptedExpertPegV1 state machine end-to-end:
  SCAN (100 steps) → APPROACH (80) → DESCEND_TO_CONTACT (60) → INSERT (200) → DONE
Total: ~440 steps + buffer (default --max_steps 500 is comfortable)

Validates:
  1. Peg follows gripper (FixedJoint attachment helper)
  2. Arm visits varied poses during SCAN (z range, xy spread)
  3. Arm reaches above-hole hover, descends, then inserts (with force-feedback
     spiral search compliance if it jams on a wall edge)
  4. Insertion success: peg-center z below hole top AND peg xy within hole bounds

Reports per-env final peg position vs (randomized) per-env hole xy.

Args of interest:
  --num_envs N         number of parallel environments (default 4)
  --seed S             RNG seed (controls SCAN pose sampling + hole randomization)
  --hole_noise_std σ   Gaussian noise (in meters, per axis, per episode) added to
                       the hole xy passed to the expert. Simulates perception
                       error. Default 0.0 = privileged hole pose.
  --enable_cameras     required when wrist camera is in the scene
  --viz kit            opens the Isaac Sim GUI (requires X server / DCV session)
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--max_steps", type=int, default=380)
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--verbose_target_track", action="store_true",
                    help="Print action target vs actual EE every step to diagnose IK convergence")
parser.add_argument("--no_peg_attach", action="store_true",
                    help="Disable peg-attachment helper to isolate whether attachment is interfering")
parser.add_argument("--hole_noise_std", type=float, default=0.0,
                    help="Gaussian noise std (in meters) added to hole xy passed to expert. "
                         "Simulates perception-net prediction error. Per-env, fixed across episode.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import warp as wp  # noqa: E402
import torch  # noqa: E402

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


def main():
    cfg = FrankaPegInsertV1EnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = "cuda:0"
    cfg.seed = args.seed
    cfg.episode_length_s = (args.max_steps + 20) * 0.02

    env = ManagerBasedRLEnv(cfg=cfg)
    obs, _ = env.reset()

    N = args.num_envs
    dev = env.device
    peg = env.scene["object"]

    # Hole top z in robot frame = nominal center z + HOLE_DEPTH / 2
    hole_x, hole_y, hole_z_center = HOLE_NOMINAL_POS
    hole_z_top = hole_z_center + HOLE_DEPTH / 2

    expert = ScriptedExpertPegV1(
        env,
        hole_pos_b=(hole_x, hole_y, hole_z_top),
        seed=args.seed,
    )
    attach = PegFixedJointAttacher(env, peg_name="object", offset_z=-0.125)
    attach.attach()

    # Synthetic perception noise: per-env xy noise added to the hole pose passed
    # to the expert (simulates perception-net prediction error). Sampled once per
    # episode, constant during the episode. Set to 0.0 for privileged hole pose.
    noise_gen = torch.Generator(device=dev).manual_seed(args.seed + 1)
    hole_xy_noise = torch.randn((N, 2), generator=noise_gen, device=dev) * args.hole_noise_std
    if args.hole_noise_std > 0:
        print(f"[noise] applied per-env xy noise (std={args.hole_noise_std*1000:.1f}mm): "
              f"{[(float(hole_xy_noise[i,0])*1000, float(hole_xy_noise[i,1])*1000) for i in range(N)]} mm")

    print()
    print("=" * 70)
    print(f"Peg-in-hole scripted smoke test (N={N}, seed={args.seed})")
    print(f"  Hole nominal: x={hole_x:.3f} y={hole_y:.3f} z_top={hole_z_top:.3f}")
    print("=" * 70, flush=True)

    # Track EE z range during SCAN to verify scan diversity
    scan_ee_z = []
    scan_ee_xy = []

    for step in range(args.max_steps):
        # Read per-env hole position (randomized by reset event)
        hole_base_w = wp.to_torch(env.scene["hole_base"].data.root_pos_w)
        root_pos_w = wp.to_torch(env.scene["robot"].data.root_pos_w)
        root_quat_w = wp.to_torch(env.scene["robot"].data.root_quat_w)
        hole_pos_b_dyn, _ = subtract_frame_transforms(root_pos_w, root_quat_w, hole_base_w)
        # Replace base-z with hole-top-z (controller needs hole entrance, not base)
        hole_pos_b_dyn = hole_pos_b_dyn.clone()
        hole_pos_b_dyn[:, 2] = hole_z_top
        # Apply per-env xy noise (simulates perception error)
        hole_pos_b_dyn[:, :2] += hole_xy_noise

        action = expert.get_action(obs, hole_pos_b_override=hole_pos_b_dyn)
        obs, _, _, _, _ = env.step(action)

        # Read EE pose for diagnostic
        body_state_w = wp.to_torch(env.scene["robot"].data.body_state_w)
        hand_idx = env.scene["robot"].body_names.index("panda_hand")
        ee_pos_w = body_state_w[:, hand_idx, :3]
        root_pos_w = wp.to_torch(env.scene["robot"].data.root_pos_w)
        root_quat_w = wp.to_torch(env.scene["robot"].data.root_quat_w)
        ee_pos_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w)

        # Capture SCAN data (phase 0)
        if (expert.phase == expert.SCAN).all():
            scan_ee_z.append(ee_pos_b[:, 2].clone())
            scan_ee_xy.append(ee_pos_b[:, :2].clone())

        if step % 40 == 0 or step == args.max_steps - 1 or args.verbose_target_track:
            peg_pos_w = wp.to_torch(peg.data.root_pos_w)
            peg_pos_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, peg_pos_w)
            nf_all = wp.to_torch(env.scene["peg_contact"].data.net_forces_w)
            if nf_all.dim() == 3:
                nf_all = nf_all.sum(dim=1)
            print(f"[step {step:3d}] phase={expert.phase.tolist()}  spiral={expert.spiral_step.tolist()}", flush=True)
            for i in range(N):
                print(
                    f"   env[{i}]: peg=({peg_pos_b[i,0]:.3f},{peg_pos_b[i,1]:.3f},{peg_pos_b[i,2]:.3f}) "
                    f"Fz={float(nf_all[i,2]):+6.2f}N",
                    flush=True,
                )

    # ---- SCAN diversity check (was the SCAN phase actually varied?) ----
    if scan_ee_z:
        scan_z_all = torch.stack(scan_ee_z, dim=0)  # (T, N)
        scan_xy_all = torch.stack(scan_ee_xy, dim=0)  # (T, N, 2)
        print()
        print(f"SCAN diversity check (target: z std > 0.03, xy std > 0.005):")
        print(f"  EE z range: [{scan_z_all.min():.3f}, {scan_z_all.max():.3f}]  std: {scan_z_all.std():.4f}")
        print(f"  EE x std: {scan_xy_all[..., 0].std():.4f}")
        print(f"  EE y std: {scan_xy_all[..., 1].std():.4f}")

    # ---- Final success metric (per-env hole xy from randomized hole_base) ----
    peg_final_w = wp.to_torch(peg.data.root_pos_w)
    root_pos_w = wp.to_torch(env.scene["robot"].data.root_pos_w)
    root_quat_w = wp.to_torch(env.scene["robot"].data.root_quat_w)
    peg_final_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, peg_final_w)
    hole_base_w = wp.to_torch(env.scene["hole_base"].data.root_pos_w)
    hole_b_final, _ = subtract_frame_transforms(root_pos_w, root_quat_w, hole_base_w)
    hole_xy_final = hole_b_final[:, :2]  # (N, 2) — per-env hole xy in robot frame

    print(f"\n[diagnostic] peg_final_b: {peg_final_b.tolist()}", flush=True)
    print(f"[diagnostic] hole_xy_final (per-env): {hole_xy_final.tolist()}", flush=True)
    print(f"[diagnostic] hole_z_top: {hole_z_top}", flush=True)

    # Success: peg center within hole xy bounds AND peg center below hole top
    inside_xy = (
        (peg_final_b[:, 0] - hole_xy_final[:, 0]).abs() < HOLE_INNER_HALF_X
    ) & (
        (peg_final_b[:, 1] - hole_xy_final[:, 1]).abs() < HOLE_INNER_HALF_Y
    )
    below_top = peg_final_b[:, 2] < hole_z_top
    success = inside_xy & below_top

    print()
    print("=" * 70)
    print(f"FINAL RESULT (N={N}):")
    print(f"  Inside xy bounds:  {int(inside_xy.sum())}/{N}")
    print(f"  Below hole top z:  {int(below_top.sum())}/{N}")
    print(f"  FULL SUCCESS:      {int(success.sum())}/{N} ({100*success.float().mean():.1f}%)")
    print("=" * 70, flush=True)

    # Per-env breakdown
    print("\nPer-env final peg position vs hole:")
    for i in range(N):
        dx = peg_final_b[i, 0] - hole_xy_final[i, 0]
        dy = peg_final_b[i, 1] - hole_xy_final[i, 1]
        dz = peg_final_b[i, 2] - hole_z_top
        print(
            f"  env[{i}]: hole=({float(hole_xy_final[i,0]):.3f},{float(hole_xy_final[i,1]):.3f}) "
            f"peg=({peg_final_b[i,0]:.3f},{peg_final_b[i,1]:.3f},{peg_final_b[i,2]:.3f}) "
            f"Δ=({dx:+.4f},{dy:+.4f},dz_below={dz:+.4f}) "
            f"success={bool(success[i])}"
        )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
