"""
Hole position randomization event.

Randomizes the hole xy on each reset by writing new poses for all 17 hole
components: 4 main walls + 1 base + 12 stepped chamfer rings (3 levels × 4
sides). z is held fixed (the hole sits on the table at a fixed depth);
only xy varies per episode.

Earlier iterations used 45°-rotated chamfer cuboids — see docs/iteration-log.md
for why the stepped chamfer (identity quats) replaced them. The current
implementation uses identity rotation throughout.

The hole geometry constants must match what's used in
joint_pos_camera_env_cfg_peg_v1.py.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv

# Must match constants in joint_pos_camera_env_cfg_peg_v1.py
_HOLE_INNER_HALF = 0.011
_HOLE_WALL_THICKNESS = 0.010
_HOLE_DEPTH = 0.040
_HOLE_BASE_HEIGHT = 0.005
_HOLE_Z_CENTER = _HOLE_DEPTH / 2  # 0.020

# Stepped chamfer geometry (must match env cfg)
_STEP_H = 0.005
_STEP_T = 0.005
_STEP_LEVELS = [
    (_HOLE_INNER_HALF + 0.005, _HOLE_Z_CENTER + _HOLE_DEPTH / 2 + 0.5 * _STEP_H),
    (_HOLE_INNER_HALF + 0.012, _HOLE_Z_CENTER + _HOLE_DEPTH / 2 + 1.5 * _STEP_H),
    (_HOLE_INNER_HALF + 0.020, _HOLE_Z_CENTER + _HOLE_DEPTH / 2 + 2.5 * _STEP_H),
]


def randomize_hole_position(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    x_range: tuple = (0.40, 0.60),
    y_range: tuple = (-0.20, 0.05),
    hole_inner_half: float | None = None,
):
    """Randomize the hole (4 walls + base) xy position per env on reset.

    Args:
        x_range: (min, max) in robot frame x.
        y_range: (min, max) in robot frame y.
        hole_inner_half: optional override for HOLE_INNER_HALF (in case of clearance sweep).
    """
    if env_ids is None or len(env_ids) == 0:
        return

    inner_half = _HOLE_INNER_HALF if hole_inner_half is None else hole_inner_half
    half_offset = inner_half + _HOLE_WALL_THICKNESS / 2

    n = len(env_ids)
    dev = env.device

    new_x = torch.empty(n, device=dev).uniform_(*x_range)
    new_y = torch.empty(n, device=dev).uniform_(*y_range)

    # Identity quat (w, x, y, z) per env — all components are axis-aligned
    quat_id = torch.zeros((n, 4), device=dev)
    quat_id[:, 0] = 1.0

    # 5 hole components + 12 stepped chamfer rings (3 levels × 4 sides)
    components = [
        ("hole_wall_xp", +half_offset, 0.0, _HOLE_Z_CENTER, quat_id),
        ("hole_wall_xn", -half_offset, 0.0, _HOLE_Z_CENTER, quat_id),
        ("hole_wall_yp", 0.0, +half_offset, _HOLE_Z_CENTER, quat_id),
        ("hole_wall_yn", 0.0, -half_offset, _HOLE_Z_CENTER, quat_id),
        ("hole_base", 0.0, 0.0, _HOLE_BASE_HEIGHT / 2, quat_id),
    ]
    for lvl_i, (lvl_ih, lvl_z) in enumerate(_STEP_LEVELS):
        ring_off = lvl_ih + _STEP_T / 2
        components.extend([
            (f"step_xp_{lvl_i}", +ring_off, 0.0, lvl_z, quat_id),
            (f"step_xn_{lvl_i}", -ring_off, 0.0, lvl_z, quat_id),
            (f"step_yp_{lvl_i}", 0.0, +ring_off, lvl_z, quat_id),
            (f"step_yn_{lvl_i}", 0.0, -ring_off, lvl_z, quat_id),
        ])

    env_origins = env.scene.env_origins[env_ids]  # (n, 3)

    for name, dx, dy, dz, q in components:
        asset = env.scene[name]
        positions = torch.stack([
            new_x + dx + env_origins[:, 0],
            new_y + dy + env_origins[:, 1],
            torch.full((n,), dz, device=dev) + env_origins[:, 2],
        ], dim=1)
        pose = torch.cat([positions, q], dim=1)
        asset.write_root_pose_to_sim(pose, env_ids=env_ids)
        asset.write_root_velocity_to_sim(torch.zeros((n, 6), device=dev), env_ids=env_ids)
