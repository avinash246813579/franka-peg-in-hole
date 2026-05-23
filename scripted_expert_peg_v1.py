# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Scripted peg-insertion expert with SCAN pre-phase and force-feedback compliance.

Critical design: the SCAN pre-phase explicitly samples STATIC poses across the
deployment viewpoint distribution. A perception net trained on data collected
from this expert will have seen the arm holding still at varied (xy, z) around
the hole, so deployment queries from those poses are in-distribution. (In
predecessor stacking work, training data covered trajectories but deployment
asked the net to predict from static poses — a generalization gap that capped
performance. The SCAN pre-phase is the fix.)

State machine (current phase budgets — see PHASE_STEPS below):
  SCAN              — visit 10 randomized static poses around hole (100 steps total)
  APPROACH          — move above hole at hover height (~10cm above hole top)
  DESCEND_TO_CONTACT — slow descent until contact detected, or near hole top
  INSERT             — push down into hole, with force-feedback spiral search
                       and lift-and-recover compliance when peg jams on wall edge
  DONE               — hold at insertion depth

Scan pose sampling (per-episode, randomized so the perception net learns the
underlying mapping rather than memorizing specific scan positions):
  - z: hole_top + 0.02 to hole_top + 0.15 (13cm range; mirrors deployment hover heights)
  - xy: uniform within ±2cm of hole_xy
  - orientation: down-facing throughout

Compliance: if the peg pushes upward against the wall during INSERT (|F_z| > 2N)
while still above the hole top, the controller treats this as "jammed on wall
edge" and (a) lifts the target z to clear the wall, (b) spirals the xy target
to search for the actual hole opening. See iteration log for the diagnostic
that led to this design and the noise-tolerance envelope it produces.
"""

from __future__ import annotations

import torch
import warp as wp

from isaaclab.utils.math import subtract_frame_transforms


class ScriptedExpertPegV1:
    # Phase IDs
    SCAN = 0
    APPROACH = 1
    DESCEND_TO_CONTACT = 2
    INSERT = 3
    DONE = 4

    PHASE_STEPS = {
        SCAN: 100,                  # 10 poses × 10 steps each — increased for IK convergence
        APPROACH: 80,               # 2x from initial — arm needs time to reach target
        DESCEND_TO_CONTACT: 60,
        INSERT: 200,                # extra budget to allow force-feedback spiral search
        # DONE: arm holds, no budget
    }

    # SCAN parameters
    NUM_SCAN_POSES = 10
    STEPS_PER_SCAN_POSE = 10  # NUM_SCAN_POSES * STEPS_PER_SCAN_POSE must == PHASE_STEPS[SCAN]
    SCAN_Z_MIN_ABOVE_HOLE_TOP = 0.02  # 2cm above hole top
    SCAN_Z_MAX_ABOVE_HOLE_TOP = 0.15  # 15cm above hole top
    SCAN_XY_RANGE = 0.02              # ±2cm around hole xy

    # Insertion geometry (defined in PEG-CENTER frame, then converted to action target)
    APPROACH_HEIGHT_ABOVE_HOLE_TOP = 0.10   # peg-center hovers 10cm above hole top
    DESCEND_HEIGHT_ABOVE_HOLE_TOP = 0.02    # peg-center descends to 2cm above hole top
    INSERT_DEPTH_BELOW_HOLE_TOP = 0.02      # peg-center pushed 2cm below hole top

    # Frame conversion (peg-center z → IK action target z, which is fingertip z)
    # peg is attached 12.5cm below panda_hand (offset_z=-0.125) so peg top sits at fingertip level
    # fingertip is ~10.7cm below panda_hand (measured from prior run)
    # so peg-center is 1.8cm below fingertip → action_target_z = peg_target_z + 0.018
    PEG_TO_ACTION_Z_OFFSET = 0.018

    # Force-feedback compliance — sized for perception noise tolerance.
    # With gripper open (no finger-pinch perturbation), spiral can safely grow past
    # the hole-inner-half because the "climb over wall" failure mode is gone.
    FORCE_THRESHOLD_N = 2.0           # if F_z > 2N (peg pushed up by wall edge), spiral
    SPIRAL_R_BASE = 0.0002            # 0.2mm starting radius
    SPIRAL_R_GROWTH = 0.0002          # 0.2mm growth per spiral step
    SPIRAL_R_MAX = 0.015              # cap at 15mm — covers typical perception noise (1-5mm) + margin
    SPIRAL_OMEGA = 0.4                # rad per spiral step
    # Slow-ramp insert: target_z descends 0.5mm/step from DESCEND end to final insert depth,
    # so the arm doesn't ram into the wall on the phase transition.
    INSERT_Z_RAMP_PER_STEP = 0.0005   # 0.5mm/step downward

    def __init__(
        self,
        env,
        hole_pos_b: tuple[float, float, float] = (0.50, -0.075, 0.020),
        seed: int | None = None,
    ):
        """
        Args:
            env: the Isaac Lab env (ManagerBasedRLEnv).
            hole_pos_b: initial / fallback hole position in robot-root frame
                (x, y, z_of_hole_top). At runtime, the smoke test passes a
                per-env hole pose via `hole_pos_b_override=` in get_action().
                When perception is wired in, this is where the perception net's
                predicted hole pose would be substituted.
            seed: RNG seed for SCAN-pose sampling.
        """
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device

        self.robot = env.scene["robot"]
        self._hand_idx = self.robot.body_names.index("panda_hand")

        # Hole pose (initial / fallback — overridden per-step at runtime)
        self._hole_pos_b = torch.tensor(
            hole_pos_b, device=self.device, dtype=torch.float32
        ).unsqueeze(0).expand(self.num_envs, 3).contiguous()

        # Phase state per env
        self.phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.phase_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Downward-facing quat (180° around x)
        self._down_quat = torch.tensor(
            [0.0, 1.0, 0.0, 0.0], device=self.device
        ).unsqueeze(0).expand(self.num_envs, 4).contiguous()

        # SCAN pose sampling — generate once per episode in reset()
        self._gen = torch.Generator(device=self.device)
        if seed is not None:
            self._gen.manual_seed(seed)
        self._scan_poses: torch.Tensor | None = None  # (N, NUM_SCAN_POSES, 3) xyz targets
        self._sample_scan_poses(torch.arange(self.num_envs, device=self.device))

        # Force-feedback compliance state — spiral search counter (grows while
        # peg is jammed against a wall edge; held when peg is in free air)
        self.spiral_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _sample_scan_poses(self, env_ids: torch.Tensor) -> None:
        """Generate per-env scan poses, uniformly sampled around the hole."""
        n = env_ids.numel()
        if n == 0:
            return
        if self._scan_poses is None:
            self._scan_poses = torch.zeros(
                (self.num_envs, self.NUM_SCAN_POSES, 3), device=self.device
            )
        # Sample (N, NUM_SCAN_POSES, 3): xy uniform in ±SCAN_XY_RANGE, z uniform in z_range
        hole_xy = self._hole_pos_b[env_ids, :2].unsqueeze(1)  # (n, 1, 2)
        hole_z_top = self._hole_pos_b[env_ids, 2].unsqueeze(1)  # (n, 1)

        xy_offset = (
            torch.rand((n, self.NUM_SCAN_POSES, 2), device=self.device, generator=self._gen)
            * 2 * self.SCAN_XY_RANGE - self.SCAN_XY_RANGE
        )
        scan_xy = hole_xy + xy_offset  # (n, NUM_SCAN_POSES, 2)

        z_offset = (
            torch.rand((n, self.NUM_SCAN_POSES), device=self.device, generator=self._gen)
            * (self.SCAN_Z_MAX_ABOVE_HOLE_TOP - self.SCAN_Z_MIN_ABOVE_HOLE_TOP)
            + self.SCAN_Z_MIN_ABOVE_HOLE_TOP
        )
        scan_z = hole_z_top + z_offset  # (n, NUM_SCAN_POSES)

        self._scan_poses[env_ids] = torch.cat(
            [scan_xy, scan_z.unsqueeze(-1)], dim=-1
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.phase[env_ids] = 0
        self.phase_step[env_ids] = 0
        self.spiral_step[env_ids] = 0
        self._sample_scan_poses(env_ids)

    @property
    def all_done(self) -> bool:
        return bool((self.phase >= self.DONE).all().item())

    def get_action(
        self,
        obs_dict: dict | None = None,
        hole_pos_b_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build (N, 8) action: pos (3) + quat (4) + gripper (1).

        Args:
            hole_pos_b_override: if provided, use this hole pose (per-env) instead
                of the stored fallback. This is the hook for swapping in a
                perception net's predicted hole pose at runtime. The smoke test
                uses it to feed either the privileged (sim-truth) hole pose or
                noised versions of it.
        """
        if hole_pos_b_override is not None:
            hole_pos_b = hole_pos_b_override
        else:
            hole_pos_b = self._hole_pos_b

        N = self.num_envs
        dev = self.device
        target_pos = torch.zeros((N, 3), device=dev)
        # gripper fully open: fingers don't touch peg (FixedJoint holds it).
        # Eliminates ~12N background finger-pinch force that perturbs the arm.
        gripper = torch.full((N,), 1.0, device=dev)

        # NOTE: all z targets below are PEG-CENTER positions.
        # We add PEG_TO_ACTION_Z_OFFSET at the end to convert to action-frame (fingertip) target.

        # ---- SCAN: visit static poses around hole ----
        m = self.phase == self.SCAN
        if m.any():
            pose_idx = self.phase_step[m] // self.STEPS_PER_SCAN_POSE
            pose_idx = pose_idx.clamp(0, self.NUM_SCAN_POSES - 1)
            env_ids_m = torch.nonzero(m, as_tuple=True)[0]
            scan_target = self._scan_poses[env_ids_m, pose_idx]  # peg-center frame
            target_pos[m] = scan_target

        # ---- APPROACH: hover above hole ----
        m = self.phase == self.APPROACH
        target_pos[m, 0] = hole_pos_b[m, 0]
        target_pos[m, 1] = hole_pos_b[m, 1]
        target_pos[m, 2] = hole_pos_b[m, 2] + self.APPROACH_HEIGHT_ABOVE_HOLE_TOP

        # ---- DESCEND_TO_CONTACT ----
        m = self.phase == self.DESCEND_TO_CONTACT
        target_pos[m, 0] = hole_pos_b[m, 0]
        target_pos[m, 1] = hole_pos_b[m, 1]
        target_pos[m, 2] = hole_pos_b[m, 2] + self.DESCEND_HEIGHT_ABOVE_HOLE_TOP

        # ---- INSERT (slow-ramp descent + force-feedback spiral search compliance) ----
        # target_z ramps down from where DESCEND left off (hole_z_top + DESCEND_HEIGHT)
        # to final insert depth (hole_z_top - INSERT_DEPTH), at INSERT_Z_RAMP_PER_STEP per step.
        m = self.phase == self.INSERT
        target_pos[m, 0] = hole_pos_b[m, 0]
        target_pos[m, 1] = hole_pos_b[m, 1]
        z_start = hole_pos_b[m, 2] + self.DESCEND_HEIGHT_ABOVE_HOLE_TOP
        z_final = hole_pos_b[m, 2] - self.INSERT_DEPTH_BELOW_HOLE_TOP
        z_ramped = z_start - self.phase_step[m].float() * self.INSERT_Z_RAMP_PER_STEP
        target_pos[m, 2] = torch.maximum(z_ramped, z_final)

        # Force-feedback spiral search: applies during both DESCEND_TO_CONTACT and INSERT
        # (peg can jam on wall edge during the lower part of descend, before INSERT starts).
        m_search = (self.phase == self.INSERT) | (self.phase == self.DESCEND_TO_CONTACT)
        if m_search.any():
            nf = wp.to_torch(self.env.scene["peg_contact"].data.net_forces_w)
            if nf.dim() == 3:
                nf = nf.sum(dim=1)
            upward_force = nf[:, 2]  # +z in world = wall-edge pushing peg up

            peg = self.env.scene["object"]
            peg_pos_w = wp.to_torch(peg.data.root_pos_w)
            root_pos_w = wp.to_torch(self.robot.data.root_pos_w)
            root_quat_w = wp.to_torch(self.robot.data.root_quat_w)
            peg_pos_b_now, _ = subtract_frame_transforms(root_pos_w, root_quat_w, peg_pos_w)

            above_hole = peg_pos_b_now[:, 2] > hole_pos_b[:, 2]
            force_high = upward_force > self.FORCE_THRESHOLD_N
            jammed = m_search & above_hole & force_high

            # Grow spiral counter ONLY when actively jammed; don't reset on un-jam, so search
            # progress is preserved across jam/lift/probe cycles.
            self.spiral_step[jammed] += 1

            # Apply spiral xy offset to ALL envs that have ever activated the search
            # (spiral_step > 0). This keeps the discovered xy when peg briefly clears the wall.
            ever_jammed = (self.spiral_step > 0) & m_search

            r = torch.clamp(
                self.SPIRAL_R_BASE + self.SPIRAL_R_GROWTH * self.spiral_step.float(),
                max=self.SPIRAL_R_MAX,
            )
            theta = self.SPIRAL_OMEGA * self.spiral_step.float()
            target_pos[ever_jammed, 0] += r[ever_jammed] * torch.cos(theta[ever_jammed])
            target_pos[ever_jammed, 1] += r[ever_jammed] * torch.sin(theta[ever_jammed])

            # Lift-and-search: when jammed, lift z so peg_bottom (peg_center - 0.025) is above
            # the wall top (z=0.040 = hole_z_top). Target peg_center at hole_top + 0.040 →
            # peg_bottom at hole_top + 0.015 → clear of wall. Spiral xy continues; once peg is
            # over true hole, force drops, jammed flag clears, z-ramp resumes the descent.
            target_pos[jammed, 2] = hole_pos_b[jammed, 2] + 0.040

        # ---- DONE ----
        m = self.phase >= self.DONE
        target_pos[m, 0] = hole_pos_b[m, 0]
        target_pos[m, 1] = hole_pos_b[m, 1]
        target_pos[m, 2] = hole_pos_b[m, 2] - self.INSERT_DEPTH_BELOW_HOLE_TOP

        # Convert peg-center z target to action-frame (fingertip) z target
        target_pos[:, 2] += self.PEG_TO_ACTION_Z_OFFSET

        action = torch.cat([target_pos, self._down_quat, gripper.unsqueeze(1)], dim=1)

        # Advance phase counters
        self.phase_step += 1
        for phase_id, budget in self.PHASE_STEPS.items():
            advance = (self.phase == phase_id) & (self.phase_step >= budget)
            self.phase[advance] += 1
            self.phase_step[advance] = 0

        return action
