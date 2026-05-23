"""
Attach the peg to panda_hand via a PhysX FixedJoint at startup.

This creates a real rigid-body attachment between the peg and the gripper.
After attach() is called, the peg is a dynamic rigid body rigidly coupled to
the gripper hand link — it has gravity, collides with the hole walls, and
feeds contact forces back through the arm for wrist F/T sensing.

This replaces the simpler kinematic-pose-write variant used in initial
controller-only validation, where the peg's pose was just driven to match
the gripper every step (no physics). The kinematic variant was useful for
isolating controller bugs from physics interactions; the FixedJoint variant
here is what's used for everything downstream — clearance sweeps, noise
diagnostic, and the eventual sim-to-real path.
"""

from __future__ import annotations

import torch
import warp as wp

import omni.usd
from pxr import Gf, Sdf, UsdPhysics


class PegFixedJointAttacher:
    def __init__(
        self,
        env,
        peg_name: str = "object",
        hand_link_name: str = "panda_hand",
        offset_z: float = -0.125,  # peg top at fingertip level, peg extends below for insertion
    ):
        self.env = env
        self.peg_name = peg_name
        self.hand_link_name = hand_link_name
        self.offset_z = offset_z
        self.num_envs = env.num_envs

    def _align_peg_to_hand(self):
        """One-shot pose-write so peg starts where the joint will hold it.
        Applies offset along the hand link's local +Z axis (finger-pointing direction),
        not world Z — so peg is correctly placed regardless of hand orientation.
        """
        robot = self.env.scene["robot"]
        peg = self.env.scene[self.peg_name]
        hand_idx = robot.body_names.index(self.hand_link_name)

        body_state_w = wp.to_torch(robot.data.body_state_w)
        hand_pos_w = body_state_w[:, hand_idx, :3]
        hand_quat_w = body_state_w[:, hand_idx, 3:7]  # (w, x, y, z)

        # Rotate +Z_hand by hand_quat to get the world-space finger-axis direction
        # quat (w,x,y,z) rotating local-z (0,0,1) gives (2*(xz+wy), 2*(yz-wx), 1-2*(x^2+y^2))
        w, x, y, z = hand_quat_w[:, 0], hand_quat_w[:, 1], hand_quat_w[:, 2], hand_quat_w[:, 3]
        finger_dir_w = torch.stack([
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        ], dim=1)
        # Move peg along +finger_dir by abs(offset_z) so it hangs out the fingers
        peg_pos_w = hand_pos_w + finger_dir_w * abs(self.offset_z)

        zeros = torch.zeros((self.num_envs, 6), device=self.env.device)
        peg_root_state = torch.cat([peg_pos_w, hand_quat_w, zeros], dim=-1)
        env_ids = torch.arange(self.num_envs, device=self.env.device)
        peg.write_root_state_to_sim(peg_root_state, env_ids=env_ids)

    def attach(self):
        """Create FixedJoint between peg and panda_hand for each env. Call once."""
        self._align_peg_to_hand()

        stage = omni.usd.get_context().get_stage()
        created = 0
        for i in range(self.num_envs):
            joint_path = f"/World/envs/env_{i}/PegFixedJoint"
            peg_path = f"/World/envs/env_{i}/Peg"
            hand_path = f"/World/envs/env_{i}/Robot/{self.hand_link_name}"

            if stage.GetPrimAtPath(joint_path):
                continue
            if not stage.GetPrimAtPath(peg_path):
                raise RuntimeError(f"Peg prim not found at {peg_path}")
            if not stage.GetPrimAtPath(hand_path):
                raise RuntimeError(f"Hand prim not found at {hand_path}")

            joint = UsdPhysics.FixedJoint.Define(stage, Sdf.Path(joint_path))
            joint.CreateBody0Rel().SetTargets([Sdf.Path(hand_path)])
            joint.CreateBody1Rel().SetTargets([Sdf.Path(peg_path)])

            # +Z in panda_hand's local frame points along the finger axis (toward workpiece),
            # so we use +abs(offset_z) here. Sign of offset_z parameter is preserved for the
            # user-facing convention "peg hangs |offset_z| out from the hand toward the fingers".
            joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, abs(self.offset_z)))
            joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

            joint.CreateCollisionEnabledAttr().Set(False)
            joint.CreateBreakForceAttr().Set(1e20)
            joint.CreateBreakTorqueAttr().Set(1e20)

            created += 1

        print(f"[PegFixedJointAttacher] Created {created} FixedJoints (offset_z={self.offset_z}).")
