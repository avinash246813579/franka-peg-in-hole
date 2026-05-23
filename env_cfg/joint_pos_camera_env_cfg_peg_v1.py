# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Franka peg-in-hole env config.

Inherits the stacking env's Franka Panda, table, and wrist RGB-D camera setup, and
replaces the stacking objects with a square-peg + 4-wall hole receptacle:

  - Peg: dynamic rigid body with collision + gravity. At runtime, a PhysX
    FixedJoint binds it to panda_hand (see peg_fixed_joint_helper.py).
  - Hole: 4 walls + base plate. XY is randomized per episode by a reset event
    (see peg_hole_events.randomize_hole_position). Z is fixed.
  - Chamfer: 3-level stepped funnel above the hole entry that guides off-center
    pegs inward as they descend.

Design notes:
  - Square hole, square peg. Canonical industrial peg-in-hole.
  - Default 1 mm clearance per side (HOLE_INNER_HALF_X = 0.011 with 20 mm peg).
    Tightening only requires editing HOLE_INNER_HALF_X / Y — the chamfer levels
    and wall offsets scale automatically.
  - Keeps the parent env's HIGH_PD Franka config + gripper PD bump.
"""

from __future__ import annotations

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import schemas, spawners
from isaaclab.utils import configclass

# Inherit from the stacking v2 env (gives us Franka, table, wrist camera, observations)
from .joint_pos_camera_env_cfg_stack_v2 import FrankaCubeStackV2EnvCfg

# ============================================================================
# Geometry constants — peg-in-hole specific
# ============================================================================

# Peg (rigidly attached to gripper)
PEG_HALF_SIZE_X = 0.010   # 20mm total width
PEG_HALF_SIZE_Y = 0.010
PEG_HALF_SIZE_Z = 0.025   # 50mm total height
PEG_GRIPPER_OFFSET_Z = 0.10  # peg extends 10cm below gripper tip (tune in sim)

# Hole receptacle (4 walls + bottom plate)
# Default: 1mm clearance per side (peg 20mm wide, hole 22mm wide). Industry-relevant tight tolerance.
HOLE_INNER_HALF_X = 0.011
HOLE_INNER_HALF_Y = 0.011
HOLE_DEPTH = 0.040           # 40mm deep
HOLE_WALL_THICKNESS = 0.010  # 10mm walls
HOLE_BASE_HEIGHT = 0.005     # 5mm base plate

# Hole nominal position — used as the spawn pose; randomized per episode by the
# reset event (see peg_hole_events.randomize_hole_position).
HOLE_NOMINAL_POS = (0.50, -0.075, HOLE_DEPTH / 2)  # workspace center, half-depth-up


@configclass
class FrankaPegInsertV1EnvCfg(FrankaCubeStackV2EnvCfg):
    """Franka peg-in-hole env: dynamic peg via FixedJoint, randomized hole xy, stepped chamfer."""

    # We override v1's gripper PD bump as well since precision insertion benefits from stiff fingers
    GRIPPER_STIFFNESS = 8e3
    GRIPPER_DAMPING = 2.8e2

    def __post_init__(self):
        super().__post_init__()

        # ============================================================
        # Remove cube_a and cube_b from scene (we're not stacking)
        # ============================================================
        # Note: parent env defined self.scene.object (cube A) and self.scene.object_b (cube B)
        # We'll keep "object" name for the PEG (so the obs manager's reference still works)
        # but redefine its shape, and delete object_b.

        if hasattr(self.scene, "object_b"):
            self.scene.object_b = None  # remove cube B

        # ============================================================
        # PEG — dynamic rigid body attached to gripper via PhysX FixedJoint
        # ============================================================
        # The peg is spawned as a dynamic rigid body (kinematic_enabled=False, gravity on,
        # collision on). At runtime, peg_fixed_joint_helper.PegFixedJointAttacher creates
        # a PhysX FixedJoint between the peg and panda_hand. This gives realistic contact
        # forces with hole walls (vs the kinematic-peg variant used in earlier validation).
        # Initial spawn pos is approximately where the joint will hold it, to avoid a
        # large snap impulse on the first physics step.

        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Peg",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.4, 0.0, 0.38),  # approx panda_hand rest pose + offset_z=-0.125
            ),
            spawn=spawners.CuboidCfg(
                size=(PEG_HALF_SIZE_X * 2, PEG_HALF_SIZE_Y * 2, PEG_HALF_SIZE_Z * 2),
                rigid_props=schemas.RigidBodyPropertiesCfg(
                    kinematic_enabled=False,
                    disable_gravity=False,
                ),
                activate_contact_sensors=True,
                mass_props=schemas.MassPropertiesCfg(mass=0.05),
                collision_props=schemas.CollisionPropertiesCfg(),
                visual_material=spawners.PreviewSurfaceCfg(
                    diffuse_color=(0.85, 0.45, 0.1),  # orange peg for visibility in DCV
                ),
            ),
        )

        # ============================================================
        # HOLE receptacle — 4 walls + base, randomized xy per reset
        # ============================================================
        # 5 kinematic RigidObjectCfgs (4 walls + 1 base) forming the hole.
        # Spawned at HOLE_NOMINAL_POS as initial geometry; the reset event
        # (`peg_hole_events.randomize_hole_position` registered below)
        # writes new xy per env at the start of each episode. Hole z is fixed
        # (the hole sits on the table at a fixed depth).

        hole_x, hole_y, hole_z_center = HOLE_NOMINAL_POS

        # Wall +X
        self.scene.hole_wall_xp = self._make_hole_wall(
            "HoleWallXp",
            pos=(hole_x + HOLE_INNER_HALF_X + HOLE_WALL_THICKNESS / 2, hole_y, hole_z_center),
            size=(HOLE_WALL_THICKNESS, (HOLE_INNER_HALF_Y + HOLE_WALL_THICKNESS) * 2, HOLE_DEPTH),
        )
        # Wall -X
        self.scene.hole_wall_xn = self._make_hole_wall(
            "HoleWallXn",
            pos=(hole_x - HOLE_INNER_HALF_X - HOLE_WALL_THICKNESS / 2, hole_y, hole_z_center),
            size=(HOLE_WALL_THICKNESS, (HOLE_INNER_HALF_Y + HOLE_WALL_THICKNESS) * 2, HOLE_DEPTH),
        )
        # Wall +Y
        self.scene.hole_wall_yp = self._make_hole_wall(
            "HoleWallYp",
            pos=(hole_x, hole_y + HOLE_INNER_HALF_Y + HOLE_WALL_THICKNESS / 2, hole_z_center),
            size=(HOLE_INNER_HALF_X * 2, HOLE_WALL_THICKNESS, HOLE_DEPTH),
        )
        # Wall -Y
        self.scene.hole_wall_yn = self._make_hole_wall(
            "HoleWallYn",
            pos=(hole_x, hole_y - HOLE_INNER_HALF_Y - HOLE_WALL_THICKNESS / 2, hole_z_center),
            size=(HOLE_INNER_HALF_X * 2, HOLE_WALL_THICKNESS, HOLE_DEPTH),
        )
        # Base plate
        self.scene.hole_base = self._make_hole_wall(
            "HoleBase",
            pos=(hole_x, hole_y, HOLE_BASE_HEIGHT / 2),
            size=(HOLE_INNER_HALF_X * 2, HOLE_INNER_HALF_Y * 2, HOLE_BASE_HEIGHT),
        )

        # ============================================================
        # Stepped chamfer (simple cuboid rings, no rotation).
        # Three levels above the existing walls. Each level has 4 thin walls
        # with progressively larger inner_half. Creates a stepped funnel
        # that geometrically guides off-center pegs toward the hole center.
        #
        # No rotation math = no geometric interference issues. Trade-off:
        # the chamfer is "stepped" rather than smooth, but functionally
        # equivalent for guiding pegs.
        # ============================================================
        step_h = 0.005            # 5mm per step
        step_t = 0.005            # 5mm radial thickness per step wall
        step_levels = [
            # (inner_half_at_level, level_z_center)
            (HOLE_INNER_HALF_X + 0.005, hole_z_center + HOLE_DEPTH / 2 + 0.5 * step_h),  # +5mm
            (HOLE_INNER_HALF_X + 0.012, hole_z_center + HOLE_DEPTH / 2 + 1.5 * step_h),  # +12mm
            (HOLE_INNER_HALF_X + 0.020, hole_z_center + HOLE_DEPTH / 2 + 2.5 * step_h),  # +20mm
        ]
        for lvl_i, (lvl_ih, lvl_z) in enumerate(step_levels):
            wall_L = 2 * (lvl_ih + step_t)
            setattr(self.scene, f"step_xp_{lvl_i}", self._make_chamfer_wall(
                f"StepXp{lvl_i}",
                pos=(hole_x + lvl_ih + step_t / 2, hole_y, lvl_z),
                size=(step_t, wall_L, step_h),
                rot=(1.0, 0.0, 0.0, 0.0),
            ))
            setattr(self.scene, f"step_xn_{lvl_i}", self._make_chamfer_wall(
                f"StepXn{lvl_i}",
                pos=(hole_x - lvl_ih - step_t / 2, hole_y, lvl_z),
                size=(step_t, wall_L, step_h),
                rot=(1.0, 0.0, 0.0, 0.0),
            ))
            setattr(self.scene, f"step_yp_{lvl_i}", self._make_chamfer_wall(
                f"StepYp{lvl_i}",
                pos=(hole_x, hole_y + lvl_ih + step_t / 2, lvl_z),
                size=(wall_L, step_t, step_h),
                rot=(1.0, 0.0, 0.0, 0.0),
            ))
            setattr(self.scene, f"step_yn_{lvl_i}", self._make_chamfer_wall(
                f"StepYn{lvl_i}",
                pos=(hole_x, hole_y - lvl_ih - step_t / 2, lvl_z),
                size=(wall_L, step_t, step_h),
                rot=(1.0, 0.0, 0.0, 0.0),
            ))

        # ============================================================
        # Peg contact sensor — net contact force on the peg.
        # Since peg is rigidly attached to the wrist via FixedJoint, this
        # is what a real wrist-mounted F/T sensor would feel.
        # ============================================================
        self.scene.peg_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Peg",
            update_period=0.0,
            history_length=1,
        )

        # ============================================================
        # Gripper PD bump (carry over from v1 push #3b)
        # ============================================================
        self.scene.robot.actuators["panda_hand"].stiffness = self.GRIPPER_STIFFNESS
        self.scene.robot.actuators["panda_hand"].damping = self.GRIPPER_DAMPING

        # ============================================================
        # Remove cube_a spawn randomization event (we're not spawning cubes)
        # ============================================================
        if hasattr(self.events, "reset_object_position"):
            self.events.reset_object_position = None

        # ============================================================
        # Randomize hole xy on every episode reset
        # ============================================================
        # Import here to avoid circular import at module load.
        from peg_hole_events import randomize_hole_position

        self.events.randomize_hole = EventTerm(
            func=randomize_hole_position,
            mode="reset",
            params={
                "x_range": (0.40, 0.60),
                "y_range": (-0.20, 0.05),
                "hole_inner_half": HOLE_INNER_HALF_X,
            },
        )

    @staticmethod
    def _make_chamfer_wall(name: str, pos: tuple, size: tuple, rot: tuple) -> RigidObjectCfg:
        return RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
            spawn=spawners.CuboidCfg(
                size=size,
                rigid_props=schemas.RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=schemas.MassPropertiesCfg(mass=1.0),
                visual_material=spawners.PreviewSurfaceCfg(
                    diffuse_color=(0.45, 0.45, 0.5),  # slightly lighter to distinguish from walls
                ),
                collision_props=schemas.CollisionPropertiesCfg(),
            ),
        )

    @staticmethod
    def _make_hole_wall(name: str, pos: tuple, size: tuple) -> RigidObjectCfg:
        return RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
            spawn=spawners.CuboidCfg(
                size=size,
                rigid_props=schemas.RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=schemas.MassPropertiesCfg(mass=1.0),
                visual_material=spawners.PreviewSurfaceCfg(
                    diffuse_color=(0.3, 0.3, 0.35),
                ),
                collision_props=schemas.CollisionPropertiesCfg(),
            ),
        )
