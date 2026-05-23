"""Quick prim-path inspection for env_0: find panda_hand and Peg actual USD paths."""
from __future__ import annotations
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_camera_env_cfg_peg_v1 import (  # noqa: E402
    FrankaPegInsertV1EnvCfg,
)
import omni.usd  # noqa: E402

cfg = FrankaPegInsertV1EnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = "cuda:0"
cfg.episode_length_s = 5.0
env = ManagerBasedRLEnv(cfg=cfg)
env.reset()

stage = omni.usd.get_context().get_stage()
print("\n=== prim paths in env_0 ===", flush=True)
env0 = stage.GetPrimAtPath("/World/envs/env_0")
for prim in env0.GetAllChildren():
    print(f"  {prim.GetPath()}")
print("\n=== robot children ===", flush=True)
robot = stage.GetPrimAtPath("/World/envs/env_0/Robot")
if robot:
    for prim in robot.GetAllChildren():
        print(f"  {prim.GetPath()}")

# Try to find panda_hand explicitly
print("\n=== search for panda_hand ===", flush=True)
for prim in stage.Traverse():
    p = str(prim.GetPath())
    if "env_0" in p and "panda_hand" in p:
        print(f"  {p}")

env.close()
simulation_app.close()
