# Sim-Validated Peg-in-Hole Manipulation Pipeline

A NVIDIA Isaac Sim / Isaac Lab pipeline for sub-millimeter peg-in-hole insertion on a Franka Panda arm. Combines a wrist-mounted RGB-D camera, a force-feedback compliant controller, and a stepped-chamfer hole geometry to handle realistic perception error without a perception net trained yet — privileged-state validated as the upper bound for the next stage.

> **Status:** simulation only. Real-hardware transfer and perception net training are the next chunks (see [Roadmap](#roadmap--whats-next)).

---

## TL;DR

- **Privileged-state controller hits 100% success at 1 mm peg-hole clearance** across 20 trials (5 seeds × 4 parallel envs) with randomized hole positions.
- **Precision ceiling characterized to ≤ 20 µm/side clearance** — the controller is not the limit at industrial-relevant tolerances.
- **Noise-injection diagnostic** establishes the compliance envelope *before* training a perception net: **80% success at 2 mm perception noise, 90% at 1 mm noise**. Saves the "train a net you can't use" failure mode.
- **Geometry-first compliance**: a stepped chamfer (industrial-standard funnel entry) adds 5–15 percentage points over tuned algorithmic compliance at the 1–2 mm noise regime that real perception nets occupy.

---

## Overview

The goal of this repo is to demonstrate the *manipulation-side* portion of an industrial peg-in-hole assembly stack — the part that turns a noisy estimate of where a hole is into a successful insertion. The pipeline is built around three ideas:

1. **SCAN-then-INSERT controller**: an explicit "scan" pre-phase visits randomized static poses around the hole before approach. This was the fix for a generalization gap in earlier work (the perception net saw trajectories but was deployed on static viewpoints).
2. **Real-physics fixed-joint peg + force feedback**: the peg is attached to the wrist via a PhysX `FixedJoint` (rigid, dynamic, collidable). A force-feedback spiral search compliance layer activates when the peg jams against a hole wall.
3. **Stepped chamfer**: the hole entry has a 3-level stepped funnel (mechanical guidance) that catches off-center pegs and slides them inward — the industrial standard since the 1970s, often skipped in modern ML manipulation research.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                  NVIDIA Isaac Sim / Isaac Lab                  │
└────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┴──────────────────────┐
        ▼                                              ▼
┌──────────────────────────┐         ┌─────────────────────────────┐
│  Franka Panda (per env)  │         │  Hole receptacle (per env)  │
│                          │         │                             │
│  • Wrist RGB-D camera    │         │  • 4 walls + base plate     │
│    (224 × 224, RealSense │         │  • 3-level stepped chamfer  │
│     D435-style)          │         │  • XY randomized per reset  │
│  • Peg (FixedJoint)      │         │    over workspace bounds    │
│  • Wrist F/T sensor      │         │                             │
└──────────────────────────┘         └─────────────────────────────┘
                                │
                                ▼
        ┌──────────────────────────────────────────────────┐
        │  ScriptedExpertPegV1 (state machine):            │
        │                                                  │
        │    SCAN → APPROACH → DESCEND → INSERT → DONE     │
        │                                                  │
        │  + force-feedback spiral search                  │
        │    (activates when |F_z| > 2 N above hole top)   │
        │  + lift-and-recover override                     │
        │    (lifts target z to clear wall during search)  │
        └──────────────────────────────────────────────────┘
                                │
                                ▼
        ┌──────────────────────────────────────────────────┐
        │  Differential IK action (absolute pose target)   │
        │  → Franka joint commands                          │
        └──────────────────────────────────────────────────┘
```

Action space: 8-D `(pos_xyz, quat_wxyz, gripper)`. The gripper command stays slightly open during insertion (≈ peg width) so the fingers don't apply lateral perturbation to the peg — see Phase 7.2 in [`docs/iteration-log.md`](docs/iteration-log.md).

---

## Results

All numbers below come from running `scripts/smoke_test_peg_scripted.py` with the seed and `--hole_noise_std` shown. Each cell = success rate over 5 seeds × 4 envs = 20 trials, unless stated otherwise.

### Clearance sweep (privileged hole pose, no noise)

Tests how tight the peg-hole clearance can go before the controller breaks. Peg is 20 mm wide; "clearance per side" is the wall gap (e.g., 1 mm/side = 22 mm hole inner).

| Clearance per side | Success (4 envs, seed 12345) |
|---|---|
| 20 mm | 4/4 (100%) |
| 10 mm | 4/4 (100%) |
|  5 mm | 4/4 (100%) |
|  2 mm | 4/4 (100%) |
|  **1 mm**  | **4/4 (100%)** |

After the Phase 7.2 finger-pinch fix (opening the gripper to remove lateral perturbation on the peg), the precision ceiling was characterized below 1 mm/side. The controller continues to succeed at progressively tighter clearances down to **≤ 20 µm per side**. See [`docs/iteration-log.md`](docs/iteration-log.md) for the precision-ceiling characterization and the diagnostic story.

### Hole-randomization robustness (1 mm clearance, no perception noise)

The privileged controller is tested across randomized hole positions per env:

| Seeds | Total trials | Success |
|---|---|---|
| 5 seeds × 4 envs | 20 | **20/20 (100%)** |

Hole-xy range: `x ∈ [0.40, 0.60]`, `y ∈ [-0.20, 0.05]` in robot frame. Seeds: `{12345, 67890, 11111, 99999, 42}`. (Equivalent to the `noise=0 mm` row in the table below — same configuration with zero added perception noise.)

### Noise tolerance (1 mm clearance + stepped chamfer)

Synthetic per-episode Gaussian noise added to the privileged hole xy passed to the expert (simulates perception net prediction error). With stepped chamfer:

| Perception noise σ | Success (5 seeds × 4 envs = 20 trials) |
|---|---|
| 0 mm  | **20/20 (100%)** |
| 1 mm  | **18/20 (90%)** |
| 2 mm  | **16/20 (80%)** |
| 5 mm  | 10/20 (50%) |
| 10 mm | 4/20 (20%) |

**Reading:** for a realistic perception net hitting 1–2 mm precision (achievable with a single wrist camera on a clean indoor scene), the pipeline lands at **80–90% insertion success at 1 mm/side clearance** without a perception net trained yet. Beyond 2 mm of noise, success drops faster — that's the operating envelope where additional vision modalities (overhead camera, multi-view fusion) or a more sophisticated compliance controller (true Cartesian impedance) would be required.

Raw per-seed data is in [`results/eval_outputs.md`](results/eval_outputs.md). Earlier ablations *without* the stepped chamfer bottomed out around 2–3 mm of noise — beyond that, no algorithmic compliance tuning recovered. See [`docs/iteration-log.md`](docs/iteration-log.md) for the chamfer-vs-algorithm ablation.

---

## Setup & reproduce

### Prerequisites

- **NVIDIA Isaac Sim** 5.1.0 (also tested with 5.0.0)
- **Isaac Lab** 3.0 (release branch)
- **Python** 3.10 (bundled with Isaac Sim)
- **CUDA-capable GPU** (tested on NVIDIA A10G, 24 GB VRAM)
- **Ubuntu 22.04 or 24.04** (sim is Linux-first)
- **Docker** with NVIDIA Container Toolkit (recommended; this is how the repo was developed)

### File placement

The env config (`env_cfg/joint_pos_camera_env_cfg_peg_v1.py`) needs to be placed inside the Isaac Lab tasks tree:

```
/path/to/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/joint_pos_camera_env_cfg_peg_v1.py
```

The other Python files (`scripted_expert_peg_v1.py`, `peg_fixed_joint_helper.py`, `peg_hole_events.py`, and `scripts/smoke_test_peg_scripted.py`) should sit together in a working directory; the env config imports `peg_hole_events` from PYTHONPATH at runtime, so add that working directory to PYTHONPATH or run scripts from inside it.

> **Parent-class dependency**: The env config inherits from `FrankaCubeStackV2EnvCfg`, a stacking env from earlier work in the same task tree. That parent (and *its* parent chain back into the Isaac Lab Franka-lift task family) is part of a broader codebase not included in this repo. The env config is published here primarily to show the **design** (peg + chamfer geometry, joint attachment, hole randomization, gripper PD config). To run end-to-end as-is, you'd need either (a) access to the parent stacking env file, or (b) to adapt the inheritance to a public Isaac Lab base task (e.g., `FrankaCubeLiftEnvCfg`) that exposes the same Franka + table + wrist-camera scaffolding. The self-contained pieces are the controller and helpers (`scripted_expert_peg_v1.py`, `peg_fixed_joint_helper.py`, `peg_hole_events.py`) — those work against any compatible env config.

### Running the smoke test

From inside the Isaac Sim container, with the working directory containing the support files:

```bash
/isaac-sim/python.sh scripts/smoke_test_peg_scripted.py \
    --num_envs 4 \
    --max_steps 500 \
    --seed 12345 \
    --enable_cameras \
    --hole_noise_std 0.000
```

To run with synthetic perception noise (units in meters):

```bash
/isaac-sim/python.sh scripts/smoke_test_peg_scripted.py \
    --num_envs 4 --max_steps 500 --seed 12345 --enable_cameras \
    --hole_noise_std 0.002      # 2mm Gaussian noise per axis per episode
```

To view the simulation GUI (requires X server / DCV session):

```bash
/isaac-sim/python.sh scripts/smoke_test_peg_scripted.py \
    --num_envs 4 --max_steps 500 --seed 12345 --enable_cameras \
    --viz kit                   # opens the Isaac Sim Kit window
```

### Tightening clearance

Edit `env_cfg/joint_pos_camera_env_cfg_peg_v1.py`:

```python
HOLE_INNER_HALF_X = 0.011   # 1 mm/side. Set 0.012 for 2mm/side, 0.0105 for 0.5mm/side, etc.
HOLE_INNER_HALF_Y = 0.011
```

The randomization-event geometry inherits this constant, so the chamfer levels and wall offsets all scale together.

---

## Key engineering decisions (and why)

### 1. Why a `FixedJoint`-attached peg instead of a grasped peg?

Real industrial peg-in-hole tasks often start with the peg already in a fixed pose relative to the gripper (e.g., from a feeder or fixture). Modeling this as a rigid `FixedJoint` constraint is closer to the real deployment scenario *and* avoids conflating grasp uncertainty with insertion uncertainty during controller development. The grasp-then-insert variant is planned (see Roadmap).

### 2. Why a SCAN pre-phase?

In earlier stacking work, the perception net was trained on trajectory data and deployed for static-pose prediction — viewpoints the training distribution barely contained. The SCAN pre-phase explicitly visits randomized static poses around the hole during training-data collection, so the perception net sees the deployment-viewpoint distribution. See iteration log for details.

### 3. Why open the gripper during insertion?

Phase 7.2 diagnostic: with the gripper closed around the peg AND the peg attached via `FixedJoint`, two physical attachments fight each other. The fingers apply ~12 N of lateral perturbation that masquerades as a controller precision limit. Opening the gripper (letting the rigid joint hold the peg alone) restored precision from ~50% at 1 mm clearance to 100%, with the precision ceiling pushed down to 20 µm/side.

### 4. Why stepped chamfer (vs smooth slope)?

Two earlier attempts with rotated cuboid blocks (45° tilted) had geometric interference with the existing walls — the rotated geometry didn't stack cleanly. The stepped chamfer (3 horizontal cuboid rings at increasing widths) is simpler, has no rotation math to debug, and works as well as a smooth slope for guiding pegs. When geometry math fights you, simplify the geometry.

---

## Limitations

- **Simulation only.** No real-hardware transfer has been done. The pipeline is structured to support transfer (action mode compatible with Franka FCI / ROS2 ros_franka), but sim-to-real domain randomization and a real-hardware bench-test loop are pending.
- **No trained perception network yet.** Hole pose is read from privileged state (`env.scene["hole_base"].data.root_pos_w`). The noise-injection diagnostic establishes the perception precision requirement (≤ 2 mm σ for demo-grade reliability), but the actual perception net has not yet been trained.
- **Single wrist camera.** The noise-injection result suggests this is sufficient given the chamfer + compliance; multi-view perception (overhead + wrist) would push precision lower but adds infrastructure cost. Validated *post hoc*, not pre-decided.
- **Single peg geometry.** Pegs and holes are uniform across episodes. Geometric variation (peg scaling, hole machining drift) randomization is in the env config skeleton but not exercised in the current sweeps.
- **Gripper is partially open during insertion** — a workaround for the finger-pinch perturbation rather than a "real" grasp. The Phase 7.4 grasp-then-insert pivot is on the roadmap and would replace this.

---

## Roadmap / What's next

In rough order:

1. **Data recording infrastructure.** Capture `(wrist_rgb, depth, ee_pose_b, hole_xy_b)` per step during SCAN+APPROACH phases. Save NPZ per `(episode, env)` to disk.
2. **Perception net training.** Small CNN-based hole-pose regressor. Target: ≤ 2 mm prediction error to clear the compliance threshold.
3. **Sim-to-real transfer.** Deploy on a physical Franka Panda or UR5e cobot. Domain randomization in sim, then 20–50 real demonstrations for fine-tuning.
4. **Grasp-then-insert.** Replace the `FixedJoint`-attached peg with an actual grasp phase. Ports the grasp controller from earlier stacking work; failure modes are well-characterized.
5. **Geometric & material randomization.** Per-episode peg scaling, hole machining noise, friction coefficient, lighting variation. Required for any real-factory deployment.

---

## Tech stack

| Layer | Tool |
|---|---|
| Simulator | NVIDIA Isaac Sim 5.1 |
| Robotics framework | Isaac Lab 3.0 |
| Physics | PhysX 5 (GPU) |
| Robot model | Franka Panda (HIGH_PD config, gripper PD bumped per `joint_pos_camera_env_cfg_peg_v1.py`) |
| Action mode | Differential IK (absolute pose) |
| Perception target | Wrist RGB-D (RealSense D435-style intrinsics, 224 × 224) |
| Force sensing | Built-in wrist F/T via PhysX articulation joint reaction forces |
| Language | Python 3.10 |
| Compute | Single GPU (developed on AWS EC2 g5.xlarge / A10G) |

---

## File index

```
.
├── README.md                              ← this file
├── LICENSE                                ← MIT
├── .gitignore
├── env_cfg/
│   └── joint_pos_camera_env_cfg_peg_v1.py ← env config (drop into IsaacLab task tree)
├── scripted_expert_peg_v1.py              ← state-machine expert + spiral search
├── peg_fixed_joint_helper.py              ← PhysX FixedJoint between peg and panda_hand
├── peg_hole_events.py                     ← reset event: randomize hole xy + chamfer rings
├── scripts/
│   ├── smoke_test_peg_scripted.py         ← driver: runs N envs, prints success metrics
│   └── inspect_prims.py                   ← utility: dumps USD prim paths
├── docs/
│   └── iteration-log.md                   ← phase-by-phase history with lessons
└── results/
    └── (raw eval outputs, written by sweeps)
```

---

## License

MIT. See [`LICENSE`](LICENSE).

## Contact

Avinash Kumar · Trigun AI · [LinkedIn](https://www.linkedin.com/in/avinash-kumar-973b5761/) · avinash@trigunai.com

If you're a robotics or AI hiring team and want to chat about this work or about deployment to your hardware, I'd love to hear from you.
