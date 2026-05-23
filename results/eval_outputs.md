# Raw evaluation outputs

This directory documents the raw output from the evaluation sweeps that back the
results table in [`../README.md`](../README.md). All numbers in the README trace
to a corresponding entry here.

## Methodology

- All evaluations run via `scripts/smoke_test_peg_scripted.py`, parallel envs in
  NVIDIA Isaac Sim 5.1 / Isaac Lab 3.0.
- Each cell in the noise table = 5 seeds × 4 parallel envs = 20 independent
  trials.
- Seeds: `{12345, 67890, 11111, 99999, 42}`.
- Hardware: single NVIDIA A10G GPU (24 GB), AWS g5.xlarge instance.
- Clearance: 1 mm per side (`HOLE_INNER_HALF_X = HOLE_INNER_HALF_Y = 0.011`),
  peg 20 mm × 20 mm.
- Stepped chamfer is active (default env config).
- Gripper held partially open (away from the peg) to avoid the finger-pinch
  perturbation identified in [`../docs/iteration-log.md`](../docs/iteration-log.md).

## Noise sweep at 1 mm clearance

Per-episode Gaussian xy noise added to the privileged hole pose passed to the
expert. The privileged hole-xy from sim remains correct; only the value the
controller "sees" is noised.

Results (clean sweep, run 2026-05-23):

| Perception noise σ | Per-seed (12345, 67890, 11111, 99999, 42) | Total | Success |
|---|---|---|---|
| 0 mm  | 4, 4, 4, 4, 4 | 20 | **20/20 (100%)** |
| 1 mm  | 4, 3, 4, 4, 3 | 20 | **18/20 (90%)**  |
| 2 mm  | 3, 3, 4, 4, 2 | 20 | **16/20 (80%)**  |
| 5 mm  | 3, 1, 3, 3, 0 | 20 | 10/20 (50%)      |
| 10 mm | 1, 0, 2, 1, 0 | 20 |  4/20 (20%)      |

Raw log: [`noise_sweep_2026-05-23.txt`](noise_sweep_2026-05-23.txt).

The full sweep took ~30 minutes (~70 seconds per smoke test × 5 seeds × 5 noise levels) on a single A10G.

## Reproduction

To reproduce any cell:

```bash
/isaac-sim/python.sh scripts/smoke_test_peg_scripted.py \
    --num_envs 4 --max_steps 500 --seed <SEED> --enable_cameras \
    --hole_noise_std <NOISE_STD_IN_METERS>
```

The script prints `FULL SUCCESS: X/4` on the final line. Aggregate across the
5 seeds for the 20-trial total.

## Single-seed clearance sweep

Single-seed (12345) clearance validation, with `HOLE_INNER_HALF_X` varied:

| Clearance per side (m) | HOLE_INNER_HALF_X | Success (seed 12345, 4 envs) |
|---|---|---|
| 0.020 | 0.030 | 4/4 |
| 0.010 | 0.020 | 4/4 |
| 0.005 | 0.015 | 4/4 |
| 0.002 | 0.012 | 4/4 |
| 0.001 | 0.011 | 4/4 |

(Reproduces the clearance sweep step. Below 1 mm per side, the precision
ceiling characterized in Phase 7.2 — see iteration log — pushed reliable
operation down to ≤ 20 µm per side. Those tighter-clearance results were
captured during iteration and have not been re-run multi-seed in this
release.)
