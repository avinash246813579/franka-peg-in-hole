# Perception net iterations

Three controlled iterations on the wrist-camera perception net for predicting hole xy in robot frame. **Goal: hit ≤ 2 mm prediction MAE** (the controller's noise tolerance established by the [noise-injection sweep](eval_outputs.md)). **Outcome: all three iterations plateau at ~30 mm val MAE → 0% end-to-end success.**

This is the honest first-pass story. The architecture is data-diversity-limited, not data-quantity- or input-channel-limited. The conclusions point clearly to the next-iteration design.

## Architecture (constant across iterations)

A small CNN (4 conv layers, ~785k params) + MLP head conditioned on EE pose. Input channels and dataset size vary across iterations; everything else is identical.

```
RGB(D) (B, C, 224, 224)  ─→  Conv2d ×4 + ReLU  ─→  AdaptiveAvgPool(4,4)  ─→  Flatten (2048)
                                                                                │
                                       EE pose (B, 7)  ────────────────┐       │
                                                                       ▼       ▼
                                                                  MLP [2048+7 → 256 → 64 → 2]
                                                                                │
                                                                                ▼
                                                                         hole_xy (B, 2)
```

Loss: MSE in meters. Optimizer: Adam, lr=1e-3. Batch size 64 (Iter 1,3) or 32 (Iter 2, after OOM). Train/val split: 80/20 at the episode level (no frame leakage between adjacent timesteps).

## Results

| Iter | Setup | Train MAE | Val MAE | Per-frame deployment MAE | Latched MAE | End-to-end success |
|---|---|---|---|---|---|---|
| **1** | 20 episodes × 4 envs = 80 trajectories, **RGB only** | 30.13 mm | 29.86 mm | 58.56 mm (mean) | 57.13 mm (mean) | **0/20 (0%)** |
| **2** | 100 episodes × 4 envs = 400 trajectories, **RGB only** | 22.92 mm | 32.61 mm | 50.12 mm | 48.97 mm | **0/40 (0%)** |
| **3** | 100 episodes × 4 envs = 400 trajectories, **RGB + depth** | 31.16 mm (epoch 11) | 33.24 mm (epoch 5, best) | 51.12 mm | 49.64 mm | **0/40 (0%)** |

(Iter 3 was stopped at epoch 11/40 — the val MAE trajectory had already plateaued in the same ~30 mm regime and the per-epoch time had blown out to 215 sec because the combined 25 GB RGB+depth memmap no longer fit in the 16 GB instance's OS page cache. The best checkpoint at epoch 5 was used for end-to-end validation.)

## What each iteration tells us

### Iter 1 — End-to-end pipeline plumbed; data-limited at small scale

Pipeline works: scripted privileged controller → 4 envs × 20 episode resets → NPZ trajectories saved → CNN trained → policy plugs net's prediction back into expert via `hole_pos_b_override`. Result: 30 mm val MAE, ~57 mm at deployment, 0% success. The 15× gap from the 2 mm target was expected — only 64 unique hole positions in training. Next obvious move: more data.

### Iter 2 — 5× data does NOT help

100 episodes (5× Iter 1 data, 400 trajectories, 72k frames). Train MAE drops to 22.92 mm (model is fitting harder), but val MAE *worsens slightly* to 32.61 mm. Deployment MAE 50 mm. **5× more frames of the same 100 unique hole positions buys nothing**. Train/val divergence (22 vs 33 mm) confirms the model is starting to memorize seen positions rather than learning to triangulate from arbitrary viewpoints.

Implementation note: Iter 2 needed a memmap-backed dataset (the eager loader OOM'd at 72k frames × 224×224×3 = 11 GB on a 16 GB instance). See `preprocess_demos_peg.py`.

### Iter 3 — Depth doesn't help either

Same 100 episodes, same architecture, but depth stacked as the 4th input channel (normalized 0.05–2.0 m → 0–1). Val MAE still ~33 mm, deployment MAE still ~50 mm. Depth gradient at the hole edge is locally informative but doesn't compensate for the small number of unique hole positions the model has been exposed to.

## Diagnosis

**It's a data *diversity* bottleneck, not data *quantity* or input-modality.** Three independent evidence sources point at this:

1. Train MAE in Iter 2 (22 mm) is meaningfully lower than val MAE (33 mm) → the network *can* fit seen positions, it just doesn't generalize to unseen ones.
2. Val MAE (33 mm) is worse than deployment MAE (~50 mm) — and deployment episodes use a different per-episode seed sequence than train/val. The model gets worse the further the test hole position is from the train distribution.
3. Adding a full extra input channel with completely independent geometric information (depth) produces no improvement, suggesting the bottleneck is not "the model lacks information" but "the model lacks exposure to enough hole-position variation to learn the inverse-projection."

## What the next iteration should try

In rough order of expected payoff:

1. **Vastly more unique hole positions** — 1000+ episodes (10× the current data). Sample new positions per env every reset, not per outer loop. Estimated: ~12 hours of collection on the current AWS instance.
2. **Temporal fusion** — instead of per-frame regression, aggregate features across the full SCAN+APPROACH frame sequence (an LSTM or a small transformer over the ~180-frame context). The hole-pose label is constant over an episode, so the model can effectively triangulate from multiple viewpoints.
3. **Pretrained vision backbone** — replace the 785k-param custom CNN with a frozen ImageNet-pretrained ResNet18 or DINOv2 features. May generalize better even at the same data scale.
4. **Relative-pose target** — predict offset from current EE position to hole xy, instead of absolute hole xy. Should be easier to learn (rotation-equivariant under hole-relative-to-camera transforms).

The end-to-end pipeline (recorder → preprocess → memmap dataset → CNN → policy plug-in) is in place. Future iterations are *just* swapping the model + adding more data, no infra rework needed.

## Reproduce

```bash
# 1. Collect demos (~20 min for 100 episodes × 4 envs on A10G)
python perception/record_demos_peg.py --num_envs 4 --num_episodes 100 --seed 12345 --enable_cameras

# 2. Preprocess to memmap (~5 min)
python perception/preprocess_demos_peg.py --in_dir demos_peg --out_dir demos_peg_mmap

# 3. Train (~25 min for 40 epochs on A10G; add --use_depth for the RGBD variant)
python perception/train_perception_peg.py --mmap_dir demos_peg_mmap --output_ckpt perception_net.pt --epochs 40

# 4. Validate end-to-end (~3 min)
python perception/validate_perception_peg.py --ckpt perception_net.pt --num_envs 4 --num_episodes 10 --enable_cameras
```

Logs from the three iterations are in `train_perception_iter*.log` (omitted from the repo — too large; structure shown in `perception_iterations.md`).
