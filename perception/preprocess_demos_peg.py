"""
One-time preprocessor: converts the per-trajectory compressed NPZ demos into
a single uncompressed RGB memmap plus sidecar arrays.

The NPZ lazy-load Dataset was disk-I/O bound during training (~22 min/epoch
on 100-episode dataset). A memmap RGB file lets the DataLoader workers do
near-zero-cost random access, bringing per-epoch time back down to GPU speed.

Output layout (in --out_dir):
    rgb.dat          uint8  (N_total, H, W, 3)   <- memmap, ~11 GB for 72k frames
    ee_pose.npy      float32 (N_total, 7)
    hole_xy.npy      float32 (N_total, 2)
    phase.npy        uint8   (N_total,)
    file_idx.npy     int32   (N_total,)          per-frame source-file id
    meta.npy         dict (n_files, n_frames, height, width)

Usage:
    /isaac-sim/python.sh preprocess_demos_peg.py \
        --in_dir /workspace/imitation/demos_peg \
        --out_dir /workspace/imitation/demos_peg_mmap
"""

from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", default="/workspace/imitation/demos_peg")
    parser.add_argument("--out_dir", default="/workspace/imitation/demos_peg_mmap")
    return parser.parse_args()


def main():
    args = _parse_args()
    files = sorted(glob.glob(os.path.join(args.in_dir, "ep*_env*.npz")))
    if not files:
        raise RuntimeError(f"No NPZ files found in {args.in_dir}")
    print(f"Found {len(files)} NPZ files in {args.in_dir}")

    # First pass: scan shapes, count total frames
    total = 0
    H, W = None, None
    for fp in files:
        with np.load(fp) as d:
            T, h, w, c = d["rgb"].shape
            assert c == 3, f"Expected 3-channel RGB, got {c}"
            if H is None:
                H, W = h, w
            else:
                assert (h, w) == (H, W), f"Inconsistent shapes: {(h, w)} vs {(H, W)}"
            total += T
    print(f"Total frames: {total:,}, frame shape: ({H}, {W}, 3)")

    os.makedirs(args.out_dir, exist_ok=True)

    rgb_path = os.path.join(args.out_dir, "rgb.dat")
    depth_path = os.path.join(args.out_dir, "depth.dat")
    rgb_mm = np.memmap(rgb_path, dtype=np.uint8, mode="w+", shape=(total, H, W, 3))
    depth_mm = np.memmap(depth_path, dtype=np.float32, mode="w+", shape=(total, H, W))
    ee_all = np.zeros((total, 7), dtype=np.float32)
    hole_all = np.zeros((total, 2), dtype=np.float32)
    phase_all = np.zeros((total,), dtype=np.uint8)
    file_idx_all = np.zeros((total,), dtype=np.int32)

    # Camera clipping range from joint_pos_camera_env_cfg.py — depth values
    # outside this are clipped (inf at far plane -> max distance).
    depth_min, depth_max = 0.05, 2.0

    t0 = time.time()
    offset = 0
    for f_idx, fp in enumerate(files):
        with np.load(fp) as d:
            T = d["rgb"].shape[0]
            rgb_mm[offset:offset + T] = d["rgb"]
            # Depth: drop trailing-1 channel if present, replace inf, clip range
            depth = d["depth"]
            if depth.ndim == 4 and depth.shape[-1] == 1:
                depth = depth[..., 0]
            depth = np.where(np.isfinite(depth), depth, depth_max)
            depth = np.clip(depth, depth_min, depth_max).astype(np.float32)
            depth_mm[offset:offset + T] = depth
            ee_all[offset:offset + T] = np.concatenate(
                [d["ee_pos_b"], d["ee_quat_b"]], axis=-1
            ).astype(np.float32)
            hole_all[offset:offset + T] = d["hole_xy_b"].astype(np.float32)
            phase_all[offset:offset + T] = d["phase"]
            file_idx_all[offset:offset + T] = f_idx
            offset += T
        if (f_idx + 1) % 50 == 0:
            print(f"  processed {f_idx + 1}/{len(files)} files ({offset:,} frames, "
                  f"{time.time() - t0:.1f}s elapsed)", flush=True)

    rgb_mm.flush()
    depth_mm.flush()
    del rgb_mm
    del depth_mm

    np.save(os.path.join(args.out_dir, "ee_pose.npy"), ee_all)
    np.save(os.path.join(args.out_dir, "hole_xy.npy"), hole_all)
    np.save(os.path.join(args.out_dir, "phase.npy"), phase_all)
    np.save(os.path.join(args.out_dir, "file_idx.npy"), file_idx_all)
    meta = {"n_files": len(files), "n_frames": total, "height": H, "width": W}
    np.save(os.path.join(args.out_dir, "meta.npy"), np.array([meta], dtype=object), allow_pickle=True)

    rgb_size_gb = os.path.getsize(rgb_path) / 1e9
    depth_size_gb = os.path.getsize(depth_path) / 1e9
    print()
    print("=" * 70)
    print(f"DONE. Wrote {total:,} frames to {args.out_dir}")
    print(f"  rgb.dat   = {rgb_size_gb:.2f} GB (uint8 RGB)")
    print(f"  depth.dat = {depth_size_gb:.2f} GB (float32 depth, clipped to {depth_min}-{depth_max} m)")
    print(f"  elapsed   = {time.time() - t0:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
