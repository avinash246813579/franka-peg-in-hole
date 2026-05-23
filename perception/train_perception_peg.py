"""
Train a small CNN to predict hole xy (in robot frame) from a wrist RGB image
and the current EE pose.

Input:
    rgb       (B, 3, 224, 224)   float in [0, 1]
    ee_pose   (B, 7)             pos_xyz + quat_wxyz in robot frame
Output:
    hole_xy   (B, 2)             hole xy in robot frame

Loss: MSE in meters. Training MAE reported in millimeters (the unit the
controller's noise-tolerance envelope is expressed in).

Trains on .npz files produced by `record_demos_peg.py`. Episode-level
train/val split (no frame leakage between adjacent timesteps).

Usage:
    /isaac-sim/python.sh train_perception_peg.py \
        --data_dir /workspace/imitation/demos_peg \
        --output_ckpt /workspace/imitation/perception_net_peg.pt \
        --epochs 30 --batch_size 64
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmap_dir", default="/workspace/imitation/demos_peg_mmap",
                        help="Directory of memmap dataset (from preprocess_demos_peg.py).")
    parser.add_argument("--output_ckpt", default="/workspace/imitation/perception_net_peg.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_frac", type=float, default=0.2,
                        help="Fraction of source FILES held out for validation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_depth", action="store_true",
                        help="Stack normalized depth as a 4th input channel.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PegPerceptionNet(nn.Module):
    """Small CNN backbone + MLP head conditioned on EE pose.

    ~785k parameters. Designed to run fast on a single GPU (training + inference)
    and to be a clean honest baseline — not a "we threw a transformer at it"
    architecture. The point is to characterize how much precision a single
    wrist camera can give us under the controller's noise tolerance.

    Input channels:
        in_ch=3 → RGB only
        in_ch=4 → RGB + depth (depth as 4th channel)
    """

    def __init__(self, in_ch: int = 3, ee_pose_dim: int = 7, out_dim: int = 2):
        super().__init__()
        self.in_ch = in_ch
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(inplace=True),  # 224 -> 112
            nn.Conv2d(32,    64, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),  # 112 -> 56
            nn.Conv2d(64,   128, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),  # 56  -> 28
            nn.Conv2d(128,  128, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),  # 28  -> 14
            nn.AdaptiveAvgPool2d((4, 4)),                                                      # 14  -> 4
            nn.Flatten(),                                                                      # 128*16 = 2048
        )
        self.head = nn.Sequential(
            nn.Linear(2048 + ee_pose_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 64),                  nn.ReLU(inplace=True),
            nn.Linear(64, out_dim),
        )

    def forward(self, image: torch.Tensor, ee_pose: torch.Tensor) -> torch.Tensor:
        """`image` is (B, in_ch, H, W). RGB-only -> 3 channels; RGB+depth -> 4 channels."""
        feat = self.cnn(image)
        x = torch.cat([feat, ee_pose], dim=-1)
        return self.head(x)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PegDemoMemmapDataset(Dataset):
    """Memmap-backed dataset — reads RGB directly from a single uncompressed file.

    Requires running `preprocess_demos_peg.py` first to convert the per-trajectory
    NPZ files into a single `rgb.dat` memmap + sidecar npy files. Once converted,
    __getitem__ is effectively free (mmap'd random access), making per-epoch time
    GPU-compute-bound rather than I/O-bound.

    `select_file_indices` filters frames to those originating from a subset of
    source NPZ files — used for train/val split at the EPISODE level (no frame
    leakage between adjacent timesteps from the same trajectory).
    """

    def __init__(self, mmap_dir: str, select_file_indices: np.ndarray | None = None,
                 use_depth: bool = False):
        meta = np.load(os.path.join(mmap_dir, "meta.npy"), allow_pickle=True)[0]
        n_total, H, W = meta["n_frames"], meta["height"], meta["width"]
        self.use_depth = use_depth

        ee_all = np.load(os.path.join(mmap_dir, "ee_pose.npy"))
        hole_all = np.load(os.path.join(mmap_dir, "hole_xy.npy"))
        file_idx_all = np.load(os.path.join(mmap_dir, "file_idx.npy"))

        self.rgb_mm = np.memmap(
            os.path.join(mmap_dir, "rgb.dat"),
            dtype=np.uint8, mode="r", shape=(n_total, H, W, 3),
        )
        if use_depth:
            depth_path = os.path.join(mmap_dir, "depth.dat")
            if not os.path.exists(depth_path):
                raise RuntimeError(
                    f"--use_depth requested but {depth_path} not found. "
                    f"Re-run preprocess_demos_peg.py to generate the depth memmap."
                )
            self.depth_mm = np.memmap(depth_path, dtype=np.float32, mode="r", shape=(n_total, H, W))
        else:
            self.depth_mm = None

        if select_file_indices is not None:
            mask = np.isin(file_idx_all, select_file_indices)
            self.frame_indices = np.where(mask)[0].astype(np.int64)
        else:
            self.frame_indices = np.arange(n_total, dtype=np.int64)

        self.ee_poses = ee_all[self.frame_indices]
        self.hole_xys = hole_all[self.frame_indices]
        channels = "RGB+depth" if use_depth else "RGB"
        print(f"  indexed {len(self.frame_indices):>6d} frames ({channels}, "
              f"{len(set(file_idx_all[self.frame_indices].tolist()))} unique source files)")

    def __len__(self) -> int:
        return len(self.frame_indices)

    def __getitem__(self, idx: int):
        global_idx = self.frame_indices[idx]
        rgb = self.rgb_mm[global_idx].astype(np.float32).transpose(2, 0, 1) / 255.0  # (3, H, W)
        if self.use_depth:
            # Normalize depth from [0.05, 2.0] m to [0, 1]; stack as 4th channel
            depth = self.depth_mm[global_idx].astype(np.float32)
            depth = (depth - 0.05) / (2.0 - 0.05)  # → [0, 1]
            depth = depth[None, :, :]  # (1, H, W)
            image = np.concatenate([rgb, depth], axis=0)  # (4, H, W)
        else:
            image = rgb
        return image, self.ee_poses[idx], self.hole_xys[idx]


# Backwards-compatibility alias so older invocations still work
PegDemoDataset = PegDemoMemmapDataset


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    meta_path = os.path.join(args.mmap_dir, "meta.npy")
    if not os.path.exists(meta_path):
        raise RuntimeError(
            f"Memmap dataset not found at {args.mmap_dir}. "
            f"Run preprocess_demos_peg.py first."
        )
    meta = np.load(meta_path, allow_pickle=True)[0]
    n_files = meta["n_files"]
    print(f"Memmap dataset: {n_files} source files, {meta['n_frames']:,} frames "
          f"({meta['height']}x{meta['width']})")

    # Episode-level (source-file-level) train/val split — no frame leakage
    rng = np.random.default_rng(args.seed)
    file_perm = rng.permutation(n_files)
    n_val = max(1, int(n_files * args.val_frac))
    val_file_idxs = file_perm[:n_val]
    train_file_idxs = file_perm[n_val:]
    print(f"Split: train={len(train_file_idxs)} val={len(val_file_idxs)} source files")

    print("Loading train set:")
    train_ds = PegDemoMemmapDataset(args.mmap_dir, select_file_indices=train_file_idxs,
                                     use_depth=args.use_depth)
    print("Loading val set:")
    val_ds = PegDemoMemmapDataset(args.mmap_dir, select_file_indices=val_file_idxs,
                                   use_depth=args.use_depth)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    device = torch.device(args.device)
    in_ch = 4 if args.use_depth else 3
    model = PegPerceptionNet(in_ch=in_ch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print()

    best_val_mae_mm = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_sum_loss = 0.0
        train_sum_mae = 0.0
        train_n = 0
        for rgb, ee, target in train_loader:
            rgb = rgb.to(device, non_blocking=True)
            ee = ee.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            pred = model(rgb, ee)
            loss = F.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            B = rgb.shape[0]
            train_sum_loss += loss.item() * B
            train_sum_mae += (pred - target).abs().mean().item() * B
            train_n += B

        train_mse = train_sum_loss / train_n
        train_mae_mm = (train_sum_mae / train_n) * 1000

        model.eval()
        val_sum_loss = 0.0
        val_sum_mae = 0.0
        val_n = 0
        with torch.no_grad():
            for rgb, ee, target in val_loader:
                rgb = rgb.to(device, non_blocking=True)
                ee = ee.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                pred = model(rgb, ee)
                B = rgb.shape[0]
                val_sum_loss += F.mse_loss(pred, target).item() * B
                val_sum_mae += (pred - target).abs().mean().item() * B
                val_n += B

        val_mse = val_sum_loss / val_n
        val_mae_mm = (val_sum_mae / val_n) * 1000
        dt = time.time() - t0

        is_best = val_mae_mm < best_val_mae_mm
        best_val_mae_mm = min(best_val_mae_mm, val_mae_mm)
        flag = "  <-- best" if is_best else ""
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_mse={train_mse:.6f}  train_mae={train_mae_mm:6.2f} mm  "
              f"val_mse={val_mse:.6f}  val_mae={val_mae_mm:6.2f} mm  "
              f"({dt:.1f}s){flag}",
              flush=True)

        if is_best:
            torch.save({
                "model_state": model.state_dict(),
                "config": {
                    "input_size": 224,
                    "in_ch": in_ch,
                    "use_depth": args.use_depth,
                    "ee_pose_dim": 7,
                    "out_dim": 2,
                    "epoch": epoch,
                    "val_mae_mm": val_mae_mm,
                },
            }, args.output_ckpt)

    print()
    print("=" * 70)
    print(f"DONE. Best val MAE: {best_val_mae_mm:.2f} mm")
    print(f"Saved best checkpoint to {args.output_ckpt}")
    print("=" * 70)


if __name__ == "__main__":
    main()
