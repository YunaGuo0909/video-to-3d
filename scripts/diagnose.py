"""
diagnose.py
===========
Inspect a pipeline output directory and report the status of each stage.

Usage
-----
    python scripts/diagnose.py --output /transfer/vt3/outputs/room01

Checks
------
1. Frame extraction  — how many frames were extracted vs rejected
2. Pose estimation   — registered frames, registration rate, pose sanity
3. Sparse PLY        — initialisation point cloud for Gaussian Splatting
4. Training          — latest nerfstudio checkpoint
5. Visualization     — splat.ply and fly-through video in viz/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# ── Per-stage checks ──────────────────────────────────────────────────────────

def check_images(output_dir: Path) -> int:
    images_dir = output_dir / "images"
    if not images_dir.exists():
        _fail("images/ directory not found — frame extraction did not run")
        return 0

    frames = sorted(images_dir.glob("frame_*.jpg")) + sorted(images_dir.glob("frame_*.png"))
    n = len(frames)
    status = "OK" if n >= 50 else "WARN"
    _line(status, f"{n} frames extracted")
    if n < 50:
        _warn(f"< 50 frames — reconstruction will be unstable (recommended ≥ 100)")
    return n


def check_poses(output_dir: Path, total_images: int) -> int:
    transforms_path = output_dir / "transforms.json"
    if not transforms_path.exists():
        _fail("transforms.json not found — pose estimation did not run")
        return 0

    with open(transforms_path) as f:
        data = json.load(f)

    frames = data.get("frames", [])
    n = len(frames)
    rate = n / total_images * 100 if total_images > 0 else 0.0

    status = "OK" if n >= 50 else ("WARN" if n >= 10 else "FAIL")
    _line(status, f"{n} / {total_images} frames registered ({rate:.0f}%)")

    if n < 2:
        _fail("< 2 frames registered — COLMAP failed completely; re-record at higher resolution")
        return n

    if rate < 50:
        _warn("Registration rate < 50% — consider re-recording with slower camera movement")

    # Rotation matrix sanity: determinant should be ~1.0
    first = frames[0].get("transform_matrix")
    if first:
        R = np.array(first)[:3, :3]
        det = float(np.linalg.det(R))
        det_status = "OK" if 0.95 < abs(det) < 1.05 else "WARN"
        _line(det_status, f"rotation determinant: {det:.4f}  (expected ~1.0)")

    fl_x = data.get("fl_x") or frames[0].get("fl_x", 0)
    w = data.get("w", "?")
    h = data.get("h", "?")
    _line("INFO", f"fl_x={fl_x:.1f}  resolution={w}×{h}")

    return n


def check_sparse_ply(output_dir: Path) -> None:
    ply_path = output_dir / "sparse_pt_cloud.ply"
    if not ply_path.exists():
        _warn("sparse_pt_cloud.ply not found — Gaussians will initialise from a random sphere")
        return

    size_kb = ply_path.stat().st_size / 1024
    status = "OK" if size_kb > 10 else "WARN"
    _line(status, f"sparse_pt_cloud.ply  ({size_kb:.0f} KB)")
    if size_kb < 10:
        _warn("PLY is very small — sparse reconstruction may have produced few points")


def check_training(output_dir: Path) -> None:
    nerfstudio_dir = output_dir / "nerfstudio"
    if not nerfstudio_dir.exists():
        _fail("nerfstudio/ not found — training has not run")
        return

    checkpoints = sorted(nerfstudio_dir.glob("**/step-*.ckpt"))
    if not checkpoints:
        _fail("no checkpoints found — training may have crashed")
        return

    latest = checkpoints[-1]
    step_str = latest.stem.replace("step-", "")
    try:
        step = int(step_str)
    except ValueError:
        step = -1

    size_mb = latest.stat().st_size / (1024 * 1024)
    _line("OK", f"latest checkpoint: step {step}  ({size_mb:.0f} MB)")
    _line("INFO", f"  {latest}")


def check_viz(output_dir: Path) -> None:
    viz_dir = output_dir / "viz"
    if not viz_dir.exists():
        _fail("viz/ not found — export_visualization.py has not run")
        return

    ply_path = viz_dir / "splat.ply"
    if not ply_path.exists():
        _fail("viz/splat.ply not found")
    else:
        size_mb = ply_path.stat().st_size / (1024 * 1024)
        _line("OK", f"viz/splat.ply  ({size_mb:.1f} MB)")

    video_path = viz_dir / "flythrough.mp4"
    if video_path.exists():
        size_mb = video_path.stat().st_size / (1024 * 1024)
        _line("OK", f"viz/flythrough.mp4  ({size_mb:.1f} MB)")
    else:
        _line("INFO", "viz/flythrough.mp4 not found (expected in quality mode only)")


# ── Formatting helpers ────────────────────────────────────────────────────────

def _line(status: str, msg: str) -> None:
    tag = f"[{status:<4}]"
    print(f"  {tag}  {msg}")

def _fail(msg: str) -> None:
    _line("FAIL", msg)

def _warn(msg: str) -> None:
    _line("WARN", msg)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose a video-to-3D pipeline output directory.")
    parser.add_argument("--output", type=Path, required=True, help="Pipeline output directory.")
    args = parser.parse_args()

    output_dir = args.output
    if not output_dir.exists():
        print(f"[ERROR] Output directory not found: {output_dir}")
        raise SystemExit(1)

    print(f"\nDiagnostics for: {output_dir}")
    print("=" * 60)

    print("\n[1] Frame Extraction")
    n_images = check_images(output_dir)

    print("\n[2] Pose Estimation")
    n_registered = check_poses(output_dir, n_images)

    print("\n[3] Sparse Point Cloud")
    check_sparse_ply(output_dir)

    print("\n[4] Training")
    check_training(output_dir)

    print("\n[5] Visualization")
    check_viz(output_dir)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
