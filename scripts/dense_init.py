"""
dense_init.py
=============
Replace the sparse COLMAP point cloud with a dense initialisation derived
from monocular depth maps (Depth Anything V2) and known camera poses.

Problem solved
--------------
3DGS initialises Gaussians at COLMAP sparse feature points, which only cover
textured surfaces.  Flat walls, ceilings and empty space have no initial
points, so Gaussians densify into those regions to explain training-view
colours — producing floaters.

Dense depth-map coverage puts initial Gaussians on *every visible surface*,
leaving far less room for floating artefacts.

What this script does
---------------------
1. Load transforms.json for camera poses and intrinsics.
2. Estimate a depth scale-factor by comparing sparse-PLY depths with
   depth-map values at the same projected pixels.
3. For each frame, back-project a subsampled pixel grid to 3D world space
   using the scaled depth and the camera pose (nerfstudio OpenGL convention).
4. Remove outliers outside 3-sigma of the scene centroid.
5. Subsample to --target-points and write a new sparse_pt_cloud.ply.
   The original is backed up as sparse_pt_cloud_colmap.ply.

Usage
-----
    python scripts/dense_init.py \
        --output-dir /transfer/vt3/outputs/room05_quality \
        --target-points 300000 \
        --stride 16

Then run splatfacto as usual — it will pick up the denser PLY automatically.

Flags
-----
--stride N        Sample every Nth pixel per axis (default: 16).
                  Lower = denser init but slower and more memory.
                  Recommended: 8-32 for 4K video.
--target-points N Subsample final cloud to this size (default: 300 000).
--no-scale        Skip scale estimation and use depth values as-is.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ── Back-projection ───────────────────────────────────────────────────────────


def backproject_depth(
    depth: np.ndarray,
    c2w: np.ndarray,
    fl_x: float,
    fl_y: float,
    cx: float,
    cy: float,
    img_w: int,
    img_h: int,
    stride: int = 16,
) -> np.ndarray:
    """Back-project a depth map to 3D world points.

    Parameters
    ----------
    depth:
        (H, W) float32 array — depth values in COLMAP-aligned metric units.
        Larger value = farther from camera (Z-depth, not Euclidean).
    c2w:
        (4, 4) camera-to-world matrix in nerfstudio OpenGL convention
        (camera looks in -Z, Y up).
    fl_x, fl_y, cx, cy:
        Camera intrinsics at resolution img_w × img_h.
    img_w, img_h:
        Resolution at which intrinsics are defined (may differ from depth
        array shape if the depth model resized the image).
    stride:
        Sample every *stride*-th pixel per axis.

    Returns
    -------
    np.ndarray
        (N, 3) world-space 3D points.
    """
    dh, dw = depth.shape

    # ── Edge-aware flying-pixel mask ──────────────────────────────────────────
    # Pixels at depth discontinuities (object edges) have neighbours with very
    # different depth values.  These "flying pixels" back-project to mid-air
    # positions and are the primary cause of floater artefacts in the init PLY.
    # We exclude pixels where the relative depth gradient exceeds a threshold.
    grad_x = np.zeros_like(depth)
    grad_y = np.zeros_like(depth)
    grad_x[:, :-1] = np.abs(depth[:, 1:] - depth[:, :-1]) / (depth[:, :-1] + 1e-6)
    grad_y[:-1, :] = np.abs(depth[1:, :] - depth[:-1, :]) / (depth[:-1, :] + 1e-6)
    edge_mask = (np.maximum(grad_x, grad_y) < 0.1)  # keep smooth regions only

    us = np.arange(0, dw, stride, dtype=np.float32)
    vs = np.arange(0, dh, stride, dtype=np.float32)
    ug, vg = np.meshgrid(us, vs)
    ug = ug.ravel()
    vg = vg.ravel()

    ui, vi = ug.astype(np.int32), vg.astype(np.int32)
    d = depth[vi, ui]
    valid = (d > 1e-6) & edge_mask[vi, ui]
    ug, vg, d = ug[valid], vg[valid], d[valid]

    # Scale pixel coords to the intrinsics resolution
    u_img = ug * img_w / dw
    v_img = vg * img_h / dh

    # Back-project in OpenCV camera space (Z forward, Y down)
    x_cv = (u_img - cx) / fl_x * d
    y_cv = (v_img - cy) / fl_y * d
    z_cv = d

    # OpenCV → nerfstudio OpenGL camera space (flip Y and Z)
    x_gl = x_cv
    y_gl = -y_cv
    z_gl = -z_cv

    # Camera → world
    pts_cam = np.stack([x_gl, y_gl, z_gl, np.ones_like(z_gl)], axis=1)  # (N, 4)
    pts_world = (c2w @ pts_cam.T).T[:, :3]

    return pts_world.astype(np.float32)


# ── Scale alignment ───────────────────────────────────────────────────────────


def estimate_scale(
    sparse_ply: Path,
    transforms: dict,
    depth_dir: Path,
    n_frames: int = 20,
) -> float:
    """Estimate depth_map * scale ≈ COLMAP metric depth.

    Projects sparse PLY 3D points into sampled frames, reads the depth-map
    value at those projected pixels, and returns the median ratio.
    """
    from plyfile import PlyData

    plydata = PlyData.read(str(sparse_ply))
    pts = np.stack([
        np.array(plydata["vertex"]["x"], dtype=np.float64),
        np.array(plydata["vertex"]["y"], dtype=np.float64),
        np.array(plydata["vertex"]["z"], dtype=np.float64),
    ], axis=1)  # (M, 3)

    if len(pts) == 0:
        return 1.0

    frames = transforms["frames"]
    step = max(1, len(frames) // n_frames)
    sample_frames = frames[::step][:n_frames]

    img_w = int(transforms.get("w", 1))
    img_h = int(transforms.get("h", 1))
    ratios: list[float] = []

    for frame in sample_frames:
        stem = Path(frame["file_path"]).stem
        depth_path = depth_dir / (stem + ".npy")
        if not depth_path.exists():
            continue

        depth = np.load(str(depth_path)).astype(np.float64)
        dh, dw = depth.shape

        c2w = np.array(frame["transform_matrix"], dtype=np.float64)
        w2c = np.linalg.inv(c2w)

        fl_x = float(frame.get("fl_x") or transforms.get("fl_x", 1.0))
        fl_y = float(frame.get("fl_y") or transforms.get("fl_y", fl_x))
        cx   = float(frame.get("cx")  or transforms.get("cx", img_w / 2.0))
        cy   = float(frame.get("cy")  or transforms.get("cy", img_h / 2.0))

        # Transform sparse points to OpenGL camera space
        pts_h = np.hstack([pts, np.ones((len(pts), 1))])
        pts_cam_gl = (w2c @ pts_h.T).T  # (M, 4) OpenGL cam

        # In OpenGL camera space camera looks in -Z; depth = -z_cam
        z_gl = pts_cam_gl[:, 2]
        depth_colmap = -z_gl           # COLMAP metric depth (positive = in front)

        # Convert to OpenCV for projection
        x_cv = pts_cam_gl[:, 0]
        y_cv = -pts_cam_gl[:, 1]
        z_cv = -pts_cam_gl[:, 2]

        in_front = z_cv > 0.01
        if not np.any(in_front):
            continue
        x_cv = x_cv[in_front]
        y_cv = y_cv[in_front]
        z_cv = z_cv[in_front]
        depth_c = depth_colmap[in_front]

        u_img = cx + fl_x * x_cv / z_cv
        v_img = cy + fl_y * y_cv / z_cv

        # Map to depth-map pixel coords
        u_dm = (u_img * dw / img_w).astype(int)
        v_dm = (v_img * dh / img_h).astype(int)

        in_bounds = (u_dm >= 0) & (u_dm < dw) & (v_dm >= 0) & (v_dm < dh)
        if not np.any(in_bounds):
            continue

        dm_vals = depth[v_dm[in_bounds], u_dm[in_bounds]]
        dc_vals = depth_c[in_bounds]

        good = (dm_vals > 1e-6) & (dc_vals > 0.01)
        if not np.any(good):
            continue

        r = dc_vals[good] / dm_vals[good]
        # Filter absurd outliers before accumulating
        ratios.extend(r[(r > 1e-4) & (r < 1e4)].tolist())

    if not ratios:
        logger.warning("Scale estimation had no valid correspondences — using 1.0")
        return 1.0

    scale = float(np.median(ratios))
    logger.info("Depth scale factor: %.5f  (from %d correspondences)", scale, len(ratios))
    return scale


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a dense init PLY from depth maps for 3DGS training."
    )
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Pipeline output directory (contains transforms.json).")
    parser.add_argument("--depth-dir", type=Path, default=None,
                        help="Depth maps directory (default: <output_dir>/depth_raw).")
    parser.add_argument("--stride", type=int, default=16,
                        help="Pixel stride for back-projection (default: 16).")
    parser.add_argument("--target-points", type=int, default=300_000,
                        help="Final subsampled point count (default: 300000).")
    parser.add_argument("--no-scale", action="store_true",
                        help="Skip scale alignment and use raw depth values.")
    args = parser.parse_args()

    output_dir = args.output_dir
    depth_dir  = args.depth_dir or (output_dir / "depth_raw")

    transforms_path = output_dir / "transforms.json"
    sparse_ply_path = output_dir / "sparse_pt_cloud.ply"

    if not transforms_path.exists():
        raise FileNotFoundError(f"transforms.json not found: {transforms_path}")
    if not depth_dir.exists():
        raise FileNotFoundError(f"Depth directory not found: {depth_dir}")

    with open(transforms_path) as f:
        transforms = json.load(f)

    # ── Scale estimation ──────────────────────────────────────────────────────
    if args.no_scale or not sparse_ply_path.exists():
        scale = 1.0
        if not sparse_ply_path.exists():
            logger.warning("sparse_pt_cloud.ply not found — using scale=1.0")
    else:
        scale = estimate_scale(sparse_ply_path, transforms, depth_dir)

    # ── Back-project all frames ───────────────────────────────────────────────
    frames  = transforms["frames"]
    img_w   = int(transforms.get("w", 1))
    img_h   = int(transforms.get("h", 1))
    all_pts: list[np.ndarray] = []

    logger.info("Back-projecting %d depth maps (stride=%d)...", len(frames), args.stride)
    for frame in frames:
        stem = Path(frame["file_path"]).stem
        depth_path = depth_dir / (stem + ".npy")
        if not depth_path.exists():
            continue

        depth  = np.load(str(depth_path)).astype(np.float32) * scale
        c2w    = np.array(frame["transform_matrix"], dtype=np.float64)
        fl_x   = float(frame.get("fl_x") or transforms.get("fl_x", 1.0))
        fl_y   = float(frame.get("fl_y") or transforms.get("fl_y", fl_x))
        cx     = float(frame.get("cx")  or transforms.get("cx", img_w / 2.0))
        cy     = float(frame.get("cy")  or transforms.get("cy", img_h / 2.0))

        pts = backproject_depth(depth, c2w, fl_x, fl_y, cx, cy, img_w, img_h, args.stride)
        all_pts.append(pts)

    if not all_pts:
        raise RuntimeError("No depth maps matched frames in transforms.json.")

    pts = np.vstack(all_pts)
    logger.info("Raw back-projected points: %d", len(pts))

    # ── Outlier removal (3-sigma from centroid) ───────────────────────────────
    centroid = np.median(pts, axis=0)
    dists    = np.linalg.norm(pts - centroid, axis=1)
    keep     = dists < (np.median(dists) + 3.0 * np.std(dists))
    pts      = pts[keep]
    logger.info("After outlier removal: %d points", len(pts))

    # ── Subsample ─────────────────────────────────────────────────────────────
    if len(pts) > args.target_points:
        idx = np.random.default_rng(0).choice(len(pts), args.target_points, replace=False)
        pts = pts[idx]
    logger.info("Final cloud: %d points", len(pts))

    # ── Write PLY ─────────────────────────────────────────────────────────────
    from plyfile import PlyData, PlyElement

    verts = np.empty(
        len(pts),
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")],
    )
    verts["x"] = pts[:, 0]
    verts["y"] = pts[:, 1]
    verts["z"] = pts[:, 2]

    # Back up original sparse PLY before overwriting
    backup_ply = output_dir / "sparse_pt_cloud_colmap.ply"
    if sparse_ply_path.exists() and not backup_ply.exists():
        shutil.copy2(sparse_ply_path, backup_ply)
        logger.info("Backed up original sparse PLY → %s", backup_ply.name)

    PlyData([PlyElement.describe(verts, "vertex")]).write(str(sparse_ply_path))
    logger.info("Dense init PLY written → %s", sparse_ply_path)
    logger.info(
        "Run splatfacto as usual — it will use the denser initialisation automatically."
    )


if __name__ == "__main__":
    main()
