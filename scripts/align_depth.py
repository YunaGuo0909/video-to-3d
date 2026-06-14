"""
align_depth.py
==============
Three-level depth scale alignment for dn-splatter depth supervision.

The three spaces that must be aligned
--------------------------------------
1. Depth Anything V2 raw output  — arbitrary relative units
2. COLMAP reconstruction space   — up-to-scale metric (arbitrary absolute)
3. nerfstudio normalised space   — unit-sphere normalisation applied at init

If monocular depth maps are passed to dn-splatter without aligning all three
levels, the depth loss gradient will push Gaussians away from the scene
(the "scene explosion" bug described in the DN-Splatter blog discussion).

How this script works
---------------------
1. Read COLMAP sparse PLY → project points into each camera → measure
   median(COLMAP_depth / depth_map_value) → colmap_scale  (Level 1 → 2)
2. Read nerfstudio dataparser_transforms.json → extract scene scale
   (Level 2 → 3)
3. For every depth map:
       depth_aligned = depth_raw * colmap_scale * ns_scale
4. Save aligned maps to <output_dir>/depth_aligned/
5. Update transforms.json to point depth_file_path at depth_aligned/

Usage
-----
    python scripts/align_depth.py \
        --output-dir /transfer/vt3/outputs/room06_quality \
        --dataparser-transforms /transfer/vt3/outputs/room06_quality/nerfstudio/room_reconstruction/splatfacto/<timestamp>/dataparser_transforms.json

After running this script, launch dn-splatter with the same data directory:
    ns-train dn-splatter --data <output_dir> ... nerfstudio-data
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


# ── Level 1→2: estimate COLMAP scale ─────────────────────────────────────────


def estimate_colmap_scale(
    sparse_ply: Path,
    transforms: dict,
    depth_dir: Path,
    n_frames: int = 20,
) -> float:
    """Return scale s.t.  depth_raw * s ≈ COLMAP metric depth.

    Projects sparse COLMAP 3-D points into sampled frames and computes
    the median ratio of projected depth to raw depth-map value.
    """
    from plyfile import PlyData

    plydata = PlyData.read(str(sparse_ply))
    pts = np.stack([
        np.array(plydata["vertex"]["x"], dtype=np.float64),
        np.array(plydata["vertex"]["y"], dtype=np.float64),
        np.array(plydata["vertex"]["z"], dtype=np.float64),
    ], axis=1)

    if len(pts) == 0:
        logger.warning("Sparse PLY is empty — using colmap_scale=1.0")
        return 1.0

    frames = transforms["frames"]
    img_w = int(transforms.get("w", 1))
    img_h = int(transforms.get("h", 1))
    step = max(1, len(frames) // n_frames)
    ratios: list[float] = []

    for frame in frames[::step][:n_frames]:
        stem = Path(frame["file_path"]).stem
        dp = depth_dir / (stem + ".npy")
        if not dp.exists():
            continue

        depth = np.load(str(dp)).astype(np.float64)
        dh, dw = depth.shape

        c2w = np.array(frame["transform_matrix"], dtype=np.float64)
        w2c = np.linalg.inv(c2w)
        fl_x = float(frame.get("fl_x") or transforms.get("fl_x", 1.0))
        fl_y = float(frame.get("fl_y") or transforms.get("fl_y", fl_x))
        cx   = float(frame.get("cx")  or transforms.get("cx", img_w / 2.0))
        cy   = float(frame.get("cy")  or transforms.get("cy", img_h / 2.0))

        # Project sparse points into this camera (OpenGL world → OpenGL cam)
        pts_h  = np.hstack([pts, np.ones((len(pts), 1))])
        pts_gl = (w2c @ pts_h.T).T                    # (M, 4) OpenGL cam

        depth_colmap = -pts_gl[:, 2]                  # positive for in-front
        x_cv = pts_gl[:, 0]
        y_cv = -pts_gl[:, 1]
        z_cv = -pts_gl[:, 2]

        in_front = z_cv > 0.01
        if not np.any(in_front):
            continue

        x_cv, y_cv, z_cv = x_cv[in_front], y_cv[in_front], z_cv[in_front]
        dc = depth_colmap[in_front]

        u_img = cx + fl_x * x_cv / z_cv
        v_img = cy + fl_y * y_cv / z_cv
        u_dm  = (u_img * dw / img_w).astype(int)
        v_dm  = (v_img * dh / img_h).astype(int)

        ok = (u_dm >= 0) & (u_dm < dw) & (v_dm >= 0) & (v_dm < dh)
        if not np.any(ok):
            continue

        dm_vals = depth[v_dm[ok], u_dm[ok]]
        dc_vals = dc[ok]
        good = (dm_vals > 1e-6) & (dc_vals > 0.01)
        if not np.any(good):
            continue

        r = dc_vals[good] / dm_vals[good]
        ratios.extend(r[(r > 1e-4) & (r < 1e4)].tolist())

    if not ratios:
        logger.warning("No COLMAP correspondences found — using colmap_scale=1.0")
        return 1.0

    scale = float(np.median(ratios))
    logger.info("Level 1→2  colmap_scale = %.5f  (%d correspondences)", scale, len(ratios))
    return scale


# ── Level 2→3: read nerfstudio normalisation scale ───────────────────────────


def read_ns_scale(dataparser_transforms_path: Path) -> float:
    """Extract the nerfstudio scene normalisation scale from dataparser_transforms.json.

    nerfstudio stores the scale it applied to camera positions when normalising
    the scene to a unit sphere.  Depth maps must be multiplied by this factor
    so their values are in the same coordinate space as the trained Gaussians.
    """
    with open(dataparser_transforms_path) as f:
        data = json.load(f)
    scale = float(data["scale"])
    logger.info("Level 2→3  ns_scale = %.6f  (from dataparser_transforms.json)", scale)
    return scale


# ── Main alignment + transforms.json update ──────────────────────────────────


def align_and_attach(
    output_dir: Path,
    dataparser_transforms_path: Path,
    depth_raw_dir: Path | None = None,
) -> None:
    depth_raw_dir = depth_raw_dir or (output_dir / "depth_raw")
    transforms_path = output_dir / "transforms.json"
    sparse_ply     = output_dir / "sparse_pt_cloud.ply"

    if not depth_raw_dir.exists():
        raise FileNotFoundError(f"depth_raw/ not found: {depth_raw_dir}")
    if not transforms_path.exists():
        raise FileNotFoundError(f"transforms.json not found: {transforms_path}")
    if not dataparser_transforms_path.exists():
        raise FileNotFoundError(
            f"dataparser_transforms.json not found: {dataparser_transforms_path}"
        )

    with open(transforms_path) as f:
        transforms = json.load(f)

    # ── Scale factors ─────────────────────────────────────────────────────────
    if sparse_ply.exists():
        colmap_scale = estimate_colmap_scale(sparse_ply, transforms, depth_raw_dir)
    else:
        logger.warning("sparse_pt_cloud.ply not found — colmap_scale=1.0")
        colmap_scale = 1.0

    ns_scale    = read_ns_scale(dataparser_transforms_path)
    total_scale = colmap_scale * ns_scale
    logger.info("Total scale applied: %.5f × %.6f = %.6f", colmap_scale, ns_scale, total_scale)

    # ── Write aligned depth maps ──────────────────────────────────────────────
    aligned_dir = output_dir / "depth_aligned"
    aligned_dir.mkdir(exist_ok=True)

    raw_files = sorted(depth_raw_dir.glob("*.npy"))
    if not raw_files:
        raise FileNotFoundError(f"No .npy files in {depth_raw_dir}")

    for src in raw_files:
        depth_raw = np.load(str(src)).astype(np.float32)
        depth_aligned = depth_raw * total_scale
        np.save(str(aligned_dir / src.name), depth_aligned)

    logger.info("Wrote %d aligned depth maps to %s", len(raw_files), aligned_dir)

    # ── Update transforms.json with depth_file_path ───────────────────────────
    aligned_index = {f.stem: f"depth_aligned/{f.name}" for f in sorted(aligned_dir.glob("*.npy"))}
    updated = 0
    for frame in transforms.get("frames", []):
        stem = Path(frame["file_path"]).stem
        if stem in aligned_index:
            frame["depth_file_path"] = aligned_index[stem]
            updated += 1

    with open(transforms_path, "w") as f:
        json.dump(transforms, f, indent=2)

    logger.info(
        "Injected depth_file_path (aligned) into %d / %d frames in transforms.json.",
        updated, len(transforms.get("frames", [])),
    )
    logger.info("Ready for: ns-train dn-splatter --data %s ...", output_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Three-level depth alignment for dn-splatter supervision."
    )
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Pipeline output directory (contains transforms.json).")
    parser.add_argument("--dataparser-transforms", type=Path, required=True,
                        help="Path to nerfstudio dataparser_transforms.json.")
    parser.add_argument("--depth-raw-dir", type=Path, default=None,
                        help="Raw depth maps directory (default: <output_dir>/depth_raw).")
    args = parser.parse_args()

    align_and_attach(
        output_dir=args.output_dir,
        dataparser_transforms_path=args.dataparser_transforms,
        depth_raw_dir=args.depth_raw_dir,
    )


if __name__ == "__main__":
    main()
