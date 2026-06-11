"""
clean_splat.py
==============
Remove floaters and outer noise shell from a 3DGS PLY file.

Two complementary filters applied in sequence:

1. Distance filter  — removes the outer noise shell.
   Computes each Gaussian's distance from the scene centroid (median position).
   Gaussians beyond  median_dist + distance_factor * std  are discarded.

2. Opacity filter  — removes transparent floaters in empty space.
   3DGS stores raw logit opacity; sigmoid(-4.0) ≈ 0.018 (nearly invisible).
   Gaussians below the threshold contribute nothing visually but add noise.

Usage
-----
    python scripts/clean_splat.py \
        --input  /transfer/vt3/outputs/room04_quality/viz_v2/splat.ply \
        --output /transfer/vt3/outputs/room04_quality/viz_v2/splat_clean.ply

Tune the filters
----------------
    --distance-factor 2.5   # tighter crop (removes more of the outer shell)
    --opacity-thresh  -3.0  # keep slightly more semi-transparent Gaussians
    --distance-factor 4.0   # looser crop (preserve more background detail)
"""

from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path


def clean_splat(
    input_ply: Path,
    output_ply: Path,
    distance_factor: float = 3.0,
    opacity_thresh: float = -4.0,
) -> None:
    """Load a 3DGS PLY, filter outlier Gaussians, and save the result.

    Parameters
    ----------
    input_ply:
        Source PLY file exported by nerfstudio (ns-export gaussian-splat).
    output_ply:
        Destination path for the cleaned PLY.
    distance_factor:
        Controls the outer-shell crop radius.
        Gaussians beyond  median(dist) + distance_factor * std(dist)  are
        removed.  Lower = more aggressive crop.  Default: 3.0 (≈ 3-sigma).
    opacity_thresh:
        Raw logit opacity cutoff.  Gaussians with opacity < threshold are
        removed.  sigmoid(-4.0) ≈ 1.8%.  Lower = keep more floaters.
    """
    from plyfile import PlyData, PlyElement

    print(f"Loading {input_ply} ...")
    plydata = PlyData.read(str(input_ply))
    vertex = plydata["vertex"]
    n_original = len(vertex.data)

    x = np.array(vertex["x"])
    y = np.array(vertex["y"])
    z = np.array(vertex["z"])
    positions = np.stack([x, y, z], axis=1)

    # ── Filter 1: distance from scene centroid ────────────────────────────────
    centroid = np.median(positions, axis=0)
    distances = np.linalg.norm(positions - centroid, axis=1)
    dist_threshold = np.median(distances) + distance_factor * np.std(distances)
    dist_mask = distances < dist_threshold

    # ── Filter 2: opacity (remove transparent floaters) ──────────────────────
    if "opacity" in vertex.data.dtype.names:
        raw_opacity = np.array(vertex["opacity"])
        opacity_mask = raw_opacity > opacity_thresh
    else:
        # PLY may store pre-activated opacity; skip this filter if field absent
        print("  [WARN] 'opacity' field not found — skipping opacity filter")
        opacity_mask = np.ones(n_original, dtype=bool)

    mask = dist_mask & opacity_mask
    n_kept = int(mask.sum())

    print(f"  Original Gaussians   : {n_original:>10,}")
    print(f"  Distance filter (-{(~dist_mask).sum():,}): removed outer shell")
    print(f"  Opacity  filter (-{(~opacity_mask).sum():,}): removed transparent floaters")
    print(f"  Kept                 : {n_kept:>10,}  ({100 * n_kept / n_original:.1f}%)")

    # ── Write filtered PLY ────────────────────────────────────────────────────
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    filtered = vertex.data[mask]
    new_vertex = PlyElement.describe(filtered, "vertex")
    PlyData([new_vertex], text=False).write(str(output_ply))
    print(f"Saved → {output_ply}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean 3DGS PLY: remove outer noise shell and transparent floaters."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input PLY file.")
    parser.add_argument("--output", type=Path, required=True, help="Output cleaned PLY file.")
    parser.add_argument(
        "--distance-factor",
        type=float,
        default=3.0,
        help="Sigma multiplier for outer-shell removal (default: 3.0). Lower = tighter crop.",
    )
    parser.add_argument(
        "--opacity-thresh",
        type=float,
        default=-4.0,
        help="Raw logit opacity cutoff (default: -4.0 ≈ 1.8%% opacity). Lower = keep more.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input PLY not found: {args.input}")

    clean_splat(args.input, args.output, args.distance_factor, args.opacity_thresh)


if __name__ == "__main__":
    main()
