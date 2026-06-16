"""
export_gsplat_ply.py
====================
Export a gsplat checkpoint (.pt) to a standard 3DGS PLY file that can be
viewed in SuperSplat, antimatter15 viewer, or any 3DGS-compatible tool.

Usage
-----
    python scripts/export_gsplat_ply.py \
        --ckpt /path/to/ckpt_29999_rank0.pt \
        --output /path/to/output.ply
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement


def export(ckpt_path: Path, output_path: Path) -> None:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    splats = ckpt["splats"]

    means = splats["means"].numpy()
    scales = splats["scales"].numpy()
    quats = splats["quats"].numpy()
    opacities = splats["opacities"].numpy().reshape(-1)
    sh0 = splats["sh0"].numpy()
    shN = splats.get("shN")
    if shN is not None:
        shN = shN.numpy()

    N = len(means)
    print(f"Gaussians: {N}")
    print(f"  sh0: {sh0.shape}, shN: {shN.shape if shN is not None else 'None'}")

    # Build structured dtype for the standard 3DGS PLY format
    attrs: list[tuple[str, str]] = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
    ]
    n_rest_coeffs = shN.shape[1] * 3 if shN is not None else 0
    for i in range(n_rest_coeffs):
        attrs.append((f"f_rest_{i}", "f4"))
    attrs.append(("opacity", "f4"))
    attrs += [("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4")]
    attrs += [("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4")]

    v = np.empty(N, dtype=attrs)

    # Position
    v["x"], v["y"], v["z"] = means[:, 0], means[:, 1], means[:, 2]
    v["nx"] = v["ny"] = v["nz"] = 0.0

    # Spherical harmonics (DC)
    dc = sh0.reshape(N, 3)
    v["f_dc_0"], v["f_dc_1"], v["f_dc_2"] = dc[:, 0], dc[:, 1], dc[:, 2]

    # Spherical harmonics (higher order)
    if shN is not None:
        rest = shN.reshape(N, -1, 3)
        for i in range(rest.shape[1]):
            v[f"f_rest_{i * 3}"] = rest[:, i, 0]
            v[f"f_rest_{i * 3 + 1}"] = rest[:, i, 1]
            v[f"f_rest_{i * 3 + 2}"] = rest[:, i, 2]

    # Opacity (logit space) and scale (log space)
    v["opacity"] = opacities
    v["scale_0"], v["scale_1"], v["scale_2"] = scales[:, 0], scales[:, 1], scales[:, 2]

    # Rotation quaternion
    v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"] = (
        quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(v, "vertex")]).write(str(output_path))
    print(f"Saved {output_path}  ({output_path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export gsplat checkpoint to 3DGS PLY.")
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to gsplat .pt checkpoint.")
    parser.add_argument("--output", type=Path, required=True, help="Output PLY path.")
    args = parser.parse_args()
    export(args.ckpt, args.output)


if __name__ == "__main__":
    main()
