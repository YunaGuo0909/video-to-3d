"""
export_visualization.py
========================
Export reconstruction artifacts and render a fly-through camera path.

Steps
-----
1. Read the trained nerfstudio experiment directory.
2. Export a ``splat.ply`` (Gaussian point cloud) for external viewers.
3. Render a camera trajectory (spiral or existing) to produce a demo video.

Usage
-----
    uv run python scripts/export_visualization.py \
        --experiment-dir outputs/nerfstudio/splatfacto/room_reconstruction/<timestamp> \
        --output-dir outputs/viz

Requirements
------------
    nerfstudio installed (ns-export, ns-render)
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def export_ply(experiment_dir: Path, output_dir: Path) -> Path:
    """Export 3DGS Gaussians to a PLY file via ns-export."""
    config = _find_config(experiment_dir)
    ply_path = output_dir / "splat.ply"
    ply_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ns-export", "gaussian-splat",
        "--load-config", str(config),
        "--output-dir", str(output_dir),
    ]
    logger.info("Exporting PLY: %s", " ".join(cmd))
    _run(cmd)

    logger.info("PLY saved to %s", ply_path)
    return ply_path


def render_video(experiment_dir: Path, output_dir: Path, seconds: float = 8.0) -> Path:
    """Render a spiral fly-through video using nerfstudio's ns-render.

    Parameters
    ----------
    experiment_dir:
        Path to the nerfstudio experiment (contains config.yml).
    output_dir:
        Destination directory for the rendered video.
    seconds:
        Duration of the rendered fly-through in seconds.

    Returns
    -------
    Path
        Path to the rendered ``flythrough.mp4``.
    """
    config = _find_config(experiment_dir)
    video_path = output_dir / "flythrough.mp4"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ns-render", "spiral",
        "--load-config", str(config),
        "--output-path", str(video_path),
        "--seconds", str(seconds),
        "--output-format", "video",
    ]
    logger.info("Rendering fly-through: %s", " ".join(cmd))
    _run(cmd)

    logger.info("Fly-through saved to %s", video_path)
    return video_path


# ── Internal helpers ──────────────────────────────────────────────────────────


def _find_config(experiment_dir: Path) -> Path:
    configs = list(experiment_dir.glob("**/config.yml"))
    if not configs:
        raise FileNotFoundError(f"No config.yml found under {experiment_dir}")
    return configs[0]


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Export 3DGS visualization artifacts.")
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="nerfstudio experiment directory (contains config.yml).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/viz"),
        help="Where to write PLY and fly-through video.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=8.0,
        help="Duration of the fly-through video in seconds.",
    )
    parser.add_argument(
        "--skip-ply",
        action="store_true",
        help="Skip PLY export (only render video).",
    )
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Skip video render (only export PLY).",
    )
    args = parser.parse_args()

    if not args.skip_ply:
        export_ply(args.experiment_dir, args.output_dir)

    if not args.skip_video:
        render_video(args.experiment_dir, args.output_dir, seconds=args.seconds)


if __name__ == "__main__":
    main()
