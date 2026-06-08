"""
dataset_builder.py
==================
Validates the processed dataset and confirms it is ready for nerfstudio training.

Responsibilities
----------------
- Verify that every frame listed in transforms.json has a matching image file.
- Warn if the number of registered frames is too low for stable reconstruction.
- Optionally copy/symlink depth maps (from DepthPrior) into the dataset directory
  so that DN-Splatter can consume them during training.

The dataset layout expected by nerfstudio splatfacto:

  <output_dir>/
    images/
      frame_000000.jpg
      frame_000001.jpg
      ...
    transforms.json
    depth/           (optional — populated by DepthPrior)
      frame_000000.npy
      ...
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

MIN_RECOMMENDED_FRAMES = 50


@dataclass
class DatasetStats:
    registered_frames: int
    missing_images: list[str]
    has_depth: bool

    @property
    def is_valid(self) -> bool:
        return len(self.missing_images) == 0 and self.registered_frames > 0

    def summary(self) -> str:
        status = "OK" if self.is_valid else "INVALID"
        return (
            f"[{status}] {self.registered_frames} registered frames | "
            f"{len(self.missing_images)} missing images | "
            f"depth maps: {'yes' if self.has_depth else 'no'}"
        )


class DatasetBuilder:
    """Validate and optionally augment a processed nerfstudio dataset.

    Parameters
    ----------
    output_dir:
        Root of the processed dataset (contains images/ and transforms.json).
    """

    def __init__(self, output_dir: Path | str) -> None:
        self.output_dir = Path(output_dir)

    def validate(self) -> DatasetStats:
        """Check dataset integrity and return a summary.

        Raises
        ------
        FileNotFoundError
            If transforms.json is missing.
        """
        transforms_path = self.output_dir / "transforms.json"
        if not transforms_path.exists():
            raise FileNotFoundError(
                f"transforms.json not found in {self.output_dir}. "
                "Run PoseEstimator.estimate() first."
            )

        with open(transforms_path) as f:
            transforms = json.load(f)

        frames = transforms.get("frames", [])
        missing: list[str] = []

        for frame in frames:
            img_path = self.output_dir / frame["file_path"]
            if not img_path.exists():
                missing.append(frame["file_path"])

        depth_dir = self.output_dir / "depth"
        has_depth = depth_dir.exists() and any(depth_dir.iterdir())

        stats = DatasetStats(
            registered_frames=len(frames),
            missing_images=missing,
            has_depth=has_depth,
        )

        logger.info(stats.summary())

        if missing:
            logger.error("Missing images: %s", missing[:5])

        if stats.registered_frames < MIN_RECOMMENDED_FRAMES:
            logger.warning(
                "Only %d frames registered (recommended ≥ %d). "
                "Reconstruction may be unstable.",
                stats.registered_frames,
                MIN_RECOMMENDED_FRAMES,
            )

        return stats

    def attach_depth_maps(self, depth_dir: Path | str) -> None:
        """Copy depth maps from *depth_dir* into the dataset depth/ directory.

        This is called after DepthPrior.predict() to make depth priors
        available to the splatfacto trainer (via DN-Splatter integration).

        Parameters
        ----------
        depth_dir:
            Source directory containing ``frame_XXXXXX.npy`` depth maps.
        """
        depth_dir = Path(depth_dir)
        dest = self.output_dir / "depth"
        dest.mkdir(exist_ok=True)

        depth_files = sorted(depth_dir.glob("*.npy"))
        if not depth_files:
            logger.warning("No .npy depth maps found in %s", depth_dir)
            return

        for src in depth_files:
            shutil.copy2(src, dest / src.name)

        logger.info("Attached %d depth maps to dataset.", len(depth_files))
