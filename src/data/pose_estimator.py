"""
pose_estimator.py
=================
Camera pose estimation from a set of extracted frames.

Supported backends
------------------
MASt3R-SfM (primary)
    Feed-forward transformer-based SfM (ICLR 2025). Robust to textureless
    indoor surfaces and requires no camera calibration. Outputs camera poses
    and sparse point cloud directly from image pairs.
    Reference: Duisterhof et al., ICLR 2025 — https://github.com/naver/mast3r

COLMAP (fallback)
    Classic SfM via nerfstudio's ``ns-process-data video`` command.
    More sensitive to textureless regions but widely deployed and well-tested.

Both backends produce a ``transforms.json`` file in the nerfstudio format,
which is then consumed by ``DatasetBuilder`` and the splatfacto trainer.
"""

from __future__ import annotations

import json
import logging
import subprocess
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class PoseBackend(str, Enum):
    MAST3R = "mast3r"    # Recommended: ICLR 2025 SOTA
    COLMAP = "colmap"    # Fallback: classic SfM via nerfstudio


class PoseEstimator:
    """Estimate camera poses from a directory of extracted frames.

    Parameters
    ----------
    backend:
        Which pose estimation backend to use (see ``PoseBackend``).
    mast3r_repo:
        Path to a local clone of https://github.com/naver/mast3r.
        Required only when ``backend=PoseBackend.MAST3R``.
    colmap_executable:
        Path to the COLMAP binary. ``"colmap"`` assumes it is on ``$PATH``.
    verbose:
        Forward subprocess stdout/stderr to the logger.
    """

    def __init__(
        self,
        backend: PoseBackend = PoseBackend.MAST3R,
        mast3r_repo: Path | str | None = None,
        colmap_executable: str = "colmap",
        verbose: bool = True,
    ) -> None:
        self.backend = backend
        self.mast3r_repo = Path(mast3r_repo) if mast3r_repo else None
        self.colmap_executable = colmap_executable
        self.verbose = verbose

    # ── Public interface ─────────────────────────────────────────────────────

    def estimate(self, images_dir: Path | str, output_dir: Path | str) -> Path:
        """Run pose estimation and write ``transforms.json`` to *output_dir*.

        Parameters
        ----------
        images_dir:
            Directory containing extracted JPEG/PNG frames.
        output_dir:
            Root directory of the processed dataset.
            ``transforms.json`` is written here.

        Returns
        -------
        Path
            Absolute path to the generated ``transforms.json``.
        """
        images_dir = Path(images_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Pose estimation backend: %s", self.backend.value)

        if self.backend == PoseBackend.MAST3R:
            return self._run_mast3r(images_dir, output_dir)
        else:
            return self._run_colmap(images_dir, output_dir)

    # ── MASt3R-SfM backend ───────────────────────────────────────────────────

    def _run_mast3r(self, images_dir: Path, output_dir: Path) -> Path:
        """Run MASt3R-SfM and convert output to nerfstudio transforms.json.

        MASt3R-SfM outputs a reconstruction directory containing:
          - scene.npz : camera poses (c2w matrices) + intrinsics + points3d

        We read scene.npz and convert it to the nerfstudio transforms.json
        format expected by ``ns-train splatfacto``.
        """
        if self.mast3r_repo is None:
            raise ValueError(
                "mast3r_repo must be set to the path of a local MASt3R clone "
                "when using PoseBackend.MAST3R. "
                "Clone from: https://github.com/naver/mast3r"
            )

        mast3r_out = output_dir / "mast3r_raw"
        mast3r_out.mkdir(exist_ok=True)

        # MASt3R-SfM entry point: demo/demo_mast3r_sfm.py
        script = self.mast3r_repo / "demo" / "demo_mast3r_sfm.py"
        cmd = [
            "python", str(script),
            "--image_dir", str(images_dir),
            "--output_dir", str(mast3r_out),
            "--device", "cuda",
        ]

        logger.info("Running MASt3R-SfM: %s", " ".join(cmd))
        self._run_subprocess(cmd, cwd=self.mast3r_repo)

        # Convert MASt3R output → nerfstudio transforms.json
        transforms_path = output_dir / "transforms.json"
        self._convert_mast3r_to_nerfstudio(mast3r_out, images_dir, transforms_path)

        logger.info("transforms.json written to %s", transforms_path)
        return transforms_path

    def _convert_mast3r_to_nerfstudio(
        self,
        mast3r_out: Path,
        images_dir: Path,
        transforms_path: Path,
    ) -> None:
        """Parse MASt3R scene.npz and write nerfstudio transforms.json.

        MASt3R stores poses as camera-to-world (c2w) 4x4 matrices in the
        OpenCV coordinate convention (X right, Y down, Z forward).
        nerfstudio splatfacto uses the same convention, so no axis flip
        is needed — we copy the matrices directly.
        """
        scene_file = mast3r_out / "scene.npz"
        if not scene_file.exists():
            raise FileNotFoundError(
                f"MASt3R output not found: {scene_file}. "
                "Check that the MASt3R script ran successfully."
            )

        data = np.load(str(scene_file), allow_pickle=True)

        # MASt3R npz keys: "poses" (N,4,4), "intrinsics" (N,3,3), "image_paths" (N,)
        poses: np.ndarray = data["poses"]          # c2w matrices, shape (N, 4, 4)
        intrinsics: np.ndarray = data["intrinsics"] # K matrices, shape (N, 3, 3)
        image_paths: list[str] = list(data["image_paths"])

        # Use the first frame's intrinsics as the shared intrinsic (valid for
        # fixed-focal videos; per-frame intrinsics are written if they differ).
        K0 = intrinsics[0]
        h, w = self._read_image_hw(images_dir / Path(image_paths[0]).name)

        frames: list[dict[str, Any]] = []
        for img_path, pose, K in zip(image_paths, poses, intrinsics):
            frame_name = Path(img_path).name
            frames.append(
                {
                    "file_path": f"images/{frame_name}",
                    "transform_matrix": pose.tolist(),
                    # Per-frame intrinsics (nerfstudio ignores these if
                    # top-level fl_x/fl_y are set, but we write both for
                    # compatibility with other loaders).
                    "fl_x": float(K[0, 0]),
                    "fl_y": float(K[1, 1]),
                    "cx": float(K[0, 2]),
                    "cy": float(K[1, 2]),
                }
            )

        transforms: dict[str, Any] = {
            "camera_model": "OPENCV",
            "fl_x": float(K0[0, 0]),
            "fl_y": float(K0[1, 1]),
            "cx": float(K0[0, 2]),
            "cy": float(K0[1, 2]),
            "w": w,
            "h": h,
            "frames": frames,
        }

        with open(transforms_path, "w") as f:
            json.dump(transforms, f, indent=2)

    # ── COLMAP / nerfstudio fallback backend ─────────────────────────────────

    def _run_colmap(self, images_dir: Path, output_dir: Path) -> Path:
        """Use nerfstudio's ``ns-process-data images`` for COLMAP-based SfM.

        nerfstudio handles COLMAP installation, feature extraction, matching,
        and mapping, and writes transforms.json automatically.
        """
        cmd = [
            "ns-process-data", "images",
            "--data", str(images_dir),
            "--output-dir", str(output_dir),
            "--num-downscales", "0",  # keep original resolution
        ]

        logger.info("Running COLMAP via nerfstudio: %s", " ".join(cmd))
        self._run_subprocess(cmd)

        transforms_path = output_dir / "transforms.json"
        if not transforms_path.exists():
            raise RuntimeError(
                "ns-process-data did not produce transforms.json. "
                "Check the COLMAP output above for errors."
            )

        logger.info("transforms.json written to %s", transforms_path)
        return transforms_path

    # ── Utilities ────────────────────────────────────────────────────────────

    def _run_subprocess(self, cmd: list[str], cwd: Path | None = None) -> None:
        """Run a shell command, streaming output to the logger."""
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=not self.verbose,
            text=True,
        )
        if result.returncode != 0:
            stderr = result.stderr or "(no stderr)"
            raise RuntimeError(
                f"Command failed (exit {result.returncode}): {' '.join(cmd)}\n{stderr}"
            )

    @staticmethod
    def _read_image_hw(image_path: Path) -> tuple[int, int]:
        """Return (height, width) of an image without loading pixel data."""
        import cv2
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        return img.shape[0], img.shape[1]
