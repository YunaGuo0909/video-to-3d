"""
pose_estimator.py
=================
Camera pose estimation from a set of extracted frames.

Supported backends
------------------
MASt3R-SfM (primary)
    Feed-forward transformer-based SfM (ICLR 2025). Robust to textureless
    indoor surfaces and requires no camera calibration.
    Reference: Duisterhof et al., ICLR 2025 — https://github.com/naver/mast3r

COLMAP (fallback)
    Classic SfM using pycolmap Python bindings (no COLMAP binary required).
    Install with: pip install pycolmap

Both backends produce a ``transforms.json`` file in the nerfstudio format.
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
    COLMAP = "colmap"    # Fallback: pycolmap Python API (no binary needed)


class PoseEstimator:
    """Estimate camera poses from a directory of extracted frames.

    Parameters
    ----------
    backend:
        Which pose estimation backend to use (see ``PoseBackend``).
    mast3r_repo:
        Path to a local clone of https://github.com/naver/mast3r.
        Required only when ``backend=PoseBackend.MAST3R``.
    verbose:
        Forward subprocess stdout/stderr to the logger.
    """

    def __init__(
        self,
        backend: PoseBackend = PoseBackend.COLMAP,
        mast3r_repo: Path | str | None = None,
        verbose: bool = True,
    ) -> None:
        self.backend = backend
        self.mast3r_repo = Path(mast3r_repo) if mast3r_repo else None
        self.verbose = verbose

    # ── Public interface ─────────────────────────────────────────────────────

    def estimate(self, images_dir: Path | str, output_dir: Path | str) -> Path:
        """Run pose estimation and write ``transforms.json`` to *output_dir*.

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
        """Run MASt3R-SfM and convert output to nerfstudio transforms.json."""
        if self.mast3r_repo is None:
            raise ValueError(
                "mast3r_repo must be set when using PoseBackend.MAST3R. "
                "Clone from: https://github.com/naver/mast3r"
            )

        mast3r_out = output_dir / "mast3r_raw"
        mast3r_out.mkdir(exist_ok=True)

        script = self.mast3r_repo / "demo" / "demo_mast3r_sfm.py"
        cmd = [
            "python", str(script),
            "--image_dir", str(images_dir),
            "--output_dir", str(mast3r_out),
            "--device", "cuda",
        ]

        logger.info("Running MASt3R-SfM: %s", " ".join(cmd))
        self._run_subprocess(cmd, cwd=self.mast3r_repo)

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
        """Parse MASt3R scene.npz and write nerfstudio transforms.json."""
        scene_file = mast3r_out / "scene.npz"
        if not scene_file.exists():
            raise FileNotFoundError(f"MASt3R output not found: {scene_file}")

        data = np.load(str(scene_file), allow_pickle=True)
        poses: np.ndarray = data["poses"]
        intrinsics: np.ndarray = data["intrinsics"]
        image_paths: list[str] = list(data["image_paths"])

        K0 = intrinsics[0]
        h, w = self._read_image_hw(images_dir / Path(image_paths[0]).name)

        frames: list[dict[str, Any]] = []
        for img_path, pose, K in zip(image_paths, poses, intrinsics):
            frame_name = Path(img_path).name
            frames.append({
                "file_path": f"images/{frame_name}",
                "transform_matrix": pose.tolist(),
                "fl_x": float(K[0, 0]),
                "fl_y": float(K[1, 1]),
                "cx": float(K[0, 2]),
                "cy": float(K[1, 2]),
            })

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

    # ── COLMAP backend (via pycolmap Python API) ──────────────────────────────

    def _run_colmap(self, images_dir: Path, output_dir: Path) -> Path:
        """Run SfM using pycolmap Python bindings (no COLMAP binary needed).

        pycolmap wraps the COLMAP C++ library directly. The pipeline is:
          1. Feature extraction (SIFT)
          2. Exhaustive feature matching
          3. Incremental sparse reconstruction
          4. Convert reconstruction to nerfstudio transforms.json

        Coordinate conversion: COLMAP uses OpenCV convention (Y down, Z fwd).
        nerfstudio uses OpenGL convention (Y up, Z backward). We apply the
        standard flip of Y and Z axes on the camera-to-world matrix.
        """
        try:
            import pycolmap
        except ImportError:
            raise ImportError(
                "pycolmap is required for COLMAP-based pose estimation. "
                "Install with: pip install pycolmap"
            )

        colmap_dir = output_dir / "colmap"
        colmap_dir.mkdir(parents=True, exist_ok=True)
        database_path = colmap_dir / "database.db"
        sparse_path = colmap_dir / "sparse"
        sparse_path.mkdir(exist_ok=True)

        # ── Step 1: Feature extraction ────────────────────────────────────────
        logger.info("Extracting SIFT features (%s)...", images_dir)
        pycolmap.extract_features(
            database_path=str(database_path),
            image_path=str(images_dir),
        )

        # ── Step 2: Feature matching ──────────────────────────────────────────
        # Exhaustive matching is appropriate for small scenes (<200 frames).
        logger.info("Matching features (exhaustive)...")
        pycolmap.match_exhaustive(database_path=str(database_path))

        # ── Step 3: Incremental sparse reconstruction ─────────────────────────
        logger.info("Running incremental mapping...")
        maps = pycolmap.incremental_mapping(
            database_path=str(database_path),
            image_path=str(images_dir),
            output_path=str(sparse_path),
        )

        if not maps:
            raise RuntimeError(
                "pycolmap reconstruction failed — no maps produced. "
                "Check that frames have sufficient overlap and texture."
            )

        # Pick the reconstruction with the most registered images.
        best = max(maps.values(), key=lambda m: len(m.images))
        logger.info(
            "Reconstruction: %d cameras, %d images, %d 3D points.",
            len(best.cameras),
            len(best.images),
            len(best.points3D),
        )

        # ── Step 4: Convert to nerfstudio format ──────────────────────────────
        transforms_path = output_dir / "transforms.json"
        self._convert_colmap_reconstruction_to_nerfstudio(
            best, images_dir, transforms_path
        )

        logger.info("transforms.json written to %s", transforms_path)
        return transforms_path

    def _convert_colmap_reconstruction_to_nerfstudio(
        self,
        reconstruction,
        images_dir: Path,
        transforms_path: Path,
    ) -> None:
        """Convert a pycolmap Reconstruction to nerfstudio transforms.json.

        COLMAP poses are world-to-camera (w2c). We invert to camera-to-world
        (c2w) and then convert axes from OpenCV to OpenGL convention by
        flipping the Y and Z columns of the rotation part.
        """
        frames: list[dict[str, Any]] = []

        for _img_id, image in reconstruction.images.items():
            cam = reconstruction.cameras[image.camera_id]

            # Build 4x4 world-to-camera matrix from pycolmap Rigid3d.
            # cam_from_world is the w2c transform stored as Rigid3d.
            try:
                # pycolmap >= 3.x API
                R = image.cam_from_world.rotation.matrix()   # (3,3)
                t = np.array(image.cam_from_world.translation)  # (3,)
            except AttributeError:
                # pycolmap < 3.x fallback (qvec / tvec)
                from scipy.spatial.transform import Rotation
                R = Rotation.from_quat(
                    [image.qvec[1], image.qvec[2], image.qvec[3], image.qvec[0]]
                ).as_matrix()
                t = np.array(image.tvec)

            w2c = np.eye(4, dtype=np.float64)
            w2c[:3, :3] = R
            w2c[:3, 3] = t
            c2w = np.linalg.inv(w2c)

            # Convert COLMAP (OpenCV) → nerfstudio (OpenGL): flip Y and Z.
            c2w[:3, 1:3] *= -1

            # Parse camera intrinsics (handle common COLMAP camera models).
            fl_x, fl_y, cx, cy, k1, k2, p1, p2 = self._parse_colmap_camera(cam)

            frames.append({
                "file_path": f"images/{image.name}",
                "transform_matrix": c2w.tolist(),
                "fl_x": fl_x,
                "fl_y": fl_y,
                "cx": cx,
                "cy": cy,
                "k1": k1,
                "k2": k2,
                "p1": p1,
                "p2": p2,
            })

        # Shared intrinsics from the first camera.
        first_cam = list(reconstruction.cameras.values())[0]
        fl_x0, fl_y0, cx0, cy0, *_ = self._parse_colmap_camera(first_cam)

        transforms: dict[str, Any] = {
            "camera_model": "OPENCV",
            "fl_x": fl_x0,
            "fl_y": fl_y0,
            "cx": cx0,
            "cy": cy0,
            "w": int(first_cam.width),
            "h": int(first_cam.height),
            "frames": frames,
        }

        with open(transforms_path, "w") as f:
            json.dump(transforms, f, indent=2)

    @staticmethod
    def _parse_colmap_camera(cam) -> tuple[float, ...]:
        """Extract (fl_x, fl_y, cx, cy, k1, k2, p1, p2) from a pycolmap camera."""
        model_name = cam.model.name if hasattr(cam.model, "name") else str(cam.model)
        p = list(cam.params)

        if "SIMPLE_PINHOLE" in model_name:
            # params: f, cx, cy
            return p[0], p[0], p[1], p[2], 0.0, 0.0, 0.0, 0.0
        elif "PINHOLE" in model_name:
            # params: fx, fy, cx, cy
            return p[0], p[1], p[2], p[3], 0.0, 0.0, 0.0, 0.0
        elif "SIMPLE_RADIAL" in model_name:
            # params: f, cx, cy, k1
            return p[0], p[0], p[1], p[2], p[3], 0.0, 0.0, 0.0
        elif "RADIAL" in model_name:
            # params: f, cx, cy, k1, k2
            return p[0], p[0], p[1], p[2], p[3], p[4], 0.0, 0.0
        elif "OPENCV" in model_name or "FULL" in model_name:
            # params: fx, fy, cx, cy, k1, k2, p1, p2
            return p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]
        else:
            # Generic fallback: assume fx, fy, cx, cy
            return p[0], p[1] if len(p) > 1 else p[0], p[2] if len(p) > 2 else 0.0, p[3] if len(p) > 3 else 0.0, 0.0, 0.0, 0.0, 0.0

    # ── Shared utilities ──────────────────────────────────────────────────────

    def _run_subprocess(self, cmd: list[str], cwd: Path | None = None) -> None:
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
        import cv2
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        return img.shape[0], img.shape[1]
