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
import shutil
import subprocess
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class PoseBackend(str, Enum):
    MAST3R = "mast3r"    # ICLR 2025 SOTA — robust to textureless surfaces
    COLMAP = "colmap"    # Classic SfM via pycolmap Python API
    AUTO   = "auto"      # Try COLMAP; fall back to MASt3R if registration < 70%


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

        if self.backend == PoseBackend.AUTO:
            return self._estimate_auto(images_dir, output_dir)
        elif self.backend == PoseBackend.MAST3R:
            return self._run_mast3r(images_dir, output_dir)
        else:
            return self._run_colmap(images_dir, output_dir)

    # ── AUTO backend ──────────────────────────────────────────────────────────

    def _estimate_auto(self, images_dir: Path, output_dir: Path) -> Path:
        """Run COLMAP; if registration rate < 70 %, also try MASt3R and keep
        whichever result registered more frames.

        Both intermediate transforms files are preserved as
        ``transforms_colmap.json`` / ``transforms_mast3r.json`` for debugging.
        """
        transforms_path = output_dir / "transforms.json"

        # ── Step 1: COLMAP ────────────────────────────────────────────────────
        logger.info("AUTO: running COLMAP...")
        self._run_colmap(images_dir, output_dir)
        colmap_backup = output_dir / "transforms_colmap.json"
        if transforms_path.exists():
            shutil.copy2(transforms_path, colmap_backup)

        colmap_rate = self._registration_rate(transforms_path, images_dir)
        logger.info("COLMAP registration rate: %.1f%%", colmap_rate * 100)

        if colmap_rate >= 0.70 or self.mast3r_repo is None:
            if colmap_rate < 0.70:
                logger.warning(
                    "Registration rate %.1f%% < 70%% but MASt3R repo not set "
                    "(pass --mast3r-repo to enable fallback).",
                    colmap_rate * 100,
                )
            logger.info("AUTO: using COLMAP result (rate=%.1f%%)", colmap_rate * 100)
            return transforms_path

        # ── Step 2: MASt3R fallback ───────────────────────────────────────────
        logger.info(
            "AUTO: COLMAP rate %.1f%% < 70%% — trying MASt3R...", colmap_rate * 100
        )
        mast3r_backup = output_dir / "transforms_mast3r.json"
        try:
            self._run_mast3r(images_dir, output_dir)
            if transforms_path.exists():
                shutil.copy2(transforms_path, mast3r_backup)
            mast3r_rate = self._registration_rate(transforms_path, images_dir)
            logger.info("MASt3R registration rate: %.1f%%", mast3r_rate * 100)
        except Exception as exc:
            logger.warning("MASt3R failed (%s) — keeping COLMAP result.", exc)
            shutil.copy2(colmap_backup, transforms_path)
            return transforms_path

        # ── Pick winner ───────────────────────────────────────────────────────
        if mast3r_rate >= colmap_rate:
            logger.info(
                "AUTO: MASt3R wins (%.1f%% vs COLMAP %.1f%%).",
                mast3r_rate * 100, colmap_rate * 100,
            )
            shutil.copy2(mast3r_backup, transforms_path)
        else:
            logger.info(
                "AUTO: COLMAP wins (%.1f%% vs MASt3R %.1f%%).",
                colmap_rate * 100, mast3r_rate * 100,
            )
            shutil.copy2(colmap_backup, transforms_path)

        return transforms_path

    @staticmethod
    def _registration_rate(transforms_path: Path, images_dir: Path) -> float:
        """Return fraction of images in *images_dir* that appear in transforms.json."""
        if not transforms_path.exists():
            return 0.0
        with open(transforms_path) as f:
            data = json.load(f)
        n_registered = len(data.get("frames", []))
        all_images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        n_total = len(all_images)
        return n_registered / n_total if n_total > 0 else 0.0

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
        """Run SfM with pycolmap, convert poses from COLMAP text files.

        Pipeline:
          1. pycolmap: feature extraction + matching + incremental mapping
          2. best.write_text() → dump cameras/images/points3D as plain text
          3. Parse text files with standard quaternion math (no pycolmap API)
          4. Write nerfstudio transforms.json

        Using text files avoids all pycolmap version API differences.
        The quaternion→rotation conversion uses scipy, which is deterministic.
        """
        try:
            import pycolmap
        except ImportError:
            raise ImportError(
                "pycolmap is required. Install with: pip install pycolmap"
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
        logger.info("Matching features (exhaustive)...")
        pycolmap.match_exhaustive(database_path=str(database_path))

        # ── Step 3: Incremental mapping ───────────────────────────────────────
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

        best_id = max(maps.keys(), key=lambda k: len(maps[k].images))
        best = maps[best_id]
        logger.info(
            "Best reconstruction: %d images, %d 3D points.",
            len(best.images), len(best.points3D),
        )

        # ── Step 4: Export as text (API-version-independent) ─────────────────
        text_dir = colmap_dir / "text"
        text_dir.mkdir(exist_ok=True)
        best.write_text(str(text_dir))
        logger.info("COLMAP text model written to %s", text_dir)

        # ── Step 5: Export sparse PLY for Gaussian initialisation ─────────────
        ply_path = output_dir / "sparse_pt_cloud.ply"
        self._export_sparse_ply(best, ply_path)

        # ── Step 6: Parse text files → transforms.json ───────────────────────
        transforms_path = output_dir / "transforms.json"
        self._convert_colmap_text_to_nerfstudio(
            text_dir, images_dir, transforms_path,
            ply_file_path="sparse_pt_cloud.ply",
        )

        logger.info("transforms.json written to %s", transforms_path)
        return transforms_path

    def _convert_colmap_text_to_nerfstudio(
        self,
        text_dir: Path,
        images_dir: Path,
        transforms_path: Path,
        ply_file_path: str | None = None,
    ) -> None:
        """Parse COLMAP text files and write nerfstudio transforms.json.

        COLMAP images.txt stores the WORLD-TO-CAMERA pose as:
          QW QX QY QZ TX TY TZ

        Conversion to nerfstudio camera-to-world (OpenGL convention):
          1. Build rotation matrix from quaternion (scipy, [x,y,z,w] convention)
          2. Invert: R_wc = R_cw^T,  t_wc = -R_cw^T @ t_cw
          3. Flip Y and Z columns: COLMAP(Y↓,Z→) → nerfstudio(Y↑,Z←)
        """
        from scipy.spatial.transform import Rotation

        # ── Parse cameras.txt ─────────────────────────────────────────────────
        cameras: dict[int, dict] = {}
        with open(text_dir / "cameras.txt") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                cam_id = int(parts[0])
                model = parts[1]
                w, h = int(parts[2]), int(parts[3])
                p = [float(x) for x in parts[4:]]
                cameras[cam_id] = {
                    "model": model, "w": w, "h": h, "params": p
                }

        # ── Parse images.txt ──────────────────────────────────────────────────
        # Odd lines: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        # Even lines: 2D point observations (skip)
        frames: list[dict] = []
        with open(text_dir / "images.txt") as f:
            lines = [l for l in f if not l.startswith("#") and l.strip()]

        for i in range(0, len(lines), 2):
            parts = lines[i].split()
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            cam_id = int(parts[8])
            name = parts[9]

            cam = cameras[cam_id]

            # World-to-camera rotation (scipy uses [x,y,z,w] scalar-last)
            R_cw = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            t_cw = np.array([tx, ty, tz])

            # Camera-to-world
            R_wc = R_cw.T
            t_wc = -R_wc @ t_cw
            c2w = np.eye(4, dtype=np.float64)
            c2w[:3, :3] = R_wc
            c2w[:3, 3] = t_wc

            # COLMAP OpenCV (Y↓,Z→) → nerfstudio OpenGL (Y↑,Z←)
            c2w[:3, 1:3] *= -1

            fl_x, fl_y, cx, cy = self._intrinsics_from_params(cam["model"], cam["params"], cam["w"], cam["h"])

            frames.append({
                "file_path": f"images/{name}",
                "transform_matrix": c2w.tolist(),
                "fl_x": fl_x,
                "fl_y": fl_y,
                "cx": cx,
                "cy": cy,
            })

        if not frames:
            raise RuntimeError("No frames parsed from images.txt.")

        # Shared intrinsics from first camera
        first_cam = cameras[min(cameras.keys())]
        fl_x0, fl_y0, cx0, cy0 = self._intrinsics_from_params(
            first_cam["model"], first_cam["params"], first_cam["w"], first_cam["h"]
        )

        transforms: dict[str, Any] = {
            "camera_model": "OPENCV",
            "fl_x": fl_x0, "fl_y": fl_y0,
            "cx": cx0, "cy": cy0,
            "w": first_cam["w"], "h": first_cam["h"],
            "frames": frames,
        }
        if ply_file_path:
            transforms["ply_file_path"] = ply_file_path

        with open(transforms_path, "w") as f:
            json.dump(transforms, f, indent=2)

        logger.info("Wrote %d frames to transforms.json", len(frames))

    @staticmethod
    def _intrinsics_from_params(model: str, p: list, w: int, h: int) -> tuple:
        """Return (fl_x, fl_y, cx, cy) from COLMAP camera params."""
        m = model.upper()
        if "SIMPLE_PINHOLE" in m:   # f cx cy
            return p[0], p[0], p[1], p[2]
        if "PINHOLE" in m:          # fx fy cx cy
            return p[0], p[1], p[2], p[3]
        if "SIMPLE_RADIAL" in m:    # f cx cy k
            return p[0], p[0], p[1], p[2]
        if "RADIAL" in m:           # f cx cy k1 k2
            return p[0], p[0], p[1], p[2]
        if "OPENCV" in m:           # fx fy cx cy k1 k2 p1 p2
            return p[0], p[1], p[2], p[3]
        # Fallback: assume f cx cy
        return p[0], p[0], w / 2.0, h / 2.0

    @staticmethod
    def _export_sparse_ply(reconstruction, ply_path: Path) -> None:
        """Write sparse 3D points from pycolmap Reconstruction to a PLY file.

        nerfstudio reads this file to initialise Gaussian positions, which
        produces much better results than a random sphere initialisation.
        """
        try:
            from plyfile import PlyData, PlyElement
        except ImportError:
            logger.warning("plyfile not installed — skipping PLY export.")
            return

        pts = reconstruction.points3D
        if not pts:
            logger.warning("Reconstruction has no 3D points — skipping PLY export.")
            return

        xyz = np.array([p.xyz for p in pts.values()], dtype=np.float32)
        rgb = np.array([p.color[:3] for p in pts.values()], dtype=np.uint8)

        vertices = np.empty(
            len(xyz),
            dtype=[("x","f4"),("y","f4"),("z","f4"),
                   ("red","u1"),("green","u1"),("blue","u1")],
        )
        vertices["x"], vertices["y"], vertices["z"] = xyz[:,0], xyz[:,1], xyz[:,2]
        vertices["red"], vertices["green"], vertices["blue"] = rgb[:,0], rgb[:,1], rgb[:,2]

        PlyData([PlyElement.describe(vertices, "vertex")]).write(str(ply_path))
        logger.info("Sparse PLY exported: %d points → %s", len(xyz), ply_path)

    def _convert_colmap_reconstruction_to_nerfstudio(
        self,
        reconstruction,
        images_dir: Path,
        transforms_path: Path,
        ply_file_path: str | None = None,
    ) -> None:
        """Convert a pycolmap Reconstruction to nerfstudio transforms.json.

        COLMAP poses are world-to-camera (w2c). We invert to camera-to-world
        (c2w) and then convert axes from OpenCV to OpenGL convention by
        flipping the Y and Z columns of the rotation part.
        """
        frames: list[dict[str, Any]] = []

        for _img_id, image in reconstruction.images.items():
            cam = reconstruction.cameras[image.camera_id]

            R, t = self._extract_w2c(image)

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
        if ply_file_path:
            transforms["ply_file_path"] = ply_file_path

        with open(transforms_path, "w") as f:
            json.dump(transforms, f, indent=2)

    @staticmethod
    def _extract_w2c(image) -> tuple[np.ndarray, np.ndarray]:
        """Extract (R, t) world-to-camera from a pycolmap Image.

        Handles API differences across pycolmap versions:
          - 4.x: cam_from_world is a callable method → call it, then access
                 .rotation.matrix() and .translation
          - 3.x: cam_from_world is a Rigid3d property (not callable)
          - fallback: reconstruct from qvec/tvec (older pycolmap)
        """
        cfw = image.cam_from_world
        # In pycolmap 4.x cam_from_world is a bound method — call it.
        if callable(cfw):
            cfw = cfw()

        # Try Rigid3d with Rotation3d sub-object (.rotation.matrix())
        try:
            rot = cfw.rotation
            R = np.array(rot.matrix() if callable(rot.matrix) else rot.matrix)
            t = np.array(cfw.translation)
            return R, t
        except AttributeError:
            pass

        # Try Rigid3d that exposes .matrix() directly (3×4 or 4×4)
        try:
            mat = np.array(cfw.matrix() if callable(cfw.matrix) else cfw.matrix)
            return mat[:3, :3], mat[:3, 3]
        except AttributeError:
            pass

        # Oldest pycolmap: qvec + tvec stored on the Image directly
        try:
            from scipy.spatial.transform import Rotation
            R = Rotation.from_quat(
                [image.qvec[1], image.qvec[2], image.qvec[3], image.qvec[0]]
            ).as_matrix()
            return R, np.array(image.tvec)
        except AttributeError:
            pass

        raise RuntimeError(
            f"Cannot extract pose from pycolmap Image (cam_from_world type: "
            f"{type(cfw)}). Please report your pycolmap version."
        )

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
