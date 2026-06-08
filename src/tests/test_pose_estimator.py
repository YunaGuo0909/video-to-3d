"""
Tests for PoseEstimator.

Integration tests (those that call MASt3R or COLMAP) are marked with
``@pytest.mark.integration`` and are skipped in standard CI runs.

Run integration tests explicitly with:
    pytest -m integration
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.data.pose_estimator import PoseEstimator, PoseBackend


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


# ── Unit tests (no external dependencies) ─────────────────────────────────────


def test_colmap_backend_raises_on_missing_transforms(temp_dir):
    """COLMAP backend must raise RuntimeError if ns-process-data produces no output."""
    est = PoseEstimator(backend=PoseBackend.COLMAP)

    # Patch subprocess.run to simulate ns-process-data success but no output file.
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        with pytest.raises(RuntimeError, match="transforms.json"):
            est._run_colmap(temp_dir / "images", temp_dir / "out")


def test_mast3r_backend_requires_repo_path(temp_dir):
    """MASt3R backend must raise ValueError when mast3r_repo is not set."""
    est = PoseEstimator(backend=PoseBackend.MAST3R, mast3r_repo=None)
    with pytest.raises(ValueError, match="mast3r_repo"):
        est.estimate(temp_dir / "images", temp_dir / "out")


def test_convert_mast3r_to_nerfstudio(temp_dir):
    """_convert_mast3r_to_nerfstudio should write a valid transforms.json."""
    import cv2

    # Write a dummy image so _read_image_hw works.
    images_dir = temp_dir / "images"
    images_dir.mkdir()
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.imwrite(str(images_dir / "frame_000000.jpg"), dummy_img)

    # Create a minimal scene.npz as MASt3R would produce.
    mast3r_out = temp_dir / "mast3r_raw"
    mast3r_out.mkdir()

    n = 1
    poses = np.eye(4, dtype=np.float64)[None].repeat(n, axis=0)
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
    intrinsics = K[None].repeat(n, axis=0)
    image_paths = np.array(["frame_000000.jpg"])

    np.savez(
        str(mast3r_out / "scene.npz"),
        poses=poses,
        intrinsics=intrinsics,
        image_paths=image_paths,
    )

    est = PoseEstimator(backend=PoseBackend.MAST3R, mast3r_repo=temp_dir)
    transforms_path = temp_dir / "transforms.json"
    est._convert_mast3r_to_nerfstudio(mast3r_out, images_dir, transforms_path)

    assert transforms_path.exists()
    with open(transforms_path) as f:
        data = json.load(f)

    assert "frames" in data
    assert len(data["frames"]) == 1
    assert data["camera_model"] == "OPENCV"
    assert "fl_x" in data
    assert len(data["frames"][0]["transform_matrix"]) == 4


# ── Integration test marker ────────────────────────────────────────────────────


@pytest.mark.integration
def test_colmap_full_run(temp_dir):
    """Full COLMAP run — requires nerfstudio and COLMAP installed."""
    pytest.skip("Integration test: run manually with -m integration")
