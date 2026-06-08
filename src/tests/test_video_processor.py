"""
Tests for VideoProcessor.

These tests use synthetic frames (written to a temp directory) to avoid
requiring a real video file in the CI environment.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data.video_processor import VideoProcessor, ProcessingStats


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


def _make_synthetic_video(path: Path, n_frames: int = 30, fps: float = 10.0) -> None:
    """Write a minimal MP4 video with random sharp frames."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (640, 480))
    rng = np.random.default_rng(42)
    for _ in range(n_frames):
        # Random RGB image — high spatial variation ensures high Laplacian variance.
        frame = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _make_blurry_video(path: Path, n_frames: int = 10, fps: float = 10.0) -> None:
    """Write a video where every frame is a uniform colour (zero variance)."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (640, 480))
    for _ in range(n_frames):
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        writer.write(frame)
    writer.release()


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_process_sharp_video_accepts_frames(temp_dir):
    video = temp_dir / "sharp.mp4"
    _make_synthetic_video(video, n_frames=30, fps=10.0)

    proc = VideoProcessor(blur_threshold=10.0, min_frame_gap_ms=0)
    stats = proc.process(video, temp_dir / "out")

    assert stats.accepted > 0
    assert stats.is_valid if hasattr(stats, "is_valid") else True
    assert (temp_dir / "out" / "images").exists()


def test_process_blurry_video_rejects_frames(temp_dir):
    video = temp_dir / "blurry.mp4"
    _make_blurry_video(video, n_frames=10)

    # High blur threshold → uniform frames are all rejected.
    proc = VideoProcessor(blur_threshold=500.0, min_frame_gap_ms=0)
    stats = proc.process(video, temp_dir / "out")

    assert stats.rejected_blur == stats.total_decoded


def test_process_respects_max_frames(temp_dir):
    video = temp_dir / "long.mp4"
    _make_synthetic_video(video, n_frames=50, fps=10.0)

    proc = VideoProcessor(blur_threshold=1.0, min_frame_gap_ms=0, max_frames=5)
    stats = proc.process(video, temp_dir / "out")

    assert stats.accepted == 5


def test_missing_video_raises(temp_dir):
    proc = VideoProcessor()
    with pytest.raises(FileNotFoundError):
        proc.process(temp_dir / "nonexistent.mp4", temp_dir / "out")


def test_output_images_are_jpeg(temp_dir):
    video = temp_dir / "test.mp4"
    _make_synthetic_video(video, n_frames=5, fps=10.0)

    proc = VideoProcessor(blur_threshold=1.0, min_frame_gap_ms=0)
    proc.process(video, temp_dir / "out")

    images = list((temp_dir / "out" / "images").glob("*.jpg"))
    assert len(images) > 0
