"""
video_processor.py
==================
Extract and filter frames from an input video file.

Quality filters applied:
  - Blur detection  : Laplacian variance — discards frames below a sharpness threshold.
  - Temporal stride  : Skips frames to avoid near-duplicate training samples.
  - Resolution check : Warns if the source video is lower than the minimum recommended.

Output layout (written to <output_dir>/images/):
  frame_000000.jpg, frame_000001.jpg, ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Minimum Laplacian variance to accept a frame as "sharp enough".
# Tuned empirically for typical indoor phone video at 1080p.
DEFAULT_BLUR_THRESHOLD = 80.0

# Take at most one frame every N milliseconds to limit near-duplicates.
DEFAULT_MIN_FRAME_GAP_MS = 250  # 4 fps effective maximum sampling rate


@dataclass
class ProcessingStats:
    """Counts extracted and rejected frames for transparency."""

    total_decoded: int = 0
    accepted: int = 0
    rejected_blur: int = 0
    rejected_stride: int = 0

    def summary(self) -> str:
        return (
            f"Decoded {self.total_decoded} frames → "
            f"accepted {self.accepted} "
            f"(rejected blur={self.rejected_blur}, stride={self.rejected_stride})"
        )


@dataclass
class VideoProcessor:
    """Extract high-quality frames from a video file.

    Parameters
    ----------
    blur_threshold:
        Laplacian variance below this value → frame is discarded as blurry.
    min_frame_gap_ms:
        Minimum time gap (milliseconds) between consecutive accepted frames.
        Prevents near-duplicate frames that waste GPU memory during training.
    max_frames:
        Hard cap on accepted frames. ``None`` means no cap.
    image_ext:
        Output image format. JPEG with high quality is the default because
        nerfstudio's COLMAP backend expects JPEG or PNG.
    jpeg_quality:
        JPEG compression quality (1–100). 95 preserves enough detail for
        COLMAP feature matching while keeping file sizes manageable.
    """

    blur_threshold: float = DEFAULT_BLUR_THRESHOLD
    min_frame_gap_ms: float = DEFAULT_MIN_FRAME_GAP_MS
    max_frames: int | None = None
    image_ext: str = "jpg"
    jpeg_quality: int = 95

    def process(self, video_path: Path | str, output_dir: Path | str) -> ProcessingStats:
        """Extract frames from *video_path* into *output_dir*/images/.

        Parameters
        ----------
        video_path:
            Path to the source video (MP4, MOV, AVI, …).
        output_dir:
            Root directory for the processed dataset. The ``images/``
            subdirectory is created automatically.

        Returns
        -------
        ProcessingStats
            Counts of accepted / rejected frames.

        Raises
        ------
        FileNotFoundError
            If *video_path* does not exist.
        RuntimeError
            If OpenCV fails to open the video.
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(
            "Video: %s  |  %.1f fps  |  %d frames  |  %dx%d",
            video_path.name,
            fps,
            total_frames,
            width,
            height,
        )
        if min(width, height) < 720:
            logger.warning(
                "Video resolution (%dx%d) is below recommended 720p. "
                "Reconstruction quality may be reduced.",
                width,
                height,
            )

        stats = ProcessingStats()
        last_accepted_ms = -self.min_frame_gap_ms  # allow first frame

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]

        with tqdm(total=total_frames, desc="Extracting frames", unit="fr") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                stats.total_decoded += 1
                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

                # ── Temporal stride filter ──────────────────────────────────
                if timestamp_ms - last_accepted_ms < self.min_frame_gap_ms:
                    stats.rejected_stride += 1
                    pbar.update(1)
                    continue

                # ── Blur filter ─────────────────────────────────────────────
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                if sharpness < self.blur_threshold:
                    stats.rejected_blur += 1
                    pbar.update(1)
                    continue

                # ── Accept frame ────────────────────────────────────────────
                out_path = images_dir / f"frame_{stats.accepted:06d}.{self.image_ext}"
                cv2.imwrite(str(out_path), frame, encode_params)
                last_accepted_ms = timestamp_ms
                stats.accepted += 1

                if self.max_frames and stats.accepted >= self.max_frames:
                    logger.info("Reached max_frames=%d, stopping.", self.max_frames)
                    break

                pbar.update(1)

        cap.release()
        logger.info(stats.summary())

        if stats.accepted < 30:
            logger.warning(
                "Only %d frames accepted. Reconstruction may be unstable. "
                "Try lowering blur_threshold or min_frame_gap_ms.",
                stats.accepted,
            )

        return stats
