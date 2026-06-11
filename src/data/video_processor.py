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
    mean_sharpness: float = 0.0   # mean Laplacian variance of accepted frames

    def summary(self) -> str:
        return (
            f"Decoded {self.total_decoded} frames → "
            f"accepted {self.accepted} "
            f"(rejected blur={self.rejected_blur}, stride={self.rejected_stride}, "
            f"mean_sharpness={self.mean_sharpness:.1f})"
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
        sharpness_sum = 0.0

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
                sharpness_sum += sharpness
                stats.accepted += 1

                if self.max_frames and stats.accepted >= self.max_frames:
                    logger.info("Reached max_frames=%d, stopping.", self.max_frames)
                    break

                pbar.update(1)

        cap.release()
        if stats.accepted > 0:
            stats.mean_sharpness = sharpness_sum / stats.accepted
        logger.info(stats.summary())

        if stats.accepted < 30:
            logger.warning(
                "Only %d frames accepted. Reconstruction may be unstable. "
                "Try lowering blur_threshold or min_frame_gap_ms.",
                stats.accepted,
            )

        return stats

    # ── Adaptive configuration ────────────────────────────────────────────────

    @staticmethod
    def scan_video(video_path: Path | str, n_samples: int = 60) -> dict:
        """Pre-scan a video to estimate sharpness and motion statistics.

        Samples *n_samples* frames evenly across the video and computes
        Laplacian variance (sharpness) and inter-frame mean-absolute-difference
        (motion proxy). Results are used by ``from_video()`` to set thresholds
        automatically.

        Returns
        -------
        dict
            Keys: sharpness_p10/p25/p50/p75, motion_mean, motion_std,
            fps, total_frames.
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video for scanning: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        sample_indices = np.linspace(
            0, total_frames - 1, min(n_samples, total_frames), dtype=int
        )

        sharpness_scores: list[float] = []
        motion_scores: list[float] = []
        prev_gray: np.ndarray | None = None

        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness_scores.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
            if prev_gray is not None:
                diff = float(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))))
                motion_scores.append(diff)
            prev_gray = gray

        cap.release()

        sh = np.array(sharpness_scores) if sharpness_scores else np.array([100.0])
        mo = np.array(motion_scores) if motion_scores else np.array([10.0])

        return {
            "sharpness_p10": float(np.percentile(sh, 10)),
            "sharpness_p25": float(np.percentile(sh, 25)),
            "sharpness_p50": float(np.percentile(sh, 50)),
            "sharpness_p75": float(np.percentile(sh, 75)),
            "motion_mean":   float(np.mean(mo)),
            "motion_std":    float(np.std(mo)),
            "fps":           fps,
            "total_frames":  total_frames,
        }

    @classmethod
    def from_video(
        cls,
        video_path: Path | str,
        max_frames: int | None = None,
    ) -> "VideoProcessor":
        """Create a VideoProcessor with thresholds adapted to the input video.

        Pre-scans *video_path* and sets:

        - ``blur_threshold``: 25th-percentile sharpness of sampled frames —
          accepts the top 75 % of frames by sharpness, regardless of content.
        - ``min_frame_gap_ms``: shorter for fast-moving video (more frames
          needed for stable triangulation), longer for slow/static captures.

        Parameters
        ----------
        video_path:
            Input video file.
        max_frames:
            Hard cap forwarded to the constructor.
        """
        scan = cls.scan_video(video_path)

        # Keep the sharpest 75 % of frames; clamp to a safe range.
        blur_threshold = float(np.clip(scan["sharpness_p25"], 30.0, 200.0))

        # Faster motion → shorter gap to preserve sufficient parallax.
        motion = scan["motion_mean"]
        if motion > 15.0:
            min_frame_gap_ms = 150.0    # fast camera: ~6-7 fps
        elif motion > 8.0:
            min_frame_gap_ms = 200.0    # moderate: ~5 fps
        else:
            min_frame_gap_ms = 300.0    # slow / tripod: ~3 fps

        logger.info(
            "Adaptive config — blur_threshold=%.1f  (p10/p25/p50=%.0f/%.0f/%.0f)  "
            "min_frame_gap_ms=%.0f  (motion_mean=%.1f)",
            blur_threshold,
            scan["sharpness_p10"], scan["sharpness_p25"], scan["sharpness_p50"],
            min_frame_gap_ms,
            motion,
        )

        return cls(
            blur_threshold=blur_threshold,
            min_frame_gap_ms=min_frame_gap_ms,
            max_frames=max_frames,
        )
