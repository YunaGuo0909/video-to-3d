"""
depth_prior.py
==============
Monocular depth estimation using Depth Anything V2.

Predicted depth maps are used as supervision signal during splatfacto
training (via DN-Splatter), improving geometric accuracy for indoor scenes
where sparse COLMAP/MASt3R points provide weak geometric constraints.

Install extras before use:
    uv sync --extra depth

Reference
---------
- Yang et al., "Depth Anything V2", NeurIPS 2024.
  https://github.com/DepthAnything/Depth-Anything-V2
- Turkulainen et al., "DN-Splatter: Depth and Normal Priors for Gaussian
  Splatting and Meshing", 2024. https://github.com/maturk/dn-splatter
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

# HuggingFace model IDs for Depth Anything V2.
# "small" runs comfortably on an RTX 4070 with large batches.
# "large" gives higher quality at the cost of ~2x GPU memory.
MODEL_IDS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base":  "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}


class DepthPrior:
    """Predict per-frame monocular depth maps using Depth Anything V2.

    Parameters
    ----------
    model_size:
        One of ``"small"``, ``"base"``, ``"large"``.
        ``"small"`` is recommended for the RTX 4070 (8 GB VRAM).
    device:
        Torch device string. ``"cuda"`` is assumed for the remote GPU.
    batch_size:
        Number of images to process per forward pass.
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cuda",
        batch_size: int = 4,
    ) -> None:
        if model_size not in MODEL_IDS:
            raise ValueError(f"model_size must be one of {list(MODEL_IDS)}")
        self.model_id = MODEL_IDS[model_size]
        self.device = device
        self.batch_size = batch_size
        self._pipeline = None  # lazy-loaded to avoid heavy import at module level

    # ── Public interface ─────────────────────────────────────────────────────

    def predict(self, images_dir: Path | str, output_dir: Path | str) -> int:
        """Run depth estimation on all images in *images_dir*.

        Writes one ``.npy`` file per image into *output_dir*.
        The numpy array stores a float32 depth map in metres (relative scale).

        Parameters
        ----------
        images_dir:
            Directory containing ``frame_XXXXXX.jpg`` files.
        output_dir:
            Destination for ``frame_XXXXXX.npy`` depth maps.

        Returns
        -------
        int
            Number of depth maps written.
        """
        images_dir = Path(images_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(
            p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not image_paths:
            raise FileNotFoundError(f"No images found in {images_dir}")

        logger.info(
            "Predicting depth for %d images with Depth Anything V2 (%s).",
            len(image_paths),
            self.model_id,
        )

        pipe = self._load_pipeline()
        written = 0

        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i : i + self.batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]

            # HuggingFace pipeline returns a list of dicts with key "predicted_depth".
            results = pipe(images)

            for path, result in zip(batch_paths, results):
                depth_np = np.array(result["predicted_depth"], dtype=np.float32)
                out_path = output_dir / (path.stem + ".npy")
                np.save(str(out_path), depth_np)
                written += 1

        logger.info("Depth maps written to %s (%d files).", output_dir, written)
        return written

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _load_pipeline(self):
        """Lazy-load the HuggingFace depth estimation pipeline."""
        if self._pipeline is None:
            try:
                from transformers import pipeline as hf_pipeline
            except ImportError as e:
                raise ImportError(
                    "transformers is required for DepthPrior. "
                    "Install with: uv sync --extra depth"
                ) from e

            logger.info("Loading %s on %s …", self.model_id, self.device)
            self._pipeline = hf_pipeline(
                task="depth-estimation",
                model=self.model_id,
                device=0 if self.device == "cuda" else -1,
            )
        return self._pipeline
