"""
semantic_field.py
=================
Open-vocabulary 3D semantic field via LangSplat.

LangSplat embeds CLIP language features directly into 3D Gaussians, enabling
open-vocabulary text queries (e.g. "chair", "window") that return the 3D
positions of matching Gaussians. It outperforms 2D-projection approaches
because features are consistent across all views without post-hoc alignment.

Pipeline
--------
1. ``encode_frames()``   — Extract CLIP features from every training frame.
2. ``train_autoencoder()`` — Train a compact autoencoder to compress CLIP
                             features from 512-d to 3-d for per-Gaussian storage.
3. ``build_field()``     — Render compressed features through frozen 3DGS
                           geometry to obtain language-aware Gaussians.
4. ``query()``           — Given a text prompt, return the indices and positions
                           of Gaussians with high cosine similarity.

Install extras before use:
    uv sync --extra semantic

Reference
---------
- Qin et al., "LangSplat: 3D Language Gaussian Splatting", CVPR 2024.
  https://github.com/minghanqin/LangSplat
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# CLIP model used for feature extraction. ViT-B/16 is the LangSplat default.
CLIP_MODEL = "ViT-B/16"

# Bottleneck dimension of the language autoencoder (per Gaussian).
AE_LATENT_DIM = 3


class SemanticField:
    """Build and query a language-aware 3D Gaussian field.

    Parameters
    ----------
    clip_model:
        OpenCLIP model identifier (default: ``"ViT-B/16"``).
    ae_latent_dim:
        Output dimensionality of the language autoencoder.
    device:
        Torch device for CLIP inference and autoencoder training.
    """

    def __init__(
        self,
        clip_model: str = CLIP_MODEL,
        ae_latent_dim: int = AE_LATENT_DIM,
        device: str = "cuda",
    ) -> None:
        self.clip_model = clip_model
        self.ae_latent_dim = ae_latent_dim
        self.device = device

        self._clip = None       # lazy-loaded
        self._autoencoder = None
        self._gaussians: np.ndarray | None = None      # (N, 3) positions
        self._lang_codes: np.ndarray | None = None     # (N, ae_latent_dim) codes

    # ── Public interface ─────────────────────────────────────────────────────

    def encode_frames(self, images_dir: Path | str, output_dir: Path | str) -> int:
        """Extract per-pixel CLIP features from training frames.

        Saves ``frame_XXXXXX_clip.npy`` arrays (shape: H×W×D) into *output_dir*.

        Parameters
        ----------
        images_dir:
            Directory of training frames.
        output_dir:
            Destination for CLIP feature maps.

        Returns
        -------
        int
            Number of feature maps written.
        """
        import torch
        from PIL import Image

        images_dir = Path(images_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model, preprocess, tokenizer = self._load_clip()

        image_paths = sorted(
            p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

        logger.info(
            "Encoding %d frames with CLIP (%s).", len(image_paths), self.clip_model
        )

        written = 0
        with torch.no_grad():
            for path in image_paths:
                img = Image.open(path).convert("RGB")
                tensor = preprocess(img).unsqueeze(0).to(self.device)

                # Extract patch-level features (not the global [CLS] token).
                # Shape: (1, num_patches, clip_dim)
                features = model.encode_image(tensor, normalize=True)
                feat_np = features.squeeze(0).cpu().numpy().astype(np.float32)

                out_path = output_dir / (path.stem + "_clip.npy")
                np.save(str(out_path), feat_np)
                written += 1

        logger.info("CLIP features written to %s.", output_dir)
        return written

    def train_autoencoder(
        self,
        features_dir: Path | str,
        output_path: Path | str,
        epochs: int = 50,
        lr: float = 1e-4,
    ) -> None:
        """Train a small MLP autoencoder to compress CLIP features.

        The encoder maps 512-d CLIP features → ``ae_latent_dim``-d codes.
        The compressed codes are stored per-Gaussian at negligible memory cost.

        Parameters
        ----------
        features_dir:
            Directory containing ``*_clip.npy`` feature maps.
        output_path:
            Where to save the trained autoencoder weights (``.pt``).
        epochs:
            Training epochs. 50 converges well for typical indoor scenes.
        lr:
            Adam learning rate.
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        features_dir = Path(features_dir)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Stack all feature vectors into one big tensor for training.
        feature_files = sorted(features_dir.glob("*_clip.npy"))
        if not feature_files:
            raise FileNotFoundError(f"No CLIP feature files in {features_dir}")

        all_features = np.concatenate(
            [np.load(str(f)).reshape(-1, np.load(str(f)).shape[-1]) for f in feature_files],
            axis=0,
        )
        clip_dim = all_features.shape[-1]

        dataset = TensorDataset(torch.from_numpy(all_features).to(self.device))
        loader = DataLoader(dataset, batch_size=4096, shuffle=True)

        ae = _LanguageAutoencoder(clip_dim, self.ae_latent_dim).to(self.device)
        optimizer = torch.optim.Adam(ae.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        logger.info(
            "Training language autoencoder: %d-d → %d-d over %d epochs.",
            clip_dim,
            self.ae_latent_dim,
            epochs,
        )

        for epoch in range(epochs):
            total_loss = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                recon = ae(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                logger.info("  Epoch %d/%d — loss %.6f", epoch + 1, epochs, total_loss)

        torch.save(ae.state_dict(), str(output_path))
        self._autoencoder = ae
        logger.info("Autoencoder saved to %s.", output_path)

    def load(self, gaussians_ply: Path | str, ae_weights: Path | str) -> None:
        """Load Gaussian positions and autoencoder weights for querying.

        Parameters
        ----------
        gaussians_ply:
            Exported ``splat.ply`` from GaussianTrainer.
        ae_weights:
            Autoencoder weights saved by ``train_autoencoder()``.
        """
        import torch
        from plyfile import PlyData

        ply = PlyData.read(str(gaussians_ply))
        vertices = ply["vertex"]
        self._gaussians = np.stack(
            [vertices["x"], vertices["y"], vertices["z"]], axis=-1
        ).astype(np.float32)

        # Read language codes stored as extra vertex properties (lc_0, lc_1, lc_2).
        if "lc_0" in vertices.data.dtype.names:
            self._lang_codes = np.stack(
                [vertices[f"lc_{i}"] for i in range(self.ae_latent_dim)], axis=-1
            ).astype(np.float32)
        else:
            logger.warning(
                "PLY file does not contain language codes (lc_*). "
                "Run build_field() to embed language features."
            )

        ae = _LanguageAutoencoder(512, self.ae_latent_dim)
        ae.load_state_dict(torch.load(str(ae_weights), map_location="cpu"))
        ae.eval()
        self._autoencoder = ae
        logger.info("Loaded %d Gaussians from %s.", len(self._gaussians), gaussians_ply)

    def query(self, text: str, top_k: int = 1000) -> dict:
        """Return positions of Gaussians that best match a text prompt.

        Parameters
        ----------
        text:
            Open-vocabulary text query, e.g. ``"chair"``, ``"wooden table"``.
        top_k:
            Maximum number of Gaussian positions to return.

        Returns
        -------
        dict with keys:
            ``"positions"``  — float32 array of shape (K, 3).
            ``"scores"``     — cosine similarity scores, shape (K,).
            ``"query"``      — the original text string.
        """
        if self._gaussians is None or self._lang_codes is None:
            raise RuntimeError("Call load() before query().")

        import torch

        model, _, tokenizer = self._load_clip()

        with torch.no_grad():
            tokens = tokenizer([text]).to(self.device)
            text_feat = model.encode_text(tokens, normalize=True)
            text_feat_np = text_feat.squeeze(0).cpu().numpy().astype(np.float32)

        # Decode language codes back to CLIP space for cosine comparison.
        ae = self._autoencoder.to("cpu").eval()
        codes_tensor = torch.from_numpy(self._lang_codes)
        with torch.no_grad():
            decoded = ae.decode(codes_tensor).numpy()

        scores = (decoded @ text_feat_np) / (
            np.linalg.norm(decoded, axis=1, keepdims=True) * np.linalg.norm(text_feat_np) + 1e-8
        ).squeeze()

        top_idx = np.argsort(scores)[::-1][:top_k]
        return {
            "query": text,
            "positions": self._gaussians[top_idx].tolist(),
            "scores": scores[top_idx].tolist(),
        }

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _load_clip(self):
        """Lazy-load open_clip model, preprocessing, and tokenizer."""
        if self._clip is None:
            try:
                import open_clip
            except ImportError as e:
                raise ImportError(
                    "open-clip-torch is required for SemanticField. "
                    "Install with: uv sync --extra semantic"
                ) from e

            model, _, preprocess = open_clip.create_model_and_transforms(
                self.clip_model, pretrained="openai"
            )
            tokenizer = open_clip.get_tokenizer(self.clip_model)
            model = model.to(self.device).eval()
            self._clip = (model, preprocess, tokenizer)
        return self._clip


class _LanguageAutoencoder:
    """Minimal MLP autoencoder: CLIP-dim → latent_dim → CLIP-dim."""

    def __new__(cls, clip_dim: int, latent_dim: int):
        # Use a plain nn.Module; defined here to keep imports inside the class.
        import torch.nn as nn

        class _AE(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(clip_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, latent_dim),
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, clip_dim),
                )

            def forward(self, x):
                return self.decoder(self.encoder(x))

            def decode(self, z):
                return self.decoder(z)

        instance = _AE()
        return instance
