"""
Tests for SemanticField.

CLIP and autoencoder tests use small synthetic data to avoid requiring
GPU resources in CI. Actual CLIP inference is mocked.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.models.semantic_field import SemanticField, _LanguageAutoencoder


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


# ── Autoencoder unit tests ─────────────────────────────────────────────────────


def test_autoencoder_forward_shape():
    """Autoencoder output must have the same shape as its input."""
    import torch

    ae = _LanguageAutoencoder(clip_dim=512, latent_dim=3)
    x = torch.randn(8, 512)
    out = ae(x)
    assert out.shape == (8, 512)


def test_autoencoder_decode_shape():
    import torch

    ae = _LanguageAutoencoder(clip_dim=512, latent_dim=3)
    z = torch.randn(8, 3)
    decoded = ae.decode(z)
    assert decoded.shape == (8, 512)


def test_autoencoder_trains_without_error(temp_dir):
    """train_autoencoder should complete without raising on small data."""
    # Write a small synthetic features directory.
    feat_dir = temp_dir / "features"
    feat_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(3):
        feat = rng.random((10, 512), dtype=np.float32)
        np.save(str(feat_dir / f"frame_{i:06d}_clip.npy"), feat)

    sf = SemanticField(device="cpu")
    out_weights = temp_dir / "ae.pt"

    # Run for just 2 epochs to keep the test fast.
    sf.train_autoencoder(feat_dir, out_weights, epochs=2, lr=1e-3)
    assert out_weights.exists()


# ── Query unit tests ───────────────────────────────────────────────────────────


def test_query_raises_without_load():
    sf = SemanticField(device="cpu")
    with pytest.raises(RuntimeError, match="load"):
        sf.query("chair")


def test_query_returns_expected_keys(temp_dir):
    """query() should return dict with 'positions', 'scores', 'query'."""
    import torch

    sf = SemanticField(device="cpu")

    # Inject synthetic Gaussians and language codes directly.
    n = 100
    sf._gaussians = np.random.randn(n, 3).astype(np.float32)
    sf._lang_codes = np.random.randn(n, 3).astype(np.float32)
    sf._autoencoder = _LanguageAutoencoder(clip_dim=512, latent_dim=3)

    # Mock CLIP so no model is downloaded.
    mock_model = MagicMock()
    mock_model.encode_text.return_value = torch.randn(1, 512)
    sf._clip = (mock_model, MagicMock(), MagicMock())

    result = sf.query("chair", top_k=10)

    assert "positions" in result
    assert "scores" in result
    assert "query" in result
    assert len(result["positions"]) <= 10
    assert result["query"] == "chair"
