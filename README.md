# Video to 3D Scene Reconstruction

Reconstruct a geometrically coherent 3D scene from a short indoor phone video.

**Core pipeline:** frame extraction → camera pose estimation (MASt3R-SfM) →
3D Gaussian Splatting (nerfstudio splatfacto) → PLY export + fly-through render.

**Optional extension:** open-vocabulary 3D semantic queries via LangSplat — type
`"chair"` and the matching Gaussians are highlighted in 3D.

---

## Quick Start

```bash
# 1. Clone and set up environment (requires uv)
git clone https://github.com/YunaGuo0909/video-to-3d
cd video-to-3d
uv sync

# 2. Run the full pipeline
#    Input videos live at /transfer/vt3/ on the training machine.
bash scripts/run_pipeline.sh \
    --video /transfer/vt3/room.mp4 \
    --output /transfer/vt3/outputs/my_scene

# 3. Export PLY and render fly-through
uv run python scripts/export_visualization.py \
    --experiment-dir /transfer/vt3/outputs/my_scene/nerfstudio/splatfacto/room_reconstruction/<timestamp> \
    --output-dir /transfer/vt3/outputs/my_scene/viz
```

The fly-through video and `splat.ply` are written to the `viz/` subdirectory.

---

## Installation

### Requirements
- Python 3.10+
- CUDA-capable GPU (tested on RTX 4070, Linux)
- [uv](https://docs.astral.sh/uv/)

### Core (reconstruction only)
```bash
uv sync
```

### With depth prior (Depth Anything V2 — improves indoor geometry)
```bash
uv sync --extra depth
```

### With semantic field (LangSplat — open-vocabulary 3D queries)
```bash
uv sync --extra semantic
```

### Development tools
```bash
uv sync --extra dev
```

### MASt3R-SfM setup (recommended pose estimator)
```bash
git clone https://github.com/naver/mast3r /opt/mast3r
cd /opt/mast3r && uv pip install -e .
```
Pass `--backend mast3r --mast3r-repo /opt/mast3r` to the pipeline script.
COLMAP (via nerfstudio) is used automatically if `--backend` is not specified.

---

## Data

Input videos and all outputs are stored under `/transfer/vt3/` on the training machine.

```
/transfer/vt3/
  room.mp4                    # example input video
  outputs/
    my_scene/
      images/                 # extracted frames
      transforms.json         # camera poses
      nerfstudio/             # training checkpoints
      viz/
        splat.ply             # Gaussian point cloud
        flythrough.mp4        # rendered demo video
```

---

## Pipeline Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | `/transfer/vt3/room.mp4` | Path to input video (MP4, MOV, AVI) |
| `--output` | `/transfer/vt3/outputs/reconstruction` | Root directory for all outputs |
| `--backend` | `colmap` | Pose estimator: `mast3r` or `colmap` |
| `--mast3r-repo` | — | Path to local MASt3R clone (required for `mast3r` backend) |
| `--depth` | off | Enable Depth Anything V2 prior |
| `--semantic` | off | Build LangSplat semantic field |
| `--mode` | `quality` | `debug` (3k iters, ~2 min) or `quality` (30k iters, ~20 min) |
| `--max-frames` | `300` | Cap on extracted frames |

---

## Semantic Query API

After running with `--semantic`:

```bash
# Start the API server
GAUSSIANS_PLY=/transfer/vt3/outputs/my_scene/viz/splat.ply \
AE_WEIGHTS=/transfer/vt3/outputs/my_scene/ae_weights.pt \
uvicorn src.api.server:app --host 0.0.0.0 --port 8000

# Query the 3D scene
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"text": "chair", "top_k": 500}'
```

The response contains the 3D positions (x, y, z) of the top matching Gaussians,
which can be visualised as a coloured point cloud overlay on the reconstruction.

---

## Capture Guidelines

Reconstruction quality depends significantly on video capture:

- **Lighting:** diffuse, even lighting — avoid direct sunlight and deep shadows.
- **Motion:** slow, deliberate camera movement — avoid fast pans or shakes.
- **Coverage:** orbit around the room at multiple heights; include overlapping views.
- **Scene:** avoid large textureless areas (blank white walls), mirrors, and
  transparent glass where possible.
- **Duration:** 60–90 seconds at 30 fps is sufficient for most small rooms.

---

## Running Tests

```bash
uv run pytest
```

Integration tests (require GPU + installed models) are skipped by default.
Run them with:
```bash
uv run pytest -m integration
```

---

## References

- Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering", SIGGRAPH 2023.
- Tancik et al., "Nerfstudio", SIGGRAPH 2023. https://docs.nerf.studio
- Duisterhof et al., "MASt3R-SfM", ICLR 2025. https://github.com/naver/mast3r
- Qin et al., "LangSplat: 3D Language Gaussian Splatting", CVPR 2024. https://github.com/minghanqin/LangSplat
- Yang et al., "Depth Anything V2", NeurIPS 2024.
- Turkulainen et al., "DN-Splatter", 2024. https://github.com/maturk/dn-splatter
