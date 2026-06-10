#!/usr/bin/env bash
# run_pipeline.sh
# ===============
# End-to-end pipeline: video → 3D Gaussian Splatting scene.
#
# Dataset location on the training machine: /transfer/vt3/
# Place input videos there, e.g. /transfer/vt3/room.mp4
#
# Usage:
#   bash scripts/run_pipeline.sh \
#       --video /transfer/vt3/room.mp4 \
#       --output /transfer/vt3/outputs/my_scene
#
# Optional flags:
#   --backend        mast3r | colmap           (default: colmap)
#   --mast3r-repo    /path/to/mast3r           (required if --backend mast3r)
#   --depth                                    (enable Depth Anything V2 prior)
#   --semantic                                 (build LangSplat field after training)
#   --mode           debug | quality           (default: quality)
#   --max-frames     N                         (cap on extracted frames, default: 300)
#   --blur-threshold F                         (Laplacian sharpness cutoff, default: 80)
#   --min-frame-gap  N                         (min ms between frames, default: 250)
#
# Requirements:
#   Activate the project venv before running:
#     source /transfer/vt3/.venv/bin/activate

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
DATA_ROOT="/transfer/vt3"

VIDEO=""
OUTPUT=""
BACKEND="colmap"
MAST3R_REPO=""
USE_DEPTH=false
USE_SEMANTIC=false
MODE="quality"
MAX_FRAMES=300
BLUR_THRESHOLD=80.0
MIN_FRAME_GAP=250

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --video)           VIDEO="$2";          shift 2 ;;
        --output)          OUTPUT="$2";         shift 2 ;;
        --backend)         BACKEND="$2";        shift 2 ;;
        --mast3r-repo)     MAST3R_REPO="$2";   shift 2 ;;
        --depth)           USE_DEPTH=true;      shift   ;;
        --semantic)        USE_SEMANTIC=true;   shift   ;;
        --mode)            MODE="$2";           shift 2 ;;
        --max-frames)      MAX_FRAMES="$2";     shift 2 ;;
        --blur-threshold)  BLUR_THRESHOLD="$2"; shift 2 ;;
        --min-frame-gap)   MIN_FRAME_GAP="$2";  shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

VIDEO="${VIDEO:-${DATA_ROOT}/room.mp4}"
# Auto-derive output dir from video stem + mode if not explicitly set.
# e.g. room01.mp4 + quality → /transfer/vt3/outputs/room01_quality
if [[ -z "$OUTPUT" ]]; then
    VIDEO_STEM=$(basename "${VIDEO%.*}")
    OUTPUT="${DATA_ROOT}/outputs/${VIDEO_STEM}_${MODE}"
fi

echo "============================================================"
echo "  Video-to-3D Pipeline"
echo "  Data root     : $DATA_ROOT"
echo "  Video         : $VIDEO"
echo "  Output        : $OUTPUT"
echo "  Backend       : $BACKEND"
echo "  Mode          : $MODE"
echo "  Blur threshold: $BLUR_THRESHOLD"
echo "  Min frame gap : ${MIN_FRAME_GAP}ms"
echo "  Python        : $(python --version)"
echo "============================================================"

# ── Step 1: Frame extraction ──────────────────────────────────────────────────
echo ""
echo "[1/5] Extracting frames..."
python - <<EOF
from pathlib import Path
from src.data.video_processor import VideoProcessor

proc = VideoProcessor(
    blur_threshold=$BLUR_THRESHOLD,
    min_frame_gap_ms=$MIN_FRAME_GAP,
    max_frames=$MAX_FRAMES,
)
stats = proc.process(Path("$VIDEO"), Path("$OUTPUT"))
print(stats.summary())
EOF

# ── Step 2: Pose estimation ───────────────────────────────────────────────────
echo ""
echo "[2/5] Estimating camera poses (backend: $BACKEND)..."
python - <<EOF
from pathlib import Path
from src.data.pose_estimator import PoseEstimator, PoseBackend
from src.data.dataset_builder import DatasetBuilder

backend = PoseBackend("$BACKEND")
mast3r_repo = "$MAST3R_REPO" or None

est = PoseEstimator(backend=backend, mast3r_repo=mast3r_repo)
est.estimate(Path("$OUTPUT") / "images", Path("$OUTPUT"))

db = DatasetBuilder(Path("$OUTPUT"))
stats = db.validate()
print(stats.summary())
if not stats.is_valid:
    raise RuntimeError("Dataset validation failed — check logs above.")
EOF

# ── Step 3a (optional): Depth prior ──────────────────────────────────────────
if [[ "$USE_DEPTH" == "true" ]]; then
    echo ""
    echo "[3a/5] Predicting depth maps (Depth Anything V2)..."
    python - <<EOF
from pathlib import Path
from src.models.depth_prior import DepthPrior
from src.data.dataset_builder import DatasetBuilder

dp = DepthPrior(model_size="small")
dp.predict(Path("$OUTPUT") / "images", Path("$OUTPUT") / "depth_raw")

db = DatasetBuilder(Path("$OUTPUT"))
db.attach_depth_maps(Path("$OUTPUT") / "depth_raw")
print("Depth maps attached to dataset.")
EOF
fi

# ── Step 3: Gaussian Splatting training ───────────────────────────────────────
echo ""
echo "[3/5] Training 3D Gaussian Splatting..."

if [[ "$MODE" == "debug" ]]; then
    MAX_ITERS=3000
else
    MAX_ITERS=30000
fi

python - <<EOF
from pathlib import Path
from src.models.gaussian_trainer import GaussianTrainer, TrainingConfig

cfg = TrainingConfig(
    max_num_iterations=$MAX_ITERS,
    output_dir=Path("$OUTPUT") / "nerfstudio",
    use_depth_prior=$([[ "$USE_DEPTH" == "true" ]] && echo "True" || echo "False"),
)
trainer = GaussianTrainer(cfg)
exp_dir = trainer.train(Path("$OUTPUT"))
ply_path = trainer.export_ply(exp_dir)
print(f"PLY exported: {ply_path}")
(Path("$OUTPUT") / ".ply_path").write_text(str(ply_path))
(Path("$OUTPUT") / ".exp_dir").write_text(str(exp_dir))
EOF

# ── Step 4: Export visualization ──────────────────────────────────────────────
echo ""
echo "[4/5] Exporting visualization to viz/..."
EXP_DIR=$(cat "$OUTPUT/.exp_dir")

if [[ "$MODE" == "debug" ]]; then
    # Debug: PLY only, skip the slow fly-through render
    python scripts/export_visualization.py \
        --experiment-dir "$EXP_DIR" \
        --output-dir "$OUTPUT/viz" \
        --skip-video
else
    # Quality: PLY + fly-through video
    python scripts/export_visualization.py \
        --experiment-dir "$EXP_DIR" \
        --output-dir "$OUTPUT/viz"
fi

echo "  PLY  : $OUTPUT/viz/splat.ply"
if [[ "$MODE" != "debug" ]]; then
    echo "  Video: $OUTPUT/viz/flythrough.mp4"
fi

# ── Step 5 (optional): Semantic field ────────────────────────────────────────
if [[ "$USE_SEMANTIC" == "true" ]]; then
    echo ""
    echo "[5/5] Building LangSplat semantic field..."
    PLY_PATH=$(cat "$OUTPUT/.ply_path")

    python - <<EOF
from pathlib import Path
from src.models.semantic_field import SemanticField

sf = SemanticField(device="cuda")

feat_dir = Path("$OUTPUT") / "clip_features"
sf.encode_frames(Path("$OUTPUT") / "images", feat_dir)

ae_weights = Path("$OUTPUT") / "ae_weights.pt"
sf.train_autoencoder(feat_dir, ae_weights)

print("Semantic field ready.")
print(f"To serve: GAUSSIANS_PLY=$PLY_PATH AE_WEIGHTS={ae_weights} uvicorn src.api.server:app")
EOF
fi

echo ""
echo "============================================================"
echo "  Pipeline complete!"
echo "  Outputs : $OUTPUT/"
echo "  PLY     : $OUTPUT/viz/splat.ply"
echo "============================================================"
