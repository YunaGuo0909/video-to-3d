"""
gaussian_trainer.py
====================
Wrapper around nerfstudio's ``ns-train splatfacto`` command.

3D Gaussian Splatting (3DGS) learns a set of 3D Gaussians with position,
colour, opacity, and covariance from multi-view images. splatfacto is
nerfstudio's production implementation, offering faster training than the
original 3DGS repo and a built-in web viewer.

Reference
---------
- Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering",
  SIGGRAPH 2023.
- nerfstudio splatfacto: https://docs.nerf.studio/nerfology/methods/splat.html
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# nerfstudio splatfacto default training iterations.
ITER_DEBUG = 3_000   # ~2 min on RTX 4070 — for fast iteration
ITER_QUALITY = 30_000  # ~20 min on RTX 4070 — for final submission


@dataclass
class TrainingConfig:
    """Subset of splatfacto hyperparameters exposed for easy tuning.

    All fields map directly to nerfstudio CLI flags:
      ns-train splatfacto --<field> <value> ...

    Parameters
    ----------
    max_num_iterations:
        Total training steps. Use ITER_DEBUG for sanity checks.
    output_dir:
        Root directory where nerfstudio writes checkpoints and exports.
    experiment_name:
        Sub-folder name under output_dir.
    use_depth_prior:
        If True, pass ``--pipeline.model.use-depth-loss True`` to enable
        DN-Splatter-style monocular depth supervision. Requires depth/
        directory in the dataset.
    cull_alpha_thresh:
        Prune Gaussians with opacity below this value. Higher = more
        aggressive pruning of floaters. Default raised from 0.005 → 0.01.
    densify_grad_thresh:
        Spawn new Gaussians when the view-space gradient exceeds this.
        Lower = more Gaussians in fine-detail regions. Default lowered
        from 0.0002 → 0.0001.
    densify_until_iter:
        Stop densification after this iteration. Extended from 15 000 →
        20 000 to allow more densification passes on orbital captures.
    """

    max_num_iterations: int = ITER_QUALITY
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    experiment_name: str = "room_reconstruction"
    use_depth_prior: bool = False
    cull_alpha_thresh: float = 0.01
    densify_grad_thresh: float = 0.0001
    densify_until_iter: int = 20_000


class GaussianTrainer:
    """Launch and monitor a nerfstudio splatfacto training run.

    Parameters
    ----------
    config:
        Training configuration. Defaults to quality settings.
    """

    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = config or TrainingConfig()
        self._check_nerfstudio()

    # ── Public interface ─────────────────────────────────────────────────────

    def train(self, dataset_dir: Path | str) -> Path:
        """Start splatfacto training on a processed nerfstudio dataset.

        Parameters
        ----------
        dataset_dir:
            Directory containing ``transforms.json`` and ``images/``.

        Returns
        -------
        Path
            Path to the nerfstudio experiment directory (contains checkpoints
            and the exported ``splat.ply``).
        """
        dataset_dir = Path(dataset_dir)
        self._validate_dataset(dataset_dir)

        cmd = self._build_command(dataset_dir)
        logger.info("Starting splatfacto training:\n  %s", " ".join(cmd))

        result = subprocess.run(cmd, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                "ns-train splatfacto failed. Check the output above for details."
            )

        output_path = self._find_output_dir()
        logger.info("Training complete. Outputs at: %s", output_path)
        return output_path

    def export_ply(self, experiment_dir: Path | str) -> Path:
        """Export the trained Gaussians to a ``.ply`` point cloud file.

        The PLY file can be viewed in Blender, MeshLab, or the online
        3DGS viewer at https://antimatter15.com/splat/.

        Parameters
        ----------
        experiment_dir:
            nerfstudio experiment directory (returned by ``train()``).

        Returns
        -------
        Path
            Path to the exported ``splat.ply``.
        """
        experiment_dir = Path(experiment_dir)
        config_path = self._find_config_yaml(experiment_dir)

        ply_out = experiment_dir / "exports" / "splat.ply"
        ply_out.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ns-export", "gaussian-splat",
            "--load-config", str(config_path),
            "--output-dir", str(ply_out.parent),
        ]

        logger.info("Exporting Gaussian PLY: %s", " ".join(cmd))
        result = subprocess.run(cmd, text=True)
        if result.returncode != 0:
            raise RuntimeError("ns-export failed. Check output above.")

        logger.info("PLY exported to %s", ply_out)
        return ply_out

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _build_command(self, dataset_dir: Path) -> list[str]:
        cfg = self.config
        cmd = [
            "ns-train", "splatfacto",
            "--data", str(dataset_dir),
            "--output-dir", str(cfg.output_dir),
            "--experiment-name", cfg.experiment_name,
            "--max-num-iterations", str(cfg.max_num_iterations),
            "--pipeline.model.cull-alpha-thresh", str(cfg.cull_alpha_thresh),
            "--pipeline.model.densify-grad-thresh", str(cfg.densify_grad_thresh),
            "--pipeline.model.stop-split-at", str(cfg.densify_until_iter),
        ]

        if cfg.use_depth_prior:
            # DN-Splatter depth supervision flag
            cmd += ["--pipeline.model.use-depth-loss", "True"]

        # Data parser sub-command must come last
        cmd += ["nerfstudio-data"]

        return cmd

    def _validate_dataset(self, dataset_dir: Path) -> None:
        transforms = dataset_dir / "transforms.json"
        images = dataset_dir / "images"
        if not transforms.exists() or not images.exists():
            raise FileNotFoundError(
                f"Dataset missing transforms.json or images/ in {dataset_dir}. "
                "Run DatasetBuilder.validate() first."
            )

    def _find_output_dir(self) -> Path:
        """Locate the most recent nerfstudio experiment directory."""
        base = self.config.output_dir / "splatfacto" / self.config.experiment_name
        if not base.exists():
            return base  # Return expected path even if not yet created
        # nerfstudio appends a timestamp sub-folder; pick the latest.
        subdirs = sorted(base.iterdir(), key=lambda p: p.stat().st_mtime)
        return subdirs[-1] if subdirs else base

    def _find_config_yaml(self, experiment_dir: Path) -> Path:
        configs = list(experiment_dir.glob("**/config.yml"))
        if not configs:
            raise FileNotFoundError(f"No config.yml found in {experiment_dir}")
        return configs[0]

    @staticmethod
    def _check_nerfstudio() -> None:
        if not shutil.which("ns-train"):
            raise EnvironmentError(
                "nerfstudio is not installed or not on PATH. "
                "Install with: uv pip install nerfstudio"
            )
