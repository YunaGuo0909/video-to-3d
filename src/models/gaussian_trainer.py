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

import functools
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=4)
def _probe_method_flags(method: str) -> frozenset:
    """Return all --flag strings found in 'ns-train <method> --help'.

    Cached per method (up to 4) so the subprocess runs at most once per
    process per method.  Falls back to an empty set on any error.
    """
    try:
        result = subprocess.run(
            ["ns-train", method, "--help"],
            capture_output=True, text=True, timeout=30,
        )
        return frozenset(re.findall(r"--[\w.-]+", result.stdout + result.stderr))
    except Exception:
        return frozenset()


def _probe_splatfacto_flags() -> frozenset:
    """Convenience wrapper — probes splatfacto flags."""
    return _probe_method_flags("splatfacto")


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
    """

    max_num_iterations: int = ITER_QUALITY
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    experiment_name: str = "room_reconstruction"
    use_depth_prior: bool = False
    use_dn_splatter: bool = False           # use ns-train dn-splatter instead of splatfacto
    use_scale_regularization: bool = True   # penalise needle-shaped Gaussians
    cull_alpha_thresh: float = 0.05         # prune near-transparent floaters
    stop_split_at: int = 10_000            # stop densification early (default 15k causes late needles)
    densify_grad_thresh: float = 0.0004    # less aggressive densification (default 0.0002 too high)
    camera_res_scale_factor: float = 0.5   # downscale training images (0.5 = half res, 4x less tile memory)
    depth_lambda: float = 0.2              # weight for depth loss (dn-splatter requires > 0 when use_depth_loss=True)


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

        import os
        env = os.environ.copy()
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        result = subprocess.run(cmd, text=True, env=env)
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
        method = "dn-splatter" if cfg.use_dn_splatter else "splatfacto"
        cmd = [
            "ns-train", method,
            "--data", str(dataset_dir),
            "--output-dir", str(cfg.output_dir),
            "--experiment-name", cfg.experiment_name,
            "--max-num-iterations", str(cfg.max_num_iterations),
        ]

        if cfg.use_dn_splatter:
            # dn-splatter activates depth supervision automatically when
            # transforms.json contains depth_file_path entries — no extra flag.
            # Probe dn-splatter's own flags for optional quality settings.
            supported = _probe_method_flags("dn-splatter")
        else:
            supported = _probe_splatfacto_flags()

        scale_flag = "--pipeline.model.use-scale-regularization"
        if cfg.use_scale_regularization:
            if not supported or scale_flag in supported:
                cmd += [scale_flag, "True"]
            else:
                logger.warning("%s not available in this method — skipping.", scale_flag)

        cull_flag = "--pipeline.model.cull-alpha-thresh"
        if not supported or cull_flag in supported:
            cmd += [cull_flag, str(cfg.cull_alpha_thresh)]
        else:
            logger.warning("%s not available — skipping.", cull_flag)

        stop_flag = "--pipeline.model.stop-split-at"
        if not supported or stop_flag in supported:
            cmd += [stop_flag, str(cfg.stop_split_at)]
        else:
            logger.warning("%s not available — skipping.", stop_flag)

        densify_flag = "--pipeline.model.densify-grad-thresh"
        if not supported or densify_flag in supported:
            cmd += [densify_flag, str(cfg.densify_grad_thresh)]
        else:
            logger.warning("%s not available — skipping.", densify_flag)

        if cfg.use_dn_splatter:
            # dn-splatter requires --pipeline.model.use-depth-loss True to
            # activate depth supervision and the normal-nerfstudio dataparser
            # which properly loads depth_file_path into the training batch.
            # The standard nerfstudio-data dataparser does NOT load depth.
            depth_flag = "--pipeline.model.use-depth-loss"
            if not supported or depth_flag in supported:
                cmd += [depth_flag, "True"]
            else:
                logger.warning("%s not available in dn-splatter — depth supervision disabled.", depth_flag)
            # depth_lambda must be > 0 when use_depth_loss=True (asserted in dn_model.py)
            cmd += ["--pipeline.model.depth-lambda", str(cfg.depth_lambda)]
        elif cfg.use_depth_prior:
            # splatfacto fallback: depth-loss flag (may not be available)
            depth_flag = "--pipeline.model.use-depth-loss"
            if not supported or depth_flag in supported:
                cmd += [depth_flag, "True"]
            else:
                logger.warning(
                    "Depth prior requested but %s not available — "
                    "use dn-splatter for depth supervision.",
                    depth_flag,
                )

        # Downscale training images to reduce tile-rasterization VRAM.
        # dn-splatter OOMs on 16 GB at full res with a dense init PLY.
        # camera-res-scale-factor is a datamanager flag → must come before dataparser.
        if cfg.camera_res_scale_factor != 1.0:
            cmd += ["--pipeline.datamanager.camera-res-scale-factor",
                    str(cfg.camera_res_scale_factor)]

        # Data parser sub-command must come last.
        # dn-splatter requires its own normal-nerfstudio dataparser to load
        # depth maps from transforms.json into the training batch.
        dataparser = "normal-nerfstudio" if cfg.use_dn_splatter else "nerfstudio-data"
        cmd += [dataparser]

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
        method = "dn-splatter" if self.config.use_dn_splatter else "splatfacto"
        base = self.config.output_dir / self.config.experiment_name / method
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
