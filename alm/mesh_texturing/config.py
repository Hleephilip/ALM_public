from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

PROMPT_PREFIX = "Best quality, extremely detailed "
NEGATIVE_PROMPT = "oversmoothed, blurry, depth of field, out of focus, low quality, bloom, glowing effect."


@dataclass(frozen=True)
class MeshTexturingConfig:
    """Runtime configuration."""
    mesh: Path
    prompt: str
    output_dir: Path
    seed: int = 0
    num_samples: int = 1
    steps: int = 30
    guidance_scale: float = 15.5
    conditioning_scale: float = 0.7
    latent_view_size: int = 96
    latent_texture_size: int = 1536
    rgb_view_size: int = 768
    rgb_texture_size: int = 1024
    mesh_scale: float = 1.0
    auto_uv: bool = False
    disable_voronoi: bool = False
    w1: float = 0.5
    w2: float = 0.001

    mvd_end: float = 0.8
    camera_azimuths: tuple[int, ...] = (-180, -135, -90, -45, 0, 45, 90, 135)
    top_cameras: bool = True
    control_guidance_start: float = 0.0
    control_guidance_end: float = 0.99


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mesh_texturing"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=15.5)
    parser.add_argument("--conditioning-scale", type=float, default=0.7)
    parser.add_argument("--latent-view-size", type=int, default=96)
    parser.add_argument("--latent-texture-size", type=int, default=1536)
    parser.add_argument("--rgb-view-size", type=int, default=768)
    parser.add_argument("--rgb-texture-size", type=int, default=1024)
    parser.add_argument("--mesh-scale", type=float, default=1.0)
    parser.add_argument("--w1", type=float, default=0.5)
    parser.add_argument("--w2", type=float, default=0.001)
    parser.add_argument("--auto-uv", action="store_true", help="Rebuild UV coordinates with xatlas even when the OBJ contains UVs.")
    parser.add_argument("--disable-voronoi", action="store_true", help="Disable nearest-neighbor filling of unobserved texture pixels.")
    return parser


def parse_config(argv: Sequence[str] | None = None) -> MeshTexturingConfig:
    args = build_parser().parse_args(argv)
    config = MeshTexturingConfig(
        mesh=args.mesh,
        prompt=args.prompt,
        output_dir=args.output_dir,
        seed=args.seed,
        num_samples=args.num_samples,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        conditioning_scale=args.conditioning_scale,
        latent_view_size=args.latent_view_size,
        latent_texture_size=args.latent_texture_size,
        rgb_view_size=args.rgb_view_size,
        rgb_texture_size=args.rgb_texture_size,
        mesh_scale=args.mesh_scale,
        auto_uv=args.auto_uv,
        disable_voronoi=args.disable_voronoi,
        w1=args.w1,
        w2=args.w2,
    )
    validate_config(config)
    return config


def validate_config(config: MeshTexturingConfig) -> None:
    if not config.mesh.is_file():
        raise FileNotFoundError(f"Mesh does not exist: {config.mesh}")
    if config.mesh.suffix.lower() != ".obj":
        raise ValueError("ALM mesh texturing currently supports OBJ meshes only.")
    if config.num_samples < 1:
        raise ValueError("--num-samples must be positive.")
    if config.steps < 1:
        raise ValueError("--steps must be positive.")
    if config.guidance_scale <= 1:
        raise ValueError("--guidance-scale must be greater than 1.")
    if config.conditioning_scale <= 0:
        raise ValueError("ControlNet conditioning scale must be positive.")
    if config.mesh_scale <= 0:
        raise ValueError("--mesh-scale must be positive.")
    if config.w1 < 0 or config.w2 < 0:
        raise ValueError("--w1 and --w2 must be non-negative.")
    if not config.prompt.strip():
        raise ValueError("--prompt must not be empty.")
    for name, value in (
        ("latent view size", config.latent_view_size),
        ("latent texture size", config.latent_texture_size),
        ("RGB view size", config.rgb_view_size),
        ("RGB texture size", config.rgb_texture_size),
    ):
        if value < 1:
            raise ValueError(f"{name} must be positive.")
