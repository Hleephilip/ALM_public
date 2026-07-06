from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .diffusion import (
    WIDE_IMAGE_MODEL_ID,
    PromptEmbeddings,
    StableDiffusionComponents,
    ddim_sigma,
    decode_latents,
    encode_prompt,
    load_stable_diffusion,
    precision_to_dtype,
    predict_conditional_noise,
    predict_noise_with_cfg,
    require_cuda,
)
from .image_utils import save_image, tensor_to_image


class ALMWideImageGenerator:

    def __init__(
        self,
        components: StableDiffusionComponents,
        *,
        num_patches: int,
        stride_latents: int,
        guidance_scale: float,
        w1: float,
        w2: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.components = components
        self.num_patches = num_patches
        self.stride_latents = stride_latents
        self.guidance_scale = guidance_scale
        self.w1 = w1
        self.w2 = w2
        self.device = device
        self.dtype = dtype
        self.patch_latents = int(components.unet.config.sample_size)


    @torch.inference_mode()
    def sample(
        self,
        latents: torch.Tensor,
        mask: torch.Tensor,
        embeddings: PromptEmbeddings,
    ) -> torch.Tensor:
        scheduler = self.components.scheduler
        for timestep in tqdm(scheduler.timesteps, desc="Wide image generation", leave=False):
            conditional_batch = embeddings.conditional.repeat(self.num_patches, 1, 1)
            noise_y = predict_conditional_noise(self.components.unet, latents, timestep, conditional_batch)
            sigma = ddim_sigma(scheduler, timestep, device=self.device, dtype=self.dtype)
            blended_latents: list[torch.Tensor] = []

            for index in range(1, self.num_patches):
                previous = torch.zeros_like(latents[index : index + 1])
                overlap_width = self.patch_latents - self.stride_latents
                previous[..., :overlap_width] = latents[index - 1 : index, ..., self.stride_latents:]
                blended = previous * (1 - mask) + latents[index : index + 1] * mask
                blended_latents.append(blended)

                noise_e = predict_conditional_noise(self.components.unet, blended, timestep, embeddings.conditional)
                alm_update = mask * (self.w1 * (noise_y[index : index + 1] - noise_e) - self.w2 * noise_e)
                latents[index : index + 1] += sigma * alm_update

            latents_before_step = latents.clone()
            guided_noise, _ = predict_noise_with_cfg(self.components.unet, latents, timestep, embeddings, self.guidance_scale)
            latents = scheduler.step(guided_noise, timestep, latents).prev_sample

            for index, blended in enumerate(blended_latents, start=1):
                latents[index : index + 1] -= self.w1 * sigma * (1 - mask) * (latents_before_step[index : index + 1] - blended)

        canvas_width = self.patch_latents + self.stride_latents * (self.num_patches - 1)
        canvas = torch.zeros((1, latents.shape[1], self.patch_latents, canvas_width), device=self.device, dtype=self.dtype)
        for index in range(self.num_patches - 1, -1, -1):
            start = self.stride_latents * index
            canvas[..., start : start + self.patch_latents] = latents[index : index + 1]

        return decode_latents(canvas, self.components.vae)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/wide_image"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-images", type=int, default=1)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--w1", type=float, default=1.0)
    parser.add_argument("--w2", type=float, default=0.001)
    parser.add_argument("--num-patches", type=int, default=5)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.num_images < 1:
        raise ValueError("--num-images must be positive.")
    if args.num_inference_steps < 1:
        raise ValueError("--num-inference-steps must be positive.")
    if args.guidance_scale < 1:
        raise ValueError("--guidance-scale must be at least 1.")
    if args.w1 < 0 or args.w2 < 0:
        raise ValueError("--w1 and --w2 must be non-negative.")
    if args.num_patches < 2:
        raise ValueError("--num-patches must be at least 2.")
    if not 0 < args.stride < 512:
        raise ValueError("--stride must be between 1 and 511 pixels.")
    if args.stride % 8:
        raise ValueError("--stride must be divisible by the VAE scale factor (8).")


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    device = require_cuda()
    dtype = precision_to_dtype(args.precision)

    print(f"[!] Loading {WIDE_IMAGE_MODEL_ID}")
    components = load_stable_diffusion(WIDE_IMAGE_MODEL_ID, device=device, dtype=dtype)
    components.scheduler.set_timesteps(args.num_inference_steps)
    embeddings = encode_prompt(args.prompt, components, device=device)

    vae_scale_factor = 2 ** (len(components.vae.config.block_out_channels) - 1)
    patch_latents = int(components.unet.config.sample_size)
    patch_pixels = patch_latents * vae_scale_factor
    if patch_pixels != 512:
        raise ValueError(f"Expected a 512px Stable Diffusion model, got {patch_pixels}px patches.")
    
    stride_latents = args.stride // vae_scale_factor
    sampler = ALMWideImageGenerator(
        components,
        num_patches=args.num_patches,
        stride_latents=stride_latents,
        guidance_scale=args.guidance_scale,
        w1=args.w1,
        w2=args.w2,
        device=device,
        dtype=dtype,
    )

    mask = torch.zeros((1, components.unet.config.in_channels, patch_latents, patch_latents), device=device, dtype=dtype)
    mask[..., patch_latents - stride_latents :] = 1
    canvas_width = patch_latents + stride_latents * (args.num_patches - 1)

    for seed in range(args.seed, args.seed + args.num_images):
        generator = torch.Generator(device=device).manual_seed(seed)
        canvas_noise = torch.randn((1, components.unet.config.in_channels, patch_latents, canvas_width), generator=generator, device=device, dtype=dtype)
        patch_noise = torch.cat([canvas_noise[..., stride_latents * index : stride_latents * index + patch_latents] for index in range(args.num_patches)], dim=0)
        generated = sampler.sample(patch_noise, mask, embeddings)
        image = tensor_to_image(generated)[0]
        output_path = args.output_dir / f"{seed}.png"
        save_image(image, output_path)

    print(f"[!] Done")


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
