from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .diffusion import (
    INPAINTING_MODEL_ID,
    PromptEmbeddings,
    StableDiffusionComponents,
    ddim_sigma,
    decode_latents,
    encode_image,
    encode_prompt,
    load_stable_diffusion,
    precision_to_dtype,
    predict_conditional_noise,
    predict_noise_with_cfg,
    require_cuda,
)
from .image_utils import (
    find_images,
    image_to_tensor,
    load_mask,
    load_rgb,
    save_image,
    tensor_to_image,
)

class ALMInpainter:
    def __init__(
        self,
        components: StableDiffusionComponents,
        *,
        guidance_scale: float,
        w1: float,
        w2: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.components = components
        self.guidance_scale = guidance_scale
        self.w1 = w1
        self.w2 = w2
        self.device = device
        self.dtype = dtype

    @torch.inference_mode()
    def invert(
        self,
        latent: torch.Tensor,
        embeddings: PromptEmbeddings,
    ) -> list[torch.Tensor]:
        scheduler = self.components.scheduler
        trajectory: list[torch.Tensor] = []

        for timestep in tqdm(scheduler.timesteps.flip(0), desc="DDIM inversion", leave=False):
            noise = predict_conditional_noise(self.components.unet, latent, timestep, embeddings.conditional)
            next_timestep = int(timestep.item())
            current_timestep = max(0, next_timestep - scheduler.config.num_train_timesteps // len(scheduler.timesteps))
            alpha_current = scheduler.alphas_cumprod[current_timestep].to(device=self.device, dtype=self.dtype)
            alpha_next = scheduler.alphas_cumprod[next_timestep].to(device=self.device, dtype=self.dtype)
            latent = (latent - (1 - alpha_current).sqrt() * noise) * (alpha_next.sqrt() / alpha_current.sqrt()) + (1 - alpha_next).sqrt() * noise
            trajectory.append(latent.clone())

        return trajectory

    @torch.inference_mode()
    def sample(
        self,
        latent: torch.Tensor,
        mask: torch.Tensor,
        source_trajectory: list[torch.Tensor],
        embeddings: PromptEmbeddings,
    ) -> torch.Tensor:
        scheduler = self.components.scheduler
        if len(source_trajectory) != len(scheduler.timesteps):
            raise ValueError("The inversion and sampling trajectories must have equal length.")

        for index, timestep in enumerate(tqdm(scheduler.timesteps, desc="Inpainting", leave=False)):
            source_latent = source_trajectory[-(index + 1)]
            noise_y = predict_conditional_noise(self.components.unet, latent, timestep, embeddings.conditional)
            blended_latent = source_latent * (1 - mask) + latent * mask
            noise_e = predict_conditional_noise(self.components.unet, blended_latent, timestep, embeddings.conditional)
            sigma = ddim_sigma(scheduler, timestep, device=self.device, dtype=self.dtype)
            alm_update = mask * (self.w1 * (noise_y - noise_e) - self.w2 * noise_e)
            latent = latent + sigma * alm_update

            latent_before_step = latent.clone()
            guided_noise, _ = predict_noise_with_cfg(self.components.unet, latent, timestep, embeddings, self.guidance_scale)
            latent = scheduler.step(guided_noise, timestep, latent).prev_sample
            latent = latent - self.w1 * sigma * (1 - mask) * (latent_before_step - source_latent)

        return decode_latents(latent, self.components.vae)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/inpainting"))
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--w1", type=float, default=1.0)
    parser.add_argument("--w2", type=float, default=0.005)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.num_inference_steps < 1:
        raise ValueError("--num-inference-steps must be positive.")
    if args.guidance_scale < 1:
        raise ValueError("--guidance-scale must be at least 1.")
    if args.w1 < 0 or args.w2 < 0:
        raise ValueError("--w1 and --w2 must be non-negative.")


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    input_paths = find_images(args.input)
    binary_mask, blend_mask = load_mask(args.mask)
    device = require_cuda()
    dtype = precision_to_dtype(args.precision)

    print(f"[!] Loading {INPAINTING_MODEL_ID}")
    components = load_stable_diffusion(INPAINTING_MODEL_ID, device=device, dtype=dtype)
    components.scheduler.set_timesteps(args.num_inference_steps)
    embeddings = encode_prompt(args.prompt, components, device=device)
    sampler = ALMInpainter(
        components,
        guidance_scale=args.guidance_scale,
        w1=args.w1,
        w2=args.w2,
        device=device,
        dtype=dtype
    )

    pixel_mask = binary_mask.astype(np.float32)
    mask_tensor = torch.from_numpy(binary_mask).permute(2, 0, 1).unsqueeze(0)
    mask_tensor = mask_tensor.to(device=device, dtype=dtype)

    for input_path in input_paths:
        source = load_rgb(input_path)
        masked_source = source.astype(np.float32) * (1 - pixel_mask)
        masked_source_tensor = image_to_tensor(masked_source, device=device, dtype=dtype)
        masked_source_latent = encode_image(masked_source_tensor, components.vae)
        latent_mask = F.interpolate(mask_tensor, size=masked_source_latent.shape[-2:], mode="nearest")

        source_trajectory = sampler.invert(masked_source_latent, embeddings)
        generator = torch.Generator(device=device).manual_seed(args.seed)
        initial_latent = torch.randn(masked_source_latent.shape, generator=generator, device=device, dtype=dtype)
        generated_tensor = sampler.sample(
            initial_latent,
            latent_mask,
            source_trajectory,
            embeddings,
        )
        generated = tensor_to_image(generated_tensor)[0]
        blended = generated.astype(np.float32) * blend_mask + masked_source * (1 - blend_mask)
        blended = np.clip(np.rint(blended), 0, 255).astype(np.uint8)

        stem = input_path.stem
        save_image(source, args.output_dir / "source" / f"{stem}.png")
        save_image(masked_source.astype(np.uint8), args.output_dir / "masked_source" / f"{stem}.png")
        save_image(generated, args.output_dir / "generated" / f"{stem}.png")
        save_image(blended, args.output_dir / "blended" / f"{stem}.png")

        print("[!] Done")
    
    save_image((binary_mask[..., 0] * 255).astype(np.uint8), args.output_dir / "mask.png")


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
