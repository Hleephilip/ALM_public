from __future__ import annotations
import random
from pathlib import Path

import torch
from diffusers import ControlNetModel

from .config import MeshTexturingConfig, NEGATIVE_PROMPT, PROMPT_PREFIX
from .diffusion import load_mesh_diffusion_pipeline
from .renderer import UVProjection
from .utils import (
    DIRECTION_NAMES,
    build_rgb_texture,
    composite_rendered_views,
    encode_latents,
    get_conditioning_images,
    prepare_directional_prompts,
    select_negative_prompt,
    select_positive_prompt,
    tensor_to_pil,
)

COLOR_VALUES = {
    "black": (-1, -1, -1),
    "white": (1, 1, 1),
    "maroon": (0, -1, -1),
    "red": (1, -1, -1),
    "olive": (0, 0, -1),
    "yellow": (1, 1, -1),
    "green": (-1, 0, -1),
    "lime": (-1, 1, -1),
    "teal": (-1, 0, 0),
    "aqua": (-1, 1, 1),
    "navy": (-1, -1, 0),
    "blue": (-1, -1, 1),
    "purple": (0, -1, 0),
    "fuchsia": (1, -1, 1),
}


class ALMMeshTexturingPipeline:
    def __init__(self, config: MeshTexturingConfig) -> None:
        self.config = config
        self.device = torch.device("cuda")
        self.mesh_path = config.mesh.resolve()
        self.model = load_mesh_diffusion_pipeline()
        self._initialize_projections()

    def _initialize_projections(self) -> None:
        self.camera_poses = []
        self.attention_mask = []
        camera_count = len(self.config.camera_azimuths)
        front_index = 0
        back_index = 0
        front_difference = 360
        back_difference = 360

        for index, raw_azimuth in enumerate(self.config.camera_azimuths):
            azimuth = raw_azimuth + 360 if raw_azimuth < 0 else raw_azimuth
            self.camera_poses.append((0, azimuth))
            self.attention_mask.append([(camera_count + index - 1) % camera_count, index, (index + 1) % camera_count])
            if abs(azimuth) < front_difference:
                front_index = index
                front_difference = abs(azimuth)
            if abs(azimuth - 180) < back_difference:
                back_index = index
                back_difference = abs(azimuth - 180)

        if self.config.top_cameras:
            self.camera_poses.extend(((30, 0), (30, 180)))
            self.attention_mask.extend(([front_index, camera_count], [back_index, camera_count + 1]))

        device = self.model._execution_device
        self.latent_projection = UVProjection(
            texture_size=self.config.latent_texture_size,
            render_size=self.config.latent_view_size,
            sampling_mode="nearest",
            channels=4,
            device=device,
        )
        self.latent_projection.load_mesh(
            str(self.mesh_path),
            scale_factor=self.config.mesh_scale,
            autouv=self.config.auto_uv,
        )
        self.latent_projection.set_cameras_and_render_settings(
            self.camera_poses,
            centers=((0, 0, 0),),
            camera_distance=2.0,
        )

        self.rgb_projection = UVProjection(
            texture_size=self.config.rgb_texture_size,
            render_size=self.config.rgb_view_size,
            sampling_mode="nearest",
            channels=3,
            device=device,
        )
        self.rgb_projection.mesh = self.latent_projection.mesh.clone()
        self.rgb_projection.set_cameras_and_render_settings(
            self.camera_poses,
            centers=((0, 0, 0),),
            camera_distance=2.0,
        )
        _, _, _, cosine_maps, _, _ = self.rgb_projection.render_geometry()
        self.rgb_projection.calculate_cosine_weights(
            cosine_maps,
            fill_unobserved=False,
            disable_voronoi=self.config.disable_voronoi,
        )

        colors = torch.tensor(
            list(COLOR_VALUES.values()),
            dtype=self.model.text_encoder.dtype,
            device=device,
        ).reshape(-1, 3, 1, 1)
        color_images = colors * (0.5 * colors + 0.5)
        color_images = color_images.expand(-1, -1, self.config.latent_view_size * 8, self.config.latent_view_size * 8)
        color_latents = encode_latents(self.model.vae, color_images)
        self.color_latents = dict(zip(COLOR_VALUES, color_latents))

    def _forward_mapping(
        self,
        texture: torch.Tensor,
        view_index: int,
        backgrounds: torch.Tensor | None,
        *,
        current_index: int,
        background_colors: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if texture.ndim == 4:
            texture = texture.squeeze(0)
        self.latent_projection.set_texture_map(texture)
        rendered_views = self.latent_projection.render_textured_views()
        foregrounds = [view[:-1] for view in rendered_views]
        alpha_masks = [view[-1:] for view in rendered_views]
        timestep = self.model.scheduler.timesteps[current_index]

        object_mask = ((foregrounds[view_index][0] != 0) & (foregrounds[view_index][0] != 1)).to(torch.uint8)
        add_noise = False
        if backgrounds is None:
            backgrounds = torch.stack([self.color_latents[color] for color in background_colors])
            add_noise = True
        elif current_index == 0:
            timestep = timestep + 1

        latents = composite_rendered_views(
            self.model.scheduler,
            backgrounds,
            foregrounds,
            alpha_masks,
            timestep,
            add_noise=add_noise,
        ).type(backgrounds[0].dtype)
        return latents, object_mask

    def _inverse_mapping(
        self,
        views: torch.Tensor,
    ) -> torch.Tensor:
        return self.latent_projection.bake_texture(
            views=[view.to(self.latent_projection.device) for view in views],
            exponent=1,
            fill_unobserved=False,
            disable_voronoi=self.config.disable_voronoi,
        )

    def _noise_predictions(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        params: dict,
        *,
        classifier_free_guidance: bool,
        view_index: int | None = None,
    ) -> torch.Tensor:
        positive = params["positive_prompt_embeds"]
        negative = params["negative_prompt_embeds"]
        conditioning = params["conditioning_images"]
        attention_mask = self.attention_mask
        if view_index is not None:
            positive = positive[view_index : view_index + 1]
            negative = negative[view_index : view_index + 1]
            conditioning = conditioning[view_index : view_index + 1]
            attention_mask = [[0]]

        return self.model.compute_noise_predictions(
            latents,
            timestep,
            guidance_scale=self.config.guidance_scale if classifier_free_guidance else 1.0,
            positive_prompt_embeds=positive,
            negative_prompt_embeds=negative,
            conditioning_images=conditioning,
            attention_mask=attention_mask,
            controlnet_keep=params["controlnet_keep"],
            conditioning_scale=params["conditioning_scale"],
            classifier_free_guidance=classifier_free_guidance,
            current_index=params["current_index"],
        )

    def _ddim_step(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        scheduler = self.model.scheduler
        timestep_index = int(timestep.item())
        previous_timestep = timestep_index - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
        alpha = scheduler.alphas_cumprod[timestep_index]
        sigma = (1 - alpha).sqrt()
        predicted_original = (latents - sigma * noise) / alpha.sqrt()
        alpha_previous = scheduler.alphas_cumprod[previous_timestep] if previous_timestep >= 0 else scheduler.final_alpha_cumprod
        latent_coefficient = (1 - alpha_previous).sqrt() / (1 - alpha).sqrt()
        original_coefficient = alpha_previous.sqrt() - (alpha.sqrt() * (1 - alpha_previous).sqrt() / (1 - alpha).sqrt())
        return latent_coefficient * latents + original_coefficient * predicted_original

    def _alm_step(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        params: dict,
    ) -> torch.Tensor:
        scheduler = self.model.scheduler
        timestep_index = int(timestep.item())
        previous_timestep = timestep_index - scheduler.config.num_train_timesteps // len(scheduler.timesteps)
        alpha = scheduler.alphas_cumprod[timestep_index]
        alpha_previous = scheduler.alphas_cumprod[previous_timestep] if previous_timestep >= 0 else scheduler.final_alpha_cumprod
        sigma = ((1 - alpha_previous) / (1 - alpha)).sqrt()
        sigma = sigma * (1 - alpha / alpha_previous).sqrt()

        noise_y = self._noise_predictions(latents, timestep, params, classifier_free_guidance=False)
        for view_index in range(1, len(self.camera_poses)):
            texture = self._inverse_mapping(latents[:view_index])
            mapped, object_mask = self._forward_mapping(
                texture,
                view_index,
                None,
                current_index=params["current_index"],
                background_colors=params["background_colors"],
            )
            object_mask = object_mask.unsqueeze(0).unsqueeze(0)
            generated_mask = 1 - object_mask
            blended = mapped[view_index : view_index + 1] * object_mask + latents[view_index : view_index + 1] * generated_mask
            noise_e = self._noise_predictions(
                blended,
                timestep,
                params,
                classifier_free_guidance=False,
                view_index=view_index,
            )
            update = generated_mask * (self.config.w1 * (noise_y[view_index : view_index + 1] - noise_e) - self.config.w2 * noise_e)
            latents[view_index : view_index + 1] += sigma * update

        guided_noise = self._noise_predictions(
            latents,
            timestep,
            params,
            classifier_free_guidance=True,
        )
        latents_before_step = latents.clone()
        previous_latents = self._ddim_step(latents, timestep, guided_noise)

        for view_index in range(1, len(self.camera_poses)):
            texture = self._inverse_mapping(latents_before_step[:view_index])
            mapped, object_mask = self._forward_mapping(
                texture,
                view_index,
                None,
                current_index=params["current_index"],
                background_colors=params["background_colors"],
            )
            preservation = sigma * object_mask * (latents_before_step[view_index : view_index + 1] - mapped[view_index : view_index + 1])
            previous_latents[view_index : view_index + 1] -= self.config.w1 * preservation
        return previous_latents

    @torch.no_grad()
    def run(self) -> list[Path]:
        output_directories = []
        for sample_index in range(self.config.num_samples):
            sample_seed = self.config.seed + sample_index
            output_dir = self.config.output_dir
            if self.config.num_samples > 1:
                output_dir = output_dir / f"{sample_index:02d}"
            output_dir.mkdir(parents=True, exist_ok=True)
            self._run_sample(output_dir, sample_seed)
            output_directories.append(output_dir)

        print(f"[!] Done")
        return output_directories

    def _run_sample(self, output_dir: Path, seed: int) -> None:
        num_timesteps = self.model.scheduler.config.num_train_timesteps
        height = self.config.latent_view_size * 8
        width = height
        prompt = PROMPT_PREFIX + self.config.prompt
        negative_prompt = NEGATIVE_PROMPT
        control_guidance_start = [self.config.control_guidance_start]
        control_guidance_end = [self.config.control_guidance_end]

        self.model.check_inputs(
            prompt,
            torch.zeros((1, 3, height, width), device=self.model._execution_device),
            1,
            negative_prompt,
            None,
            None,
            self.config.conditioning_scale,
            control_guidance_start,
            control_guidance_end,
        )
        device = self.model._execution_device
        classifier_free_guidance = self.config.guidance_scale > 1
        controlnet = self.model.controlnet
        guess_mode = controlnet.config.global_pool_conditions if isinstance(controlnet, ControlNetModel) else False
        if guess_mode:
            raise ValueError("The fixed ALM mesh pipeline does not use guess mode.")

        directional_prompt, directional_negative = prepare_directional_prompts(prompt, negative_prompt)
        prompt_embeds = self.model._encode_prompt(
            directional_prompt,
            device,
            1,
            classifier_free_guidance,
            directional_negative,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=None,
        )
        negative_embeds, positive_embeds = torch.chunk(prompt_embeds, 2)
        positive_by_direction = dict(zip(DIRECTION_NAMES, positive_embeds))
        negative_by_direction = dict(zip(DIRECTION_NAMES, negative_embeds))

        self.latent_projection.to(device)
        conditioning_images = get_conditioning_images(self.latent_projection, height).type(positive_embeds.dtype)

        self.model.scheduler.set_timesteps(self.config.steps, device=device)
        timesteps = self.model.scheduler.timesteps
        generator = torch.manual_seed(seed)
        latents = self.model.prepare_latents(
            len(self.camera_poses),
            self.model.unet.config.in_channels,
            height,
            width,
            positive_embeds.dtype,
            device,
            generator,
            None,
        )
        self.latent_projection.set_noise_texture()
        noise_views = self.latent_projection.render_textured_views()
        latents = composite_rendered_views(
            self.model.scheduler,
            latents,
            [view[:-1] for view in noise_views],
            [view[-1:] for view in noise_views],
            timesteps[0] + 1,
            add_noise=True,
        ).type(latents.dtype)

        extra_step_kwargs = self.model.prepare_extra_step_kwargs(generator, 0.0)
        controlnet_keep = []
        for index in range(len(timesteps)):
            keep = 1.0 - float(index / len(timesteps) < control_guidance_start[0] or (index + 1) / len(timesteps) > control_guidance_end[0])
            controlnet_keep.append(keep)

        positive_prompt_embeds = torch.stack([select_positive_prompt(positive_by_direction, pose) for pose in self.camera_poses])
        negative_prompt_embeds = torch.stack([select_negative_prompt(negative_by_direction, pose) for pose in self.camera_poses])
        background_colors = [random.choice(list(COLOR_VALUES)) for _ in self.camera_poses]
        conditioning_scale = self.config.conditioning_scale

        with self.model.progress_bar(total=len(timesteps)) as progress_bar:
            for index, timestep in enumerate(timesteps):
                params = {
                    "positive_prompt_embeds": positive_prompt_embeds,
                    "negative_prompt_embeds": negative_prompt_embeds,
                    "conditioning_images": conditioning_images,
                    "controlnet_keep": controlnet_keep,
                    "conditioning_scale": conditioning_scale,
                    "current_index": index,
                    "background_colors": background_colors,
                }
                if timestep > (1 - self.config.mvd_end) * num_timesteps:
                    latents = self._alm_step(latents, timestep, params)
                else:
                    noise = self._noise_predictions(
                        latents,
                        timestep,
                        params,
                        classifier_free_guidance=True,
                    )
                    step = self.model.scheduler.step(noise, timestep, latents, **extra_step_kwargs, return_dict=True)
                    latents = step["prev_sample"]
                progress_bar.update()

        rgb_texture = build_rgb_texture(
            self.model.vae,
            self.rgb_projection,
            latents,
            disable_voronoi=self.config.disable_voronoi,
        )
        self.latent_projection.save_mesh(str(output_dir / "textured.obj"), rgb_texture.permute(1, 2, 0))
        self.rgb_projection.set_texture_map(rgb_texture)
        textured_views = self.rgb_projection.render_textured_views()
        preview = torch.cat(textured_views, dim=-1)[:3]
        tensor_to_pil(preview).save(output_dir / "textured_views_rgb.png")

