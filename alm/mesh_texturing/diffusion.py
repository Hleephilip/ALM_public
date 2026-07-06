from __future__ import annotations

from typing import List, Tuple, Union

import torch
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDIMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from alm.diffusion import INPAINTING_MODEL_ID

from .utils import replace_self_attention_processors

CONTROLNET_MODEL_ID = "lllyasviel/control_v11f1p_sd15_depth"


class ALMControlNetPipeline(StableDiffusionControlNetPipeline):
    """ControlNet pipeline exposing the noise predictions used by ALM."""

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet: Union[
            ControlNetModel,
            List[ControlNetModel],
            Tuple[ControlNetModel],
        ],
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = False,
    ) -> None:
        super().__init__(
            vae,
            text_encoder,
            tokenizer,
            unet,
            controlnet,
            scheduler,
            safety_checker,
            feature_extractor,
            requires_safety_checker,
        )
        self.scheduler = scheduler
        self.model_cpu_offload_seq = "vae->text_encoder->unet->vae"
        self.enable_model_cpu_offload()
        self.enable_vae_slicing()
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def compute_noise_predictions(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        *,
        guidance_scale: float,
        positive_prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        conditioning_images: torch.Tensor,
        attention_mask: list[list[int]],
        controlnet_keep: list[float],
        conditioning_scale: float,
        classifier_free_guidance: bool,
        current_index: int,
    ) -> torch.Tensor:
        """Predict conditional noise, optionally applying classifier-free guidance."""

        model_input = self.scheduler.scale_model_input(latents, timestep).type(
            self.unet.dtype
        )
        prompt_groups = {"positive": positive_prompt_embeds}
        if classifier_free_guidance:
            prompt_groups["negative"] = negative_prompt_embeds

        predictions = {}
        for name, prompt_embeds in prompt_groups.items():
            scale = conditioning_scale * controlnet_keep[current_index]
            down_samples, middle_sample = self.controlnet(
                model_input,
                timestep,
                encoder_hidden_states=prompt_embeds,
                controlnet_cond=conditioning_images,
                conditioning_scale=scale,
                return_dict=False,
            )
            replace_self_attention_processors(self.unet, attention_mask)
            predictions[name] = self.unet(
                model_input,
                timestep,
                encoder_hidden_states=prompt_embeds,
                down_block_additional_residuals=down_samples,
                mid_block_additional_residual=middle_sample,
                return_dict=False,
            )[0]

        positive = predictions["positive"]
        if not classifier_free_guidance:
            return positive
        return predictions["negative"] + guidance_scale * (positive - predictions["negative"])


def load_mesh_diffusion_pipeline() -> ALMControlNetPipeline:

    controlnet = ControlNetModel.from_pretrained(CONTROLNET_MODEL_ID, variant="fp16", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(INPAINTING_MODEL_ID, controlnet=controlnet, torch_dtype=torch.float16)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    model = ALMControlNetPipeline(**pipe.components)
    model.text_encoder.requires_grad_(False)
    model.unet.requires_grad_(False)
    model.vae.requires_grad_(False)
    model.controlnet.requires_grad_(False)
    return model
