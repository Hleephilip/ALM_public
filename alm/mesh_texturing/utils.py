from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as functional
from diffusers.models.attention_processor import Attention
from PIL import Image
from torchvision.transforms import Compose, GaussianBlur, InterpolationMode, Resize

DIRECTION_NAMES = ("", "front", "side", "back", "top", "bottom")


@torch.no_grad()
def composite_rendered_views(
    scheduler,
    backgrounds: torch.Tensor,
    foregrounds: Sequence[torch.Tensor],
    masks: Sequence[torch.Tensor],
    timestep: torch.Tensor,
    *,
    add_noise: bool,
) -> torch.Tensor:
    """Composite rendered foreground latents over diffusion backgrounds."""

    composited_images = []
    for background, foreground, mask in zip(backgrounds, foregrounds, masks):
        if timestep > 0 and add_noise:
            alpha = scheduler.alphas_cumprod[timestep]
            noise = torch.normal(0, 1, background.shape, device=background.device)
            background = (1 - alpha) * noise + alpha * background
        composited_images.append(foreground * mask + background * (1 - mask))
    return torch.stack(composited_images)


@torch.no_grad()
def get_conditioning_images(
    projection,
    output_size: int,
    blur_filter: int = 5,
) -> torch.Tensor:
    """Render depth maps for ControlNet conditioning."""

    _, _, depths, _, _, _ = projection.render_geometry(image_size=output_size)
    transforms = Compose([
        Resize((output_size, output_size), interpolation=InterpolationMode.BILINEAR, antialias=True),
        GaussianBlur(blur_filter, blur_filter // 3 + 1),
    ])

    condition = projection.decode_normalized_depth(depths).permute(0, 3, 1, 2)
    return transforms(condition)


@torch.no_grad()
def encode_latents(vae, images: torch.Tensor) -> torch.Tensor:
    images = (images - 0.5) * 2
    return vae.encode(images).latent_dist.sample() * vae.config.scaling_factor


@torch.no_grad()
def build_rgb_texture(
    vae,
    rgb_projection,
    latents: torch.Tensor,
    *,
    disable_voronoi: bool,
) -> torch.Tensor:
    """Decode final views and bake them into one RGB texture map."""

    result_views = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    resize = Resize(
        (rgb_projection.render_size, rgb_projection.render_size),
        interpolation=InterpolationMode.NEAREST_EXACT,
        antialias=True,
    )
    result_views = resize(result_views / 2 + 0.5).clamp(0, 1).unbind(0)
    result_texture = rgb_projection.bake_texture(
        views=result_views,
        exponent=6,
        fill_unobserved=True,
        disable_voronoi=disable_voronoi,
    )
    return result_texture


def replace_self_attention_processors(unet, attention_mask: list[list[int]]) -> None:
    """Install the cross-view self-attention processor on U-Net self-attention."""

    processors = dict(unet.attn_processors)
    for name in processors:
        if "attn1" in name:
            processors[name] = CrossViewAttnProcessor(attention_mask)
    unet.set_attn_processor(processors)


class CrossViewAttnProcessor:
    """Apply self-attention over neighboring rendered views."""

    def __init__(self, attention_mask: list[list[int]]) -> None:
        if not hasattr(functional, "scaled_dot_product_attention"):
            raise ImportError("Cross-view attention requires PyTorch 2.0 or newer.")
        self.attention_mask = attention_mask

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, channels = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        if len(self.attention_mask) != batch_size:
            raise ValueError("Attention-mask view count does not match the U-Net batch size.")

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        outputs = []
        for batch_index, neighbor_indices in enumerate(self.attention_mask):
            selected_keys = torch.stack([key[index] for index in neighbor_indices])
            selected_values = torch.stack([value[index] for index in neighbor_indices])
            selected_keys = selected_keys.view(-1, selected_keys.shape[1], attn.heads, head_dim).permute(2, 0, 1, 3).contiguous().view(1, attn.heads, -1, head_dim)
            selected_values = selected_values.view(-1, selected_values.shape[1], attn.heads, head_dim).permute(2, 0, 1, 3).contiguous().view(1, attn.heads, -1, head_dim)
            outputs.append(
                functional.scaled_dot_product_attention(
                    query[batch_index : batch_index + 1],
                    selected_keys,
                    selected_values,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                )
            )

        hidden_states = torch.cat(outputs)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


def prepare_directional_prompts(
    prompt: str,
    negative_prompt: str,
) -> tuple[list[str], list[str]]:
    positive = [prompt + f", {view} view." for view in DIRECTION_NAMES]
    negative = [negative_prompt + f", {view} view." for view in DIRECTION_NAMES]
    return positive, negative


@torch.no_grad()
def select_positive_prompt(embeddings: dict[str, torch.Tensor], pose) -> torch.Tensor:
    elevation, azimuth = pose
    if elevation > 30:
        return embeddings["top"]
    if elevation < -30:
        return embeddings["bottom"]
    if azimuth > 180:
        azimuth -= 360
    if -30 <= azimuth <= 30:
        return embeddings["front"]
    if azimuth <= -120 or azimuth >= 120:
        return embeddings["back"]
    return embeddings["side"]


@torch.no_grad()
def select_negative_prompt(embeddings: dict[str, torch.Tensor], pose) -> torch.Tensor:
    _, azimuth = pose
    if azimuth > 180:
        azimuth -= 360
    if -30 < azimuth < 30:
        return embeddings[""]
    return embeddings["front"]


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a CHW tensor in [0, 1] into an RGB PIL image."""

    tensor = tensor.detach().float().cpu().clamp(0, 1)
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError("Expected one image when converting a batch to PIL.")
        tensor = tensor[0]
    array = (tensor.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    return Image.fromarray(array[..., :3])

