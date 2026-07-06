from __future__ import annotations
from dataclasses import dataclass
import torch
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel

try:
    from huggingface_hub.errors import HfHubHTTPError
except ImportError: 
    from huggingface_hub.utils import HfHubHTTPError
from transformers import CLIPTextModel, CLIPTokenizer

INPAINTING_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
WIDE_IMAGE_MODEL_ID = "stabilityai/stable-diffusion-2-1-base"
LATENT_SCALING_FACTOR = 0.18215


@dataclass
class StableDiffusionComponents:

    vae: AutoencoderKL
    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModel
    unet: UNet2DConditionModel
    scheduler: DDIMScheduler


@dataclass(frozen=True)
class PromptEmbeddings:
    """Conditional and unconditional CLIP embeddings for one prompt."""

    conditional: torch.Tensor
    unconditional: torch.Tensor

    def repeat(self, batch_size: int) -> PromptEmbeddings:
        return PromptEmbeddings(conditional=self.conditional.repeat(batch_size, 1, 1), unconditional=self.unconditional.repeat(batch_size, 1, 1))


def load_stable_diffusion(
    model_id: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> StableDiffusionComponents:
    """Load only the Stable Diffusion components used during ALM inference."""

    tokenizer = _load_pretrained(CLIPTokenizer, model_id, subfolder="tokenizer")
    text_encoder = _load_pretrained(CLIPTextModel, model_id, subfolder="text_encoder", torch_dtype=dtype)
    vae = _load_pretrained(AutoencoderKL, model_id, subfolder="vae", torch_dtype=dtype)
    unet = _load_pretrained(UNet2DConditionModel, model_id, subfolder="unet", torch_dtype=dtype)
    scheduler = _load_pretrained(DDIMScheduler, model_id, subfolder="scheduler")

    text_encoder.to(device).eval().requires_grad_(False)
    vae.to(device).eval().requires_grad_(False)
    unet.to(device).eval().requires_grad_(False)

    return StableDiffusionComponents(
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        unet=unet,
        scheduler=scheduler,
    )


def _load_pretrained(component_class, model_id: str, **kwargs):
    """Load from Hugging Face, falling back to an existing local snapshot."""

    try:
        return component_class.from_pretrained(model_id, **kwargs)
    except HfHubHTTPError as online_error:
        try:
            return component_class.from_pretrained(model_id, local_files_only=True, **kwargs)
        except OSError:
            raise RuntimeError(f"Could not access pretrained model '{model_id}'. Accept its Hugging Face terms and run `hf auth login`, then retry.") from online_error


@torch.inference_mode()
def encode_prompt(
    prompt: str,
    components: StableDiffusionComponents,
    *,
    device: torch.device,
) -> PromptEmbeddings:
    """Encode a text prompt and the empty prompt used for CFG."""

    text_inputs = components.tokenizer(
        [prompt],
        padding="max_length",
        max_length=components.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    unconditional_inputs = components.tokenizer(
        [""],
        padding="max_length",
        max_length=components.tokenizer.model_max_length,
        return_tensors="pt",
    )

    conditional = components.text_encoder(text_inputs.input_ids.to(device))[0]
    unconditional = components.text_encoder(unconditional_inputs.input_ids.to(device))[0]
    return PromptEmbeddings(conditional=conditional, unconditional=unconditional)


@torch.inference_mode()
def encode_image(image: torch.Tensor, vae: AutoencoderKL) -> torch.Tensor:
    """Encode an image tensor in [-1, 1] into Stable Diffusion latents."""

    return vae.encode(image).latent_dist.mode() * LATENT_SCALING_FACTOR


@torch.inference_mode()
def decode_latents(latents: torch.Tensor, vae: AutoencoderKL) -> torch.Tensor:
    """Decode Stable Diffusion latents into an image tensor in [-1, 1]."""

    return vae.decode(latents / LATENT_SCALING_FACTOR).sample


@torch.inference_mode()
def predict_conditional_noise(
    unet: UNet2DConditionModel,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    conditional_embeddings: torch.Tensor,
) -> torch.Tensor:

    timestep = timestep.to(latents.device)
    return unet(latents, timestep, encoder_hidden_states=conditional_embeddings).sample


@torch.inference_mode()
def predict_noise_with_cfg(
    unet: UNet2DConditionModel,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    embeddings: PromptEmbeddings,
    guidance_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:

    batch_size = latents.shape[0]
    embeddings = embeddings.repeat(batch_size)
    model_input = torch.cat([latents, latents], dim=0)
    encoder_hidden_states = torch.cat([embeddings.conditional, embeddings.unconditional], dim=0)
    timestep = timestep.to(latents.device)
    noise = unet(model_input, timestep, encoder_hidden_states=encoder_hidden_states).sample
    conditional_noise, unconditional_noise = noise.chunk(2)
    guided_noise = unconditional_noise + guidance_scale * (conditional_noise - unconditional_noise)
    return guided_noise, conditional_noise


def ddim_sigma(
    scheduler: DDIMScheduler,
    timestep: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:

    current_timestep = int(timestep.item())
    step_size = scheduler.config.num_train_timesteps // len(scheduler.timesteps)
    previous_timestep = current_timestep - step_size
    alpha_current = scheduler.alphas_cumprod[current_timestep]
    if previous_timestep >= 0:
        alpha_previous = scheduler.alphas_cumprod[previous_timestep]
    else:
        alpha_previous = scheduler.final_alpha_cumprod

    sigma = ((1 - alpha_previous) / (1 - alpha_current)).sqrt()
    sigma = sigma * (1 - alpha_current / alpha_previous).sqrt()
    return sigma.to(device=device, dtype=dtype)


def require_cuda() -> torch.device:

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for these Stable Diffusion experiments. Set CUDA_VISIBLE_DEVICES to select a GPU.")
    return torch.device("cuda")


def precision_to_dtype(precision: str) -> torch.dtype:

    if precision == "fp16":
        return torch.float16
    if precision == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported precision: {precision}")

