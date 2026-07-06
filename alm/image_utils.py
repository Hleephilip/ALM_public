from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def find_images(path: Path) -> list[Path]:
    """Resolve one input image or all supported images in a directory."""

    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {path.suffix}")
        return [path]

    images = sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise ValueError(f"No JPG or PNG images found in: {path}")
    return images


def load_rgb(path: Path, size: int = 512) -> np.ndarray:
    """Load an RGB image and resize."""

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)


def load_mask(path: Path, size: int = 512) -> tuple[np.ndarray, np.ndarray]:

    if not path.is_file():
        raise FileNotFoundError(f"Mask does not exist: {path}")
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {path}")
    if mask.shape != (size, size):
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)

    binary = (mask >= 128).astype(np.float32)[..., None]
    blurred = cv2.GaussianBlur(binary[..., 0], (21, 21), 0)[..., None]
    blend_mask = 1.0 - (1.0 - binary) * (1.0 - blurred)
    return binary, blend_mask


def image_to_tensor(
    image: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert an RGB uint8/float image to BCHW in [-1, 1]."""

    tensor = torch.from_numpy(np.ascontiguousarray(image)).to(device=device, dtype=dtype)
    tensor = tensor / 127.5 - 1.0
    return tensor.permute(2, 0, 1).unsqueeze(0)


def tensor_to_image(image: torch.Tensor) -> np.ndarray:
    """Convert BCHW in [-1, 1] to an RGB uint8 array."""

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    return (image * 255).round().astype(np.uint8)


def save_image(image: np.ndarray, path: Path) -> None:
    """Save an RGB uint8 image, creating its parent directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)
