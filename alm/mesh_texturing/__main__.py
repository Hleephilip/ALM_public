from __future__ import annotations

import os
import random

import numpy as np
import torch

from .config import parse_config

INITIALIZATION_SEED = 2024


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    config = parse_config()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for mesh texturing. Set CUDA_VISIBLE_DEVICES to select a GPU.")

    seed_everything(INITIALIZATION_SEED)
    from .pipeline import ALMMeshTexturingPipeline

    pipeline = ALMMeshTexturingPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
