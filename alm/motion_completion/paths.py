from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parent

@dataclass(frozen=True)
class MotionPaths:
    """Resolved runtime locations for datasets, models, and outputs."""

    data_root: Path
    glove_root: Path
    evaluator_root: Path
    body_model_root: Path
    checkpoint: Path
    output_root: Path
    dataset_resources: Path
    smplify_resources: Path


def _path_from_environment(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


def get_motion_paths() -> MotionPaths:
    """Return explicit CLI overrides or the repository's canonical defaults."""

    motion_data = REPOSITORY_ROOT / "data" / "motion"
    return MotionPaths(
        data_root=_path_from_environment("ALM_MOTION_DATA_ROOT", motion_data / "HumanML3D"),
        glove_root=_path_from_environment("ALM_MOTION_GLOVE_ROOT", motion_data / "glove"),
        evaluator_root=_path_from_environment("ALM_MOTION_EVALUATOR_ROOT", motion_data / "evaluators"),
        body_model_root=_path_from_environment("ALM_MOTION_BODY_MODEL_ROOT", motion_data / "body_models"),
        checkpoint=_path_from_environment("ALM_MOTION_CHECKPOINT", REPOSITORY_ROOT / "checkpoints" / "motion" / "condmdi_uncond" / "model000500000.pt"),
        output_root=_path_from_environment("ALM_MOTION_OUTPUT_ROOT", REPOSITORY_ROOT / "outputs" / "motion_completion"),
        dataset_resources=(PACKAGE_ROOT / "_condmdi" / "resources" / "dataset").resolve(),
        smplify_resources=(PACKAGE_ROOT / "_condmdi" / "visualize" / "joints2smpl" / "smpl_models").resolve(),
    )

