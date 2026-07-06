from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from .paths import REPOSITORY_ROOT, get_motion_paths


GLOVE_URL = (
    "https://drive.google.com/file/d/"
    "1cmXKUT31pqd7_XpJAiWEo1K81TMYHA5n/view?usp=sharing"
)
T2M_EVALUATOR_URL = (
    "https://drive.google.com/file/d/"
    "1DSaKqWX2HlwBtVH5l7DdW96jeYUIXsOP/view"
)
KIT_EVALUATOR_URL = (
    "https://drive.google.com/file/d/"
    "1tX79xk0fflp07EZ660Xz1RAFE33iEyJR/view"
)
SMPL_URL = "https://drive.google.com/uc?id=1INYlGA76ak_cKGzvpOV2Pe6RkYTlXTW2"
CHECKPOINT_URL = (
    "https://drive.google.com/file/d/"
    "1B0PYpmCXXwV0a5mhkgea_J2pOwhYy-k5/view?usp=sharing"
)


def _download(url: str, destination: Path) -> Path:
    try:
        import gdown
    except ImportError as error:
        raise RuntimeError("gdown is required; install environment-motion.yml first.") from error

    destination.parent.mkdir(parents=True, exist_ok=True)
    downloaded = gdown.download(
        url=url,
        output=str(destination),
        quiet=False,
        fuzzy=True,
    )
    if downloaded is None or not destination.is_file():
        raise RuntimeError(f"Download failed: {url}")
    return destination


def _find_extracted_directory(staging: Path, name: str) -> Path:
    direct = staging / name
    if direct.is_dir():
        return direct
    matches = [path for path in staging.rglob(name) if path.is_dir()]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one '{name}' directory in the downloaded archive; found {len(matches)}.")
    return matches[0]


def _install_zip_directory(
    archive: Path,
    directory_name: str,
    destination: Path,
    force: bool,
) -> None:
    staging = archive.parent / f"extract-{directory_name}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(staging)
        source = _find_extracted_directory(staging, directory_name)
        if destination.exists():
            if not force:
                print(f"Keeping existing [{destination}]")
                return
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        print(f"Installed [{destination}]")
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        archive.unlink(missing_ok=True)


def _download_directory(
    url: str,
    archive_name: str,
    directory_name: str,
    destination: Path,
    force: bool,
) -> None:
    if destination.exists() and not force:
        print(f"Keeping existing [{destination}]")
        return
    cache = REPOSITORY_ROOT / ".cache" / "alm-motion"
    archive = _download(url, cache / archive_name)
    _install_zip_directory(archive, directory_name, destination, force=force)


def download_support(force: bool) -> None:
    paths = get_motion_paths()
    _download_directory(GLOVE_URL, "glove.zip", "glove", paths.glove_root, force)
    _download_directory(T2M_EVALUATOR_URL, "t2m.zip", "t2m", paths.evaluator_root / "t2m", force)
    _download_directory(KIT_EVALUATOR_URL, "kit.zip", "kit", paths.evaluator_root / "kit", force)


def download_smpl(force: bool) -> None:
    paths = get_motion_paths()
    _download_directory(SMPL_URL, "smpl.zip", "smpl", paths.body_model_root / "smpl", force)


def download_checkpoint(force: bool) -> None:
    paths = get_motion_paths()
    destination = paths.checkpoint.parent
    if destination.exists() and not force:
        print(f"Keeping existing [{destination}]")
        return

    cache = REPOSITORY_ROOT / ".cache" / "alm-motion"
    archive = _download(CHECKPOINT_URL, cache / "condmdi_uncond.zip")
    if not zipfile.is_zipfile(archive):
        archive.unlink(missing_ok=True)
        raise RuntimeError("The CondMDI checkpoint download is not a ZIP archive.")
    _install_zip_directory(archive, "condmdi_uncond", destination, force=force)


def _missing_files(root: Path, relative_paths: tuple[str, ...]) -> list[Path]:
    return [root / relative for relative in relative_paths if not (root / relative).is_file()]


def validate() -> None:
    paths = get_motion_paths()
    missing = []
    missing.extend(_missing_files(paths.data_root, ("Mean.npy","Std.npy", "Mean_abs_3d.npy", "Std_abs_3d.npy", "test.txt")))
    for directory in ("new_joint_vecs", "new_joint_vecs_abs_3d", "texts"):
        candidate = paths.data_root / directory
        if not candidate.is_dir():
            missing.append(candidate)
    missing.extend(_missing_files(paths.glove_root, ("our_vab_data.npy", "our_vab_idx.pkl", "our_vab_words.pkl")))
    missing.extend(_missing_files(paths.evaluator_root, ("t2m/text_mot_match/model/finest.tar")))
    missing.extend(_missing_files(paths.body_model_root, ("smpl/J_regressor_extra.npy", "smpl/SMPL_NEUTRAL.pkl", "smpl/kintree_table.pkl", "smpl/smplfaces.npy")))
    missing.extend(_missing_files(paths.checkpoint.parent, ("args.json", paths.checkpoint.name)))
    if missing:
        lines = "\n".join(f"  - {path}" for path in missing)
        raise RuntimeError(f"Motion setup is incomplete:\n{lines}")

    with (paths.checkpoint.parent / "args.json").open() as stream:
        checkpoint_arguments = json.load(stream)
    expected = {
        "dataset": "humanml",
        "arch": "unet",
        "diffusion_steps": 1000,
        "noise_schedule": "cosine",
        "predict_xstart": True,
        "use_ddim": False,
        "abs_3d": True,
    }
    mismatches = {
        name: (checkpoint_arguments.get(name), value)
        for name, value in expected.items()
        if checkpoint_arguments.get(name) != value
    }
    if mismatches:
        details = "\n".join(
            f"  - {name}: found {actual!r}, expected {expected_value!r}"
            for name, (actual, expected_value) in mismatches.items()
        )
        raise RuntimeError(f"Unexpected CondMDI checkpoint configuration:\n{details}")
    print("Motion assets are complete and the checkpoint uses 1,000-step cosine DDPM.")


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--download_support", action="store_true")
    parser.add_argument("--download_checkpoint", action="store_true")
    parser.add_argument("--download_smpl", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()

    if arguments.download_support:
        download_support(arguments.force)
    if arguments.download_checkpoint:
        download_checkpoint(arguments.force)
    if arguments.download_smpl:
        download_smpl(arguments.force)
    if arguments.check or not any((arguments.download_support, arguments.download_checkpoint, arguments.download_smpl)):
        validate()


if __name__ == "__main__":
    main()
