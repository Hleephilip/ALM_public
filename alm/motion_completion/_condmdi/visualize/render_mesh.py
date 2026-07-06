import argparse
import os
import re
import shutil

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Convert a generated motion video into per-frame SMPL meshes."
    )
    parser.add_argument(
        "--input_path",
        required=True,
        help="Generated sample##_rep##.mp4 file.",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device index.")
    parser.add_argument("--cpu", action="store_true", help="Run SMPLify on CPU.")
    args = parser.parse_args()

    match = re.fullmatch(r"sample(\d+)_rep(\d+)\.mp4", os.path.basename(args.input_path))
    if match is None:
        parser.error("--input_path must end with sample##_rep##.mp4")
    sample_index, repetition_index = (int(value) for value in match.groups())

    results_path = os.path.join(os.path.dirname(args.input_path), "results.npy")
    if not os.path.isfile(results_path):
        raise FileNotFoundError(f"Missing generated motion data: {results_path}")

    from alm.motion_completion._condmdi.visualize import vis_utils

    output_params = args.input_path.replace(".mp4", "_smpl_params.npy")
    output_meshes = args.input_path.replace(".mp4", "_obj")
    if os.path.exists(output_meshes):
        shutil.rmtree(output_meshes)
    os.makedirs(os.path.join(output_meshes, "loc"))

    converter = vis_utils.npy2obj(
        results_path,
        sample_index,
        repetition_index,
        device=args.device,
        cuda=not args.cpu,
    )
    print(f"Saving OBJ files to [{os.path.abspath(output_meshes)}]")
    for frame_index in tqdm(range(converter.real_num_frames)):
        converter.save_obj(
            os.path.join(output_meshes, f"frame{frame_index:03d}.obj"),
            frame_index,
        )

    print(f"Saving SMPL parameters to [{os.path.abspath(output_params)}]")
    converter.save_npy(output_params)


if __name__ == "__main__":
    main()
