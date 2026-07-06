from __future__ import annotations

import os
import runpy
import sys
from collections.abc import Callable, Sequence

from .paths import get_motion_paths


_COMMAND_HELP = {
    "complete": """usage: python -m alm.motion_completion complete [options]

Complete HumanML3D test motions with ALM.

options:
  --checkpoint PATH         CondMDI model checkpoint
  --data-root PATH          Prepared HumanML3D directory
  --glove-root PATH         CondMDI GloVe directory
  --output-dir PATH         Directory for this run
  --edit-mode MODE          first_half, middle_half, or last_half
  --transition-length N     Transition length for applicable edit modes
  --num-samples N           Number of test motions (default: 10)
  --num-repetitions N       Samples per motion (default: 3)
  --w_1 VALUE               ALM likelihood weight (default: 1.0)
  --w_2 VALUE               ALM score regularization (default: 0.005)
  --seed N                  Random seed (default: 10)
""",
    "edit": """usage: python -m alm.motion_completion edit [options]

Run the retained CondMDI text-conditioned editing interface.
Use --checkpoint, --data-root, --glove-root, and --output-dir to override paths.
""",
    "evaluate": """usage: python -m alm.motion_completion evaluate [options]

Evaluate ALM motion completion on HumanML3D.

options:
  --checkpoint PATH         CondMDI model checkpoint
  --data-root PATH          Prepared HumanML3D directory
  --glove-root PATH         CondMDI GloVe directory
  --evaluator-root PATH     Directory containing the t2m evaluator
  --output-dir PATH         Evaluation output directory
  --edit-mode MODE          first_half, middle_half, or last_half
  --transition-length N     Transition length for applicable edit modes
  --w_1 VALUE               ALM likelihood weight (default: 1.0)
  --w_2 VALUE               ALM score regularization (default: 0.005)
  --seed N                  Random seed (default: 10)
""",
    "render": """usage: python -m alm.motion_completion render --input VIDEO [options]

Convert a generated sample##_rep##.mp4 and its results.npy into SMPL meshes.

options:
  --input PATH              Generated sample video
  --body-model-root PATH    Directory containing body_models/smpl files
  --device N                CUDA device index (default: 0)
  --cpu                     Run SMPLify on CPU
""",
    "prepare": """usage: python -m alm.motion_completion prepare [options]

Prepare or validate CondMDI-compatible motion assets.

options:
  --download-support        Download GloVe and T2M evaluator archives
  --download-checkpoint     Download the CondMDI unconditional checkpoint
  --download-smpl           Download the SMPL archive used for rendering
  --check                   Validate all canonical runtime assets
  --force                   Replace an existing downloaded asset
  --data-root PATH          Override the HumanML3D directory
  --checkpoint PATH         Override the model checkpoint
""",
}

_PATH_OPTIONS = {
    "--data_root": "ALM_MOTION_DATA_ROOT",
    "--glove_root": "ALM_MOTION_GLOVE_ROOT",
    "--evaluator_root": "ALM_MOTION_EVALUATOR_ROOT",
    "--body_model_root": "ALM_MOTION_BODY_MODEL_ROOT",
    "--output_root": "ALM_MOTION_OUTPUT_ROOT",
    "--checkpoint": "ALM_MOTION_CHECKPOINT",
}


def _normalize_options(arguments: Sequence[str]) -> list[str]:
    normalized = []
    for argument in arguments:
        if argument.startswith("--"):
            name, separator, value = argument.partition("=")
            name = "--" + name[2:].replace("-", "_")
            argument = name + (separator + value if separator else "")
        normalized.append(argument)
    return normalized


def _pop_option(arguments: list[str], option: str) -> str | None:
    for index, argument in enumerate(arguments):
        if argument == option:
            if index + 1 >= len(arguments):
                raise SystemExit(f"{option.replace('_', '-')} requires a path")
            value = arguments[index + 1]
            del arguments[index:index + 2]
            return value
        prefix = option + "="
        if argument.startswith(prefix):
            value = argument[len(prefix):]
            del arguments[index]
            return value
    return None


def _configure_runtime_paths(arguments: list[str]) -> None:
    for option, environment_name in _PATH_OPTIONS.items():
        value = _pop_option(arguments, option)
        if value is not None:
            os.environ[environment_name] = value


def _invoke(function: Callable[[], None], arguments: list[str]) -> None:
    previous = sys.argv
    try:
        sys.argv = [previous[0], *arguments]
        function()
    finally:
        sys.argv = previous


def run(command: str, arguments: Sequence[str]) -> None:
    if command not in _COMMAND_HELP:
        raise SystemExit(f"Unknown motion command: {command}")
    if any(argument in {"-h", "--help"} for argument in arguments):
        print(_COMMAND_HELP[command], end="")
        return
    if command == "prepare":
        from .prepare import main as prepare_main

        prepare_arguments = _normalize_options(arguments)
        _configure_runtime_paths(prepare_arguments)
        _invoke(prepare_main, prepare_arguments)
        return

    backend_arguments = _normalize_options(arguments)
    _configure_runtime_paths(backend_arguments)

    if command != "render" and not any(
        value == "--model_path" or value.startswith("--model_path=")
        for value in backend_arguments
    ):
        backend_arguments.extend(
            ["--model_path", str(get_motion_paths().checkpoint)]
        )

    if command in {"complete", "evaluate"}:
        edit_mode = _pop_option(backend_arguments, "--edit_mode")
        if edit_mode is None:
            edit_mode = "last_half" if command == "complete" else "first_half"
        allowed_modes = {"first_half", "middle_half", "last_half"}
        if edit_mode not in allowed_modes:
            choices = ", ".join(sorted(allowed_modes))
            raise SystemExit(f"--edit-mode must be one of: {choices}")
        backend_arguments.extend(["--edit_mode", edit_mode])

    if command == "render":
        input_path = _pop_option(backend_arguments, "--input")
        if input_path is not None:
            backend_arguments.extend(["--input_path", input_path])
        from ._condmdi.visualize.render_mesh import main as render_main

        _invoke(render_main, backend_arguments)
    elif command == "complete":
        if "--alm" not in backend_arguments:
            backend_arguments.append("--alm")
        from ._condmdi.sample.edit_auto_text import main as complete_main

        _invoke(complete_main, backend_arguments)
    elif command == "edit":
        from ._condmdi.sample.edit import main as edit_main

        _invoke(edit_main, backend_arguments)
    else:
        if "--alm" not in backend_arguments:
            backend_arguments.append("--alm")
        previous = sys.argv
        try:
            sys.argv = [previous[0], *backend_arguments]
            runpy.run_module(
                "alm.motion_completion._condmdi.eval.eval_humanml_condmdi",
                run_name="__main__",
            )
        finally:
            sys.argv = previous


def main(arguments: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(
            "usage: python -m alm.motion_completion "
            "{complete,edit,evaluate,render,prepare} [options]\n"
        )
        print("ALM motion completion with the CondMDI HumanML3D backbone.")
        return
    run(arguments[0], arguments[1:])
