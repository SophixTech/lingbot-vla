#!/usr/bin/env python3
"""Run the bundled A2D converter with a CPU H.264 encoder.

The bundled bytecode currently requests NVENC unconditionally.  This wrapper
only changes that ffmpeg encoder selection to libx264; input data and the
converter's alignment logic are unchanged.
"""
from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path


def main() -> None:
    converter = Path(__file__).with_name("__pycache__") / "batch_convert_a2d_raw_to_lerobot.cpython-312.pyc"
    if not converter.exists():
        raise FileNotFoundError(converter)
    original_run = subprocess.run

    def run(*args, **kwargs):
        if args and isinstance(args[0], (list, tuple)):
            command = list(args[0])
            if "h264_nvenc" in command:
                command[command.index("h264_nvenc")] = "libx264"
                args = (command, *args[1:])
        return original_run(*args, **kwargs)

    subprocess.run = run
    runpy.run_path(str(converter), run_name="__main__")


if __name__ == "__main__":
    main()
