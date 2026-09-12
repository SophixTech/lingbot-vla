#!/usr/bin/env python3
"""Create a read-only-source ACT dataset with 224x224 H.264 camera videos."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


def link_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--crf", type=int, default=23)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output == source:
        raise ValueError("output must differ from source")
    output.mkdir(parents=True, exist_ok=True)
    for subtree in ("data", "meta"):
        for path in (source / subtree).rglob("*"):
            if path.is_file():
                link_copy(path, output / path.relative_to(source))
    info_path = output / "meta/info.json"
    info = json.loads(info_path.read_text())
    for key, feature in info["features"].items():
        if key.startswith("observation.images."):
            feature["shape"] = [args.size, args.size, 3]
            feature["info"]["video.height"] = args.size
            feature["info"]["video.width"] = args.size
    info_path.unlink()
    info_path.write_text(json.dumps(info, indent=4) + "\n")
    videos = sorted((source / "videos").rglob("*.mp4"))
    for index, video in enumerate(videos, start=1):
        target = output / video.relative_to(source)
        if target.exists() and target.stat().st_size > 4096:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".partial.mp4")
        temporary.unlink(missing_ok=True)
        command = [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
            "-vf", f"scale={args.size}:{args.size}:force_original_aspect_ratio=decrease,pad={args.size}:{args.size}:(ow-iw)/2:(oh-ih)/2",
            "-an", "-c:v", "h264_nvenc", "-preset", "p4", "-cq", str(args.crf), "-pix_fmt", "yuv420p", str(temporary),
        ]
        print(f"[{index}/{len(videos)}] {video.relative_to(source)}", flush=True)
        subprocess.run(command, check=True)
        temporary.replace(target)


if __name__ == "__main__":
    main()
