#!/usr/bin/env python3
"""Extract the ACT224 videos into deterministic per-frame JPEG caches."""
from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
from pathlib import Path


def extract(video: Path, source_root: Path, output_root: Path, quality: int) -> str:
    relative = video.relative_to(source_root / "videos").with_suffix("")
    target = output_root / relative
    marker = target / ".complete"
    if marker.exists():
        return f"skip {relative}"
    target.mkdir(parents=True, exist_ok=True)
    for stale in target.glob("*.jpg"):
        stale.unlink()
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
        "-vsync", "0", "-q:v", str(quality), "-threads", "1", str(target / "%08d.jpg"),
    ]
    subprocess.run(command, check=True)
    marker.write_text("ok\n")
    return f"done {relative}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--quality", type=int, default=4)
    args = parser.parse_args()
    videos = sorted((args.source / "videos").rglob("*.mp4"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(extract, video, args.source, args.output, args.quality) for video in videos]
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            print(f"[{index}/{len(videos)}] {future.result()}", flush=True)


if __name__ == "__main__":
    main()
