#!/usr/bin/env python3
"""Summarize episode duration distribution for package raw data.

The script reads only per-episode meta_info.json files. Duration is taken from
the explicit metadata field first, and clip_end_time - clip_start_time is used
only as a fallback.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median


DEFAULT_ROOT = Path("/home/bjtc/Sophix/data/package_raw_data")


@dataclass(frozen=True)
class EpisodeDuration:
    episode_dir: str
    duration_s: float
    source: str
    clip_start_time: float | None
    clip_end_time: float | None
    metadata_duration_s: float | None


def parse_number(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return None
    else:
        return None
    if not math.isfinite(number):
        return None
    return number


def load_episode_duration(meta_path: Path) -> tuple[EpisodeDuration | None, str | None]:
    try:
        meta = json.loads(meta_path.read_text())
    except Exception as exc:
        return None, f"{meta_path}: cannot read JSON: {exc}"

    metadata_duration = parse_number(meta.get("duration"))
    clip_start = parse_number(meta.get("clip_start_time"))
    clip_end = parse_number(meta.get("clip_end_time"))
    clip_duration = None
    if clip_start is not None and clip_end is not None:
        clip_duration = clip_end - clip_start

    if metadata_duration is not None:
        duration = metadata_duration
        source = "duration"
    elif clip_duration is not None:
        duration = clip_duration
        source = "clip_time_delta"
    else:
        return None, f"{meta_path}: missing duration and clip timestamps"

    if duration < 0:
        return None, f"{meta_path}: negative duration {duration}"

    warning = None
    if metadata_duration is not None and clip_duration is not None and abs(metadata_duration - clip_duration) > 1e-6:
        warning = (
            f"{meta_path}: duration={metadata_duration:g} differs from "
            f"clip_end_time-clip_start_time={clip_duration:g}"
        )

    row = EpisodeDuration(
        episode_dir=meta_path.parent.name,
        duration_s=duration,
        source=source,
        clip_start_time=clip_start,
        clip_end_time=clip_end,
        metadata_duration_s=metadata_duration,
    )
    return row, warning


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("percentile() needs at least one value")
    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def collect(root: Path) -> tuple[list[EpisodeDuration], list[str]]:
    rows: list[EpisodeDuration] = []
    warnings: list[str] = []
    for meta_path in sorted(root.glob("*/meta_info.json")):
        row, warning = load_episode_duration(meta_path)
        if row is not None:
            rows.append(row)
        if warning is not None:
            warnings.append(warning)
    return rows, warnings


def make_bins(values: list[float], bin_size: float, origin: float) -> list[tuple[float, float, int]]:
    counts: Counter[int] = Counter()
    for value in values:
        bin_index = math.floor((value - origin) / bin_size)
        counts[bin_index] += 1
    return [(origin + i * bin_size, origin + (i + 1) * bin_size, count) for i, count in sorted(counts.items())]


def format_seconds(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def bar(count: int, total: int, width: int) -> str:
    if total <= 0:
        return ""
    filled = round(width * count / total)
    return "#" * filled


def print_summary(
    root: Path,
    rows: list[EpisodeDuration],
    warnings: list[str],
    bin_size: float,
    bin_origin: float,
    top_n: int,
) -> None:
    values = sorted(row.duration_s for row in rows)
    total = len(values)
    print(f"root: {root}")
    print(f"episodes_with_duration: {total}")
    print(f"warnings: {len(warnings)}")
    if not values:
        return

    print()
    print("summary_seconds")
    print(f"  min:    {format_seconds(values[0])}")
    print(f"  p05:    {format_seconds(percentile(values, 0.05))}")
    print(f"  p25:    {format_seconds(percentile(values, 0.25))}")
    print(f"  median: {format_seconds(median(values))}")
    print(f"  mean:   {mean(values):.3f}")
    print(f"  p75:    {format_seconds(percentile(values, 0.75))}")
    print(f"  p95:    {format_seconds(percentile(values, 0.95))}")
    print(f"  max:    {format_seconds(values[-1])}")

    print()
    print(f"histogram_seconds bin_size={format_seconds(bin_size)} origin={format_seconds(bin_origin)}")
    print("  range           count  percent  bar")
    for lo, hi, count in make_bins(values, bin_size, bin_origin):
        pct = 100.0 * count / total
        label = f"[{format_seconds(lo)}, {format_seconds(hi)})"
        print(f"  {label:<14} {count:5d}  {pct:6.2f}%  {bar(count, total, 40)}")

    exact_counts = Counter(values)
    print()
    print("exact_duration_counts")
    print("  seconds  count  percent")
    for duration, count in sorted(exact_counts.items()):
        pct = 100.0 * count / total
        print(f"  {format_seconds(duration):>7}  {count:5d}  {pct:6.2f}%")

    if top_n > 0:
        print()
        print(f"shortest_{top_n}")
        for row in sorted(rows, key=lambda r: (r.duration_s, r.episode_dir))[:top_n]:
            print(f"  {format_seconds(row.duration_s):>7}s  {row.episode_dir}")
        print()
        print(f"longest_{top_n}")
        for row in sorted(rows, key=lambda r: (-r.duration_s, r.episode_dir))[:top_n]:
            print(f"  {format_seconds(row.duration_s):>7}s  {row.episode_dir}")

    if warnings:
        print()
        print("metadata_warnings", file=sys.stderr)
        for warning in warnings:
            print(f"  {warning}", file=sys.stderr)


def write_csv(path: Path, rows: list[EpisodeDuration]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode_dir",
                "duration_s",
                "source",
                "clip_start_time",
                "clip_end_time",
                "metadata_duration_s",
            ],
        )
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r.episode_dir):
            writer.writerow(
                {
                    "episode_dir": row.episode_dir,
                    "duration_s": row.duration_s,
                    "source": row.source,
                    "clip_start_time": row.clip_start_time,
                    "clip_end_time": row.clip_end_time,
                    "metadata_duration_s": row.metadata_duration_s,
                }
            )


def write_json(
    path: Path,
    root: Path,
    rows: list[EpisodeDuration],
    warnings: list[str],
    bin_size: float,
    bin_origin: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = sorted(row.duration_s for row in rows)
    payload = {
        "root": str(root),
        "episodes_with_duration": len(rows),
        "warnings": warnings,
        "summary_seconds": None,
        "histogram_seconds": [],
        "exact_duration_counts": [],
        "episodes": [row.__dict__ for row in sorted(rows, key=lambda r: r.episode_dir)],
    }
    if values:
        payload["summary_seconds"] = {
            "min": values[0],
            "p05": percentile(values, 0.05),
            "p25": percentile(values, 0.25),
            "median": median(values),
            "mean": mean(values),
            "p75": percentile(values, 0.75),
            "p95": percentile(values, 0.95),
            "max": values[-1],
        }
        payload["histogram_seconds"] = [
            {"start_s": lo, "end_s": hi, "count": count, "percent": 100.0 * count / len(values)}
            for lo, hi, count in make_bins(values, bin_size, bin_origin)
        ]
        payload["exact_duration_counts"] = [
            {"duration_s": duration, "count": count, "percent": 100.0 * count / len(values)}
            for duration, count in sorted(Counter(values).items())
        ]
    path.write_text(json.dumps(payload, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="package raw data root")
    parser.add_argument("--bin-size", type=float, default=5.0, help="histogram bin size in seconds")
    parser.add_argument("--bin-origin", type=float, default=0.0, help="histogram origin in seconds")
    parser.add_argument("--top-n", type=int, default=10, help="show N shortest and longest episodes")
    parser.add_argument("--csv-out", type=Path, help="optional per-episode CSV output path")
    parser.add_argument("--json-out", type=Path, help="optional machine-readable JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bin_size <= 0:
        raise SystemExit("--bin-size must be positive")
    if args.top_n < 0:
        raise SystemExit("--top-n must be non-negative")
    if not args.root.exists():
        raise SystemExit(f"root does not exist: {args.root}")

    rows, warnings = collect(args.root)
    print_summary(args.root, rows, warnings, args.bin_size, args.bin_origin, args.top_n)

    if args.csv_out:
        write_csv(args.csv_out, rows)
        print(f"\nwrote CSV: {args.csv_out}")
    if args.json_out:
        write_json(args.json_out, args.root, rows, warnings, args.bin_size, args.bin_origin)
        print(f"wrote JSON: {args.json_out}")


if __name__ == "__main__":
    main()
