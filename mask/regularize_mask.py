"""Connect nearby keep regions and export a fixed PNG/JSON training mask."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

def ranges(keep):
    keep = np.asarray(keep, dtype=bool)
    height, width = keep.shape
    rows = []
    for row in keep:
        edges = np.flatnonzero(np.diff(np.pad(row.astype(np.int8), (1, 1))))
        rows.append(edges.reshape(-1, 2).tolist())
    return {'width': width, 'height': height, 'rows': rows,
            'keep_pixels': int(keep.sum()), 'keep_fraction': float(keep.mean())}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def regularize(mask, close_size=35, epsilon=3.0, min_area=300.0):
    if close_size < 1 or close_size % 2 != 1 or epsilon < 0 or min_area < 0:
        raise ValueError('Use a positive odd close-size and nonnegative epsilon/min-area')
    binary = (mask > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(binary)
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            polygon = cv2.approxPolyDP(contour, epsilon, True)
            cv2.fillPoly(result, [polygon], 255)
    return result > 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--close-size', type=int, default=35)
    parser.add_argument('--epsilon', type=float, default=3.0)
    parser.add_argument('--min-area', type=float, default=300.0)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.output.suffix.lower() != '.png':
        parser.error('Output must be a PNG')
    intervals = args.output.with_name(args.output.stem + '_ranges.json')
    if not args.overwrite and any(p.exists() for p in (args.output, intervals)):
        parser.error('Output exists; use --overwrite to replace it')
    with Image.open(args.input) as image:
        source = np.asarray(image.convert('L'))
    result = regularize(source, args.close_size, args.epsilon, args.min_area)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result.astype(np.uint8) * 255).save(args.output)
    write_json(intervals, ranges(result))
    print(f'Keep: {(source > 0).mean():.2%} -> {result.mean():.2%}; {args.output}')


if __name__ == '__main__':
    main()
