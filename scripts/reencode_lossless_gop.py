"""Build a separate LeRobot copy with lossless H.264 and bounded GOP.

Every output is checked against source decoded frame hashes and timestamps.
Only the training root passed on the command line is copied. Source is read-only.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def run(args):
    return subprocess.check_output(args, stderr=subprocess.PIPE, text=True)


def probe(path):
    return json.loads(run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                           '-show_streams', '-of', 'json', str(path)]))['streams'][0]


def frames(path, time_base):
    output = run(['ffmpeg', '-v', 'error', '-nostdin', '-threads', '1', '-copyts',
                  '-i', str(path), '-map', '0:v:0', '-vsync', '0', '-threads', '1',
                  '-enc_time_base', time_base.replace('/', ':'), '-f', 'framemd5', '-'])
    tb = None
    result = []
    for line in output.splitlines():
        if line.startswith('#tb 0:'):
            tb = Fraction(line.split(':', 1)[1].strip())
        elif line and not line.startswith('#'):
            values = [v.strip() for v in line.split(',')]
            assert tb is not None
            result.append((int(values[2]) * tb, int(values[3]) * tb,
                           int(values[4]), values[5]))
    assert result, f'No decoded frames: {path}'
    return result


def verify(source, target, gop):
    original, converted = probe(source), probe(target)
    for field in ['width', 'height', 'pix_fmt', 'avg_frame_rate', 'start_time',
                  'duration', 'color_range', 'color_space', 'color_transfer',
                  'color_primaries', 'chroma_location']:
        assert original.get(field) == converted.get(field), (source, field, original.get(field), converted.get(field))
    a, b = frames(source, original['time_base']), frames(target, original['time_base'])
    assert a == b, f'Decoded pixels, frame count, or timestamps differ: {source}'
    packets = json.loads(run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                              '-show_entries', 'packet=flags', '-of', 'json', str(target)]))['packets']
    keys = [i for i, packet in enumerate(packets) if 'K' in packet['flags']]
    assert keys and keys[0] == 0 and len(packets) == len(a)
    assert max(y-x for x, y in zip(keys, keys[1:] + [len(packets)])) <= gop
    return len(a)


def convert(source, target, gop):
    start = time.monotonic()
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        info = probe(source)
        temp = target.with_name(target.stem + '.partial.mp4')
        run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
             '-threads', '1', '-copyts', '-i', str(source), '-map', '0:v:0', '-an',
             '-vsync', '0', '-c:v', 'libx264', '-preset', 'fast', '-qp', '0',
             '-g', str(gop), '-keyint_min', str(gop), '-sc_threshold', '0', '-bf', '0',
             '-pix_fmt', info['pix_fmt'], '-threads', '1', '-enc_time_base',
             info['time_base'].replace('/', ':'), '-movflags', '+faststart', str(temp)])
        count = verify(source, temp, gop)
        os.replace(temp, target)
    else:
        count = verify(source, target, gop)
    return {'source': str(source), 'target': str(target), 'frames': count,
            'source_bytes': source.stat().st_size, 'target_bytes': target.stat().st_size,
            'seconds': time.monotonic()-start, 'verified': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('target', type=Path)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--gop', type=int, default=30)
    args = parser.parse_args()
    source, target = args.source.resolve(), args.target.resolve()
    assert source != target and source not in target.parents and target not in source.parents
    assert args.workers > 0 and args.gop > 0
    assert (source/'meta/info.json').exists()
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = target.parent/'lossless_gop_manifest.json'
    # Do not leave a success marker from an earlier run during revalidation.
    manifest_path.write_text(json.dumps({'complete': False, 'source': str(source), 'target': str(target)}))
    for item in source.iterdir():
        if item.name == 'videos':
            continue
        dest = target/item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)
    videos = sorted((source/'videos').rglob('*.mp4'))
    assert videos
    records = []
    start = time.monotonic()
    with (target.parent/'conversion_verified.jsonl').open('a', buffering=1) as log:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(convert, p, target/p.relative_to(source), args.gop) for p in videos]
            for future in as_completed(futures):
                record = future.result()
                records.append(record)
                log.write(json.dumps(record)+'\n')
                if len(records) % 25 == 0 or len(records) == len(videos):
                    print(f'Verified {len(records)}/{len(videos)} videos; elapsed {time.monotonic()-start:.1f}s', flush=True)
    manifest = {'complete': True, 'source': str(source), 'target': str(target),
                'gop': args.gop, 'encoder': 'libx264', 'qp': 0, 'preset': 'fast',
                'validation': 'All decoded native pixel hashes, frame PTS/duration, stream color metadata and GOP checked',
                'videos': len(records), 'camera_frames': sum(x['frames'] for x in records),
                'source_video_bytes': sum(x['source_bytes'] for x in records),
                'target_video_bytes': sum(x['target_bytes'] for x in records),
                'elapsed_seconds': time.monotonic()-start}
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
