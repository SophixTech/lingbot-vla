#!/usr/bin/env python3
"""Split the original datav2_1015 into independent train/test LeRobot roots.

Standalone dependencies: numpy, pyarrow (no repository imports).
Example:
  python split_datav2_1015.py --source /DATA/datav2_1015 --output /DATA/1015spilt

Defaults reproduce the 492/10 split made on 2026-10-16. All video files are
copied, never linked. Source files are read-only. This writes per-split raw
meta/stats.json, NOT the model's joint-training norm_stats.json.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_TEST_IDS = [42, 44, 47, 100, 215, 217, 324, 348, 382, 427]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows))


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def indexed(rows, count, name):
    result = {r['episode_index']: r for r in rows}
    require(len(rows) == len(result) == count and set(result) == set(range(count)),
            f'{name}: incomplete or duplicate episode IDs')
    return result


def describe(values):
    require(np.isfinite(values).all(), 'Non-finite numeric data')
    return {k: v.tolist() for k, v in {
        'mean': values.mean(0), 'std': values.std(0),
        'min': values.min(0), 'max': values.max(0),
        **dict(zip(('q01', 'q99', 'q02', 'q98'),
                   np.quantile(values, [.01, .99, .02, .98], axis=0))),
        'count': np.array([len(values)]),
    }.items()}


def split_dataset(source, output, seed=42, test_count=10):
    source, output = Path(source).resolve(), Path(output).resolve()
    require(source.is_dir(), f'Source not found: {source}')
    require(not output.exists(), f'Refusing to overwrite: {output}')
    require(not output.is_relative_to(source) and not source.is_relative_to(output),
            'Source and output must be separate directories')
    info = json.loads((source / 'meta/info.json').read_text())
    count = int(info['total_episodes'])
    require(0 < test_count < count, 'test-count must be between 1 and total_episodes - 1')
    require(info['codebase_version'] == 'v3.0', 'Expected LeRobot v3.0')
    metadata = indexed([r for p in sorted((source / 'meta/episodes').rglob('*.parquet'))
                        for r in pq.read_table(p).to_pylist()], count, 'Episode metadata')
    provenance = indexed(read_rows(source / 'meta/episodes.jsonl'), count, 'Provenance')
    labels = indexed(read_rows(source / 'meta/annotations/episode_labels.jsonl'), count, 'Labels')
    source_ids = [provenance[i]['source_episode_id'] for i in range(count)]
    require(len(set(source_ids)) == count, 'Duplicate source_episode_id')
    test_ids = sorted(map(int, np.random.default_rng(seed).choice(count, test_count, replace=False)))
    if count == 502 and seed == 42 and test_count == 10:
        require(test_ids == DEFAULT_TEST_IDS, 'Unexpected random selection; refusing different default split')
    test_set = set(test_ids)
    cameras = [k for k, v in info['features'].items() if v['dtype'] == 'video']
    require(len(cameras) == 3, 'Expected three datav2 cameras')
    # Read and validate the original packed parquet before creating any output.
    original, cache, previous_end = {}, {}, 0
    for old in range(count):
        m = metadata[old]
        start, end = m['dataset_from_index'], m['dataset_to_index']
        require(start == previous_end and end - start == m['length'] > 0,
                f'Invalid bounds for episode {old}')
        previous_end = end
        path = source / info['data_path'].format(chunk_index=m['data/chunk_index'], file_index=m['data/file_index'])
        if path not in cache:
            cache[path] = pq.read_table(path)
        packed = cache[path]
        offset = start - int(packed['index'][0].as_py())
        require(offset >= 0, f'Invalid offset: episode {old}')
        table = packed.slice(offset, end - start)
        require(len(table) == end - start and
                np.array_equal(table['index'].to_numpy(), np.arange(start, end)) and
                np.array_equal(table['frame_index'].to_numpy(), np.arange(end - start)) and
                np.all(table['episode_index'].to_numpy() == old), f'Invalid indices: episode {old}')
        require(provenance[old]['frame_count'] == len(table) and
                labels[old]['source_episode_id'] == source_ids[old], f'Identity mismatch: episode {old}')
        original[old] = table
    require(previous_end == info['total_frames'], 'Frame total mismatch')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name + '.partial-', dir=output.parent))
    manifest = {'source': str(source), 'seed': seed, 'test_original_episode_indices': test_ids,
                'selection': 'numpy default_rng choice without replacement', 'splits': {}}
    print(f'Test original episode IDs: {test_ids}', flush=True)
    try:
        for split, ids in [('train', [i for i in range(count) if i not in test_set]), ('test', test_ids)]:
            root = stage / split
            tables, metas, prov, labs, mapping, videos = [], [], [], [], [], []
            offset = 0
            for new, old in enumerate(ids):
                table = original[old]
                n = len(table)
                for key, value in [('episode_index', np.full(n, new, dtype=np.int64)),
                                   ('index', np.arange(offset, offset + n, dtype=np.int64))]:
                    table = table.set_column(table.schema.get_field_index(key), table.schema.field(key), pa.array(value))
                tables.append(table)
                m = copy.deepcopy(metadata[old])
                m.update(episode_index=new, dataset_from_index=offset, dataset_to_index=offset + n)
                m['data/chunk_index'] = m['data/file_index'] = 0
                pr, lab = copy.deepcopy(provenance[old]), copy.deepcopy(labels[old])
                for row in (pr, lab):
                    row.update(episode_index=new, source_episode_index=old)
                pr['data/chunk_index'] = pr['data/file_index'] = 0
                for camera in cameras:
                    prefix = 'videos/' + camera
                    src = source / info['video_path'].format(video_key=camera,
                        chunk_index=metadata[old][prefix + '/chunk_index'], file_index=metadata[old][prefix + '/file_index'])
                    relative = info['video_path'].format(video_key=camera, chunk_index=0, file_index=new)
                    dst = root / relative
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    digest = sha256(src)
                    require(sha256(dst) == digest, f'Video copy mismatch: {dst}')
                    videos.append({'path': relative, 'source': str(src), 'sha256': digest})
                    m[prefix + '/chunk_index'], m[prefix + '/file_index'] = 0, new
                    pr['videos'][camera].update(chunk_index=0, file_index=new)
                align = json.loads((source / f'meta/alignments/episode-{old:06d}.json').read_text())
                require(align['episode_index'] == old and align['source_episode_id'] == source_ids[old]
                        and align['frame_count'] == n, f'Alignment mismatch: episode {old}')
                align.update(episode_index=new, source_episode_index=old)
                write_json(root / f'meta/alignments/episode-{new:06d}.json', align)
                metas.append(m); prov.append(pr); labs.append(lab)
                mapping.append({'episode_index': new, 'source_episode_index': old,
                                'source_episode_id': source_ids[old], 'frames': n})
                offset += n
                if (new + 1) % 50 == 0:
                    print(f'{split}: copied {new + 1}/{len(ids)} episodes', flush=True)
            combined = pa.concat_tables(tables)
            data_path = root / info['data_path'].format(chunk_index=0, file_index=0)
            data_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(combined, data_path)
            meta_path = root / 'meta/episodes/chunk-000/file-000.parquet'
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(metas), meta_path)
            # Verify every saved numeric column, including rewritten indices.
            require(pq.read_table(data_path).equals(combined), f'Parquet roundtrip failed: {split}')
            require(pq.read_table(meta_path).to_pylist() == metas, 'Metadata roundtrip failed')
            write_rows(root / 'meta/episodes.jsonl', prov)
            write_rows(root / 'meta/annotations/episode_labels.jsonl', labs)
            write_rows(root / 'episode_mapping.jsonl', mapping)
            write_rows(root / 'video_checksums.jsonl', videos)
            shutil.copy2(source / 'meta/tasks.parquet', root / 'meta/tasks.parquet')
            newinfo = copy.deepcopy(info)
            newinfo.update(total_episodes=len(ids), total_frames=offset,
                           splits={split: f'0:{len(ids)}'}, source_dataset_root=str(source))
            write_json(root / 'meta/info.json', newinfo)
            stats = {}
            for key in ('observation.state', 'action'):
                values = np.asarray(combined[key].combine_chunks().values.to_numpy(), dtype=np.float64).reshape(offset, -1)
                stats[key] = describe(values)
            stats['timestamp'] = describe(combined['timestamp'].to_numpy().astype(np.float64).reshape(-1, 1))
            write_json(root / 'meta/stats.json', stats)
            manifest['splits'][split] = {'episodes': len(ids), 'frames': offset,
                                        'source_episode_ids': [source_ids[i] for i in ids]}
        require(sum(s['frames'] for s in manifest['splits'].values()) == info['total_frames'], 'Lost frames')
        manifest['validation'] = {'disjoint_and_complete': True, 'all_video_sha256_match': True,
                                  'numeric_parquet_roundtrip': True, 'independent_copies': True}
        write_json(stage / 'split_manifest.json', manifest)
        require(not output.exists(), f'Output appeared during conversion: {output}')
        stage.rename(output)
    except BaseException:
        # Only remove the unique staging directory owned by this invocation.
        shutil.rmtree(stage)
        raise
    for name, result in manifest['splits'].items():
        print(f"{name}: {result['episodes']} episodes, {result['frames']} frames")
    print(f'Output: {output}\nSource unchanged. Joint training normalization was not generated.')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', required=True, type=Path, help='Original datav2_1015 root containing meta/data/videos')
    parser.add_argument('--output', required=True, type=Path, help='New directory; must not already exist')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test-count', type=int, default=10)
    args = parser.parse_args()
    split_dataset(args.source, args.output, args.seed, args.test_count)


if __name__ == '__main__':
    main()
