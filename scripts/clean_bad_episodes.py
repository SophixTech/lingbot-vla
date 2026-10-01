"""Build clean, index-audited copies of the marked and interpolation datasets."""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path('/home/bjtc/Sophix')
DATASETS = ROOT / 'datasets'
BAD_SOURCE_EPISODES = {148, 228, 340, 341, 481, 497, 755, 806, 865}
MARKED_SOURCE = DATASETS / 'marked_data_0916'
MARKED_CLEAN = DATASETS / 'marked_data_0916_clean'
SPLIT_SOURCE = DATASETS / 'marked_9_1_0922'
SPLIT_CLEAN = DATASETS / 'marked_9_1_0922_clean'
INTERP_SOURCE = DATASETS / 'package_interpolation_lerobot'
INTERP_CLEAN = DATASETS / 'package_interpolation_lerobot_clean'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records))


def copy_regular(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def find_single_data_file(root: Path, chunk: int, file_index: int) -> Path:
    matches = list((root / 'data').glob(f'chunk-{chunk:03d}/file-{file_index:03d}.parquet'))
    if len(matches) != 1:
        raise RuntimeError(f'Expected one data parquet for {chunk}/{file_index}: {matches}')
    return matches[0]


def copy_video_for_episode(src_root: Path, dst_root: Path, info: dict, ep: dict, new_index: int,
                           old_index: int, checksums: list[dict]) -> None:
    for key, feature in info['features'].items():
        if feature.get('dtype') != 'video':
            continue
        prefix = f"videos/{key}"
        source = src_root / prefix / f"chunk-{int(ep[prefix + '/chunk_index']):03d}" / f"file-{int(ep[prefix + '/file_index']):03d}.mp4"
        if not source.is_file():
            raise FileNotFoundError(source)
        target = dst_root / prefix / f'chunk-000/file-{new_index:03d}.mp4'
        copy_regular(source, target)
        sha = digest(source)
        if digest(target) != sha:
            raise RuntimeError(f'Video checksum mismatch: {source}')
        ep[prefix + '/chunk_index'] = 0
        ep[prefix + '/file_index'] = new_index
        checksums.append({'kind': 'video', 'source_episode_index': old_index,
                          'episode_index': new_index, 'feature': key,
                          'source': str(source.relative_to(src_root)), 'sha256': sha})


def copy_marked_subset(src_root: Path, dst_root: Path, source_ids: set[int], source_root: Path,
                       excluded: set[int], label_filter: set[int] | None = None) -> dict:
    if dst_root.exists():
        raise FileExistsError(f'Refusing to overwrite existing destination: {dst_root}')
    info = json.loads((src_root / 'meta/info.json').read_text())
    labels = read_jsonl(src_root / 'meta/annotations/episode_labels.jsonl')
    labels_by_index = {int(row['episode_index']): row for row in labels}
    provenance = read_jsonl(src_root / 'meta/episodes.jsonl')
    provenance_by_index = {int(row['episode_index']): row for row in provenance}
    ep_tables = [pq.read_table(path) for path in sorted((src_root / 'meta/episodes').glob('chunk-*/*.parquet'))]
    if not ep_tables:
        raise RuntimeError(f'No episode metadata parquet in {src_root}')
    ep_schema = ep_tables[0].schema
    ep_rows = {int(row['episode_index']): row for table in ep_tables for row in table.to_pylist()}
    available = set(ep_rows)
    if available != set(labels_by_index) or available != set(provenance_by_index):
        raise RuntimeError(f'Episode metadata/label/provenance sets differ in {src_root}')
    selected = sorted(available - excluded)
    if label_filter is not None:
        selected = [idx for idx in selected if idx in label_filter]
    dst_root.mkdir(parents=True)
    for name in ['data/chunk-000', 'videos', 'meta/episodes/chunk-000', 'meta/annotations', 'meta/alignments']:
        (dst_root / name).mkdir(parents=True, exist_ok=True)
    task_file = src_root / 'meta/tasks.parquet'
    if task_file.exists():
        copy_regular(task_file, dst_root / 'meta/tasks.parquet')
    records = []
    new_meta, new_prov, new_labels, checksums = [], [], [], []
    offset = 0
    old_to_new = {}
    for new_idx, old_idx in enumerate(selected):
        meta = copy.deepcopy(ep_rows[old_idx])
        source_data = find_single_data_file(src_root, int(meta['data/chunk_index']), int(meta['data/file_index']))
        table = pq.read_table(source_data)
        n = table.num_rows
        if n != int(meta['length']):
            raise RuntimeError(f'Frame count mismatch for source episode {old_idx}: parquet={n}, meta={meta["length"]}')
        got_ep = table['episode_index'].to_numpy()
        got_frame = table['frame_index'].to_numpy()
        if not np.all(got_ep == old_idx) or not np.array_equal(got_frame, np.arange(n)):
            raise RuntimeError(f'Parquet row identity mismatch for source episode {old_idx}')
        table = table.set_column(table.schema.get_field_index('episode_index'), 'episode_index', pa.array(np.full(n, new_idx, np.int64)))
        table = table.set_column(table.schema.get_field_index('index'), 'index', pa.array(np.arange(offset, offset + n, dtype=np.int64)))
        dst_data = dst_root / f'data/chunk-000/file-{new_idx:03d}.parquet'
        pq.write_table(table, dst_data, compression='zstd')
        check = pq.read_table(dst_data)
        if not check.equals(table):
            raise RuntimeError(f'Parquet round-trip mismatch for source episode {old_idx}')
        meta.update(episode_index=new_idx, **{'data/chunk_index': 0, 'data/file_index': new_idx,
                                              'dataset_from_index': offset, 'dataset_to_index': offset + n})
        copy_video_for_episode(src_root, dst_root, info, meta, new_idx, old_idx, checksums)
        old_prov = copy.deepcopy(provenance_by_index[old_idx])
        old_prov.update(episode_index=new_idx, source_episode_index=old_idx,
                        source_episode_id=provenance_by_index[old_idx]['source_episode_id'],
                        source_dataset_root=str(source_root), alignment_path=f'meta/alignments/episode-{new_idx:06d}.json')
        old_label = copy.deepcopy(labels_by_index[old_idx])
        old_label.update(episode_index=new_idx, source_episode_index=old_idx,
                         source_episode_id=provenance_by_index[old_idx]['source_episode_id'])
        src_align = src_root / provenance_by_index[old_idx]['alignment_path']
        dst_align = dst_root / f'meta/alignments/episode-{new_idx:06d}.json'
        if src_align.exists():
            copy_regular(src_align, dst_align)
            if digest(src_align) != digest(dst_align):
                raise RuntimeError(f'Alignment checksum mismatch for source episode {old_idx}')
            checksums.append({'kind': 'alignment', 'source_episode_index': old_idx,
                              'episode_index': new_idx, 'sha256': digest(src_align)})
        old_to_new[old_idx] = new_idx
        records.append({'episode_index': new_idx, 'source_episode_index': old_idx,
                        'source_episode_id': provenance_by_index[old_idx]['source_episode_id'],
                        'frames': n, 'dataset_from_index': offset,
                        'dataset_to_index': offset + n})
        new_meta.append(meta)
        new_prov.append(old_prov)
        new_labels.append(old_label)
        checksums.append({'kind': 'data', 'source_episode_index': old_idx,
                          'episode_index': new_idx, 'source': str(source_data.relative_to(src_root)),
                          'target': str(dst_data.relative_to(dst_root)), 'sha256': digest(dst_data)})
        offset += n
    pq.write_table(pa.Table.from_pylist(new_meta, schema=ep_schema),
                   dst_root / 'meta/episodes/chunk-000/file-000.parquet', compression='zstd')
    write_jsonl(dst_root / 'meta/episodes.jsonl', new_prov)
    write_jsonl(dst_root / 'meta/annotations/episode_labels.jsonl', new_labels)
    write_jsonl(dst_root / 'episode_mapping.jsonl', records)
    write_jsonl(dst_root / 'file_checksums.jsonl', checksums)
    new_info = copy.deepcopy(info)
    new_info.update(total_episodes=len(selected), total_frames=offset,
                    splits={'train': f'0:{len(selected)}'},
                    data_path='data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
                    source_dataset_root=str(source_root), excluded_source_episode_indices=sorted(excluded))
    write_json(dst_root / 'meta/info.json', new_info)
    for name in ['meta/stats.json']:
        source = src_root / name
        if source.exists():
            copy_regular(source, dst_root / name)
    if (src_root / 'meta/tasks.parquet').exists():
        pass
    write_json(dst_root / 'cleanup_manifest.json', {
        'source_dataset': str(source_root), 'source_view': str(src_root),
        'excluded_source_episode_indices': sorted(excluded), 'retained_episodes': len(selected),
        'retained_frames': offset, 'mapping_file': 'episode_mapping.jsonl',
        'mapping_identity': 'source_episode_index and source_episode_id come from canonical source provenance; episode_index is contiguous output id',
        'mapping': records,
    })
    return {'episodes': len(selected), 'frames': offset, 'mapping': records, 'old_to_new': old_to_new}


def clean_split_dataset() -> dict:
    if SPLIT_CLEAN.exists():
        raise FileExistsError(f'Refusing to overwrite existing destination: {SPLIT_CLEAN}')
    source_mapping_all = {}
    for split in ('train', 'test'):
        for row in read_jsonl(SPLIT_SOURCE / split / 'episode_mapping.jsonl'):
            old = int(row['source_episode_index'])
            source_mapping_all[old] = row['source_episode_id']
    if any(old in source_mapping_all and old in BAD_SOURCE_EPISODES for old in BAD_SOURCE_EPISODES):
        pass
    SPLIT_CLEAN.mkdir()
    report = {'source_dataset': str(SPLIT_SOURCE), 'excluded_source_episode_indices': sorted(BAD_SOURCE_EPISODES), 'splits': {}}
    for split in ('train', 'test'):
        src = SPLIT_SOURCE / split
        dst = SPLIT_CLEAN / split
        old_mapping = read_jsonl(src / 'episode_mapping.jsonl')
        source_ep_ids = {int(row['source_episode_index']): row['source_episode_id'] for row in old_mapping}
        removed = {idx for idx in BAD_SOURCE_EPISODES if idx in source_ep_ids}
        keep_old_local = sorted(int(row['episode_index']) for row in old_mapping if int(row['source_episode_index']) not in BAD_SOURCE_EPISODES)
        info = json.loads((src / 'meta/info.json').read_text())
        source_ep_table = pq.read_table(src / 'meta/episodes/chunk-000/file-000.parquet')
        meta_by_local = {int(row['episode_index']): row for row in source_ep_table.to_pylist()}
        labels_by_local = {int(r['episode_index']): r for r in read_jsonl(src / 'meta/annotations/episode_labels.jsonl')}
        prov_by_local = {int(r['episode_index']): r for r in read_jsonl(src / 'meta/episodes.jsonl')}
        dst.mkdir(parents=True)
        for d in ['data/chunk-000', 'meta/episodes/chunk-000', 'meta/annotations', 'meta/alignments']:
            (dst / d).mkdir(parents=True, exist_ok=True)
        checksums, new_meta, new_labels, new_prov, mapping = [], [], [], [], []
        offset = 0
        kept_source_indices = []
        for new_local, old_local in enumerate(keep_old_local):
            mapped = next(row for row in old_mapping if int(row['episode_index']) == old_local)
            source_ep = int(mapped['source_episode_index'])
            if source_ep in BAD_SOURCE_EPISODES:
                raise RuntimeError('Bad source episode unexpectedly selected for retention')
            meta = copy.deepcopy(meta_by_local[old_local])
            src_data = src / f"data/chunk-000/file-{int(meta['data/file_index']):03d}.parquet"
            table = pq.read_table(src_data)
            n = table.num_rows
            if n != int(meta['length']) or not np.all(table['episode_index'].to_numpy() == old_local):
                raise RuntimeError(f'Split parquet/local metadata mismatch at {split}:{old_local}')
            if not np.array_equal(table['frame_index'].to_numpy(), np.arange(n)):
                raise RuntimeError(f'Frame identity mismatch at {split}:{old_local}')
            table = table.set_column(table.schema.get_field_index('episode_index'), 'episode_index', pa.array(np.full(n, new_local, np.int64)))
            table = table.set_column(table.schema.get_field_index('index'), 'index', pa.array(np.arange(offset, offset + n, dtype=np.int64)))
            target_data = dst / f'data/chunk-000/file-{new_local:03d}.parquet'
            pq.write_table(table, target_data, compression='zstd')
            if not pq.read_table(target_data).equals(table):
                raise RuntimeError(f'Split data write verification failed {split}:{old_local}')
            meta.update(episode_index=new_local, **{'data/chunk_index': 0, 'data/file_index': new_local,
                                                    'dataset_from_index': offset, 'dataset_to_index': offset + n})
            for key, feature in info['features'].items():
                if feature.get('dtype') != 'video':
                    continue
                prefix = f'videos/{key}'
                source = src / prefix / f"chunk-{int(meta_by_local[old_local][prefix + '/chunk_index']):03d}/file-{int(meta_by_local[old_local][prefix + '/file_index']):03d}.mp4"
                target = dst / prefix / f'chunk-000/file-{new_local:03d}.mp4'
                copy_regular(source, target)
                if digest(source) != digest(target):
                    raise RuntimeError(f'Split video checksum mismatch {split}:{old_local}:{key}')
                meta[prefix + '/chunk_index'] = 0
                meta[prefix + '/file_index'] = new_local
                checksums.append({'kind': 'video', 'source_episode_index': source_ep,
                                  'old_split_episode_index': old_local, 'episode_index': new_local,
                                  'feature': key, 'sha256': digest(target)})
            new_meta.append(meta)
            for source_row, rows in ((labels_by_local[old_local], new_labels), (prov_by_local[old_local], new_prov)):
                row = copy.deepcopy(source_row)
                row.update(episode_index=new_local, source_episode_index=source_ep,
                           source_episode_id=mapped['source_episode_id'], source_dataset_root=str(MARKED_SOURCE))
                rows.append(row)
            mapping.append({'episode_index': new_local, 'old_split_episode_index': old_local,
                            'source_episode_index': source_ep, 'source_episode_id': mapped['source_episode_id'],
                            'frames': n, 'dataset_from_index': offset, 'dataset_to_index': offset + n})
            checksums.append({'kind': 'data', 'source_episode_index': source_ep,
                              'old_split_episode_index': old_local, 'episode_index': new_local,
                              'sha256': digest(target_data)})
            kept_source_indices.append(source_ep)
            offset += n
        pq.write_table(pa.Table.from_pylist(new_meta, schema=source_ep_table.schema),
                       dst / 'meta/episodes/chunk-000/file-000.parquet', compression='zstd')
        write_jsonl(dst / 'meta/episodes.jsonl', new_prov)
        write_jsonl(dst / 'meta/annotations/episode_labels.jsonl', new_labels)
        write_jsonl(dst / 'episode_mapping.jsonl', mapping)
        write_jsonl(dst / 'file_checksums.jsonl', checksums)
        if (src / 'meta/tasks.parquet').exists():
            copy_regular(src / 'meta/tasks.parquet', dst / 'meta/tasks.parquet')
        clean_info = copy.deepcopy(info)
        clean_info.update(total_episodes=len(mapping), total_frames=offset,
                          splits={split: f'0:{len(mapping)}'},
                          source_dataset_root=str(MARKED_SOURCE),
                          excluded_source_episode_indices=sorted(BAD_SOURCE_EPISODES))
        write_json(dst / 'meta/info.json', clean_info)
        # Recompute state/action/timestamp statistics from retained parquet rows.
        numeric = []
        for idx in range(len(mapping)):
            numeric.append(pq.read_table(dst / f'data/chunk-000/file-{idx:03d}.parquet', columns=['observation.state','action','timestamp']))
        def summarize(field: str) -> dict:
            vals = np.concatenate([np.asarray(t[field].combine_chunks().to_pylist(), dtype=np.float64) for t in numeric], axis=0)
            if vals.ndim == 1:
                vals = vals[:, None]
            return {'mean': vals.mean(axis=0).tolist(), 'std': vals.std(axis=0).tolist(), 'min': vals.min(axis=0).tolist(), 'max': vals.max(axis=0).tolist()}
        write_json(dst / 'meta/stats.json', {key: summarize(key) for key in ['observation.state','action','timestamp']})
        report['splits'][split] = {'episodes': len(mapping), 'frames': offset,
                                   'removed_source_episode_indices': sorted(removed),
                                   'retained_source_episode_indices': kept_source_indices,
                                   'mapping': mapping}
    # Rebuild the shared split plan/manifest from the exact retained source IDs.
    train_ids = set(report['splits']['train']['retained_source_episode_indices'])
    test_ids = set(report['splits']['test']['retained_source_episode_indices'])
    if train_ids & test_ids:
        raise RuntimeError('Train/test source episode overlap after cleaning')
    report['source_episode_overlap'] = 0
    report['train_test_source_episode_union_count'] = len(train_ids | test_ids)
    report['source_mapping_sha256'] = digest(SPLIT_SOURCE / 'split_manifest.json')
    write_json(SPLIT_CLEAN / 'cleanup_manifest.json', report)
    return report


def clean_interpolation() -> dict:
    if INTERP_CLEAN.exists():
        raise FileExistsError(f'Refusing to overwrite existing destination: {INTERP_CLEAN}')
    info = json.loads((INTERP_SOURCE / 'meta/info.json').read_text())
    table = pq.read_table(INTERP_SOURCE / 'meta/episodes/chunk-000/file-000.parquet')
    rows = table.to_pylist()
    if sorted(int(r['episode_index']) for r in rows) != list(range(len(rows))):
        raise RuntimeError('Interpolation episode table ids are not complete and contiguous')
    # Map interpolation ids by source UUID, never by assumed numeric equality.
    source_prov = {int(r['episode_index']): r for r in read_jsonl(MARKED_SOURCE / 'meta/episodes.jsonl')}
    source_by_id = {r['source_episode_id']: idx for idx, r in source_prov.items()}
    source_ids = {idx: r['source_episode_id'] for idx, r in source_prov.items()}
    report_rows = json.loads((INTERP_SOURCE / 'interpolation_conversion_report.json').read_text())['episodes_converted_this_run']
    if len(report_rows) != len(rows):
        raise RuntimeError('Conversion report and interpolation metadata episode counts differ')
    report_by_index = {i: row for i, row in enumerate(report_rows)}
    if set(report_by_index) != set(range(len(rows))):
        raise RuntimeError('Conversion report source order is not complete')
    interp_to_source, interp_source_uuid = {}, {}
    unmatched_source_ids = []
    for interp_idx, report_row in report_by_index.items():
        uuid = report_row['source_id']
        interp_source_uuid[interp_idx] = uuid
        if uuid in source_by_id:
            interp_to_source[interp_idx] = source_by_id[uuid]
        else:
            unmatched_source_ids.append(uuid)
    if len(set(interp_source_uuid.values())) != len(interp_source_uuid):
        raise RuntimeError('Duplicate source UUID in interpolation conversion report')
    bad_interp_ids = {interp for interp, source_idx in interp_to_source.items() if source_idx in BAD_SOURCE_EPISODES}
    INTERP_CLEAN.mkdir()
    for name in ['data/chunk-000', 'meta/episodes/chunk-000']:
        (INTERP_CLEAN / name).mkdir(parents=True, exist_ok=True)
    for directory in ['videos']:
        (INTERP_CLEAN / directory).mkdir(exist_ok=True)
    all_videos = [p for p in (INTERP_SOURCE / 'videos').rglob('*.mp4')]
    checksums = []
    # Copy retained data rows, preserve row order within each episode, and
    # explicitly rewrite both episode_index and global index contiguously.
    parquet_path = next((INTERP_SOURCE / 'data').glob('chunk-*/file-*.parquet'))
    data_table = pq.read_table(parquet_path)
    old_episode_col = data_table['episode_index'].to_numpy()
    old_frame_col = data_table['frame_index'].to_numpy()
    old_global_index = data_table['index'].to_numpy()
    keep_mask = ~np.isin(old_episode_col, sorted(bad_interp_ids))
    keep_rows = np.flatnonzero(keep_mask)
    kept_old_episodes = sorted(set(old_episode_col[keep_mask].tolist()))
    old_to_new = {old: new for new, old in enumerate(kept_old_episodes)}
    new_ep_col = np.asarray([old_to_new[int(old_episode_col[i])] for i in keep_rows], dtype=np.int64)
    new_indices = np.arange(len(keep_rows), dtype=np.int64)
    # filter and take keeps row order; then verify each contiguous run against metadata lengths.
    out_data = data_table.take(pa.array(keep_rows))
    out_data = out_data.set_column(out_data.schema.get_field_index('episode_index'), 'episode_index', pa.array(new_ep_col))
    out_data = out_data.set_column(out_data.schema.get_field_index('index'), 'index', pa.array(new_indices))
    target_data = INTERP_CLEAN / 'data/chunk-000/file-000.parquet'
    pq.write_table(out_data, target_data, compression='zstd')
    if not pq.read_table(target_data).equals(out_data):
        raise RuntimeError('Interpolation data parquet round-trip verification failed')
    old_rows = {int(r['episode_index']): r for r in rows}
    new_rows, mapping = [], []
    frame_cursor = 0
    for new_idx, old_idx in enumerate(kept_old_episodes):
        source_idx = interp_to_source.get(old_idx)
        source_uuid = interp_source_uuid[old_idx]
        length = int(old_rows[old_idx]['length'])
        selected = keep_rows[new_ep_col == new_idx]
        if len(selected) != length or not np.array_equal(old_frame_col[selected], np.arange(length)):
            raise RuntimeError(f'Interpolation frame mismatch for source episode {old_idx}')
        record = copy.deepcopy(old_rows[old_idx])
        record['episode_index'] = new_idx
        record['dataset_from_index'] = frame_cursor
        record['dataset_to_index'] = frame_cursor + length
        new_rows.append(record)
        mapping.append({'episode_index': new_idx, 'source_episode_index': source_idx,
                        'source_episode_id': source_uuid, 'source_episode_known': source_idx is not None,
                        'interpolation_source_episode_index': old_idx, 'frames': length,
                        'old_global_from_index': int(old_rows[old_idx]['dataset_from_index']),
                        'old_global_to_index': int(old_rows[old_idx]['dataset_to_index']),
                        'dataset_from_index': frame_cursor, 'dataset_to_index': frame_cursor + length})
        frame_cursor += length
    pq.write_table(pa.Table.from_pylist(new_rows, schema=table.schema),
                   INTERP_CLEAN / 'meta/episodes/chunk-000/file-000.parquet', compression='zstd')
    # Video mp4s in this dataset contain multiple episodes. Rebuild each camera
    # file by stream-copying only retained timestamp ranges, requiring explicit
    # ffmpeg; never retain a segment that overlaps a removed episode.
    import subprocess
    video_columns = [c for c in table.column_names if c.startswith('videos/') and c.endswith('/file_index')]
    for col in video_columns:
        key = col[len('videos/'): -len('/file_index')]
        chunk_col = f'videos/{key}/chunk_index'
        from_col = f'videos/{key}/from_timestamp'
        to_col = f'videos/{key}/to_timestamp'
        paths = set()
        for old_idx in kept_old_episodes:
            ep = old_rows[old_idx]
            old_file = int(ep[col])
            old_chunk = int(ep[chunk_col])
            source = INTERP_SOURCE / f'videos/{key}/chunk-{old_chunk:03d}/file-{old_file:03d}.mp4'
            if not source.exists():
                raise FileNotFoundError(source)
            target = INTERP_CLEAN / f'videos/{key}/chunk-000/file-{old_to_new[old_idx]:03d}.mp4'
            target.parent.mkdir(parents=True, exist_ok=True)
            start, end = float(ep[from_col]), float(ep[to_col])
            subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-ss',str(start),'-i',str(source),'-frames:v',str(int(ep['length'])),'-c','copy','-avoid_negative_ts','disabled',str(target)],check=True)
            if not target.is_file() or target.stat().st_size == 0:
                raise RuntimeError(f'Empty rebuilt video {target}')
            paths.add(target)
            record = new_rows[old_to_new[old_idx]]
            record[f'videos/{key}/chunk_index'] = 0
            record[f'videos/{key}/file_index'] = old_to_new[old_idx]
            record[f'videos/{key}/from_timestamp'] = 0.0
            record[f'videos/{key}/to_timestamp'] = end - start
            checksums.append({'kind':'video_segment','source_episode_index':interp_to_source.get(old_idx),
                              'interpolation_source_episode_index':old_idx,
                              'episode_index':old_to_new[old_idx], 'feature':key,
                              'source':str(source.relative_to(INTERP_SOURCE)),
                              'source_time_range':[start,end], 'target':str(target.relative_to(INTERP_CLEAN)),
                              'sha256':digest(target)})
    pq.write_table(pa.Table.from_pylist(new_rows, schema=table.schema),
                   INTERP_CLEAN / 'meta/episodes/chunk-000/file-000.parquet', compression='zstd')
    # Build a single video per retained episode above, with remapped video metadata.
    write_jsonl(INTERP_CLEAN / 'episode_mapping.jsonl', mapping)
    write_jsonl(INTERP_CLEAN / 'file_checksums.jsonl', checksums)
    copy_regular(INTERP_SOURCE / 'meta/tasks.parquet', INTERP_CLEAN / 'meta/tasks.parquet')
    # Copy small metadata, but statistics must be recomputed for cleaned data.
    for name in ['meta/info.json']:
        pass
    new_info = copy.deepcopy(info)
    new_info.update(total_episodes=len(mapping), total_frames=frame_cursor,
                    splits={'train': f'0:{len(mapping)}'},
                    data_path='data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
                    source_dataset_root=str(MARKED_SOURCE),
                    excluded_source_episode_indices=sorted(BAD_SOURCE_EPISODES))
    write_json(INTERP_CLEAN / 'meta/info.json', new_info)
    # Existing global statistics include removed data; do not copy them as if valid.
    write_json(INTERP_CLEAN / 'cleanup_manifest.json', {
        'source_dataset': str(INTERP_SOURCE), 'canonical_source_dataset': str(MARKED_SOURCE),
        'excluded_source_episode_indices': sorted(BAD_SOURCE_EPISODES),
        'excluded_interpolation_episode_indices': sorted(bad_interp_ids),
        'unmatched_source_episode_ids_retained': unmatched_source_ids,
        'retained_episodes': len(mapping), 'retained_frames': frame_cursor,
        'mapping_file': 'episode_mapping.jsonl', 'video_segments_rebuilt': True,
        'statistics': 'source stats omitted because they include excluded episodes; recompute before training',
        'mapping': mapping,
    })
    return {'episodes':len(mapping),'frames':frame_cursor,'mapping':mapping,
            'removed_video_files':len(all_videos)}


def main() -> None:
    for dst in (MARKED_CLEAN, SPLIT_CLEAN, INTERP_CLEAN):
        if dst.exists() and any(dst.iterdir()):
            raise FileExistsError(f'Clean output already exists; refusing overwrite: {dst}')
    src_prov = {int(r['episode_index']): r for r in read_jsonl(MARKED_SOURCE / 'meta/episodes.jsonl')}
    if not BAD_SOURCE_EPISODES <= set(src_prov):
        raise RuntimeError('Some excluded episode IDs are absent from canonical source metadata')
    source_ids = {idx: row['source_episode_id'] for idx, row in src_prov.items()}
    marked = copy_marked_subset(MARKED_SOURCE, MARKED_CLEAN, set(src_prov), MARKED_SOURCE, BAD_SOURCE_EPISODES)
    split = clean_split_dataset()
    interp = clean_interpolation()
    fast = build_clean_label_view()
    # Verify every output maps to the expected canonical source identity and no bad source id remains.
    summary = {'excluded_source_episode_indices': sorted(BAD_SOURCE_EPISODES), 'outputs': {}}
    for root in (MARKED_CLEAN, SPLIT_CLEAN / 'train', SPLIT_CLEAN / 'test'):
        mapping = read_jsonl(root / 'episode_mapping.jsonl')
        seen = set()
        for new, row in enumerate(mapping):
            source_idx = int(row['source_episode_index'])
            if source_idx in BAD_SOURCE_EPISODES:
                raise RuntimeError(f'Excluded source episode leaked into {root}: {source_idx}')
            if int(row['episode_index']) != new:
                raise RuntimeError(f'Non-contiguous output episode numbering in {root}')
            if row['source_episode_id'] != source_ids[source_idx]:
                raise RuntimeError(f'Source episode UUID mismatch in {root} at {source_idx}')
            if source_idx in seen:
                raise RuntimeError(f'Duplicate source episode {source_idx} in {root}')
            seen.add(source_idx)
        meta_path = root / 'meta/episodes/chunk-000/file-000.parquet'
        meta_rows = pq.read_table(meta_path).to_pylist()
        if len(meta_rows) != len(mapping):
            raise RuntimeError(f'Metadata row count does not match mapping in {root}')
        cursor = 0
        for i, (m, mp) in enumerate(zip(meta_rows, mapping)):
            if int(m['episode_index']) != i or int(m['length']) != int(mp['frames']):
                raise RuntimeError(f'Metadata/source frame map mismatch in {root} episode {i}')
            if int(m['dataset_from_index']) != cursor or int(m['dataset_to_index']) != cursor + int(mp['frames']):
                raise RuntimeError(f'Non-contiguous dataset bounds in {root} episode {i}')
            cursor += int(mp['frames'])
            data_path = root / f"data/chunk-{int(m['data/chunk_index']):03d}/file-{int(m['data/file_index']):03d}.parquet"
            parquet = pq.read_table(data_path, columns=['episode_index','frame_index','index'])
            if parquet.num_rows != int(mp['frames']) or not np.all(parquet['episode_index'].to_numpy() == i):
                raise RuntimeError(f'Output parquet index mismatch in {root} episode {i}')
            if not np.array_equal(parquet['frame_index'].to_numpy(), np.arange(int(mp['frames']))):
                raise RuntimeError(f'Output frame sequence mismatch in {root} episode {i}')
            if not np.array_equal(parquet['index'].to_numpy(), np.arange(int(m['dataset_from_index']), int(m['dataset_to_index']))):
                raise RuntimeError(f'Output global index mismatch in {root} episode {i}')
        info = json.loads((root / 'meta/info.json').read_text())
        if info['total_episodes'] != len(mapping) or info['total_frames'] != cursor:
            raise RuntimeError(f'Info totals mismatch in {root}')
        summary['outputs'][str(root)] = {'episodes':len(mapping),'frames':cursor,'excluded_absent':True,'index_and_uuid_mapping_verified':True}
    interp_mapping = read_jsonl(INTERP_CLEAN / 'episode_mapping.jsonl')
    interp_manifest = json.loads((INTERP_CLEAN / 'cleanup_manifest.json').read_text())
    if any(row['source_episode_index'] in BAD_SOURCE_EPISODES for row in interp_mapping if row['source_episode_index'] is not None):
        raise RuntimeError('Excluded canonical source episode remains in cleaned interpolation mapping')
    if any(int(row['interpolation_source_episode_index']) in interp_manifest['excluded_interpolation_episode_indices'] for row in interp_mapping):
        raise RuntimeError('Excluded interpolation episode remains in mapping')
    summary['outputs'][str(INTERP_CLEAN)] = {'episodes':len(interp_mapping),
        'frames':sum(int(row['frames']) for row in interp_mapping), 'excluded_absent':True,
        'mapping_by_source_uuid_verified':True,
        'unmatched_source_episode_ids_retained':interp_manifest['unmatched_source_episode_ids_retained']}
    write_json(DATASETS / 'clean_bad_episodes_report.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_clean_label_view() -> dict:
    source_report = json.loads((DATASETS / 'label_down_fast_150/selection_report.json').read_text())
    wanted_source = set(map(int, source_report['selected_episode_indices']))
    if wanted_source & BAD_SOURCE_EPISODES:
        raise RuntimeError('Label-down whitelist contains excluded source episodes')
    root = DATASETS / 'label_down_fast_150_clean'
    result = copy_marked_subset(MARKED_SOURCE, root, set(wanted_source), MARKED_SOURCE,
                               BAD_SOURCE_EPISODES, label_filter=wanted_source)
    result['expected_source_episode_indices'] = sorted(wanted_source)
    result['selected_count_matches'] = result['episodes'] == source_report['selected_count']
    if not result['selected_count_matches']:
        raise RuntimeError(f"Fast clean view count mismatch: {result['episodes']} != {source_report['selected_count']}")
    write_json(root / 'selection_report.json', {
        'source_dataset': str(MARKED_SOURCE), 'source_selection_report': str(DATASETS / 'label_down_fast_150/selection_report.json'),
        'selected_count': result['episodes'], 'selected_source_episode_indices': sorted(wanted_source),
        'excluded_source_episode_indices': sorted(BAD_SOURCE_EPISODES),
        'mapping_file': 'episode_mapping.jsonl', 'selected_frame_count': result['frames'],
    })
    return result


if __name__ == '__main__':
    main()
