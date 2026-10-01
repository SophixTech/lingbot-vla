"""Split marked_data_0916 into label-up/down LeRobot datasets."""
import copy
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SOURCE = Path('/home/bjtc/Sophix/datasets/marked_data_0916')
DEST = Path('/home/bjtc/Sophix/datasets/separated_data')
SEED = 42
HORIZON = 50


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))


def stats(values):
    values = np.asarray(values)
    q = np.quantile(values, [.01, .99, .02, .98], axis=0)
    return {
        'mean': values.mean(axis=0, dtype=np.float64).tolist(),
        'std': values.astype(np.float64).std(axis=0).tolist(),
        'min': values.min(axis=0).tolist(),
        'max': values.max(axis=0).tolist(),
        'q01': q[0].tolist(), 'q99': q[1].tolist(),
        'q02': q[2].tolist(), 'q98': q[3].tolist(),
    }


def vector(table, key):
    return table[key].combine_chunks().values.to_numpy().reshape(-1, 16)


def source_records(info):
    labels = {r['episode_index']: r for r in read_jsonl(SOURCE/'meta/annotations/episode_labels.jsonl')}
    provenance = {r['episode_index']: r for r in read_jsonl(SOURCE/'meta/episodes.jsonl')}
    tables = [pq.read_table(p) for p in sorted((SOURCE/'meta/episodes').rglob('*.parquet'))]
    ep_table = pa.concat_tables(tables)
    episodes = {int(r['episode_index']): r for r in ep_table.to_pylist()}
    assert set(labels) == set(provenance) == set(episodes) == set(range(info['total_episodes']))
    assert all(type(r['extra_labels']['self_annotation']['grasp_label_up']) is bool for r in labels.values())
    return labels, provenance, episodes, ep_table


def split_indices(episodes, labels):
    rng = np.random.default_rng(SEED)
    result = {}
    for label_up, name in ((True, 'label_up'), (False, 'label_down')):
        ids = sorted(e for e in episodes if labels[e]['extra_labels']['self_annotation']['grasp_label_up'] is label_up)
        test_count = round(len(ids) * .1)
        test = sorted(int(e) for e in rng.permutation(ids)[:test_count])
        test_set = set(test)
        result[name] = {'train': sorted(set(ids)-test_set), 'test': test}
    return result


def write_split(root, name, selected, source_info, labels, provenance, episodes, ep_schema):
    root.mkdir(parents=True)
    for d in ('data/chunk-000', 'meta/episodes/chunk-000', 'meta/annotations', 'meta/alignments'):
        (root/d).mkdir(parents=True, exist_ok=True)
    new_meta, new_labels, new_provenance, mapping, hashes = [], [], [], [], []
    offset = 0
    for new, old in enumerate(selected):
        ep = copy.deepcopy(episodes[old])
        src_data = SOURCE/f"data/chunk-{ep['data/chunk_index']:03d}/file-{ep['data/file_index']:03d}.parquet"
        table = pq.read_table(src_data)
        n = len(table)
        assert n == ep['length'] and np.all(table['episode_index'].to_numpy() == old)
        assert np.array_equal(table['frame_index'].to_numpy(), np.arange(n))
        np.testing.assert_array_equal(table['index'].to_numpy(), np.arange(ep['dataset_from_index'], ep['dataset_to_index']))
        state, action = vector(table, 'observation.state'), vector(table, 'action')
        assert np.isfinite(state).all() and np.isfinite(action).all()
        np.testing.assert_allclose(table['timestamp'].to_numpy(), np.arange(n)/source_info['fps'], atol=1e-4)
        for key, values in [('episode_index', np.full(n, new, np.int64)), ('index', np.arange(offset, offset+n, dtype=np.int64))]:
            table = table.set_column(table.schema.get_field_index(key), key, pa.array(values))
        out_data = root/f'data/chunk-000/file-{new:03d}.parquet'
        pq.write_table(table, out_data, compression='zstd')
        check = pq.read_table(out_data)
        assert check.equals(table)
        hashes.append({'source': str(src_data.relative_to(SOURCE)), 'source_sha256': digest(src_data), 'target': str(out_data.relative_to(root)), 'target_sha256': digest(out_data)})
        ep.update(episode_index=new, **{'data/chunk_index': 0, 'data/file_index': new}, dataset_from_index=offset, dataset_to_index=offset+n)
        for key, feature in source_info['features'].items():
            if feature['dtype'] != 'video':
                continue
            prefix = 'videos/'+key
            src_video = SOURCE/f"{prefix}/chunk-{episodes[old][prefix+'/chunk_index']:03d}/file-{episodes[old][prefix+'/file_index']:03d}.mp4"
            out_video = root/f'{prefix}/chunk-000/file-{new:03d}.mp4'
            out_video.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_video, out_video)
            sha = digest(src_video)
            assert digest(out_video) == sha
            hashes.append({'source': str(src_video.relative_to(SOURCE)), 'source_sha256': sha, 'target': str(out_video.relative_to(root)), 'target_sha256': sha})
            ep[prefix+'/chunk_index'], ep[prefix+'/file_index'] = 0, new
        new_meta.append(ep)
        for src, dst in ((labels[old], new_labels), (provenance[old], new_provenance)):
            record = copy.deepcopy(src)
            record.update(episode_index=new, source_episode_index=old, source_dataset_root=str(SOURCE), alignment_path=f'meta/alignments/episode-{new:06d}.json')
            dst.append(record)
        src_align = SOURCE/provenance[old]['alignment_path']
        out_align = root/f'meta/alignments/episode-{new:06d}.json'
        shutil.copy2(src_align, out_align)
        assert digest(src_align) == digest(out_align)
        mapping.append({'episode_index': new, 'source_episode_index': old, 'source_episode_id': provenance[old]['source_episode_id'], 'grasp_label_up': labels[old]['extra_labels']['self_annotation']['grasp_label_up'], 'had_regrasp': labels[old]['extra_labels']['self_annotation']['had_regrasp'], 'frames': n})
        offset += n
        if (new+1) % 100 == 0:
            print(f'{root.parent.name}/{name}: {new+1}/{len(selected)} episodes', flush=True)
    pq.write_table(pa.Table.from_pylist(new_meta, schema=ep_schema), root/'meta/episodes/chunk-000/file-000.parquet', compression='zstd')
    write_jsonl(root/'meta/annotations/episode_labels.jsonl', new_labels)
    write_jsonl(root/'meta/episodes.jsonl', new_provenance)
    write_jsonl(root/'episode_mapping.jsonl', mapping)
    write_jsonl(root/'file_checksums.jsonl', hashes)
    shutil.copy2(SOURCE/'meta/tasks.parquet', root/'meta/tasks.parquet')
    info = copy.deepcopy(source_info)
    info.update(total_episodes=len(selected), total_frames=offset, splits={name: f'0:{len(selected)}'}, source_dataset_root=str(SOURCE))
    write_json(root/'meta/info.json', info)
    stat_rows = []
    for old in selected:
        meta = episodes[old]
        table = pq.read_table(SOURCE/f"data/chunk-{meta['data/chunk_index']:03d}/file-{meta['data/file_index']:03d}.parquet")
        state, action = vector(table, 'observation.state'), vector(table, 'action')
        stat_rows.append((state, action, table['timestamp'].to_numpy()[:, None]))
    s = np.concatenate([v[0] for v in stat_rows]); a = np.concatenate([v[1] for v in stat_rows]); t = np.concatenate([v[2] for v in stat_rows])
    write_json(root/'meta/stats.json', {'observation.state': stats(s), 'action': stats(a), 'timestamp': stats(t)})
    return {'episodes': len(selected), 'frames': offset, 'source_episode_indices': selected, 'mapping': mapping}


def compute_norm(train_root, output_dir, episode_rows):
    arrays = []
    for ep in episode_rows:
        table = pq.read_table(train_root/f"data/chunk-000/file-{ep['data/file_index']:03d}.parquet")
        arrays.append((vector(table, 'observation.state'), vector(table, 'action')))
    states = np.concatenate([s for s, _ in arrays]); actions = np.concatenate([a for _, a in arrays])
    norm = {
        'observation.state.arm.position': stats(states[:, :14]),
        'observation.state.effector.position': stats(states[:, 14:]),
        'action.effector.position': stats(actions[:, 14:]),
    }
    arm = {}
    for h in range(HORIZON):
        values = np.concatenate([a[np.minimum(np.arange(len(a))+h, len(a)-1), :14]-s[:, :14] for s, a in arrays])
        sh = stats(values)
        for key, value in sh.items():
            arm.setdefault(key, []).append(value)
    norm['action.arm.position'] = arm
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir/'norm_stats.json'
    write_json(out, {'norm_stats': norm, 'count': len(states)})
    write_json(output_dir/'provenance.json', {
        'source_split': str(train_root), 'train_episodes': len(arrays), 'train_frames': len(states),
        'test_episodes_used': 0, 'chunk_size': HORIZON,
        'method': 'exact NumPy quantiles; float64 mean/std; arm action is future target minus current state with episode-tail hold',
        'sha256': digest(out),
    })


def main():
    if DEST.exists():
        raise FileExistsError(f'Refusing to overwrite {DEST}')
    source_info = json.loads((SOURCE/'meta/info.json').read_text())
    labels, provenance, episodes, ep_table = source_records(source_info)
    splits = split_indices(episodes, labels)
    manifest = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'source': str(SOURCE), 'destination': str(DEST),
        'seed': SEED, 'rng': 'numpy.default_rng / PCG64', 'test_fraction': .1, 'split_unit': 'episode',
        'label_field': 'extra_labels.self_annotation.grasp_label_up', 'include_all_labelled_episodes': True,
        'normalization': 'train only; separate per label group; action horizon 50', 'splits': {},
    }
    DEST.mkdir(parents=True)
    for group_name, group_splits in splits.items():
        group_root = DEST/group_name
        group_root.mkdir()
        manifest['splits'][group_name] = {}
        for split_name, selected in group_splits.items():
            result = write_split(group_root/split_name, split_name, selected, source_info, labels, provenance, episodes, ep_table.schema)
            manifest['splits'][group_name][split_name] = {k:v for k,v in result.items() if k != 'mapping'}
            print(group_name, split_name, result['episodes'], 'episodes', result['frames'], 'frames', flush=True)
        train_rows = pq.read_table(group_root/'train/meta/episodes/chunk-000/file-000.parquet').to_pylist()
        compute_norm(group_root/'train', group_root/'train_stats', train_rows)
        write_json(group_root/'split_manifest.json', {
            'group': group_name, 'seed': SEED, 'test_fraction': .1, 'split_unit': 'episode',
            'label_field': manifest['label_field'], 'train': manifest['splits'][group_name]['train'],
            'test': manifest['splits'][group_name]['test'], 'source_episode_overlap': 0,
        })
    write_json(DEST/'split_manifest.json', manifest)
    print('Finished:', DEST, flush=True)


if __name__ == '__main__':
    main()
