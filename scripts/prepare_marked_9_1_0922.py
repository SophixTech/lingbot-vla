"""Create disjoint, standalone LeRobot train/test datasets and train-only norms.

Preserve the existing language conditioning, action transforms, and all episodes.
Sources are read only. Existing destinations are never overwritten.
"""
import argparse
import copy
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SOURCE = Path('/home/bjtc/Sophix/datasets/marked_data_0916')
DEST = Path('/home/bjtc/Sophix/datasets/marked_9_1_0922')
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


def write_jsonl(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in data))


def stats(values, quantiles=False):
    result = {
        'mean': values.mean(axis=0, dtype=np.float64).tolist(),
        'std': values.astype(np.float64).std(axis=0).tolist(),
        'min': values.min(axis=0).tolist(),
        'max': values.max(axis=0).tolist(),
    }
    if quantiles:
        q = np.quantile(values, [.01, .99, .02, .98], axis=0)
        result.update({k: v.tolist() for k, v in zip(('q01', 'q99', 'q02', 'q98'), q)})
    return result


def vector(table, key):
    return table[key].combine_chunks().values.to_numpy().reshape(-1, 16)


def make_split():
    if DEST.exists():
        raise FileExistsError(f'Refusing to overwrite {DEST}')
    info = json.loads((SOURCE / 'meta/info.json').read_text())
    labels = {r['episode_index']: r for r in read_jsonl(SOURCE / 'meta/annotations/episode_labels.jsonl')}
    provenance = {r['episode_index']: r for r in read_jsonl(SOURCE / 'meta/episodes.jsonl')}
    tables = [pq.read_table(p) for p in sorted((SOURCE / 'meta/episodes').rglob('*.parquet'))]
    ep_table = pa.concat_tables(tables)
    episodes = {r['episode_index']: r for r in ep_table.to_pylist()}
    assert set(labels) == set(provenance) == set(episodes) == set(range(info['total_episodes']))
    assert len({r['source_episode_id'] for r in provenance.values()}) == len(episodes)
    assert all(type(r['extra_labels']['self_annotation']['grasp_label_up']) is bool for r in labels.values())
    # Largest-remainder allocation: round(1064 * .1) = 106 test episodes.
    groups = {b: sorted(e for e in episodes if labels[e]['extra_labels']['self_annotation']['grasp_label_up'] is b) for b in (False, True)}
    n_test = round(len(episodes) * .1)
    quotas = {b: int(len(ids) * .1) for b, ids in groups.items()}
    for b in sorted(groups, key=lambda b: (-(len(groups[b]) * .1 - quotas[b]), b))[:n_test - sum(quotas.values())]:
        quotas[b] += 1
    rng = np.random.default_rng(SEED)
    test = sorted(int(e) for b in (False, True) for e in rng.permutation(groups[b])[:quotas[b]])
    train = sorted(set(episodes) - set(test))
    assert not (set(train) & set(test)) and len(train) + len(test) == len(episodes)
    manifest = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source': str(SOURCE), 'destination': str(DEST),
        'dataset_name': DEST.name,
        'layout': {'train': 'train', 'test': 'test', 'norm_stats': 'train_stats/norm_stats.json'},
        'seed': SEED, 'rng': 'numpy.default_rng / PCG64', 'test_fraction': .1,
        'split_unit': 'episode', 'stratify': 'extra_labels.self_annotation.grasp_label_up',
        'exclude_regrasp': False, 'source_episode_overlap': 0,
        'source_sha256': {str(p.relative_to(SOURCE)): digest(p) for p in [SOURCE/'meta/info.json', SOURCE/'meta/annotations/episode_labels.jsonl', SOURCE/'meta/episodes.jsonl', *sorted((SOURCE/'meta/episodes').rglob('*.parquet'))]},
        'splits': {},
    }
    DEST.mkdir()
    write_json(DEST/'split_plan.json', {**manifest, 'train_source_episode_indices': train, 'test_source_episode_indices': test})
    for name, selected in [('train', train), ('test', test)]:
        root = DEST/name
        for d in ['data/chunk-000', 'meta/episodes/chunk-000', 'meta/annotations', 'meta/alignments']:
            (root/d).mkdir(parents=True)
        new_meta, new_labels, new_provenance, mapping, hashes = [], [], [], [], []
        state_arrays, action_arrays, timestamps = [], [], []
        offset = 0
        for new, old in enumerate(selected):
            ep = copy.deepcopy(episodes[old])
            src_data = SOURCE/f"data/chunk-{ep['data/chunk_index']:03d}/file-{ep['data/file_index']:03d}.parquet"
            tab = pq.read_table(src_data)
            n = len(tab)
            assert n == ep['length']
            assert np.all(tab['episode_index'].to_numpy() == old)
            np.testing.assert_array_equal(tab['index'].to_numpy(), np.arange(ep['dataset_from_index'], ep['dataset_to_index']))
            np.testing.assert_array_equal(tab['frame_index'].to_numpy(), np.arange(n))
            s, a = vector(tab, 'observation.state'), vector(tab, 'action')
            assert np.isfinite(s).all() and np.isfinite(a).all()
            ts = tab['timestamp'].to_numpy()
            np.testing.assert_allclose(ts, np.arange(n)/info['fps'], atol=1e-4)
            state_arrays.append(s); action_arrays.append(a); timestamps.append(ts[:, None])
            original = tab
            for key, values in [('episode_index', np.full(n, new, np.int64)), ('index', np.arange(offset, offset+n, dtype=np.int64))]:
                tab = tab.set_column(tab.schema.get_field_index(key), key, pa.array(values))
            out_data = root/f'data/chunk-000/file-{new:03d}.parquet'
            pq.write_table(tab, out_data, compression='zstd')
            check = pq.read_table(out_data)
            assert check.equals(tab)
            for key in original.column_names:
                if key not in ('episode_index', 'index'):
                    assert check[key].equals(original[key])
            hashes.append({'source': str(src_data.relative_to(SOURCE)), 'source_sha256': digest(src_data), 'target': str(out_data.relative_to(root)), 'target_sha256': digest(out_data)})
            ep.update(episode_index=new, **{'data/chunk_index': 0, 'data/file_index': new, 'dataset_from_index': offset, 'dataset_to_index': offset+n})
            for key, feature in info['features'].items():
                if feature['dtype'] != 'video':
                    continue
                prefix = 'videos/'+key
                src_video = SOURCE/f"{prefix}/chunk-{episodes[old][prefix+'/chunk_index']:03d}/file-{episodes[old][prefix+'/file_index']:03d}.mp4"
                target = root/f'{prefix}/chunk-000/file-{new:03d}.mp4'
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_video, target)
                before = digest(src_video)
                assert digest(target) == before
                hashes.append({'source': str(src_video.relative_to(SOURCE)), 'source_sha256': before, 'target': str(target.relative_to(root)), 'target_sha256': before})
                ep[prefix+'/chunk_index'] = 0
                ep[prefix+'/file_index'] = new
            new_meta.append(ep)
            for source_record, collection in [(labels[old], new_labels), (provenance[old], new_provenance)]:
                record = copy.deepcopy(source_record)
                record.update(episode_index=new, source_episode_index=old, source_dataset_root=str(SOURCE), alignment_path=f'meta/alignments/episode-{new:06d}.json')
                collection.append(record)
            src_align = SOURCE/provenance[old]['alignment_path']
            dst_align = root/f'meta/alignments/episode-{new:06d}.json'
            shutil.copy2(src_align, dst_align)
            assert digest(src_align) == digest(dst_align)
            hashes.append({'source': str(src_align.relative_to(SOURCE)), 'source_sha256': digest(src_align), 'target': str(dst_align.relative_to(root)), 'target_sha256': digest(dst_align)})
            mapping.append({'episode_index': new, 'source_episode_index': old, 'source_episode_id': provenance[old]['source_episode_id'], 'grasp_label_up': labels[old]['extra_labels']['self_annotation']['grasp_label_up'], 'had_regrasp': labels[old]['extra_labels']['self_annotation']['had_regrasp'], 'frames': n})
            offset += n
            if (new+1) % 100 == 0:
                print(f'{name}: copied and verified {new+1}/{len(selected)} episodes', flush=True)
        pq.write_table(pa.Table.from_pylist(new_meta, schema=ep_table.schema), root/'meta/episodes/chunk-000/file-000.parquet', compression='zstd')
        write_jsonl(root/'meta/annotations/episode_labels.jsonl', new_labels)
        write_jsonl(root/'meta/episodes.jsonl', new_provenance)
        write_jsonl(root/'episode_mapping.jsonl', mapping)
        write_jsonl(root/'file_checksums.jsonl', hashes)
        shutil.copy2(SOURCE/'meta/tasks.parquet', root/'meta/tasks.parquet')
        split_info = copy.deepcopy(info)
        split_info.update(total_episodes=len(selected), total_frames=offset, splits={name: f'0:{len(selected)}'}, source_dataset_root=str(SOURCE), data_path='data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet')
        write_json(root/'meta/info.json', split_info)
        raw_stats = {'observation.state': stats(np.concatenate(state_arrays)), 'action': stats(np.concatenate(action_arrays)), 'timestamp': stats(np.concatenate(timestamps))}
        write_json(root/'meta/stats.json', raw_stats)
        manifest['splits'][name] = {'episodes': len(selected), 'frames': offset, 'label_up': sum(r['grasp_label_up'] for r in mapping), 'label_down': sum(not r['grasp_label_up'] for r in mapping), 'regrasp': sum(r['had_regrasp'] for r in mapping), 'source_episode_indices': selected, 'videos': 3*len(selected), 'data_files': len(selected)}
        write_json(root/'meta/export_manifest.json', {'operation': 'episode_stratified_train_test_split', 'source_dataset': str(SOURCE), 'seed': SEED, 'mapping': '../episode_mapping.jsonl', **manifest['splits'][name]})
        print(name, {k:v for k,v in manifest['splits'][name].items() if k!='source_episode_indices'}, flush=True)
    for rel, sha in manifest['source_sha256'].items():
        assert digest(SOURCE/rel) == sha
    manifest['source_metadata_unchanged'] = True
    write_json(DEST/'split_manifest.json', manifest)
    print('Dataset split complete:', DEST, flush=True)


def compute_norm():
    output = DEST/'train_stats/norm_stats.json'
    if output.exists():
        raise FileExistsError(output)
    meta = pq.read_table(DEST/'train/meta/episodes/chunk-000/file-000.parquet').to_pylist()
    arrays = []
    for ep in meta:
        table = pq.read_table(DEST/f"train/data/chunk-000/file-{ep['data/file_index']:03d}.parquet")
        arrays.append((vector(table, 'observation.state'), vector(table, 'action')))
    states = np.concatenate([s for s,a in arrays]); actions = np.concatenate([a for s,a in arrays])
    norm = {'observation.state.arm.position': stats(states[:,:14], True), 'observation.state.effector.position': stats(states[:,14:], True), 'action.effector.position': stats(actions[:,14:], True)}
    arm = {}
    for h in range(HORIZON):
        values = np.concatenate([a[np.minimum(np.arange(len(a))+h,len(a)-1),:14]-s[:,:14] for s,a in arrays])
        for key,val in stats(values, True).items():
            arm.setdefault(key, []).append(val)
        if h % 5 == 0:
            print('Train-only exact delta statistics horizon', h, flush=True)
    norm['action.arm.position'] = arm
    write_json(output, {'norm_stats': norm, 'count': len(states)})
    write_json(DEST/'train_stats/provenance.json', {'source_split': str(DEST/'train'), 'split_manifest_sha256': digest(DEST/'split_manifest.json'), 'train_episodes': len(arrays), 'train_frames': len(states), 'test_episodes_used': 0, 'chunk_size': HORIZON, 'arm_action': 'future absolute joint target minus current state; clamp at episode tail', 'gripper_action': 'absolute, first horizon step', 'method': 'exact NumPy quantiles; float64 mean/std; same as marked_full0919', 'sha256': digest(output)})
    print('Norm complete:', output, 'count:',len(states), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['split', 'norm'])
    args = parser.parse_args()
    {'split': make_split, 'norm': compute_norm}[args.mode]()
