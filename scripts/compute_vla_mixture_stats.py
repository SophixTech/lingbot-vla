"""Compute shared package_a2d training statistics for separate LeRobot roots.

Run from the worktree root, e.g.:
python scripts/compute_vla_mixture_stats.py --config configs/vla/package_multi_posttrain.yaml \
    --output /path/to/shared_train_stats/norm_stats.json

No data are merged or modified. Both packed and one-episode-per-file layouts
are supported. Uses exact quantiles and float64 mean/std; no test split is added
implicitly. Match the training root list and exclude_regrasp setting exactly.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq
import yaml


def episode_tables(root, exclude_regrasp=False):
    """Yield verified complete episodes from metadata, including packed parquet."""
    root = Path(root)
    info = json.loads((root / 'meta/info.json').read_text())
    metadata = sorted((row for p in sorted((root / 'meta/episodes').rglob('*.parquet'))
                       for row in pq.read_table(p).to_pylist()), key=lambda row: row['episode_index'])
    if len(metadata) != info['total_episodes'] or [r['episode_index'] for r in metadata] != list(range(len(metadata))):
        raise ValueError(f'{root}: invalid episode metadata count or indices')
    labels_path = root / 'meta/annotations/episode_labels.jsonl'
    labels = {r['episode_index']: r for r in
              (json.loads(line) for line in labels_path.read_text().splitlines() if line.strip())} if labels_path.exists() else {}
    selected = {ep for ep,r in labels.items() if int(r.get('extra_labels',{}).get('self_annotation',{}).get('had_regrasp',0)) == 0} if exclude_regrasp and labels else None
    cache_path, cache = None, None
    end = 0
    for meta in metadata:
        start, stop = int(meta['dataset_from_index']), int(meta['dataset_to_index'])
        if start != end or stop - start != meta['length']:
            raise ValueError(f'{root}: invalid episode bounds at {meta["episode_index"]}')
        end = stop
        if selected is not None and meta['episode_index'] not in selected:
            continue
        path = root / f"data/chunk-{meta['data/chunk_index']:03d}/file-{meta['data/file_index']:03d}.parquet"
        if path != cache_path:
            cache = pq.read_table(path)
            cache_path = path
        offset = start - int(cache['index'][0].as_py())
        if offset < 0:
            raise ValueError(f'{path}: invalid offset')
        table = cache.slice(offset, stop-start)
        if len(table) != stop-start or not np.array_equal(table['index'].to_numpy(), np.arange(start,stop)):
            raise ValueError(f'{path}: invalid global indices for episode {meta["episode_index"]}')
        if not np.all(table['episode_index'].to_numpy() == meta['episode_index']) or not np.array_equal(table['frame_index'].to_numpy(), np.arange(len(table))):
            raise ValueError(f'{path}: episode/frame IDs disagree with metadata for {meta["episode_index"]}')
        yield meta, table
    if end != info['total_frames']:
        raise ValueError(f'{root}: metadata total frame count mismatch')


def numeric(table, key):
    values = np.asarray(table[key].combine_chunks().values.to_numpy(), dtype=np.float64).reshape(len(table),-1)
    if values.shape[1] != 16 or not np.isfinite(values).all():
        raise ValueError(f'{key}: expected finite 16-dimensional values')
    return values


def describe(values):
    return {key: value.tolist() for key,value in {
        'mean':values.mean(0),'std':values.std(0),'min':values.min(0),'max':values.max(0),
        **dict(zip(('q01','q99','q02','q98'),np.quantile(values,[.01,.99,.02,.98],axis=0)))
    }.items()}


def compute(roots, horizon, exclude_regrasp):
    episodes, provenance = [], []
    seen_sources = set()
    for root in roots:
        source_path = Path(root)/'meta/episodes.jsonl'
        source_ids = {r['episode_index']:r.get('source_episode_id') for r in
                      (json.loads(line) for line in source_path.read_text().splitlines() if line.strip())} if source_path.exists() else {}
        count, frames = 0, 0
        for meta,table in episode_tables(root,exclude_regrasp):
            sid = source_ids.get(meta['episode_index'])
            if sid:
                if sid in seen_sources:
                    raise ValueError(f'Duplicate source episode across selected data: {sid}')
                seen_sources.add(sid)
            episodes.append((numeric(table,'observation.state'),numeric(table,'action')))
            count += 1; frames += len(table)
        if not count:
            raise ValueError(f'{root}: no selected episodes')
        provenance.append({'path':str(root),'episodes':count,'frames':frames})
        print(provenance[-1],flush=True)
    states=np.concatenate([s for s,a in episodes]);actions=np.concatenate([a for s,a in episodes])
    norm={'observation.state.arm.position':describe(states[:,:14]),
          'observation.state.effector.position':describe(states[:,14:]),
          'action.effector.position':describe(actions[:,14:])}
    arm={}
    for h in range(horizon):
        delta=np.concatenate([a[np.minimum(np.arange(len(a))+h,len(a)-1),:14]-s[:,:14] for s,a in episodes])
        for key,val in describe(delta).items(): arm.setdefault(key,[]).append(val)
        if h % 10 == 0: print(f'Action statistics horizon {h}/{horizon-1}',flush=True)
    norm['action.arm.position']=arm
    return {'norm_stats':norm,'count':len(states)}, provenance


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    # Keep this CLI usable as a script without requiring pip install -e .
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from types import SimpleNamespace
    from lingbotvla.data.vla_data.mixed_dataset import resolve_train_paths,validate_dataset_interfaces
    cfg=yaml.safe_load(args.config.read_text());data=SimpleNamespace(**cfg['data'])
    if data.data_name != 'package_a2d': raise ValueError('This statistics CLI implements the package_a2d 14+2 mapping')
    roots=resolve_train_paths(data);validate_dataset_interfaces(roots)
    horizon=cfg['train'].get('chunk_size',50)
    if horizon <= 0: raise ValueError('chunk_size must be positive')
    result,sources=compute(roots,horizon,getattr(data,'exclude_regrasp',True))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f: json.dump(result,f,indent=2,allow_nan=False)
    args.output.with_suffix('.provenance.json').write_text(json.dumps({
        'sources':sources,'chunk_size':horizon,'exclude_regrasp':getattr(data,'exclude_regrasp',True),
        'sampling':'frame proportional','method':'float64 mean/std and exact quantiles',
        'action':'future arm targets minus current arm state; tail clamped within each episode',
        'deployment':'Use this same statistics file for training and inference.'},indent=2))
    print(f'Wrote {args.output}; set data.norm_stats_file to this path in training and deployment.')


if __name__ == '__main__':
    main()
