"""Recompute numeric normalization statistics for the cleaned training views."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

from lingbotvla.utils.normalize import RunningStats, save

ROOT = Path('/home/bjtc/Sophix')
DATASETS = ROOT / 'datasets'


def compute(root: Path, output: Path, selected_report: Path | None = None) -> dict:
    meta = pq.read_table(root / 'meta/episodes/chunk-000/file-000.parquet').to_pylist()
    if selected_report:
        selected = set(json.loads(selected_report.read_text())['selected_episode_indices'])
        mapping = {int(x['episode_index']): int(x['source_episode_index'])
                   for x in [json.loads(line) for line in (root / 'episode_mapping.jsonl').read_text().splitlines()]}
        meta = [row for row in meta if mapping[int(row['episode_index'])] in selected]
    stats = {k: RunningStats() for k in ('observation.state.arm.position',
        'observation.state.effector.position', 'action.arm.position', 'action.effector.position')}
    frames = 0
    cached_path, cached_table = None, None
    for row in meta:
        idx = int(row['episode_index'])
        p = root / f"data/chunk-{int(row['data/chunk_index']):03d}/file-{int(row['data/file_index']):03d}.parquet"
        if p != cached_path:
            cached_table = pq.read_table(p, columns=['observation.state', 'action', 'episode_index', 'index'])
            cached_path = p
        offset = int(row['dataset_from_index']) - int(cached_table['index'][0].as_py())
        t = cached_table.slice(offset, int(row['length']))
        state = np.asarray(t['observation.state'].combine_chunks().values.to_numpy(), dtype=np.float64).reshape(-1, 16)
        action = np.asarray(t['action'].combine_chunks().values.to_numpy(), dtype=np.float64).reshape(-1, 16)
        if len(state) != int(row['length']) or not np.all(t['episode_index'].to_numpy() == idx):
            raise RuntimeError(f'Index/frame mismatch at {root} episode {idx}')
        expected = np.arange(int(row['dataset_from_index']), int(row['dataset_to_index']))
        if not np.array_equal(t['index'].to_numpy(), expected):
            raise RuntimeError(f'Global index mismatch at {root} episode {idx}')
        n = len(state)
        target_idx = np.minimum(np.arange(n)[:, None] + np.arange(50)[None, :], n - 1)
        target = action[target_idx]
        stats['observation.state.arm.position'].update(state[:, :14])
        stats['observation.state.effector.position'].update(state[:, 14:])
        stats['action.arm.position'].update((target[:, :, :14] - state[:, None, :14]).reshape(n, -1))
        stats['action.effector.position'].update(target[:, 0, 14:])
        frames += n
    norm = {
        'observation.state.arm.position': stats['observation.state.arm.position'].get_statistics(),
        'observation.state.effector.position': stats['observation.state.effector.position'].get_statistics(),
        'action.arm.position': stats['action.arm.position'].get_statistics(chunk_size=50),
        'action.effector.position': stats['action.effector.position'].get_statistics(),
    }
    save(output, norm, frames)
    report = {'dataset': str(root), 'output': str(output), 'episodes': len(meta),
              'frames': frames, 'mapping_check': 'PASS', 'source_stats_excluded': True}
    output.parent.joinpath('compute_report.json').write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    reports = []
    reports.append(compute(DATASETS / 'marked_data_0916_clean',
                           DATASETS / 'marked_full0919_stats_clean/norm_stats.json'))
    reports.append(compute(DATASETS / 'marked_9_1_0922_clean/train',
                           DATASETS / 'marked_9_1_0922_clean/train_stats/norm_stats.json'))
    reports.append(compute(DATASETS / 'label_down_fast_150_clean',
                           DATASETS / 'label_down_fast_stats_clean/norm_stats.json',
                           DATASETS / 'label_down_fast_150/selection_report.json'))
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
