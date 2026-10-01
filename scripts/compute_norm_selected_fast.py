"""Fast, index-preserving norm computation for a selected episode view.

This mirrors VLADataset's 50-frame action query: actions beyond an episode's
end are held at the final action. It reads source Parquet files by the
original episode metadata and never builds a filtered/reindexed table.
"""
from pathlib import Path
import json
import numpy as np
import pyarrow.parquet as pq
from lingbotvla.utils.normalize import RunningStats, save

VIEW = Path('/home/bjtc/Sophix/datasets/label_down_fast_150')
OUT = Path('/home/bjtc/Sophix/datasets/label_down_fast_stats/norm_stats.json')
CHUNK = 50


def main():
    ep_table = pq.read_table(VIEW / 'meta/episodes/chunk-000/file-000.parquet').to_pydict()
    labels = {}
    for line in (VIEW / 'meta/annotations/episode_labels.jsonl').read_text().splitlines():
        r = json.loads(line)
        labels[int(r['episode_index'])] = r

    stats = {k: RunningStats() for k in (
        'observation.state.arm.position', 'observation.state.effector.position',
        'action.arm.position', 'action.effector.position')}
    selected = []
    total_frames = 0
    for row, ep in enumerate(ep_table['episode_index']):
        ep = int(ep)
        if ep not in labels:
            continue
        n = int(ep_table['length'][row])
        p = VIEW / f"data/chunk-{int(ep_table['data/chunk_index'][row]):03d}/file-{int(ep_table['data/file_index'][row]):03d}.parquet"
        t = pq.read_table(p, columns=['observation.state', 'action', 'episode_index', 'index']).to_pydict()
        state = np.asarray(t['observation.state'], dtype=np.float64)
        action = np.asarray(t['action'], dtype=np.float64)
        assert len(state) == n == len(action)
        assert np.all(np.asarray(t['episode_index']) == ep)
        expected = np.arange(int(ep_table['dataset_from_index'][row]), int(ep_table['dataset_to_index'][row]))
        assert np.array_equal(np.asarray(t['index'], dtype=np.int64), expected)
        # Every origin frame contributes one state and one 50-frame action chunk.
        idx = np.minimum(np.arange(n)[:, None] + np.arange(CHUNK)[None, :], n - 1)
        target = action[idx]
        state_arm = state[:, :14]
        state_eff = state[:, 14:]
        stats['observation.state.arm.position'].update(state_arm)
        stats['observation.state.effector.position'].update(state_eff)
        stats['action.arm.position'].update((target[:, :, :14] - state_arm[:, None, :]).reshape(n, -1))
        # Non-delta effector actions use only the first action in each chunk,
        # matching scripts/compute_norm.py's batch[key][:, 0] path.
        stats['action.effector.position'].update(target[:, 0, 14:])
        total_frames += n
        selected.append(ep)

    norm = {
        'observation.state.arm.position': stats['observation.state.arm.position'].get_statistics(),
        'observation.state.effector.position': stats['observation.state.effector.position'].get_statistics(),
        'action.arm.position': stats['action.arm.position'].get_statistics(chunk_size=CHUNK),
        'action.effector.position': stats['action.effector.position'].get_statistics(),
    }
    save(OUT, norm, stats['observation.state.arm.position']._count)
    report = {
        'view': str(VIEW), 'output': str(OUT), 'episodes': len(selected),
        'frames': total_frames, 'action_chunk_size': CHUNK,
        'action_arm_samples': stats['action.arm.position']._count,
        'state_samples': stats['observation.state.arm.position']._count,
        'episode_indices': selected,
        'mapping_check': 'PASS: original episode_index and global index ranges checked per source parquet',
    }
    (OUT.parent / 'compute_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
