import importlib.util
from pathlib import Path
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

spec=importlib.util.spec_from_file_location('mixture_stats',Path(__file__).resolve().parents[1]/'scripts/compute_vla_mixture_stats.py')
stats=importlib.util.module_from_spec(spec);spec.loader.exec_module(stats)


@pytest.fixture
def packed(tmp_path):
    root=tmp_path/'packed'
    (root/'meta/episodes/chunk-000').mkdir(parents=True)
    (root/'data/chunk-000').mkdir(parents=True)
    (root/'meta/annotations').mkdir()
    (root/'meta/info.json').write_text(json.dumps({'total_episodes':2,'total_frames':4}))
    (root/'meta/episodes.jsonl').write_text('\n'.join(json.dumps({'episode_index':ep,'source_episode_id':f'source-{ep}'}) for ep in range(2)))
    (root/'meta/annotations/episode_labels.jsonl').write_text('\n'.join(json.dumps({'episode_index':ep,'extra_labels':{'self_annotation':{'had_regrasp':ep}}}) for ep in range(2)))
    pq.write_table(pa.Table.from_pylist([{'episode_index':ep,'length':2,'dataset_from_index':2*ep,'dataset_to_index':2*ep+2,'data/chunk_index':0,'data/file_index':0} for ep in range(2)]),root/'meta/episodes/chunk-000/file-000.parquet')
    states=np.array([[1]*16,[2]*16,[100]*16,[200]*16],dtype=np.float32)
    actions=np.array([[3]*16,[5]*16,[300]*16,[500]*16],dtype=np.float32)
    table=pa.table({'observation.state':pa.FixedSizeListArray.from_arrays(states.ravel(),16),
                    'action':pa.FixedSizeListArray.from_arrays(actions.ravel(),16),
                    'episode_index':[0,0,1,1],'frame_index':[0,1,0,1],'index':[0,1,2,3]})
    pq.write_table(table,root/'data/chunk-000/file-000.parquet')
    return root


def test_packed_episode_slices_and_tail_clamping(packed):
    episodes=list(stats.episode_tables(packed))
    assert len(episodes)==2
    assert episodes[1][1]['index'].to_pylist()==[2,3]
    result,source=stats.compute([packed],3,False)
    assert result['count']==4 and source[0]['episodes']==2
    # Future targets never cross from the first episode into the second.
    expected=np.array([[2,3,200,300],[4,3,400,300],[4,3,400,300]]).mean(axis=1)
    np.testing.assert_allclose(np.array(result['norm_stats']['action.arm.position']['mean'])[:,0],expected)
    assert result['norm_stats']['observation.state.arm.position']['mean'][0]==75.75


def test_regrasp_filter_matches_training(packed):
    result,source=stats.compute([packed],2,True)
    assert result['count']==2 and source[0]['episodes']==1
    assert result['norm_stats']['observation.state.arm.position']['mean'][0]==1.5


def test_rejects_corrupt_episode_ids(packed):
    path=packed/'data/chunk-000/file-000.parquet';table=pq.read_table(path)
    table=table.set_column(table.schema.get_field_index('episode_index'),'episode_index',pa.array([0,0,0,0]))
    pq.write_table(table,path)
    with pytest.raises(ValueError,match='episode/frame IDs'):
        list(stats.episode_tables(packed))


def test_duplicate_source_cannot_bias_joint_statistics(packed):
    with pytest.raises(ValueError,match='Duplicate source'):
        stats.compute([packed,packed],2,False)
