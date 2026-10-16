import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

from lingbotvla.data.vla_data.mixed_dataset import build_vla_dataset, resolve_train_paths
from lingbotvla.utils.arguments import DataArguments, parse_args


@pytest.fixture
def roots(tmp_path):
    paths=[]
    for name in ('old','new','third'):
        p=tmp_path/name; (p/'meta').mkdir(parents=True)
        (p/'meta/info.json').write_text(json.dumps({
            'total_frames':3,'fps':30,'robot_type':'a2d','features':{
                'action':{'dtype':'float32','shape':[16],'names':list(map(str,range(16)))},
                'observation.state':{'dtype':'float32','shape':[16]},
                'camera':{'dtype':'video','shape':[480,768,3]}}}))
        paths.append(str(p))
    return paths


class Child(Dataset):
    calls=[]
    def __init__(self, repo_id, data_config, chunk_size, **kwargs):
        self.root=repo_id; self.horizon=chunk_size
        self.norm=data_config.norm_stats_file
        self.calls.append((repo_id,data_config,kwargs))
        data_config.train_path='mutated only in child'
    def __len__(self): return 3 if self.root.endswith('old') else 2
    def __getitem__(self,i):
        return {'root':self.root,'episode_index':0,'frame_index':i,
                'actions':torch.full((self.horizon,16),float(i))}


def config(roots):
    return SimpleNamespace(train_paths=roots,train_path='ignored legacy path',data_name='package_a2d',
                           robot_config_root='configs/robot_configs',norm_stats_file='shared.json',chunk_size=7)


@pytest.mark.parametrize('count',[1,2,3])
def test_independent_children_and_frame_proportional_sampling(roots,count):
    cfg=config(roots[:count]); Child.calls=[]
    with patch('lingbotvla.data.vla_data.mixed_dataset.VLADataset',Child):
        ds=build_vla_dataset(cfg,do_nomalize=False)
    assert isinstance(ds,ConcatDataset)==(count>1)
    assert len(ds)==3+2*(count-1)
    assert cfg.train_path=='ignored legacy path'
    assert len({id(call[1]) for call in Child.calls})==count
    assert all(call[1].norm_stats_file=='shared.json' for call in Child.calls)
    assert ds[2]['root']==roots[0]
    if count>1:
        assert ds[3]['root']==roots[1] and ds[3]['frame_index']==0
        assert ds[3]['actions'].shape==(7,16)
    indices=list(DistributedSampler(ds,num_replicas=1,rank=0,shuffle=True,seed=17))
    assert sorted(indices)==list(range(len(ds)))
    assert [sum(ds[i]['root']==r for i in indices) for r in roots[:count]]==[3]+[2]*(count-1)
    assert sum(s['fraction'] for s in ds.training_sources)==pytest.approx(1)


def test_legacy_paths_and_explicit_list_override(roots):
    assert resolve_train_paths(SimpleNamespace(train_path=','.join(roots[:2])))==roots[:2]
    assert resolve_train_paths(SimpleNamespace(train_path=roots[0]))==roots[:1]
    assert resolve_train_paths(config(roots[1:]))==roots[1:]


@pytest.mark.parametrize('values',[[],[''],['a',' '],'not-a-list'])
def test_empty_paths_rejected(values):
    with pytest.raises(ValueError): resolve_train_paths(SimpleNamespace(train_paths=values))


def test_duplicate_symlink_root_rejected(roots,tmp_path):
    alias=tmp_path/'alias';alias.symlink_to(roots[0],target_is_directory=True)
    with pytest.raises(ValueError,match='Duplicate'):
        resolve_train_paths(SimpleNamespace(train_paths=[roots[0],str(alias)]))


def test_incompatible_interfaces_rejected_before_loading(roots):
    path=Path(roots[1])/'meta/info.json';info=json.loads(path.read_text());info['fps']=20
    path.write_text(json.dumps(info))
    with patch('lingbotvla.data.vla_data.mixed_dataset.VLADataset') as child:
        with pytest.raises(ValueError,match='differ'):
            build_vla_dataset(config(roots[:2]))
        child.assert_not_called()


def test_yaml_train_paths_and_cli_override(roots,tmp_path):
    @dataclass
    class Args:
        data: DataArguments = field(default_factory=DataArguments)
    path=tmp_path/'config.yaml';path.write_text('data:\n  train_paths:\n'+''.join('    - '+r+'\n' for r in roots))
    with patch('sys.argv',['train',str(path)]):
        args=parse_args(Args)
    assert args.data.train_paths==roots and args.data.train_path==''
    with patch('sys.argv',['train',str(path),'--data.train_paths',roots[1]]):
        args=parse_args(Args)
    assert args.data.train_paths==roots[1:2]
