"""Validate split identity, every episode's loader bounds, and model inputs on CPU."""
import json
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pyarrow.parquet as pq
import torch
import yaml

from lingbotvla.data.vla_data.base_dataset import VLADataset
from lingbotvla.models import build_processor
from prepare_marked_9_1_0922 import DEST, SOURCE, digest, read_jsonl, vector, write_json

torch.set_num_threads(2)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load((PROJECT_ROOT/'configs/vla/marked_9_1_0922.yaml').read_text())
assert DEST.is_dir() and not DEST.is_symlink(), 'Use the canonical dataset directory.'
assert Path(cfg['data']['train_path']) == DEST/'train', 'Training must use only the train split.'
assert Path(cfg['data']['norm_stats_file']) == DEST/'train_stats/norm_stats.json'
assert cfg['data']['exclude_regrasp'] is False, 'Keep the existing full-data split.'
robot_config_root = Path(cfg['data']['robot_config_root'])
assert robot_config_root.is_absolute() and robot_config_root.is_dir()
manifest = json.loads((DEST/'split_manifest.json').read_text())
assert manifest['dataset_name'] == DEST.name
assert manifest['destination'] == str(DEST)
assert manifest['layout'] == {'train': 'train', 'test': 'test', 'norm_stats': 'train_stats/norm_stats.json'}
source_ids = {}
report = {'split_manifest_sha256': digest(DEST/'split_manifest.json'), 'splits': {}}
processor = build_processor(cfg['model']['tokenizer_path'])
model_config = SimpleNamespace(max_state_dim=75, max_action_dim=75, resize_imgs_with_padding=(224,224), tokenizer_max_length=72)
norm = json.loads((DEST/'train_stats/norm_stats.json').read_text())
norm_provenance = json.loads((DEST/'train_stats/provenance.json').read_text())
assert norm_provenance['source_split'] == str(DEST/'train')
assert norm_provenance['split_manifest_sha256'] == digest(DEST/'split_manifest.json')
assert norm_provenance['sha256'] == digest(DEST/'train_stats/norm_stats.json')
assert norm_provenance['test_episodes_used'] == 0
assert norm['count'] == manifest['splits']['train']['frames']
for key, fields in norm['norm_stats'].items():
    shape = (50,14) if key=='action.arm.position' else ((14,) if key.endswith('arm.position') else (2,))
    for values in fields.values():
        assert np.asarray(values).shape == shape and np.isfinite(values).all()
    assert np.all(np.asarray(fields['q99']) >= np.asarray(fields['q01']))
for name in ('train','test'):
    root = DEST/name
    mapping = read_jsonl(root/'episode_mapping.jsonl')
    source_ids[name] = {r['source_episode_id'] for r in mapping}
    data = SimpleNamespace(**cfg['data'])
    data.train_path = str(root)
    data.joints = [str(x) for x in data.joints]
    ds = VLADataset(data.train_path,data.data_name,data,str(robot_config_root),do_nomalize=False)
    assert ds.selected_episodes is None
    assert len(ds) == manifest['splits'][name]['frames']
    assert len(ds.episode_labels) == len(mapping) == manifest['splits'][name]['episodes']
    anchors = videos = 0
    for ep, m in enumerate(mapping):
        meta = ds.dataset.meta.episodes[ep]
        assert m['episode_index'] == meta['episode_index'] == ep
        source_label = ds.episode_labels[ep]
        assert source_label['source_episode_id'] == m['source_episode_id']
        assert source_label['source_episode_index'] == m['source_episode_index']
        tab = pq.read_table(root/f'data/chunk-000/file-{ep:03d}.parquet')
        s,a = vector(tab,'observation.state'), vector(tab,'action')
        np.testing.assert_array_equal(tab['index'].to_numpy(), np.arange(meta['dataset_from_index'],meta['dataset_to_index']))
        for f in sorted({0,len(tab)//2,len(tab)-1}):
            got = ds.getdata(meta['dataset_from_index']+f)
            wanted = np.minimum(np.arange(50)+f,len(tab)-1)
            np.testing.assert_array_equal(got['action.arm.position'].numpy(), a[wanted,:14]-s[f,:14])
            np.testing.assert_array_equal(got['action.effector.position'].numpy(), a[wanted,14:])
            np.testing.assert_array_equal(got['observation.state.arm.position'].numpy(), s[f,:14])
            np.testing.assert_array_equal(got['action_is_pad'].numpy(),np.arange(50)+f>=len(tab))
            assert ('flip it' in got['task']) == (not m['grasp_label_up'])
            anchors += 1
        for key in ds.dataset.meta.video_keys:
            path = root/ds.dataset.meta.get_video_file_path(ep,key)
            with av.open(path) as container:
                stream = container.streams.video[0]
                assert stream.frames == len(tab)
                assert stream.width > 0 and stream.height > 0
            videos += 1
    print(name, 'all numeric anchors', anchors, 'video headers', videos, flush=True)
    ds_image = VLADataset(data.train_path,data.data_name,data,str(robot_config_root),config=model_config,tokenizer=processor.tokenizer,image_processor=processor.image_processor)
    examples = sorted({0,len(mapping)//2,len(mapping)-1,*[next(r['episode_index'] for r in mapping if r['grasp_label_up'] is b) for b in (False,True)]})
    sampled = []
    for ep in examples:
        meta = ds_image.dataset.meta.episodes[ep]
        for f in sorted({0,meta['length']//2,meta['length']-1}):
            batch = ds_image.getdata(meta['dataset_from_index']+f)
            assert batch['actions'].shape == (50,75) and batch['state'].shape == (75,)
            assert batch['joint_mask'].sum().item() == 16
            assert batch['img_masks'].sum().item() == 3
            for k in ('images','actions','state'):
                assert torch.isfinite(batch[k]).all()
            text = processor.tokenizer.decode(batch['lang_tokens'][batch['lang_masks']].tolist())
            assert ('flip it' in text) == (not mapping[ep]['grasp_label_up'])
            sampled.append({'episode':ep,'frame':f,'label_up':mapping[ep]['grasp_label_up']})
    report['splits'][name] = {'episodes':len(mapping),'frames':len(ds),'numeric_anchors_verified':anchors,'video_headers_verified':videos,'normalized_image_language_samples':sampled}
assert not source_ids['train'] & source_ids['test']
assert source_ids['train'] | source_ids['test'] == {r['source_episode_id'] for r in read_jsonl(SOURCE/'meta/episodes.jsonl')}
for path,sha in manifest['source_sha256'].items():
    assert digest(SOURCE/path) == sha
report.update(passed=True,source_episode_overlap=0,train_only_norm_count=norm['count'],norm_sha256=digest(DEST/'train_stats/norm_stats.json'),source_metadata_unchanged=True)
write_json(DEST/'validation_report.json', report)
print('PASS', {k:{x:y for x,y in v.items() if x!='normalized_image_language_samples'} for k,v in report['splits'].items()}, flush=True)
