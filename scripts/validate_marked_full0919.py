"""Read-only source validation and independent statistics for marked_full0919."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pyarrow.parquet as pq
import yaml

ROOT = Path('/home/bjtc/Sophix/datasets/marked_data_0916')
OUT = Path('/home/bjtc/Sophix/lingbot-vla/output/marked_full0919/validation')

def statistics(v):
    q = np.quantile(v, [.01, .99, .02, .98], axis=0)
    return dict(mean=v.mean(0, dtype=np.float64).tolist(), std=v.astype(np.float64).std(0).tolist(),
                min=v.min(0).tolist(), max=v.max(0).tolist(),
                **{k:x.tolist() for k,x in zip(['q01','q99','q02','q98'], q)})

def numeric():
    from lingbotvla.data.vla_data.base_dataset import VLADataset
    cfg = yaml.safe_load(Path('configs/vla/marked_full0919.yaml').read_text())
    data = SimpleNamespace(**cfg['data'])
    data.joints = [str(j) for j in data.joints]
    ds = VLADataset(str(ROOT), data.data_name, data, 'configs/robot_configs', do_nomalize=False)
    ld = ds.dataset
    assert ds.selected_episodes is None and ld._absolute_to_relative_idx is None
    tab = ld.hf_dataset.with_format('numpy')
    indices = np.asarray(tab['index'])
    assert np.array_equal(indices, np.arange(len(ds)))
    assert len(ds) == 748377 and len(ds.episode_labels) == 1064
    episodes = []
    samples = 0
    labels = {'regrasp':0, 'label_up':0}
    for ep in range(1064):
        meta = ld.meta.episodes[ep]
        start, end = meta['dataset_from_index'], meta['dataset_to_index']
        t = pq.read_table(ROOT/f"data/chunk-{meta['data/chunk_index']:03d}/file-{meta['data/file_index']:03d}.parquet")
        raw = t.to_pydict()
        s = np.asarray(raw['observation.state'], np.float32)
        a = np.asarray(raw['action'], np.float32)
        assert len(s) == end-start
        assert np.isfinite(s).all() and np.isfinite(a).all()
        assert np.all(np.asarray(raw['episode_index']) == ep)
        assert np.array_equal(raw['index'], np.arange(start,end))
        assert np.array_equal(raw['frame_index'], np.arange(end-start))
        np.testing.assert_allclose(raw['timestamp'],np.arange(end-start)/30,atol=1e-4)
        loaded = tab.select(range(start,end))[:]
        for key in raw:
            np.testing.assert_array_equal(loaded[key], raw[key])
        for f in sorted({0, len(s)//2, len(s)-1}):
            position = start+f
            q,pad = ld._get_query_indices(position,ep)
            wanted = np.minimum(np.arange(50)+f,len(s)-1)
            np.testing.assert_array_equal(q['action'],start+wanted)
            np.testing.assert_array_equal(pad['action_is_pad'].numpy(), np.arange(50)+f>=len(s))
            got = ds.getdata(position)  # no retry: any failure aborts validation
            np.testing.assert_array_equal(got['action.arm.position'].numpy(), a[wanted,:14]-s[f,:14])
            np.testing.assert_array_equal(got['action.effector.position'].numpy(), a[wanted,14:])
            np.testing.assert_array_equal(got['observation.state.arm.position'].numpy(),s[f,:14])
            assert ('flip it' in got['task']) == (not ds.episode_labels[ep]['extra_labels']['self_annotation']['grasp_label_up'])
            samples += 1
        lab = ds.episode_labels[ep]['extra_labels']['self_annotation']
        labels['regrasp'] += int(lab['had_regrasp'])
        labels['label_up'] += int(lab['grasp_label_up'])
        episodes.append((s,a))
        if ep%200==0: print('checked episode',ep,flush=True)
    result={'episodes':len(episodes),'frames':len(ds),'labels':labels,'actual_transformed_samples_checked':samples,
            'all_rows_equal_source':True,'all_positions_equal_original_index':True,'episode_filter':None}
    (OUT/'numeric_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    s = np.concatenate([s for s,a in episodes]); a = np.concatenate([a for s,a in episodes])
    norm={'observation.state.arm.position':statistics(s[:,:14]),
          'observation.state.effector.position':statistics(s[:,14:]),
          'action.effector.position':statistics(a[:,14:])}
    norm['action.arm.position']={k:[] for k in norm['action.effector.position']}
    for h in range(50):
        values=np.concatenate([a[np.minimum(np.arange(len(a))+h,len(a)-1),:14]-s[:,:14] for s,a in episodes])
        for k,v in statistics(values).items():norm['action.arm.position'][k].append(v)
        if h%10==0:print('independent stats horizon',h,flush=True)
    (OUT/'norm_stats_exact_reference.json').write_text(json.dumps({'norm_stats':norm,'count':len(ds)},indent=2))

def decode(job):
    import av
    path,expected=job
    count=0
    with av.open(path) as c:
        stream=c.streams.video[0]
        stream.codec_context.thread_count=1
        for f in c.decode(stream):
            assert f.width>0 and f.height>0
            count+=1
    if count!=expected:raise ValueError(f'{path}: {count} != {expected}')
    return count

def videos():
    info=json.loads((ROOT/'meta/info.json').read_text())
    episodes=[r for f in (ROOT/'meta/episodes').glob('**/*.parquet') for r in pq.read_table(f).to_pylist()]
    jobs=[]
    for e in episodes:
        for k in info['features']:
            if not k.startswith('observation.images.'):continue
            prefix='videos/'+k
            p=ROOT/f"{prefix}/chunk-{e[prefix+'/chunk_index']:03d}/file-{e[prefix+'/file_index']:03d}.mp4"
            jobs.append((str(p),e['length']))
    count=0
    with ProcessPoolExecutor(max_workers=6) as pool:
        for i,n in enumerate(pool.map(decode,jobs),1):
            count+=n
            if i%300==0:print('decoded videos',i,flush=True)
    result={'videos':len(jobs),'decoded_frames':count,'all_frame_counts_match':True}
    (OUT/'video_validation.json').write_text(json.dumps(result,indent=2));print(result)

def compute_stats():
    # Same RunningStats, batch size, feature order and tail hold as compute_norm;
    # assemble chunks in NumPy to avoid 748377 individual Arrow lookups.
    from lingbotvla.utils.normalize import RunningStats, save
    metas=sorted([r for f in (ROOT/'meta/episodes').glob('**/*.parquet') for r in pq.read_table(f).to_pylist()],key=lambda r:r['episode_index'])
    states=[];actions=[];ends=[]
    for e in metas:
        p=ROOT/f"data/chunk-{e['data/chunk_index']:03d}/file-{e['data/file_index']:03d}.parquet"
        t=pq.read_table(p).to_pydict()
        states.append(np.asarray(t['observation.state'],np.float32))
        actions.append(np.asarray(t['action'],np.float32))
        ends.extend([e['dataset_to_index']-1]*e['length'])
    s=np.concatenate(states);a=np.concatenate(actions);ends=np.array(ends)
    keys=['action.arm.position','action.effector.position','observation.state.arm.position','observation.state.effector.position']
    accum={k:RunningStats() for k in keys}
    for start in range(0,len(s),2048):
        stop=min(start+2048,len(s));ids=np.arange(start,stop)
        ix=np.minimum(ids[:,None]+np.arange(50),ends[ids,None])
        delta=a[ix,:14]-s[ids,None,:14]
        for k,v in zip(keys,[delta.reshape(len(ids),-1),a[ids,14:],s[ids,:14],s[ids,14:]]):
            accum[k].update(v)
        if start//2048%40==0:print('stats frames',stop,flush=True)
    result={k:v.get_statistics(chunk_size=50 if k=='action.arm.position' else None) for k,v in accum.items()}
    save('/home/bjtc/Sophix/datasets/marked_full0919_stats/norm_stats.json',result,len(s))
    print('norm saved',len(s),flush=True)

def finalize():
    path=Path('/home/bjtc/Sophix/datasets/marked_full0919_stats/norm_stats.json')
    hist=json.loads(path.read_text())
    exact=json.loads((OUT/'norm_stats_exact_reference.json').read_text())
    assert hist['count']==exact['count']==748377
    diffs={}
    for key,stats in exact['norm_stats'].items():
        diffs[key]={}
        for field,value in stats.items():
            a=np.array(value);b=np.array(hist['norm_stats'][key][field])
            assert a.shape==b.shape and np.isfinite(a).all()
            diffs[key][field]=float(abs(a-b).max())
            if field in ('mean','std'):
                np.testing.assert_allclose(a,b,atol=3e-4,rtol=1e-4)
            elif field in ('min','max'):
                np.testing.assert_array_equal(a,b)
        assert np.all(np.array(stats['q99'])>=np.array(stats['q01']))
    (OUT/'norm_stats_histogram_reference.json').write_text(json.dumps(hist,indent=2))
    path.write_text(json.dumps(exact,indent=2))
    report={'passed':True,'count':exact['count'],'final_method':'exact NumPy quantiles, float64 moments, bounds_99 unchanged',
            'histogram_vs_exact_max_errors':diffs,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (OUT/'norm_comparison.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

def snapshot():
    import shutil
    paths=[Path('configs/vla/marked_full0919.yaml'),Path('configs/robot_configs/package_a2d.yaml'),
           Path('lingbotvla/data/vla_data/base_dataset.py'),Path('lingbotvla/utils/arguments.py'),
           Path('tasks/vla/train_lingbotvla.py'),Path('scripts/validate_marked_full0919.py'),
           ROOT/'meta/info.json',ROOT/'meta/annotations/episode_labels.jsonl',
           Path('/home/bjtc/Sophix/datasets/marked_full0919_stats/norm_stats.json')]
    hashes={}
    for i,p in enumerate(paths):
        hashes[str(p.resolve())]=hashlib.sha256(p.read_bytes()).hexdigest()
        shutil.copy2(p,OUT/f'snapshot_{i}_{p.name}')
    manifest={'source_dataset':str(ROOT),'episodes':1064,'frames':748377,'regrasp_episodes':92,
              'checkpoint_initialization':'base lingbot-vla-4b, not previous fine-tuned checkpoint',
              'systemd_service':'lingbot-vla-marked-full0919','target_steps':10000,
              'full_data_index_handling':'sort logical table by original index, assert contiguous; no episode filtering',
              'norm_method':'exact quantiles, float64 moments; same bounds_99 transform',
              'validation':['numeric_validation.json','video_validation.json','norm_comparison.json','smoke.log'],
              'smoke_checkpoint':'/home/bjtc/Sophix/lingbot-vla/output/marked_full0919_smoke/checkpoints/global_step_2/hf_ckpt',
              'sha256':hashes}
    (OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['numeric','videos','stats','finalize','snapshot']);args=parser.parse_args()
    OUT.mkdir(exist_ok=True,parents=True)
    {'numeric':numeric,'videos':videos,'stats':compute_stats,'finalize':finalize,'snapshot':snapshot}[args.mode]()
