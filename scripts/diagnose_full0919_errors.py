"""Analyze saved per-joint predictions against horizon and causal baselines."""
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

root=Path('/home/bjtc/Sophix/lingbot-vla/output/marked_full0919')
data=Path('/home/bjtc/Sophix/datasets/marked_data_0916')
ev=root/'evaluation_right_arm_per_joint'
out=root/'error_diagnosis';out.mkdir(exist_ok=True)
manifest=json.loads((ev/'manifest.json').read_text())
errors=np.load(ev/'regular_64_episodes_right_arm_errors.npz')['errors_rad']
meta={r['episode_index']:r for f in (data/'meta/episodes').glob('**/*.parquet') for r in pq.read_table(f).to_pylist()}
labels={r['episode_index']:r['extra_labels']['self_annotation'] for r in map(json.loads,(data/'meta/annotations/episode_labels.jsonl').read_text().splitlines())}
sampler=StatefulDistributedSampler(range(748377),num_replicas=1,rank=0,shuffle=True,seed=42)
sampler.set_epoch(0);seen=set(list(sampler)[:160000])
datasets={};summaries=[];cursor=0;allerr=[];hold=[];velocity=[];horizons=[];motions=[];groups=[];sample_seen=[]
def stats(v):
    return {'mae_deg':float(np.rad2deg(abs(v).mean())), 'rmse_deg':float(np.rad2deg(np.sqrt(np.square(v).mean()))),
            'p95_abs_deg':float(np.rad2deg(np.quantile(abs(v),.95))),'predicted_frames':len(v)}
for ep,frame,idx in manifest['jobs']:
    if ep in (806,865):continue
    if ep not in datasets:
        m=meta[ep];p=data/f"data/chunk-{m['data/chunk_index']:03d}/file-{m['data/file_index']:03d}.parquet"
        t=pq.read_table(p).to_pydict();datasets[ep]=(np.asarray(t['observation.state'],float),np.asarray(t['action'],float))
    s,a=datasets[ep];n=min(50,len(a)-frame);truth=a[frame:frame+n,7:14];err=errors[cursor:cursor+n];cursor+=n
    stationary=s[frame,7:14]-truth
    # Only observed states at or before the prediction origin; no future leakage.
    back=min(3,frame);speed=(s[frame,7:14]-s[frame-back,7:14])/max(1,back)
    linear=s[frame,7:14]+np.arange(n)[:,None]*speed-truth
    disp=float(np.rad2deg(np.mean(abs(truth[-1]-s[frame,7:14]))))
    label=labels[ep];group=f"regrasp_{int(label['had_regrasp'])}_up_{int(label['grasp_label_up'])}"
    rec={'episode':ep,'frame':frame,'original_index':idx,'exact_origin_seen_in_training':idx in seen,'group':group,
         'displacement_last_frame_mean_deg':disp,'model':stats(err),'hold':stats(stationary),'constant_velocity':stats(linear)}
    summaries.append(rec);allerr.append(err);hold.append(stationary);velocity.append(linear);horizons.extend(range(n));motions.extend([disp]*n);groups.extend([group]*n);sample_seen.extend([idx in seen]*n)
assert cursor==len(errors)
model=np.concatenate(allerr);hold=np.concatenate(hold);velocity=np.concatenate(velocity);horizons=np.array(horizons);motions=np.array(motions);groups=np.array(groups);sample_seen=np.array(sample_seen)
def compare(mask):
    return {'model':stats(model[mask]),'hold':stats(hold[mask]),'constant_velocity':stats(velocity[mask])}
result={'protocol':manifest['protocol'],'total':compare(np.ones(len(model),bool)),
        'horizon_bins':{},'motion_bins':{},'annotation_groups':{},'seen_origins':{},'samples':summaries}
for a,b in [(0,1),(0,5),(5,10),(10,25),(25,50)]:result['horizon_bins'][f'{a}-{b-1}']=compare((horizons>=a)&(horizons<b))
for a,b in [(0,1),(1,5),(5,1000)]:
    mask=(motions>=a)&(motions<b)
    if mask.any():result['motion_bins'][f'{a}-{b}_deg']=compare(mask)
for g in sorted(set(groups)):result['annotation_groups'][g]=compare(groups==g)
for x in [False,True]:result['seen_origins'][str(x)]=compare(sample_seen==x)
result['worst_samples']=sorted(summaries,key=lambda r:r['model']['mae_deg'],reverse=True)[:10]
result['source_hashes_unchanged']={}
import hashlib
run=json.loads((root/'validation/run_manifest.json').read_text())
for path,h in run['sha256'].items():
    if 'base_dataset' in path or 'norm_stats.json' in path or 'train_lingbotvla.py' in path:
        result['source_hashes_unchanged'][path]=hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
(out/'horizon_baseline_analysis.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ('samples','worst_samples')},indent=2))
print('worst samples',json.dumps(result['worst_samples'][:3],indent=2))
