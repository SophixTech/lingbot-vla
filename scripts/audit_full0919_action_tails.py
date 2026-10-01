"""Read-only action-tail audit; output is separate from training artifacts."""
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

root=Path('/home/bjtc/Sophix/datasets/marked_data_0916')
norm=json.loads(Path('/home/bjtc/Sophix/datasets/marked_full0919_stats/norm_stats.json').read_text())['norm_stats']['action.arm.position']
old=json.loads(Path('/home/bjtc/Sophix/datasets/package_interpolation_lerobot/norm_stats_interpolation_vla.json').read_text())['norm_stats']['action.arm.position']
lo,hi=[np.asarray(norm[k]) for k in ['q01','q99']]
oldlo,oldhi=[np.asarray(old[k]) for k in ['q01','q99']]
labels={r['episode_index']:r['extra_labels']['self_annotation'] for r in map(json.loads,(root/'meta/annotations/episode_labels.jsonl').read_text().splitlines())}
episodes=[]
chunk_magnitude=np.zeros(748377,dtype=np.float64)
frame_episode=np.zeros(748377,dtype=np.int64)
counts={str(k):0 for k in [10,100,1000]}
for p in sorted((root/'data').glob('**/*.parquet')):
    t=pq.read_table(p).to_pydict();s=np.asarray(t['observation.state'],np.float32);a=np.asarray(t['action'],np.float32)
    ep=t['episode_index'][0];n=len(s);idx=np.minimum(np.arange(n)[:,None]+np.arange(50),n-1)
    delta=a[idx,:14]-s[:,None,:14];z=2*(delta-lo)/(hi-lo+1e-6)-1
    peaks=np.max(abs(z),axis=(1,2))
    ids=np.asarray(t['index'])
    chunk_magnitude[ids]=np.mean(abs(z),axis=(1,2))*14/16
    frame_episode[ids]=ep
    for k in counts:counts[k]+=int((peaks>int(k)).sum())
    f,h,j=np.unravel_index(abs(z).argmax(),z.shape)
    episodes.append(dict(episode=ep,regrasp=labels[ep]['had_regrasp'],frames=n,
        samples_peak_gt100=int((peaks>100).sum()),max_abs_normalized=float(abs(z[f,h,j])),
        frame=int(f),horizon=int(h),joint_index=int(j),delta_rad=float(delta[f,h,j]),
        q01=float(lo[h,j]),q99=float(hi[h,j]),
        same_value_old_norm=float(2*(delta[f,h,j]-oldlo[h,j])/(oldhi[h,j]-oldlo[h,j]+1e-6)-1),
        peak_chunk_mean_abs=float(np.mean(abs(z[f])))))
report={'frames':sum(e['frames'] for e in episodes),'sample_peak_counts':counts,
        'episodes_peak_gt100':sum(e['samples_peak_gt100']>0 for e in episodes),
        'episodes':sorted(episodes,key=lambda e:e['max_abs_normalized'],reverse=True)}
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
sampler=StatefulDistributedSampler(range(748377),num_replicas=1,rank=0,shuffle=True,seed=42)
sampler.set_epoch(0)
order=np.array(list(sampler))[:160000].reshape(10000,16)
proxy=chunk_magnitude[order].mean(1)
logs=[json.loads(l) for l in Path('/home/bjtc/Sophix/lingbot-vla/output/marked_full0919/checkpoints/loss.jsonl').read_text().splitlines()]
loss=np.array([r['loss'] for r in logs]);spikes=loss>1
report['sampler_reconstruction']={'seed':42,'epoch':0,'batch_size':16,
    'loss_proxy_correlation':float(np.corrcoef(loss,proxy)[0,1]),
    'spike_steps':int(spikes.sum()),'spikes_containing_episode_806_or_865':int((spikes & np.isin(frame_episode[order],[806,865]).any(1)).sum()),
    'spikes_with_proxy_gt1':int((spikes & (proxy>1)).sum()),
    'largest_examples':[{'step':int(i+1),'loss':float(loss[i]),'normalized_target_magnitude_proxy':float(proxy[i]),'episode_ids':frame_episode[order[i]].tolist(),'original_indices':order[i].tolist()} for i in np.argsort(loss)[-3:][::-1]]}
out=Path('/home/bjtc/Sophix/lingbot-vla/output/marked_full0919/comparison/action_tail_audit.json')
out.write_text(json.dumps(report,indent=2))
print(json.dumps({**report,'episodes':report['episodes'][:2]},indent=2))
