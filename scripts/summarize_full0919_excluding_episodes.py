"""Filter historical batch metrics by reconstructed sample IDs, without retraining."""
import csv
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

root=Path('/home/bjtc/Sophix/lingbot-vla/output/marked_full0919')
out=root/'comparison'
data=Path('/home/bjtc/Sophix/datasets/marked_data_0916')
meta=sorted([r for f in (data/'meta/episodes').glob('**/*.parquet') for r in pq.read_table(f).to_pylist()],key=lambda r:r['episode_index'])
frame_episode=np.full(748377,-1,dtype=np.int64)
for e in meta:frame_episode[e['dataset_from_index']:e['dataset_to_index']]=e['episode_index']
assert (frame_episode>=0).all()
sampler=StatefulDistributedSampler(range(len(frame_episode)),num_replicas=1,rank=0,shuffle=True,seed=42)
sampler.set_epoch(0)
order=np.array(list(sampler))[:160000].reshape(10000,16)
excluded=np.isin(frame_episode[order],[806,865]).any(1)
rows=[json.loads(l) for l in (root/'checkpoints/loss.jsonl').read_text().splitlines()]
assert [r['step'] for r in rows]==list(range(1,10001))
assert 'Error occurred while getting data' not in (root/'train.log').read_text()
loss=np.array([r['loss'] for r in rows]);grad=np.array([r['grad_norm'] for r in rows]);steps=np.arange(1,10001)
def summary(mask):
    l=loss[mask];g=grad[mask]
    return dict(batches=int(mask.sum()),loss_mean=float(l.mean()),loss_median=float(np.median(l)),
                loss_std=float(l.std()),loss_p95=float(np.quantile(l,.95)),loss_max=float(l.max()),
                loss_above_1=int((l>1).sum()),grad_mean=float(g.mean()),grad_max=float(g.max()))
windows=[]
for start in range(0,10000,1000):
    mask=(steps>start)&(steps<=start+1000)
    windows.append(dict(step_start=start+1,step_end=start+1000,all=summary(mask),retained=summary(mask&~excluded)))
report={'method':'Reconstruct StatefulDistributedSampler seed=42 epoch=0, batch=16; exclude whole optimizer steps containing either episode. Not per-sample loss subtraction or retraining.',
        'excluded_episodes':[806,865],'excluded_source_frames':int(np.isin(frame_episode,[806,865]).sum()),
        'remaining_source_episodes':1062,'remaining_source_frames':int((~np.isin(frame_episode,[806,865])).sum()),
        'excluded_batches':int(excluded.sum()),'all':summary(np.ones(10000,dtype=bool)),'retained':summary(~excluded),
        'windows_1000':windows,'first_100_retained':summary((steps<=100)&~excluded),'last_100_retained':summary((steps>9900)&~excluded)}
report['first_vs_last_1000_loss_reduction_pct']=100*(1-windows[-1]['retained']['loss_mean']/windows[0]['retained']['loss_mean'])
report['penultimate_vs_last_1000_loss_reduction_pct']=100*(1-windows[-1]['retained']['loss_mean']/windows[-2]['retained']['loss_mean'])
(out/'excluded_806_865_metrics.json').write_text(json.dumps(report,indent=2))
with (out/'excluded_806_865_steps.csv').open('w') as f:
    writer=csv.writer(f);writer.writerow(['step','loss','grad_norm','excluded_due_to_episode_806_or_865'])
    writer.writerows((r['step'],r['loss'],r['grad_norm'],bool(excluded[i])) for i,r in enumerate(rows))
print(json.dumps(report,indent=2))
