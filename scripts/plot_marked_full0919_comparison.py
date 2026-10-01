import json
import re
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('/home/bjtc/Sophix/lingbot-vla/output/marked_full0919')
out=root/'comparison';out.mkdir(exist_ok=True)
source=Path('/home/bjtc/.codex/sessions/2026/08/12/rollout-2026-08-12T14-23-09-019ff4a3-95ae-7ea0-aaaa-c8c705954f23.jsonl')
pattern=re.compile(r'(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})[^\r\n]*?Step (\d+)/10000, Epoch (\d+), Loss ([\d.eE+-]+), VLA_Loss [\d.eE+-]+, Depth_Loss [\d.eE+-]+, GradNorm ([\d.eE+-]+), LR ([\d.eE+-]+), StepTime ([\d.eE+-]+)s')

def texts(value):
    if isinstance(value,str):
        yield value
        try:decoded=json.loads(value)
        except (ValueError,TypeError):return
        if isinstance(decoded,(dict,list)):yield from texts(decoded)
    elif isinstance(value,dict):
        for k,v in value.items():
            if k not in ('input','command','encrypted_content'):yield from texts(v)
    elif isinstance(value,list):
        for v in value:yield from texts(v)

points={}
for lineno,line in enumerate(source.open(),1):
    entry=json.loads(line);payload=entry.get('payload',{})
    if payload.get('type') not in ('custom_tool_call_output','function_call_output'):continue
    for s in texts(payload.get('output')):
        for m in pattern.finditer(s):
            stamp,step,epoch,loss,grad,lr,seconds=m.groups()
            t=datetime.strptime(stamp,'%m/%d/%Y %H:%M:%S')
            # Restrict to the successful original run, excluding earlier restarts
            # and later 5/20-denoising-step runs and ACT training.
            if not datetime(2026,9,2,12,27)<=t<=datetime(2026,9,3,16,57):continue
            r=dict(step=int(step),epoch=int(epoch),loss=float(loss),grad_norm=float(grad),lr=float(lr),step_time=float(seconds),timestamp=stamp,source_line=lineno)
            old=points.get(r['step'])
            if old:
                assert all(old[k]==r[k] for k in ('loss','grad_norm','step_time')),('conflicting historical points',old,r)
            points[r['step']]=r
history=sorted(points.values(),key=lambda r:r['step'])
(out/'interpolation_recovered_points.json').write_text(json.dumps({'source':str(source),'note':'Only timestamped log lines in successful original run; sparse rounded observations, no interpolation or imputation','points':history},indent=2))
rows=[json.loads(l) for l in (root/'checkpoints/loss.jsonl').read_text().splitlines()]
assert len(rows)==10000 and [r['step'] for r in rows]==list(range(1,10001))
summary={'marked_last':rows[-1],'historical_points':len(history),'historical_step_range':[history[0]['step'],history[-1]['step']] if history else None,'marked_windows':{}}
for n in (100,500,1000):
    summary['marked_windows'][str(n)]={k:float(np.mean([r[k] for r in rows[-n:]])) for k in ['loss','grad_norm','step_time']}
loss=np.array([r['loss'] for r in rows]);summary['loss_quantiles']=dict(zip(['p50','p90','p99','max'],map(float,[*np.quantile(loss,[.5,.9,.99]),loss.max()])))
summary['loss_above_1']=int((loss>1).sum());summary['nonfinite']=sum(not np.isfinite(r[k]) for r in rows for k in ['loss','grad_norm','step_time'])
summary['loss_bins_1000']=[float(loss[i:i+1000].mean()) for i in range(0,10000,1000)]
fig,axes=plt.subplots(4,1,figsize=(12,12),layout='constrained')
x=np.arange(1,len(rows)+1);hx=np.array([r['step'] for r in history])
for ax,key,title in zip(axes[:3],['loss','grad_norm','step_time'],['Training loss (normalized L1 flow matching)','Gradient norm','Compute time (seconds / optimizer step)']):
    values=np.array([r[key] for r in rows])
    ax.plot(x,values,color='#1679b5',lw=.45,alpha=.25,label='marked_full0919: per step')
    ax.plot(x[99:],np.convolve(values,np.ones(100)/100,'valid'),color='#1679b5',lw=1.5,label='marked_full0919: 100-step mean')
    if history:ax.scatter(hx,[r[key] for r in history],s=10,color='#d65a19',alpha=.65,label=f'interpolation: {len(history)} recovered individual steps',zorder=3)
    ax.set_ylabel(title);ax.set_xlabel('Optimizer step');ax.grid(alpha=.18);ax.legend(loc='upper right',fontsize=9)
# Zoom is additional; full-range loss above retains all spikes.
axes[0].set_yscale('log')
ax=axes[3];ax.plot(x,loss,color='#1679b5',lw=.45,alpha=.3)
ax.plot(x[99:],np.convolve(loss,np.ones(100)/100,'valid'),color='#1679b5',lw=1.5)
if history:ax.scatter(hx,[r['loss'] for r in history],s=10,color='#d65a19',alpha=.65)
ax.set_ylim(0,.6);ax.set_ylabel('Loss detail (0 to 0.6)');ax.set_xlabel('Optimizer step');ax.grid(alpha=.18)
fig.suptitle('marked_full0919 vs interpolation_vla\nHistorical data are sparse; gaps are not reconstructed. Different normalization limits loss comparability.',fontsize=13)
fig.savefig(out/'training_comparison.png',dpi=160)
fig.savefig(out/'training_comparison.pdf')
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
