#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

BASE = Path('/home/bjtc/Sophix/lingbot-vla/output')
FILES = {
    '5': BASE/'interpolation_vla_num_denoising_step_5/checkpoints/loss.jsonl',
    '10': BASE/'interpolation_vla/checkpoints/loss.jsonl',
    '20': BASE/'interpolation_vla_num_denoising_step_20/checkpoints/loss.jsonl',
}
COL = {'5':'#389e0d','10':'#1677ff','20':'#d4380d'}
data={k:[json.loads(x) for x in p.read_text().splitlines() if x.strip()] for k,p in FILES.items()}
n=min(7990, *(len(v) for v in data.values()))
x=np.arange(1,n+1); w=100
fig,ax=plt.subplots(3,1,figsize=(13,10),sharex=True,layout='constrained')
for k,rows in data.items():
    loss=np.array([r['loss'] for r in rows[:n]])
    grad=np.array([r['grad_norm'] for r in rows[:n]])
    mean=np.convolve(loss,np.ones(w)/w,mode='valid')
    gmean=np.convolve(grad,np.ones(w)/w,mode='valid')
    ax[0].plot(x[w-1:],mean,color=COL[k],lw=2,label=f'{k} steps')
    ax[1].plot(x,loss,color=COL[k],lw=.7,alpha=.65,label=f'{k} steps')
    ax[2].plot(x[w-1:],gmean,color=COL[k],lw=1.7,label=f'{k} steps')
ax[0].set_title(f'LingBot-VLA loss comparison at identical training steps (N={n})')
ax[0].set_ylabel('100-step mean loss')
ax[1].set_ylabel('Raw loss (symlog)')
ax[1].set_yscale('symlog',linthresh=0.1)
ax[2].set_ylabel('100-step mean grad norm')
ax[2].set_xlabel('Training step')
for a in ax:a.legend(ncol=3,fontsize=9)
out=BASE/'interpolation_vla_num_denoising_step_20/vla_5_10_20_shared_prefix.png'
fig.savefig(out,dpi=160,bbox_inches='tight')
print(out)
