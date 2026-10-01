"""Compare observed metrics; never connect historical gaps or excluded steps."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D

root=Path('/home/bjtc/Sophix/lingbot-vla/output/marked_full0919')
out=root/'comparison'
rows=[json.loads(l) for l in (root/'checkpoints/loss.jsonl').read_text().splitlines()]
with (out/'excluded_806_865_steps.csv').open() as f:
    mask=np.array([r['excluded_due_to_episode_806_or_865']=='False' for r in csv.DictReader(f)])
history=json.loads((out/'interpolation_recovered_points.json').read_text())['points']
assert mask.sum()==6515 and len(history)==370
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':11})
blue,orange='#1878B4','#E66A20'
x=np.arange(1,10001)
fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True)
fig.subplots_adjust(top=.83,bottom=.11,left=.10,right=.98,hspace=.25)
fig.suptitle('marked_full0919 与 interpolation_vla 训练指标对比',fontsize=18,y=.98)
fig.text(.5,.943,'本轮：排除含 episode 806、865 的批次，保留 6515 步；历史：恢复 370 个实际记录点',ha='center',fontsize=11)
fig.legend(handles=[Line2D([0],[0],color=blue,lw=2,label='marked_full0919：筛选后的逐步指标'),
                    Line2D([0],[0],color=blue,lw=2,ls='--',label='marked_full0919：每100原始步窗口均值'),
                    Line2D([0],[0],color=orange,lw=1.2,marker='o',ms=4,label='interpolation_vla：已有逐步记录')],
           loc='upper center',bbox_to_anchor=(.5,.922),ncol=1,frameon=False,fontsize=10)
for ax,key,label in zip(axes,['loss','grad_norm','step_time'],['训练 Loss（归一化 L1；对数轴）','梯度范数','单步计算耗时（秒，不含数据等待）']):
    raw=np.array([r[key] for r in rows]);raw[~mask]=np.nan
    old=np.full(10000,np.nan)
    for r in history:old[r['step']-1]=r[key]
    ax.plot(x,raw,color=blue,lw=.6,alpha=.32)
    ax.scatter(x[mask],raw[mask],s=2,color=blue,alpha=.12,rasterized=True)
    centers=[];means=[]
    for start in range(0,10000,100):
        centers.append(start+50.5);means.append(np.nanmean(raw[start:start+100]))
    ax.plot(centers,means,color=blue,lw=1.9,ls='--')
    ax.plot(x,old,color=orange,lw=.9,alpha=.9)
    present=np.isfinite(old)
    ax.scatter(x[present],old[present],s=10,color=orange,alpha=.8,zorder=3)
    ax.set_ylabel(label,fontsize=11);ax.grid(alpha=.2);ax.set_xlim(0,10100)
    ax.spines[['top','right']].set_visible(False)
axes[0].set_yscale('log')
axes[0].set_yticks([.1,.15,.2,.3,.5,1],labels=['0.10','0.15','0.20','0.30','0.50','1.00'])
axes[0].minorticks_off()
axes[-1].set_xlabel('原训练优化步数（未重新编号）')
fig.text(.1,.035,'橙色缺失区间留空；两组统计口径、归一化不同，此图用于比较趋势，不能直接证明任务成功率优劣。',fontsize=10)
fig.savefig(out/'training_comparison_filtered.png',dpi=180)
fig.savefig(out/'training_comparison_filtered.pdf')
print(out/'training_comparison_filtered.png')
