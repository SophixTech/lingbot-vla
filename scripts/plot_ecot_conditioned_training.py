"""Plot measured training logs; no evaluation metrics are inferred."""
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path('/var/tmp/bjtc_ecot1003_conditioned_20261009')
pattern = r'Step (\d+)/\d+, Epoch \d+, Loss ([\d.e+-]+), VLA_Loss ([\d.e+-]+), Depth_Loss ([\d.e+-]+), ECOT_Loss ([\d.e+-]+), GradNorm ([\d.e+-]+), LR ([\d.e+-]+), StepTime ([\d.e+-]+)s'
rows = np.array([[float(x) for x in row] for row in re.findall(pattern, (root/'train.log').read_text())])
assert len(rows)==10000 and np.array_equal(rows[:,0], np.arange(1,10001))
dest = Path(__file__).resolve().parents[1]/'docs/ecot_conditioned_training'
dest.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
for ax, col, title in zip(axes.flat, [1,2,4,5,6,7], ['Total loss','Action loss (L1 flow matching)','ECOT auxiliary loss (weighted)','Gradient norm','Learning rate','Logged step time (s)']):
    y=rows[:,col]
    ax.plot(rows[:,0],y,alpha=.2,lw=.6,color='#2878b5')
    ax.plot(rows[99:,0],np.convolve(y,np.ones(100)/100,mode='valid'),lw=1.6,color='#d45c28',label='100-step moving mean')
    ax.set(title=title,xlabel='Optimizer step'); ax.grid(alpha=.2)
    ax.legend(fontsize=8)
fig.suptitle('ECOT conditioned + visibility | 10,000 steps | 2026-10-09/10\nTraining metrics only; logged step time excludes some overhead',fontsize=14)
fig.savefig(dest/'training_curves.png',dpi=170)
fig.savefig(dest/'training_curves.pdf')
np.savetxt(dest/'metrics.csv',rows,delimiter=',',header='step,total_loss,action_loss,depth_loss,ecot_loss,grad_norm,lr,logged_step_seconds',comments='')
for col in [1,2,4]:
    print(col,'first100',rows[:100,col].mean(),'last100',rows[-100:,col].mean())
print(dest/'training_curves.png')
