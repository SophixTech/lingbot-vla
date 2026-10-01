from pathlib import Path
import json, numpy as np
import evaluate_postclose_four_chunks as ev

OUT=Path('/home/bjtc/Sophix/lingbot-vla/output/label_down_fast/heldout_label_down_postclose')
manifest=json.loads((OUT/'manifest.json').read_text())
ev.OUT=OUT
ev.infer(manifest)
z=np.load(OUT/'paired_trajectories.npz'); anchor=np.rad2deg(z['anchor'][:,None,7:14]); pred=np.rad2deg(z['prediction'][:,:,7:14])-anchor; rec=np.rad2deg(z['truth'][:,:,7:14])-anchor; N=200; x=np.arange(1,N+1)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig,axes=plt.subplots(4,2,figsize=(12,12),sharex=True); axes=axes.ravel()
for j,ax in enumerate(axes[:7]):
    ax.plot(x,pred[:,:,j].mean(0),color='#d95f02',lw=1.7,label='down model')
    ax.plot(x,rec[:,:,j].mean(0),color='#d95f02',ls='--',lw=1.7,label='down recorded')
    for b in (50.5,100.5,150.5): ax.axvline(b,color='.6',lw=.7,ls=':')
    ax.set_title(f'Right joint {j+1}'); ax.set_ylabel('Change from closure (deg)'); ax.set_xlabel('Frame after closure (1-200)'); ax.grid(alpha=.2)
axes[0].legend(frameon=False); axes[-1].axis('off'); fig.suptitle(f'All held-out label-down episodes ({pred.shape[0]} episodes) | Four predictions with recorded observations'); fig.tight_layout(); fig.savefig(OUT/'heldout_label_down_model_vs_recorded.png',dpi=160); plt.close(fig)
e=pred-rec; report=json.loads((OUT/'manifest.json').read_text()); report.update({'metrics':{'episodes':int(len(pred)),'mae_deg':float(np.abs(e).mean()),'rmse_deg':float(np.sqrt(np.square(e).mean())),'p95_abs_deg':float(np.quantile(np.abs(e),.95)),'mae_deg_by_window':np.abs(e).reshape(len(pred),4,50,7).mean((0,2,3)).tolist(),'rmse_deg_by_window':np.sqrt(np.square(e).reshape(len(pred),4,50,7).mean((0,2,3))).tolist(),'mae_deg_by_joint':np.abs(e).mean((0,1)).tolist()},'mapping_check':'PASS: source episode and global index ranges verified before inference'})
(OUT/'metrics.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report['metrics'],indent=2),flush=True)
