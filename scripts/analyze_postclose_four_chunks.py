"""Episode-level group contrasts on identical postclosure model/real windows."""
import json
import os
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

OUT=Path(os.environ.get('POSTCLOSE_OUT', '/home/bjtc/Sophix/lingbot-vla/output/marked_full0919/postclose_four_chunks_20260921'))


def bh(p):
    p=np.where(np.isfinite(p),p,1.);flat=p.ravel();order=np.argsort(flat)
    q=np.minimum.accumulate((flat[order]*len(flat)/np.arange(1,len(flat)+1))[::-1])[::-1]
    res=np.empty_like(flat);res[order]=np.minimum(q,1.)
    return res.reshape(p.shape)


def contrasts(x,labels,seed=9921,frames_per_window=50):
    """x: episode x 200 x 7. Frames never treated as independent samples."""
    a,b=x[labels],x[~labels]; n,m=len(a),len(b)
    windows=frames_per_window
    n_windows=x.shape[1]//windows
    delta=a.mean(0)-b.mean(0)
    rng=np.random.default_rng(seed)
    # Resample whole episodes, retaining all frames/joints/chunks together.
    wa=rng.multinomial(n,np.full(n,1/n),size=1000)/n
    wb=rng.multinomial(m,np.full(m,1/m),size=1000)/m
    boots=(wa@a.reshape(n,-1)-wb@b.reshape(m,-1)).reshape(1000,n_windows,windows,7)
    gap_boot=np.abs(boots).mean((2,3))
    # A global waveform mean-difference permutation test per 50-frame window.
    # Permute episode labels, not individual frames or joints.
    weights=np.empty((1999,n+m))
    for i in range(len(weights)):
        ix=rng.permutation(n+m);weights[i]= -1/m;weights[i,ix[:n]]=1/n
    null=(weights@np.concatenate([a,b]).reshape(n+m,-1)).reshape(1999,n_windows,windows,7)
    observed=np.square(delta.reshape(n_windows,windows,7)).mean((1,2))
    p=(1+(np.square(null).mean((2,3))>=observed).sum(0))/2000
    # Signed 50-frame means: 7 Welch tests per window, FDR over all 28 tests.
    am=a.reshape(n,n_windows,windows,7).mean(2);bm=b.reshape(m,n_windows,windows,7).mean(2)
    welch=stats.ttest_ind(am,bm,axis=0,equal_var=False)
    q=bh(welch.pvalue)
    sd=np.sqrt((am.var(0,ddof=1)+bm.var(0,ddof=1))/2)
    effect=(am.mean(0)-bm.mean(0))/np.maximum(sd,1e-12)
    rows=[]
    for k in range(n_windows):
        d=delta[k*windows:(k+1)*windows]
        rows.append(dict(window=k+1,frames_inclusive=[k*windows+1,(k+1)*windows],
            mean_abs_group_trajectory_difference_deg=float(np.abs(d).mean()),
            bootstrap_ci95_deg=np.quantile(gap_boot[:,k],[.025,.975]).tolist(),
            signed_mean_difference_by_joint_deg=d.mean(0).tolist(),
            mean_abs_difference_by_joint_deg=np.abs(d).mean(0).tolist(),
            rms_group_difference_deg=float(np.sqrt(np.square(d).mean())),
            endpoint_mean_abs_group_difference_deg=float(np.abs(d[-1]).mean()),
            functional_permutation_p=float(p[k]),functional_permutation_fdr_q=float(bh(p)[k]),
            window_mean_welch_p_by_joint=welch.pvalue[k].tolist(),
            window_mean_fdr_q_by_joint=q[k].tolist(),
            window_mean_standardized_difference_by_joint=effect[k].tolist()))
    return dict(up_episodes=n,down_episodes=m,windows=rows,
                four_window_mean_deg=float(np.abs(delta).mean()),
                four_window_mean_ci95_deg=np.quantile(gap_boot.mean(1),[.025,.975]).tolist()),delta


def main():
    manifest=json.loads((OUT/'manifest.json').read_text())
    z=np.load(OUT/'paired_trajectories.npz');labels=z['label_up']
    windows=int(manifest.get('execute_frames',50)); n_windows=z['prediction'].shape[1]//windows
    pred=np.rad2deg(z['prediction'][:,:,7:14]);truth=np.rad2deg(z['truth'][:,:,7:14]);anchor=np.rad2deg(z['anchor'][:,None,7:14])
    allz=np.load(OUT/'ground_truth.npz')
    results={'definition':'For each frame/joint take abs(mean(label_up)-mean(label_down)), then average over 50 frames and 7 right-arm joints. Four-window mean weights each window equally.',
             'relative_definition':'Subtract the same recorded action at first closure from model and truth, once per episode.',
             'uncertainty':'1000 whole-episode bootstrap resamples; 1999 whole-episode label permutations per window; FDR over 4 windows. Welch tests use one 50-frame mean per episode/joint, with FDR over 28 tests.',
             'dataset':manifest['dataset'], 'checkpoint':manifest['checkpoint'],
             'heldout_verification':manifest.get('heldout_verification'),
             'limitations':(['Non-regrasp eligible subset of held-out test episodes'] if manifest.get('heldout_verification') else ['Training episodes, not held-out evaluation'])+['Ground-truth observations refreshed every 50 frames; not autonomous closed-loop rollout','Closure is gripper command threshold proxy, not confirmed grasp','Initial scenes/states differ between label groups; this is not language-only causality']}
    deltas={}
    for name,x,l in [('model_absolute',pred,labels),('truth_absolute',truth,labels),
                     ('model_relative',pred-anchor,labels),('truth_relative',truth-anchor,labels),
                     ('all_eligible_truth_relative',np.rad2deg(allz['action'][:,:,7:14]-allz['anchor'][:,None,7:14]),allz['label_up'])]:
        results[name],deltas[name]=contrasts(x,l,frames_per_window=windows)
    error=pred-truth
    results['prediction_error']={}
    for name,mask in [('all',np.ones(len(labels),bool)),('label_up',labels),('label_down',~labels)]:
        e=error[mask].reshape(-1,n_windows,windows,7)
        results['prediction_error'][name]=dict(mae_deg_by_window=np.abs(e).mean((0,2,3)).tolist(),
                                               rmse_deg_by_window=np.sqrt(np.square(e).mean((0,2,3))).tolist(),
                                               episodes=int(mask.sum()),
                                               rmse_deg=float(np.sqrt(np.square(e).mean())),
                                               mae_deg_by_joint=np.abs(e).mean((0,1,2)).tolist(),
                                               mae_deg=float(np.abs(e).mean()))
    pd,td=deltas['model_relative'],deltas['truth_relative']
    results['group_contrast_match']=dict(rms_difference_deg=float(np.sqrt(np.square(pd-td).mean())),
        cosine_similarity=float(np.sum(pd*td)/np.sqrt(np.sum(pd**2)*np.sum(td**2))),
        mean_abs_ratio_by_window=[results['model_relative']['windows'][k]['mean_abs_group_trajectory_difference_deg']/results['truth_relative']['windows'][k]['mean_abs_group_trajectory_difference_deg'] for k in range(n_windows)])
    # A cheap causal hold-current-state baseline exposes the observation refresh contribution.
    manifest=json.loads((OUT/'manifest.json').read_text())
    states=np.stack([np.stack([np.load(OUT/'offline_predictions'/f"episode-{r['episode']:06d}-chunk-{k}.npz")['state']
                               for k in range(n_windows)]) for r in manifest['selected_rows']])
    hold=np.repeat(np.rad2deg(states[:,:,7:14]),windows,axis=1)
    results['hold_current_state_mae_deg_by_window']=np.abs(hold-truth).reshape(-1,n_windows,windows,7).mean((0,2,3)).tolist()
    results['hold_current_state_mae_deg']=float(np.abs(hold-truth).mean())
    results['episode_error_summary']={
        'median_mae_deg':float(np.median(np.abs(error).mean((1,2)))),
        'p95_mae_deg':float(np.quantile(np.abs(error).mean((1,2)),.95)),
        'max_mae_deg':float(np.abs(error).mean((1,2)).max())}
    with (OUT/'episode_errors.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['test_episode_index','source_episode_index','source_episode_id','label_up','mae_deg','rmse_deg','hold_state_mae_deg'])
        writer.writeheader()
        for i,r in enumerate(manifest['selected_rows']):
            writer.writerow(dict(test_episode_index=r['episode'],source_episode_index=r.get('source_episode_index',''),source_episode_id=r.get('source_episode_id',''),label_up=bool(labels[i]),mae_deg=float(np.abs(error[i]).mean()),rmse_deg=float(np.sqrt(np.square(error[i]).mean())),hold_state_mae_deg=float(np.abs(hold[i]-truth[i]).mean())))
    results['hold_current_state_relative_gap_deg_by_window']=np.abs((hold-anchor)[labels].mean(0)-(hold-anchor)[~labels].mean(0)).reshape(n_windows,windows,7).mean((1,2)).tolist()
    results['execute_frames'] = windows
    results['number_windows'] = n_windows
    results['definition'] = f'Absolute difference of label-group means, averaged across {windows} frames and seven joints per window.'
    results['uncertainty'] = f'Whole-episode bootstrap and permutations; FDR across {n_windows} windows or {7*n_windows} joint/window tests.'
    results['limitations'][1] = f'Recorded observations refreshed every {windows} frames; not autonomous closed-loop rollout'
    (OUT/'metrics.json').write_text(json.dumps(results,indent=2,allow_nan=False))

    fig,axes=plt.subplots(4,2,figsize=(12,12),sharex=True)
    x=np.arange(1,z['prediction'].shape[1]+1)
    for j,ax in enumerate(axes.ravel()[:7]):
        for label,color in [(True,'#1f77b4'),(False,'#d95f02')]:
            mask=labels==label
            for src,arr,style in [('model',pred-anchor,'-'),('recorded',truth-anchor,'--')]:
                a=arr[mask,:,j];mean=a.mean(0)
                ax.plot(x,mean,style,color=color,lw=1.5,label=f"{'up' if label else 'down'} {src}")
        for boundary in np.arange(windows+.5,z['prediction'].shape[1],windows):ax.axvline(boundary,color='.6',lw=.7,ls=':')
        ax.set_title(f'Right joint {j+1}');ax.set_ylabel('Change from closure (deg)');ax.set_xlabel('Frame after closure (1-200)');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=9,ncol=2,frameon=False);axes.ravel()[-1].axis('off')
    split_label='Held-out test' if manifest.get('heldout_verification') else 'Same'
    fig.suptitle(f'{split_label}: {len(labels)} non-regrasp episodes (up={labels.sum()}, down={(~labels).sum()})\n{n_windows} predictions; execute first {windows}/50 frames; recorded observations')
    fig.tight_layout();fig.savefig(OUT/'model_vs_recorded_trajectories.png',dpi=150);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for ax,mode in zip(axes,['absolute','relative']):
        for offset,src,color,label in [(-.18,'model','#756bb1','Model'),(.18,'truth','#31a354','Recorded')]:
            rr=results[f'{src}_{mode}'];means=np.array([r['mean_abs_group_trajectory_difference_deg'] for r in rr['windows']]);ci=np.array([r['bootstrap_ci95_deg'] for r in rr['windows']])
            pos=np.arange(n_windows)+offset
            ax.bar(pos,means,width=.34,color=color,label=label)
            # Bootstrap bands shown independently, without assuming symmetric intervals.
            ax.vlines(pos,ci[:,0],ci[:,1],color='.25',lw=1)
            for t,y in zip(pos,means):ax.text(t,y+.15,f'{y:.2f}',ha='center',fontsize=10)
        ax.set_xticks(np.arange(n_windows),[f'{i*windows+1}-{(i+1)*windows}' for i in range(n_windows)],rotation=45);ax.set_xlabel('Frames after first closure')
        ax.set_ylabel('Mean |up-group mean - down-group mean| (deg)')
        ax.set_title('Absolute joint targets' if mode=='absolute' else 'Change from same recorded closure pose')
        ax.grid(axis='y',alpha=.2);ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(OUT/'four_window_group_differences.png',dpi=160);plt.close(fig)
    print(json.dumps({k:results[k] for k in ['group_contrast_match','prediction_error']},indent=2))
    for mode in ['model_absolute','truth_absolute','model_relative','truth_relative','all_eligible_truth_relative']:
        print(mode,'gaps',[round(w['mean_abs_group_trajectory_difference_deg'],4) for w in results[mode]['windows']],
              'mean',results[mode]['four_window_mean_deg'],'p',[w['functional_permutation_p'] for w in results[mode]['windows']],flush=True)


if __name__=='__main__':main()
