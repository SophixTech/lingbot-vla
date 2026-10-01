"""Compare original, empty and conditional prompts on fixed episodes/noise."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_postclose_four_chunks import OUT as BASE, contrasts

OUT=BASE/'prompt_ablation'
MODES=['annotated','empty','conditional']
NAMES={'annotated':'Original label-specific prompt','empty':'Empty prompt','conditional':'Shared conditional prompt'}


def main():
    base=np.load(BASE/'paired_trajectories.npz')
    truth=np.rad2deg(base['truth'][:,:,7:14]);anchor=np.rad2deg(base['anchor'][:,None,7:14]);labels=base['label_up']
    preds={'annotated':np.rad2deg(base['prediction'][:,:,7:14])}
    for mode in MODES[1:]:
        z=np.load(OUT/mode/'paired_trajectories.npz')
        for key in ['truth','anchor','label_up']:np.testing.assert_array_equal(z[key],base[key])
        preds[mode]=np.rad2deg(z['prediction'][:,:,7:14])
    result={'protocol':'128 identical non-regrasp episodes (64 per label); 4 x 50 frames from first closure; same noise seed per observation across modes; fresh recorded images/state each chunk.',
            'difference_definition':'Mean absolute difference between label group mean trajectories, over 50 frames and 7 right-arm joints; subtract same recorded closure action once per episode.',
            'empty_prompt':'Empty task string with model boundary tokens retained; not removal of entire language branch.',
            'conditional_prompt':json.loads((OUT/'conditional/prompt_tokenization.json').read_text()),
            'limitations':['In-training-set offline evaluation; not closed loop','Both images and states remain available; cannot isolate visual recognition','Every 50 frames recorded state restores true behavior context','Empty and conditional prompts are not the exact posttraining prompts'],
            'modes':{}}
    result['truth'],td=contrasts(truth-anchor,labels)
    base_error=np.abs(preds['annotated']-truth).reshape(128,4,50,7).mean((2,3))
    for mode,pred in preds.items():
        gap,delta=contrasts(pred-anchor,labels)
        e=pred-truth;ae=np.abs(e);row={'group_contrast':gap,'by_group':{},
            'group_contrast_rmse_vs_truth_deg':float(np.sqrt(np.square(delta-td).mean())),
            'mean_absolute_prediction_change_vs_original_deg':float(np.abs(pred-preds['annotated']).mean())}
        for group,mask in [('all',np.ones(128,bool)),('label_up',labels),('label_down',~labels)]:
            a=ae[mask].reshape(-1,4,50,7);err=e[mask].reshape(-1,4,50,7)
            changes=a.mean((2,3))-base_error[mask]
            rng=np.random.default_rng(20260921)
            ix=rng.integers(0,len(a),size=(4000,len(a)))
            boot=changes[ix].mean(1)
            row['by_group'][group]={'mae_deg':float(a.mean()),'rmse_deg':float(np.sqrt(np.square(err).mean())),
                'mae_by_window_deg':a.mean((0,2,3)).tolist(),'per_joint_mae_deg':a.mean((0,1,2)).tolist(),
                'paired_mae_change_vs_original_by_window_deg':changes.mean(0).tolist(),
                'paired_mae_change_ci95_by_window_deg':np.quantile(boot,[.025,.975],axis=0).T.tolist(),
                'paired_overall_mae_change_ci95_deg':np.quantile(boot.mean(1),[.025,.975]).tolist()}
        row['mean_group_trajectory_error_deg']=float(np.mean([
            np.abs(pred[labels==l].mean(0)-truth[labels==l].mean(0)).mean() for l in [True,False]]))
        result['modes'][mode]=row
    (OUT/'comparison_metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    x=np.arange(1,201)
    # Two columns separate interventions; consistent colors separate actual labels.
    fig,axes=plt.subplots(7,2,figsize=(13,19),sharex=True,sharey='row')
    for col,mode in enumerate(MODES[1:]):
        for j in range(7):
            ax=axes[j,col]
            for lab,color in [(True,'#1f77b4'),(False,'#d95f02')]:
                mask=labels==lab
                for array,style,name in [(preds[mode],'-','prediction'),(truth,'--','recorded')]:
                    ax.plot(x,(array-anchor)[mask,:,j].mean(0),style,color=color,lw=1.5,label=f"{'up' if lab else 'down'} {name}")
            for boundary in [50.5,100.5,150.5]:ax.axvline(boundary,color='.6',ls=':',lw=.7)
            ax.set_title(f'{NAMES[mode]} | Right joint {j+1}');ax.set_ylabel('Change from closure (deg)');ax.grid(alpha=.2)
            if j==0:ax.legend(ncol=2,frameon=False,fontsize=9)
            if j==6:ax.set_xlabel('Frame after closure (1-200)')
    fig.suptitle('Same observations and noise | Solid: model; dashed: recorded | Blue: label up; orange: label down')
    fig.tight_layout(rect=(0,0,1,.975));fig.savefig(OUT/'empty_and_conditional_vs_truth.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    colors={'annotated':'#3274a1','empty':'#d95f02','conditional':'#259b65'}
    for mode in MODES:
        r=result['modes'][mode]
        axes[0].plot(range(1,5),r['by_group']['all']['mae_by_window_deg'],'o-',color=colors[mode],label=NAMES[mode])
        axes[1].plot(range(1,5),[w['mean_abs_group_trajectory_difference_deg'] for w in r['group_contrast']['windows']],'o-',color=colors[mode],label=NAMES[mode])
    axes[1].plot(range(1,5),[w['mean_abs_group_trajectory_difference_deg'] for w in result['truth']['windows']],'k--',label='Recorded')
    for ax in axes:
        ax.set_xticks(range(1,5),['1-50','51-100','101-150','151-200']);ax.set_xlabel('Frames after closure');ax.grid(alpha=.2);ax.legend(fontsize=9,frameon=False)
    axes[0].set_ylabel('Pointwise right-arm MAE (deg)');axes[0].set_title('Prediction accuracy')
    axes[1].set_ylabel('Mean absolute group-mean difference (deg)');axes[1].set_title('Label up / down trajectory separation')
    fig.tight_layout();fig.savefig(OUT/'accuracy_and_group_separation.png',dpi=160);plt.close(fig)
    # Both labels side by side; all prompt variants against true trajectory.
    fig,axes=plt.subplots(7,2,figsize=(13,19),sharex=True,sharey='row')
    for col,lab in enumerate([True,False]):
        mask=labels==lab
        for j in range(7):
            ax=axes[j,col]
            ax.plot(x,(truth-anchor)[mask,:,j].mean(0),'k--',lw=1.6,label='Recorded')
            for mode in MODES:ax.plot(x,(preds[mode]-anchor)[mask,:,j].mean(0),color=colors[mode],lw=1.3,label=NAMES[mode])
            for boundary in [50.5,100.5,150.5]:ax.axvline(boundary,color='.6',ls=':',lw=.7)
            ax.set_title(f"Label {'up' if lab else 'down'} | Right joint {j+1}");ax.set_ylabel('Change from closure (deg)');ax.grid(alpha=.2)
            if j==0:ax.legend(fontsize=8,frameon=False)
            if j==6:ax.set_xlabel('Frame after closure (1-200)')
    fig.tight_layout();fig.savefig(OUT/'all_prompts_by_label.png',dpi=140);plt.close(fig)
    for mode,r in result['modes'].items():
        print(mode,json.dumps({'error':r['by_group'],'gaps':[w['mean_abs_group_trajectory_difference_deg'] for w in r['group_contrast']['windows']],
              'q':[w['functional_permutation_fdr_q'] for w in r['group_contrast']['windows']],
              'change_vs_original':r['mean_absolute_prediction_change_vs_original_deg']},indent=2),flush=True)


if __name__=='__main__':main()
